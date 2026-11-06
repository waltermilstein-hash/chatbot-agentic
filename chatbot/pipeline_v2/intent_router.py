# =============================================================================
# =============================================================================
"""
IntentRouter — intent-decision layer of the hybrid chatbot.

Classifies each query BEFORE generating the answer and decides:
    needs_rag              -> query the document store?
    needs_context          -> use conversational history?
    allow_general_llm      -> allow the model's general knowledge?
    category               -> general_chat | general_knowledge | document_query |
                              followup | comparison | summary | analysis |
                              generation | hybrid | tool_action

Strategy: cheap heuristic first (regex) and, if ambiguous, a nano LLM that
returns JSON. Always conservative: when in doubt, needs_rag=True (don't lose RAG).

NOTE: the regex patterns below match the user's input in SPANISH; their literals
must stay in Spanish so recognition keeps working.
"""
import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

from config import Constants

logger = logging.getLogger(__name__)

# General-knowledge signals (do not require the company's documents).
_GENERAL_PATTERNS = re.compile(
    r"\b(qu[eé] es|qu[eé] son|explic[aá]|expl[ií]came|c[oó]mo funciona|"
    r"defin[eí]|diferencia entre|para qu[eé] sirve|ejemplo de|"
    r"qu[eé] significa|resum[ií]me el concepto|traduc[eí])\b",
    re.IGNORECASE,
)
# Document/corporate query signals (force RAG).
_DOC_PATTERNS = re.compile(
    r"\b(nuestro|nuestra|nuestros|nuestras|mi empresa|la empresa|pol[ií]tica|"
    r"procedimiento|contrato|factura|documento|este archivo|el pdf|seg[uú]n el|"
    r"dice el|en el documento|cl[aá]usula|expediente|cv|curr[ií]cul[ou]m?|"
    r"informe|reporte|recibo|comprobante|acta|legajo|memo|propuesta|archivo|"
    r"pasaje|ticket|receta|comprobantes?)\b",
    re.IGNORECASE,
)
# "summarize / analyze / what does <something> say" signals: in a document chatbot
# these almost always refer to an indexed document -> RAG (if there is no doc, the
# orchestrator does a generative fallback anyway).
_SUMMARY_PATTERNS = re.compile(
    r"\b(resum[ií]|res[uú]me|res[uú]mime|resumime|sintetiz[aá]|analiz[aá]|"
    r"qu[eé] dice|de qu[eé] trata|sobre qu[eé] trata|puntos? clave|extrae|extra[eé]|"
    r"qu[eé] contiene|qu[eé] informaci[oó]n|hallazgos?|conclusi[oó]n|conclusiones|"
    r"recomienda|recomendaci[oó]n|objetivo del|prop[oó]sito del)\b",
    re.IGNORECASE,
)
# "is X mentioned? / does it appear? / is it listed? / is it cited? / how many sources?":
# in a document chatbot these are questions ABOUT the document content -> RAG.
# These phrasings don't hit _DOC_PATTERNS (they don't name "documento/informe") but
# clearly interrogate the indexed material, which is why before they slipped to the
# weak LLM and ended up as "I don't see the document".
_DOC_MENTION_PATTERNS = re.compile(
    r"(\bse\s+mencionan?\b|\bmencionan?\b|\bse\s+nombran?\b|\baparecen?\b|"
    r"\bfiguran?\b|\bse\s+citan?\b|\bcitan?\b|\bse\s+incluyen?\b|"
    r"\bse\s+habla\s+de\b|\bse\s+refiere\b|\bse\s+identific[oó]\b|"
    r"\bse\s+indic[aó]\b|\bcu[aá]nt[ao]s?\s+(fuentes|veces|comparadores|productos|"
    r"supermercados|p[aá]ginas|secciones|riesgos|fuentes?\s+web)\b)",
    re.IGNORECASE,
)
# Greetings / chit-chat.
_CHAT_PATTERNS = re.compile(
    r"^\s*(hola|buenas|buen d[ií]a|gracias|ok|dale|c[oó]mo est[aá]s|qui[eé]n sos|qui[eé]n eres)\b",
    re.IGNORECASE,
)
# References to the previous conversation.
_FOLLOWUP_PATTERNS = re.compile(
    r"\b(y (el|la|los|las|eso|esto)|lo anterior|ese documento|la segunda|"
    r"el primero|continu[aá]|y qu[eé] pasa|respecto a eso)\b",
    re.IGNORECASE,
)
# Content generation (does not require RAG unless internal data is requested).
_GENERATION_PATTERNS = re.compile(
    r"\b(escrib[ei]|redact[aá]|gener[aá]|arm[aá]|cre[aá]|hac[eé] un|dame un ejemplo|"
    r"compon[eé]|borrador de|propon[eé] un)\b",
    re.IGNORECASE,
)
# Proper noun / entity: a Capitalized word (or acronym, or quoted term) that is NOT
# the start of the question. "What is Precios Claros?" names a document entity -> it is
# NOT general trivia even if it says "what is"; it is deferred to the LLM (with document
# context) instead of being cut to general_knowledge.
_ENTITY_HINT = re.compile(
    r"[\"'«“].+?[\"'»”]"                                   # quoted term
    r"|\b[A-ZÁÉÍÓÚÑ]{2,}\b"                                # uppercase acronym (DemoPost->no; BOE/CV yes)
    r"|(?<=\w\s)[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+"                   # non-initial Capitalized word
    r"|(?<=\w\s)[A-Z][a-z]*[A-Z]\w*",                      # CamelCase (DemoPost, Comparaya)
)

def _has_entity(q: str) -> bool:
    # Strip the leading question word so "Qué/Cuál" isn't counted as an entity.
    body = re.sub(r"^\s*¿?\s*(qu[eé]|cu[aá]l|cu[aá]les|c[oó]mo|qui[eé]n)\b", "", q, flags=re.IGNORECASE)
    return bool(_ENTITY_HINT.search(body))


@dataclass
class IntentDecision:
    category: str = "document_query"
    needs_rag: bool = True
    needs_context: bool = False
    needs_tools: bool = False
    allow_general_llm: bool = True
    confidence: float = 0.5
    source: str = "heuristic"

    def to_dict(self) -> dict:
        return asdict(self)


_ROUTER_SYSTEM = (
    "You are the intent router of a DOCUMENT CHATBOT: the user talks about "
    "a collection of already-loaded, indexed documents (reports, contracts, CVs, "
    "invoices, etc.). The chatbot's goal is to answer ABOUT that material. "
    "Classify the query and answer ONLY with valid JSON with these keys: "
    '{"category": "general_chat|general_knowledge|document_query|followup|memory|comparison|summary|analysis|generation|hybrid|tool_action", '
    '"needs_rag": true|false, "needs_context": true|false, "needs_tools": true|false, '
    '"allow_general_llm": true|false}.\n'
    "MAIN RULE — needs_rag=true (query the documents) for ANY question "
    "that could be answered with the loaded material: questions about its content, facts, "
    "names, entities, figures or dates; whether something is mentioned/appears/is listed/is cited; whether "
    "something exists in the material; how many times/sources/items there are; summaries, conclusions, "
    "findings, objectives, recommendations, risks; and every question whose subject is not "
    "obvious world trivia. When in DOUBT, needs_rag=true.\n"
    "needs_rag=false ONLY in these clear cases: (a) greetings and chit-chat ('hola', 'gracias', "
    "'cómo estás', 'quién sos'); (b) general world knowledge unrelated to business "
    "documents (e.g. 'capital of France', how much is 15x3, who won the "
    "World Cup, the weather); (c) requests to generate generic content that do not depend on the "
    "documents; (d) META-CONVERSATION: questions about THE CONVERSATION ITSELF, not about the "
    "documents (what was asked/said/discussed before, what was my first/last question, "
    "summarize/repeat the conversation, what we were talking about) -> category='memory', "
    "needs_rag=false, needs_context=true (answered from the HISTORY, not with documents). "
    "NOTE: 'summarize the document/report' is summary->RAG; 'summarize the chat/what we discussed' "
    "is memory. An unknown proper noun (a company, product, system or acronym) is NOT "
    "world trivia: treat it as possible document content -> needs_rag=true.\n"
    "needs_context=true if it references the previous conversation (this, the previous one, that document)."
)


class IntentRouter:
    def __init__(self, config=None):
        self.config = config
        # gpt-5.4-mini (not nano): the nano couldn't tell a question about the
        # document content ("is X mentioned?") from general trivia and returned
        # needs_rag=false, skipping RAG. Same criterion as the structured
        # chatbot's router (ChatbotService._decide_route_llm).
        self.model = getattr(Constants, "GPT_5_4_MINI", "gpt-5.4-mini")

    def _heuristic(self, query: str) -> Optional[IntentDecision]:
        q = (query or "").strip()
        if not q:
            return IntentDecision(category="general_chat", needs_rag=False, allow_general_llm=True, confidence=0.9, source="heuristic")
        if _CHAT_PATTERNS.search(q) and len(q) < 40:
            return IntentDecision(category="general_chat", needs_rag=False, needs_context=True, allow_general_llm=True, confidence=0.85, source="heuristic")
        doc = bool(_DOC_PATTERNS.search(q))
        gen = bool(_GENERAL_PATTERNS.search(q))
        follow = bool(_FOLLOWUP_PATTERNS.search(q))
        generation = bool(_GENERATION_PATTERNS.search(q))
        summary = bool(_SUMMARY_PATTERNS.search(q))
        mention = bool(_DOC_MENTION_PATTERNS.search(q))
        if doc:
            return IntentDecision(category="document_query", needs_rag=True, needs_context=follow, allow_general_llm=True, confidence=0.85, source="heuristic")
        # "is X mentioned? / does it appear? / how many sources?": they interrogate the
        # document content -> RAG (unless it is clearly a general concept).
        if mention and not gen:
            return IntentDecision(category="document_query", needs_rag=True, needs_context=follow, allow_general_llm=True, confidence=0.8, source="heuristic")
        # "summarize / analyze / what does ... say" (not a general concept) -> documental.
        if summary and not gen:
            return IntentDecision(category="summary", needs_rag=True, needs_context=follow, allow_general_llm=True, confidence=0.8, source="heuristic")
        if generation and not doc:
            return IntentDecision(category="generation", needs_rag=False, needs_context=follow, allow_general_llm=True, confidence=0.75, source="heuristic")
        if gen and not doc:
            # "what is X?" with X = entity/proper noun ("Precios Claros", DemoPost)
            # is NOT trivia: it may be document content -> let the LLM decide.
            if _has_entity(q):
                return None
            return IntentDecision(category="general_knowledge", needs_rag=False, needs_context=follow, allow_general_llm=True, confidence=0.75, source="heuristic")
        if follow and not doc and not gen:
            return IntentDecision(category="followup", needs_rag=True, needs_context=True, allow_general_llm=True, confidence=0.6, source="heuristic")
        return None  # ambiguous -> LLM

    async def classify(self, query: str, history: Optional[List[Dict]] = None) -> IntentDecision:
        h = self._heuristic(query)
        if h is not None:
            return h
        # Nano LLM fallback (JSON).
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=Constants.KEY_OPEN_AI)
            resp = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _ROUTER_SYSTEM},
                    {"role": "user", "content": query[:1500]},
                ],
                # GPT-5 (reasoner) spends tokens on "reasoning"; with a small
                # budget the text output comes back empty. Give it headroom.
                max_completion_tokens=1000,
                reasoning_effort="low",
            )
            raw = (resp.choices[0].message.content or "").strip()
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("response without JSON")
            data = json.loads(raw[start:end + 1])
            return IntentDecision(
                category=data.get("category", "document_query"),
                needs_rag=bool(data.get("needs_rag", True)),
                needs_context=bool(data.get("needs_context", False)),
                needs_tools=bool(data.get("needs_tools", False)),
                allow_general_llm=bool(data.get("allow_general_llm", True)),
                confidence=0.7, source="llm",
            )
        except Exception as exc:  # noqa: BLE001 - on failure, conservative: RAG
            logger.warning("IntentRouter LLM failed (%s); fallback to document_query", exc)
            return IntentDecision(source="fallback")
