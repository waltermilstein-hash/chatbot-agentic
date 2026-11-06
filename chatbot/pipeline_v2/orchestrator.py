# =============================================================================
# =============================================================================
"""
RagOrchestrator — Refactored orchestrator of the legal chatbot.

Usage:
    pipeline = RagOrchestrator(config, redis_client)
    result = await pipeline.query(
        query="What is the termination notice period?",
        user="dr_garcia",
        company="estudio_perez",
        conversation_id="conv_abc123",
        conversation_history=[                     # Redis history (optional)
            {"role": "user",      "content": "What does the contract say?"},
            {"role": "assistant", "content": "The contract states..."},
        ],
    )
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

import weaviate
from openai import AsyncOpenAI

from config.Constants import KEY_OPEN_AI, MODEL_ROUTER, GPT_5_4_MINI, MODEL_COREFERENCE
from chatbot.pipeline_v2.intent_router import IntentRouter
from chatbot.pipeline_v2.response_modes import build_system_prompt, normalize_mode
from chatbot.pipeline_v2.context_manager import ContextManager, ConversationState
from chatbot.pipeline_v2.llm_provider import get_provider
from chatbot.pipeline_v2.filtering.chunk_filter import ReflexiveChunkFilter
from framework.Export.intent import detect_export, wants_new_content
from framework.Export.slides import build_slide_outline
from framework.Export.exporters import to_docx, to_pdf, to_xlsx, to_pptx, to_md, MIME_TYPES, EXTENSIONS
from chatbot.pipeline_v2.search.vector_search import VectorSearchEngine, SearchParams
from framework.ConfigurationManager.ConfigurationManagerSelected import ConfigurationManagerSelected

logger = logging.getLogger(__name__)

# Elliptical follow-up: short/affirmative reply, selection, or — and this is what
# matters so the "and what else?" cases work like Claude/GPT — the family of
# CONTINUATION/EXPANSION ("what else", "tell me more", "expand", "go on", "go deeper",
# "and the technical part", "explain that", "more"…). Recognizing them by REGEX keeps the
# documental thread WITHOUT depending on the coreference LLM getting it right.
# Tolerates a "¿/¡" opening, an initial connector "y/pero/entonces" and a "?/!/…" ending.
# NOTE: the pattern matches the user's Spanish input, so its literals stay in Spanish.
_SHORT_FOLLOWUP = re.compile(
    r"^[¿¡\s]*"
    r"(y|pero|entonces|ah|ok|dale)?\s*"
    r"("
    # — affirmations / confirmation —
    r"s[ií]|sip|sii+|dale|ok(ay)?|oka|de una|hazlo|hacelo|hac[eé]lo|por favor|porfa|"
    r"claro|perfecto|obvio|afirmativo|correcto|exacto|as[ií] es|"
    # — option selection —
    r"el (primero|segundo|tercero|cuarto|breve|ejecutivo|profesional|[uú]ltimo)|"
    r"la (primera|segunda|tercera)|opci[oó]n \d+|la \d+|los dos|las dos|ambas|ambos|"
    r"ese|esa|eso|"
    # — continuation / expansion (NEW) —
    r"qu[eé] m[aá]s( pod[eé]s (decir|contar|agregar| a[ñn]adir)| sab[eé]s| hay| ten[eé]s| info(rmaci[oó]n)?)?|"
    r"algo m[aá]s|qu[eé] otra cosa|"
    r"(cont[aá]me|cont[aá]nos|cu[eé]ntame|cu[eé]ntanos|decime|dec[ií]) m[aá]s|"
    r"dame m[aá]s( detalles?| info(rmaci[oó]n)?| datos)?|dale m[aá]s|"
    r"m[aá]s detalles?|m[aá]s info(rmaci[oó]n)?|un poco m[aá]s|"
    r"ampli[aá]( eso| m[aá]s| la info(rmaci[oó]n)?)?|ampl[ií]e|amplia?lo|ampli[aá]me|ampl[ií]ame|"
    r"desarroll[aá]( eso| m[aá]s)?|detall[aá]( eso| m[aá]s)?|"
    r"segu[ií]( contando| con eso| ah[ií])?|sigue|seguime contando|continu[aá]|cont[ií]nua|"
    r"profundiz[aá]( en eso| m[aá]s)?|profundiza|"
    r"explic[aá]( eso| m[aá]s| mejor)?|explica( eso)?|explic[aá]me( eso)?|expl[ií]came( eso)?|"
    r"(y )?lo (t[eé]cnico|legal|econ[oó]mico|financiero|jur[ií]dico|pr[aá]ctico|dem[aá]s|otro)|"
    r"m[aá]s|"
    r"s[ií][, ]+(por favor|dale|gracias|hacelo|obvio)"
    r")"
    r"[\s.!?…]*$",
    re.IGNORECASE,
)


# SELECTION question over an EXPLICIT set ("which of the candidates…?",
# "which of the documents…?", "any of the invoices…?"). It is SELF-CONTAINED: the subject
# ("the candidates") already names the WHOLE collection, not the entities just
# mentioned. Rewriting these questions by coreference CONTAMINATES them — the rewriter
# tends to inject/enumerate the previous subjects ("… including Sirena and Dorian …"),
# shrinking the universe and leaving out the correct candidate. That's why they are NOT rewritten.
# Anaphoric collectives are excluded ("the two", "the mentioned ones", "the previous ones"),
# which DO refer to what was already said and need context resolution.
# NOTE: the pattern matches the user's Spanish input, so its literals stay in Spanish.
_COLLECTIVE_SELECTION = re.compile(
    r"^[¿¡\s]*"
    r"(qui[eé]n(es)?|cu[aá]l(es)?|cu[aá]nt[oa]s?|algun[oa]s?|cada\s+un[oa])\s+"
    r"de\s+(los|las)\s+"
    r"(?!dos\b|tres\b|cuatro\b|cinco\b|ambos\b|ambas\b|mism\w+|anteriores\b|"
    r"mencionad\w+|citad\w+|dich[oa]s\b|referid\w+|nombrad\w+|primer[oa]s?\b|"
    r"[uú]ltim[oa]s?\b|dos\b|restantes\b)"
    r"[a-záéíóúñ]{3,}",
    re.IGNORECASE,
)


def _should_stay_documental(is_followup: bool, last_mode) -> bool:
    """A follow-up (query rewritten by context, or a short affirmation/continuation)
    over an active DOCUMENTAL thread should keep using RAG — not fall back to general
    knowledge ('I can help you with many things')."""
    return bool(is_followup and last_mode == "documental")


# System prompt: kept separate from the user message for better instruction-following (fix 11)
_SYSTEM_PROMPT = (
    "You are {personality}. Your tone is {tone}. "
    "Answer ONLY based on the provided document excerpts. "
    "If the information is not in the documents, answer: "
    "\"I couldn't find information about this point in the available documents.\" "
    "Always cite the source document and page of each statement. "
    "Do not elaborate or infer information that is not explicitly in the documents. "
    "Format the answer in clean HTML."
)

_USER_PROMPT = (
    "REFERENCE DOCUMENTS:\n{context}\n\n"
    "LEGAL QUERY:\n{query}\n\n"
    "ANSWER (HTML, citing sources):"
)


def _sse_text(event: str) -> str:
    """Extract the text from a 'chunk'/'content' SSE event (to accumulate the answer)."""
    if not event or not event.startswith("data: "):
        return ""
    try:
        evt = json.loads(event[6:])
        if evt.get("type") in ("chunk", "content"):
            return evt.get("text") or evt.get("message") or ""
    except Exception:
        return ""
    return ""


def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


def _extract_table_rows(html: str, max_rows: int = 200) -> List[Dict[str, Any]]:
    """Extract the FIRST HTML table of an answer as a list of dicts
    (header→cell). It is the 'join point' to analyze/plot the
    RESULTS SHOWN in the bot (follow-ups like «plot that»/«analyze the
    results»). Defensive: [] on any problem or if there is no table."""
    if not html or "<table" not in html.lower():
        return []
    try:
        from bs4 import BeautifulSoup
    except Exception:  # noqa: BLE001
        return []
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:  # noqa: BLE001
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:  # noqa: BLE001
            return []
    table = soup.find("table")
    if table is None:
        return []
    trs = table.find_all("tr")
    if not trs:
        return []
    # Headers: first row with <th>, or the first row of <td>.
    header_cells = trs[0].find_all(["th", "td"])
    headers = [c.get_text(" ", strip=True) or f"col{i+1}" for i, c in enumerate(header_cells)]
    if not headers:
        return []
    rows: List[Dict[str, Any]] = []
    for tr in trs[1:]:
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        vals = [c.get_text(" ", strip=True) for c in cells]
        row = {headers[i] if i < len(headers) else f"col{i+1}": vals[i] for i in range(len(vals))}
        if any(v for v in row.values()):
            rows.append(row)
        if len(rows) >= max_rows:
            break
    return rows


def _update_latest_tool_output(collection, mongo_filter: dict, tool_output: dict):
    """Write ``tool_output`` to the MOST RECENT doc (by ``timestamp``) that
    matches ``mongo_filter`` in the ``chatbot_memory`` collection. Synchronous — runs
    in a thread. Best-effort: returns the updated doc or None."""
    return collection.find_one_and_update(
        mongo_filter,
        {"$set": {"tool_output": tool_output}},
        sort=[("timestamp", -1)],
    )


def _parse_tool_sse(event: str):
    """Parse an SSE event from the tool router and return ``(kind, payload)``.

    payload is: the text (content/chunk), the chart dict, the
    {format,title,code} diagram dict, the report dict, or the attachment's
    file_id. Defensive: on any problem it returns ``(None, None)``."""
    if not event or not event.startswith("data: "):
        return None, None
    try:
        evt = json.loads(event[6:])
    except Exception:
        return None, None
    etype = evt.get("type")
    if etype in ("content", "chunk"):
        return "text", (evt.get("text") or evt.get("message") or "")
    if etype == "chart":
        return "chart", evt.get("chart")
    if etype == "diagram":
        return "diagram", {
            "format": evt.get("format"),
            "title": evt.get("title"),
            "code": evt.get("code"),
        }
    if etype == "report":
        return "report", evt.get("report")
    if etype == "attachment":
        return "attachment", evt.get("file_id")
    # Structured sources (e.g. SQL): captured to PERSIST them with the
    # turn (they survive reloading the chat).
    if etype == "sources":
        return "sources", evt.get("sources")
    return None, None


def _is_error_event(event: str) -> bool:
    if not event or not event.startswith("data: "):
        return False
    try:
        return json.loads(event[6:]).get("type") == "error"
    except Exception:
        return False


# "No evidence" sentinels emitted by the legacy pipeline as content.
# NOTE: these must match the strings the legacy RagEngine actually emits. Kept as a
# Spanish+English superset so detection works whether or not the legacy output is translated.
_NOT_FOUND_SENTINELS = (
    "información no encontrada",
    "no encontré información",
    "no pude encontrar información relevante",
    "no se encontró información",
    "no encontré información sobre este punto",
    "information not found",
    "i couldn't find information",
    "i could not find relevant information",
    "no information was found",
    "i couldn't find information about this point",
)


def _looks_like_not_found(text: str) -> bool:
    low = (text or "").lower()
    return any(s in low for s in _NOT_FOUND_SENTINELS)


class RagOrchestrator:
    """
    Modular RAG pipeline for the legal domain.

    Fix 5: VectorSearchEngine and ReflexiveChunkFilter are instantiated once
    in __init__ — they are not recreated on every query.
    """

    def __init__(self, config, redis_client=None):
        # Accepts a dict, an AppConfig object, or a string (path to YAML)
        if isinstance(config, dict):
            raw = config
        elif isinstance(config, str):
            raw = ConfigurationManagerSelected(config).get_all_configs()
        else:
            # AppConfig object — supports bracket notation just like a dict
            raw = config

        self.config = raw

        # Extract key values — AppConfig uses brackets, a dict uses .get()
        try:
            if isinstance(raw, dict):
                self.weaviate_url = raw.get("weaviate", {}).get("url", "http://localhost:8084")
                self.openai_key = raw.get("openai_key", {}).get("key", KEY_OPEN_AI)
            else:
                self.weaviate_url = raw['weaviate']['url']
                self.openai_key = raw['openai_key']['key']
        except Exception:
            self.weaviate_url = "http://localhost:8084"
            self.openai_key = KEY_OPEN_AI

        self.redis_client = redis_client

        self.async_openai = AsyncOpenAI(api_key=self.openai_key)
        self._weaviate_client = None  # lazy — deferred connection

        # Fix 5: heavy objects created once, not on every query
        self._search_engine: Optional[VectorSearchEngine] = None
        self._chunk_filter: Optional[ReflexiveChunkFilter] = None

    # ── Public entry point ─────────────────────────────────────────────────

    async def query(
        self,
        query: str,
        user: str,
        company: str,
        conversation_id: str,
        personality: str = "Specialized legal assistant",
        tone: str = "professional",
        use_reflexive_filter: bool = True,
        model: str = MODEL_ROUTER,
        conversation_history: Optional[List[Dict]] = None,  # fix 7
        **kwargs,
    ) -> str:
        """
        Process a legal query and return the answer in HTML.

        Args:
            conversation_history: Redis conversation history — list of dicts
                                  [{"role": "user"|"assistant", "content": str}, ...]
                                  Included in the prompt for multi-turn context.
        """
        logger.info("RagOrchestrator.query | user=%s | query=%.80s",
                    user, query)

        context_docs = await self._retrieve(
            query=query,
            user=user,
            company=company,
        )

        return await self._generate_response(
            query=query,
            context_docs=context_docs,
            personality=personality,
            tone=tone,
            use_reflexive_filter=use_reflexive_filter,
            conversation_history=conversation_history or [],
        )

    # ── Retrieval ───────────────────────────────────────────────────────────

    async def _retrieve(
        self,
        query: str,
        user: str,
        company: str,
    ) -> List[Dict]:
        engine = self._get_search_engine()

        # Collection name derived from the company.
        import re as _re
        _collection = _re.sub(r"[^a-zA-Z0-9_]", "_",
                              (company or "").upper().replace(" ", "_"))

        def _to_dict(doc) -> dict:
            """Convert a SearchResult (dataclass) or dict to a flat dict compatible with the pipeline."""
            if isinstance(doc, dict):
                d = doc
            else:
                try:
                    d = {k: v for k, v in vars(doc).items()}
                except TypeError:
                    d = doc.__dict__.copy() if hasattr(doc, '__dict__') else {}
            # Normalize fields the pipeline expects as float/str, not None
            d.setdefault("certainty", d.pop("score", 0.0) or 0.0)
            d.setdefault("score", d.get("certainty", 0.0))
            d.setdefault("content", d.get("text", ""))
            d.setdefault("filename", d.get("file_name", d.get("source", d.get("name", ""))))
            # Ensure certainty/score are floats
            try:
                d["certainty"] = float(d["certainty"] or 0.0)
                d["score"]     = float(d["score"] or 0.0)
            except (TypeError, ValueError):
                d["certainty"] = 0.0
                d["score"]     = 0.0
            return d

        async def _search(q: str, **kw) -> List:
            raw = await engine.search(
                query=q,
                collection_name=_collection,
                user=user,
                company=company,
            )
            return [_to_dict(d) for d in (raw or [])]

        return await _search(query)

    # ── Generation ──────────────────────────────────────────────────────────

    async def _generate_response(
        self,
        query: str,
        context_docs: List[Dict],
        personality: str,
        tone: str,
        use_reflexive_filter: bool,
        conversation_history: List[Dict],   # fix 7
    ) -> str:
        if not context_docs:
            return (
                "<p>Not enough information was found in the available documents "
                "to answer this query. Please rephrase the question or expand "
                "the set of indexed documents.</p>"
            )

        chunks = context_docs
        if use_reflexive_filter and len(context_docs) > 5:
            chunks = await self._get_chunk_filter().filter(context_docs, query)

        context = self._build_context(chunks)

        # fix 11: system and user separated + fix 7: multi-turn history
        messages = self._build_messages(
            query=query,
            context=context,
            personality=personality,
            tone=tone,
            history=conversation_history,
        )

        try:
            response = await asyncio.wait_for(
                self.async_openai.chat.completions.create(
                    model=MODEL_ROUTER,
                    messages=messages,
                    temperature=0.1,
                    max_completion_tokens=2000,
                ),
                timeout=45.0,
            )
            return response.choices[0].message.content or ""
        except asyncio.TimeoutError:
            logger.warning("Timeout in LLM generation")
            return "<p>❌ The response time was exceeded. Try a more specific query.</p>"
        except Exception as e:
            logger.error("Error in LLM generation: %s", e)
            return f"<p>❌ Error generating the answer: {str(e)[:150]}</p>"

    def _build_messages(
        self,
        query: str,
        context: str,
        personality: str,
        tone: str,
        history: List[Dict],
    ) -> List[Dict]:
        """
        fix 11: system prompt separate from the user message.
        fix 7:  conversation history included between system and the final user message.

        Structure:
          [system]  instructions + personality + tone
          [user]    turn 1 of the history
          [assistant] turn 1 of the history
          ...
          [user]    current query with document context
        """
        messages = [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT.format(personality=personality, tone=tone),
            }
        ]

        # Last 5 turns of the history (Ghost Attention — same limit as legacy)
        for turn in history[-10:]:  # 10 items = 5 full turns
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content[:3000]})

        messages.append({
            "role": "user",
            "content": _USER_PROMPT.format(context=context, query=query),
        })
        return messages

    def _build_context(self, chunks: List[Dict]) -> str:
        parts = []
        for i, doc in enumerate(chunks[:15]):
            text = str(doc.get("content") or doc.get("contenido") or doc.get("text") or "")
            name = doc.get("filename") or doc.get("nombre") or f"Doc {i + 1}"
            page = doc.get("pagina") or doc.get("page") or ""
            ref = f"[{name}" + (f", p.{page}" if page else "") + "]"
            parts.append(f"{ref}\n{text[:800]}")
        return "\n\n---\n\n".join(parts)

    # ── Internal singletons (fix 5) ───────────────────────────────────────

    def _get_search_engine(self) -> VectorSearchEngine:
        """The EmbeddingProcessor is created once, not on every query."""
        if self._search_engine is None:
            self._search_engine = VectorSearchEngine(
                weaviate_client=self._get_weaviate(),
                openai_api_key=self.openai_key,
                params=SearchParams(limit=50, certainty=0.65),
            )
        return self._search_engine

    def _get_chunk_filter(self) -> ReflexiveChunkFilter:
        """ReflexiveChunkFilter reuses the same executor."""
        if self._chunk_filter is None:
            self._chunk_filter = ReflexiveChunkFilter(
                model_executor=self._simple_executor(),
            )
        return self._chunk_filter

    def _simple_executor(self):
        client = self.async_openai

        async def executor(prompt: str) -> str:
            try:
                r = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL_ROUTER,
                        messages=[
                            {"role": "system", "content": "Answer only YES or NO."},
                            {"role": "user",   "content": prompt},
                        ],
                        temperature=0.0,
                        max_completion_tokens=5,
                    ),
                    timeout=5.0,
                )
                return r.choices[0].message.content or ""
            except Exception:
                return "YES"

        return executor

    def _get_weaviate(self):
        if self._weaviate_client is None:
            self._weaviate_client = weaviate.Client(
                self.weaviate_url,
                startup_period=30,
                additional_headers={"X-Openai-Api-Key": self.openai_key},
            )
        return self._weaviate_client

    # ── Legacy pipeline singleton ─────────────────────────────────────────

    _legacy_instance = None  # class-level singleton (shared across instances)

    def _get_legacy_pipeline(self):
        """
        Lazy singleton of the optimized RAGPipeline.
        Created once per process — EmbeddingProcessor, Weaviate client
        and all heavy objects are initialized a single time.
        """
        if RagOrchestrator._legacy_instance is None:
            from chatbot.RAGPipeline import RAGPipeline
            RagOrchestrator._legacy_instance = RAGPipeline(self.config, self.redis_client)
            logger.info("RagOrchestrator: legacy RAGPipeline initialized (singleton)")
        return RagOrchestrator._legacy_instance

    # ── Streaming ──────────────────────────────────────────────────────────

    async def _resolve_context_query(self, query, conversation_id, legacy):
        """GENERAL coreference/context resolution on each turn: rewrites the
        query into a SELF-CONTAINED form that incorporates the thread's subject/topic, resolving
        pronouns and implicit references ("what else", "and that", "expand", "the second one",
        "give me the technical details"…). If the query is already self-contained or a new topic,
        it returns it unchanged. Reuses MODEL_COREFERENCE (same pattern as RAGPipeline).
        Returns (rewritten_query, was_rewritten). Best-effort: on error, the original."""
        q = (query or "").strip()
        if not q:
            return query, False
        # SELF-CONTAINMENT GUARD: a selection question over an explicit set
        # ("which of the candidates…?", "which of the documents…?") is already self-contained and
        # covers the WHOLE collection. It must not be rewritten: the rewriter re-anchors it to the
        # previous subjects ("… including Sirena and Dorian …"), shrinks the universe and
        # excludes the correct one (mid-conversation Sevilla bug). It is left AS-IS.
        if _COLLECTIVE_SELECTION.match(q):
            logger.info("Coreference SKIPPED (self-contained set selection): %.80s", q)
            return query, False
        # Conversation history (user/assistant), tolerant of formats.
        # NB: the history lives in legacy.custom (CustomLLMRetriever*), is async, and
        # is the SAME store the RAGPipeline itself uses. Calling it on legacy directly
        # raised AttributeError (swallowed by the except) and left coreference
        # dead: follow-ups were never rewritten.
        try:
            getter = getattr(legacy, "get_conversation_history", None) \
                or getattr(getattr(legacy, "custom", None), "get_conversation_history", None)
            if getter is None:
                return query, False
            raw = getter(conversation_id, limit=10)
            if asyncio.iscoroutine(raw):
                raw = await raw
            raw = raw or []
        except Exception:
            return query, False
        history = []
        for h in raw:
            if not isinstance(h, dict):
                continue
            if h.get("role") in ("user", "assistant") and h.get("content"):
                history.append({"role": h["role"], "content": str(h["content"])[:1500]})
            else:
                if h.get("pregunta"):
                    history.append({"role": "user", "content": str(h["pregunta"])[:1500]})
                if h.get("respuesta"):
                    history.append({"role": "assistant", "content": str(h["respuesta"])[:1500]})
        if not history:
            return query, False  # first turn: nothing to resolve
        try:
            import asyncio as _asyncio
            REWRITER_SYSTEM = (
                "Given the conversation history and the user's last question (the user writes in Spanish), "
                "rewrite the question as a SELF-CONTAINED query that needs no prior context. "
                "Resolve pronouns and implicit references, and incorporate the subject/topic that was "
                "being discussed (e.g. if 'Alejandro Aguilar' was being discussed, 'qué más podés decir' → "
                "'Dame más detalles sobre Alejandro Aguilar'). "
                "If the question is ALREADY self-contained and clear, or is a NEW unrelated topic, return it AS-IS. "
                "NEVER add, enumerate or restrict entities the question does not mention. "
                "If the question refers to a general SET or CATEGORY ('los candidatos', "
                "'las facturas', 'quién de los documentos', 'cuál de ellos'), it is a query about "
                "the WHOLE set: do NOT limit it to the entities already mentioned nor add "
                "'including X and Y' — return it AS-IS. "
                "IMPORTANT EXCEPTION: if the question is ABOUT the conversation itself (what was asked/said/"
                "discussed before, what was my first/last question, what we were talking about, summarize/repeat "
                "the chat), return it EXACTLY AS-IS — do NOT anchor it to any topic or document, "
                "because it is answered from the history, not with the documents. "
                "Answer ONLY with the rewritten question, with no explanations or quotes. "
                "Keep the rewritten question in the user's language (Spanish)."
            )
            # IMPROVEMENT 3: re-anchor to the LAST DOCUMENTAL subject. If the previous turn
            # slipped to general (or this is a short continuation like "qué más/ampliá/
            # seguí"), and there is an active documental subject in the conversation,
            # we force the rewrite to be about THAT subject (the documents), not
            # about general chit-chat — so a one-off slip doesn't contaminate the thread.
            try:
                _st = ConversationState.get(conversation_id)
                _doc_subject = _st.get("last_documental_subject")
                _prev_general = _st.get("last_mode") == "general"
                if _doc_subject and (_prev_general or _SHORT_FOLLOWUP.match(q)):
                    REWRITER_SYSTEM += (
                        "\nACTIVE DOCUMENTAL CONTEXT: the user was querying the "
                        f"documents about: \"{str(_doc_subject)[:300]}\". "
                        "If the last question is a continuation or expansion (e.g. "
                        "'qué más', 'contame más', 'ampliá', 'seguí', 'profundizá', "
                        "'y lo técnico', 'explicá eso', 'más', 'sí'), rewrite it so it "
                        "is EXPLICITLY about that documental topic, asking for MORE "
                        "information/detail from those documents — NOT about a general "
                        "chat about how you can help."
                    )
            except Exception:  # noqa: BLE001
                pass
            client = AsyncOpenAI(api_key=KEY_OPEN_AI)
            resp = await _asyncio.wait_for(
                client.chat.completions.create(
                    model=MODEL_COREFERENCE,
                    messages=[{"role": "system", "content": REWRITER_SYSTEM}, *history[-10:],
                              {"role": "user", "content": q}],
                    max_completion_tokens=600,
                ),
                timeout=12.0,
            )
            rew = (resp.choices[0].message.content or "").strip().strip('"').strip()
            if rew and rew.lower() != q.lower() and len(rew) >= 3:
                logger.info("Coreference: %.50s -> %.90s", q, rew)
                return rew, True
        except Exception as exc:  # noqa: BLE001
            logger.debug("resolve_context_query failed (non-blocking): %s", exc)
        return query, False

    async def _general_stream(self, query, conversation_id, legacy, mode,
                              personality, tone, provider, model, agent_config,
                              user=None, config_chatbot="documental",
                              timestamp=None, save_interaction=True, user_display=None):
        """Answers with general knowledge (no RAG) via LLMProvider, over SSE.
        Persists the Q&A in chatbot_memory (like the legacy path) so the
        conversation is NOT lost."""
        # Best-effort history from the legacy pipeline (lives in legacy.custom, async).
        history = []
        try:
            getter = getattr(legacy, "get_conversation_history", None) \
                or getattr(getattr(legacy, "custom", None), "get_conversation_history", None)
            raw = getter(conversation_id, limit=10) if getter else []
            if asyncio.iscoroutine(raw):
                raw = await raw
            raw = raw or []
            for h in raw:
                if not isinstance(h, dict):
                    continue
                if h.get("role") in ("user", "assistant"):
                    history.append({"role": h["role"], "content": h.get("content", "")})
                else:
                    if h.get("pregunta"):
                        history.append({"role": "user", "content": h["pregunta"]})
                    if h.get("respuesta"):
                        history.append({"role": "assistant", "content": h["respuesta"]})
        except Exception:
            history = []

        system = build_system_prompt(mode, personality=personality or "an expert assistant",
                                     tone=tone or "clear and professional")
        instructions = (agent_config or {}).get("instructions")
        if instructions:
            system = instructions.strip() + "\n\n" + system

        messages = await ContextManager().build(history, query, system)
        prov = get_provider(provider, model)
        max_tokens = int((agent_config or {}).get("max_tokens") or 1200)
        temperature = float((agent_config or {}).get("temperature", 0.7))

        yield _sse({"type": "status", "message": "Thinking…", "mode": mode, "source": "general"})
        got = False
        answer_parts = []
        try:
            async for delta in prov.stream(messages, model=model, temperature=temperature, max_tokens=max_tokens):
                if delta:
                    got = True
                    answer_parts.append(delta)
                    yield _sse({"type": "content", "text": delta, "message": delta})
        except Exception as e:  # noqa: BLE001
            logger.error("_general_stream error: %s", e)
        if not got:
            # Safety net: if the model emitted no text (e.g. a very short greeting),
            # answer something conversational instead of an error message.
            fallback = ("Hi! I'm your assistant. How can I help you? "
                        "I can answer about your documents, query your data or generate content.")
            yield _sse({"type": "content", "text": fallback, "message": fallback})
        yield _sse({"type": "done", "source": "general", "mode": mode})

        # Persist the Q&A in chatbot_memory (best-effort) so the conversation isn't lost.
        if save_interaction and got and user:
            try:
                import datetime as _dt
                await legacy.custom.remember_interaction(
                    user=user,
                    # Store the ORIGINAL user text (what they typed), not the
                    # coreference-rewritten query used for generation.
                    user_message=((user_display or "").strip() or query),
                    bot_response="".join(answer_parts),
                    conversation_id=conversation_id,
                    timestamp=timestamp or _dt.datetime.utcnow(),
                    config_chatbot=config_chatbot or "documental",
                )
            except Exception as _se:  # noqa: BLE001
                logger.warning("_general_stream: could not save the interaction: %s", _se)

    # ── Export tools (reusable layer) ─────────────────────────────────────

    async def _get_last_answer(self, conversation_id, legacy) -> str:
        """Return the LAST assistant answer in the conversation.
        Reuses the same async getter as coreference (legacy.custom)."""
        try:
            getter = getattr(legacy, "get_conversation_history", None) \
                or getattr(getattr(legacy, "custom", None), "get_conversation_history", None)
            if getter is None:
                return ""
            raw = getter(conversation_id, limit=10)
            if asyncio.iscoroutine(raw):
                raw = await raw
            raw = raw or []
        except Exception:  # noqa: BLE001
            return ""
        last = ""
        for h in raw:
            if not isinstance(h, dict):
                continue
            if h.get("role") == "assistant" and h.get("content"):
                last = str(h["content"])
            elif h.get("respuesta"):
                last = str(h["respuesta"])
        return last

    async def _generate_export_content(self, query, content_hint, source, provider, model, agent_config) -> str:
        """Generate NEW content (summary/comparison/…) with the LLM, from the
        last answer / query. Best-effort: on failure it returns `source`."""
        try:
            prov = get_provider(provider or (agent_config or {}).get("provider"), model)
            system = ("You are an assistant that produces clear, well-structured documents in clean "
                      "HTML (headings, lists, bold). Answer ONLY with the document "
                      "content, no preamble.")
            user = (f"From the following material, {query.strip()}.\n\n"
                    f"MATERIAL:\n{(source or '')[:6000]}") if source else query.strip()
            out = await prov.complete(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                model=model, temperature=0.4, max_tokens=1800,
            )
            return out or source or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("export: new-content generation failed (%s)", exc)
            return source or ""

    def _store_export(self, data: bytes, filename: str, mime: str, user=None, company=None):
        """Store the generated file in GridFS (same mechanism as the chatbot's
        attachments) and return the file_id (str) or None. Tags the owner ({owner,company})
        in the metadata so the download can validate ownership (anti-IDOR)."""
        try:
            from framework.Database.MongoDb.MongoDb import MongoDBManager
            mongo = MongoDBManager("chatbot_exports", self.config)
            meta = {"owner": user, "company": company} if user else None
            return mongo.insert_file(data, filename, content_type=mime, metadata=meta)
        except Exception as exc:  # noqa: BLE001
            logger.error("export: could not store the file in GridFS: %s", exc)
            return None

    # ── Tool router: shared ctx construction ──────────────────────────────

    async def _get_history(self, conversation_id, legacy) -> List[Dict]:
        """Best-effort history [{role,content}] (same async getter as legacy.custom)."""
        try:
            getter = getattr(legacy, "get_conversation_history", None) \
                or getattr(getattr(legacy, "custom", None), "get_conversation_history", None)
            if getter is None:
                return []
            raw = getter(conversation_id, limit=10)
            if asyncio.iscoroutine(raw):
                raw = await raw
            raw = raw or []
        except Exception:  # noqa: BLE001
            return []
        hist: List[Dict] = []
        for h in raw:
            if not isinstance(h, dict):
                continue
            if h.get("role") in ("user", "assistant") and h.get("content"):
                hist.append({"role": h["role"], "content": str(h["content"])})
            else:
                if h.get("pregunta"):
                    hist.append({"role": "user", "content": str(h["pregunta"])})
                if h.get("respuesta"):
                    hist.append({"role": "assistant", "content": str(h["respuesta"])})
        return hist

    async def _build_tool_ctx(self, *, user, company, conversation_id,
                              legacy, model, provider, agent_config, files=None) -> dict:
        """Build the ``ctx`` each tool receives (shared contract).

        Includes: identity, last answer + history, and callables
        ``llm`` / ``retrieve`` / ``websearch`` / ``save_file`` wired to the
        real infrastructure.
        """
        last_answer = await self._get_last_answer(conversation_id, legacy)
        history = await self._get_history(conversation_id, legacy)
        prov_name = provider or (agent_config or {}).get("provider")

        async def _llm(prompt: str, system: str = None, json: bool = False) -> str:
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            elif json:
                msgs.append({"role": "system", "content": "Answer ONLY with valid JSON."})
            msgs.append({"role": "user", "content": prompt})
            prov = get_provider(prov_name, model)
            return await prov.complete(msgs, model=model, temperature=0.3, max_tokens=1800)

        async def _retrieve(q: str):
            try:
                docs = await self._retrieve(
                    query=q, user=user, company=company,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("tool ctx retrieve failed (non-blocking): %s", exc)
                return []
            out = []
            for d in docs or []:
                out.append({
                    "text": str(d.get("content") or d.get("text") or ""),
                    "document": d.get("filename") or d.get("nombre") or "",
                    "page": d.get("pagina") or d.get("page") or "",
                    "score": float(d.get("score") or d.get("certainty") or 0.0),
                })
            return out

        # Deep-research: we prefer the STRUCTURED web searcher (real hits with
        # title/url/snippet). Defensive import — the module is provided by another worker and
        # may not exist yet.
        try:
            from framework.Searchers.deep_web_search import search as _web_search
        except Exception:  # noqa: BLE001
            _web_search = None

        async def _websearch(q: str):
            """Web search for deep-research.

            Returns STRUCTURED hits ``[{title,url,snippet}]`` using
            ``framework.Searchers.deep_web_search.search`` (real sources).
            If that searcher is unavailable or raises, falls back to the legacy wrapper
            ``WebSearchClient`` (a single synthesized blob with url="")."""
            if _web_search is not None:
                try:
                    hits = await _web_search(q)
                    structured = [
                        {
                            "title": str(h.get("title") or ""),
                            "url": str(h.get("url") or ""),
                            "snippet": str(h.get("snippet") or ""),
                        }
                        for h in (hits or []) if isinstance(h, dict)
                    ]
                    if structured:
                        return structured
                except Exception as exc:  # noqa: BLE001
                    logger.debug("deep_web_search failed, falling back to WebSearchClient: %s", exc)
            # Legacy fallback: a single synthesized result (no url).
            try:
                from framework.Searchers.SearchInternet import WebSearchClient
                txt = await WebSearchClient().search_and_answer_web(
                    q, context_message="Web search requested by a tool."
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("tool ctx websearch failed (non-blocking): %s", exc)
                return []
            if not txt:
                return []
            return [{"title": "Web search result", "url": "", "snippet": str(txt)}]

        def _save_file(data: bytes, filename: str, mime: str):
            return self._store_export(data, filename, mime, user=user, company=company)

        # RBAC / plan (#8): the user's roles and the company's plan, so the
        # router filters which tools the model sees. Best-effort: on any failure
        # they stay None (permissive policy → base tools available).
        roles = self._lookup_roles(user)
        plan = self._lookup_plan(company)

        return {
            "user": user,
            "company": company,
            "conversation_id": conversation_id,
            "last_answer": last_answer or "",
            # Result SHOWN in the bot (text + table) as a join point
            # for analysis/plot follow-ups ("plot that", "analyze the
            # results"): so it doesn't drift to a different topic.
            "last_result": {
                "text": last_answer or "",
                "rows": _extract_table_rows(last_answer or ""),
            },
            "history": history,
            "files": list(files or []),
            "roles": roles,
            "plan": plan,
            "llm": _llm,
            "retrieve": _retrieve,
            "websearch": _websearch,
            "save_file": _save_file,
        }

    def _lookup_roles(self, user) -> Optional[List[str]]:
        """The user's roles (by email/username) via UserManager. None if it can't
        be resolved (permissive policy in the router)."""
        if not user:
            return None
        try:
            from framework.Users.user_manager import UserManager

            doc = UserManager(self.config).get_by_email(str(user))
            if not doc:
                return None
            roles = list(doc.get("roles") or [])
            return roles
        except Exception as exc:  # noqa: BLE001
            logger.debug("tool ctx: could not resolve roles (non-blocking): %s", exc)
            return None

    def _lookup_plan(self, company) -> Optional[str]:
        """The company's plan (slug) via Billing. None if it can't be resolved."""
        if not company:
            return None
        try:
            from framework.Billing import get_or_create_subscription

            sub = get_or_create_subscription(str(company))
            return (sub or {}).get("plan_id")
        except Exception as exc:  # noqa: BLE001
            logger.debug("tool ctx: could not resolve the company's plan (non-blocking): %s", exc)
            return None

    async def _persist_tool_turn(
        self, *, legacy, user, conversation_id, user_message,
        bot_response, tool_output, files_ids, config_chatbot, timestamp,
        sources=None,
    ):
        """Persist the TOOL ROUTER turn in ``chatbot_memory`` (best-effort).

        1) ``remember_interaction`` saves the Q&A (text + ``files_ids`` of the
           attachments, if any) like the other pipeline paths.
        2) Since ``remember_interaction`` does NOT accept arbitrary fields, a follow-up
           update writes ``tool_output`` onto the just-inserted doc — the
           most recent of this conversation/user by ``timestamp``. Only the
           non-null ones are written: chart/diagram/report and the emitted ``sources``
           (e.g. the SQL query as a structured source). This way traceability
           SURVIVES reloading the conversation.

        Never breaks the stream: any error is logged and ignored."""
        import datetime as _dt
        ts = timestamp or _dt.datetime.utcnow()
        try:
            await legacy.custom.remember_interaction(
                user=user,
                user_message=user_message,
                bot_response=bot_response,
                conversation_id=conversation_id,
                timestamp=ts,
                config_chatbot=config_chatbot or "documental",
                files_ids=(list(files_ids) or None) if files_ids else None,
            )
        except Exception as _se:  # noqa: BLE001
            logger.warning("tool router: could not save the interaction: %s", _se)
            return

        # Only the structured outputs present (non-null).
        clean = {k: v for k, v in (tool_output or {}).items() if v}
        # Emitted sources (e.g. SQL): stored as-is to rehydrate the chat.
        if sources:
            clean["sources"] = sources
        if not clean:
            return
        try:
            from framework.Database.MongoDb.MongoDb import MongoDBManager
            mongo = MongoDBManager("chatbot_memory", self.config)
            mongo_filter = {"conversation_id": conversation_id}
            if user:
                mongo_filter["usuario"] = user  # Mongo field name (stored data contract)
            # The just-saved doc = the most recent by timestamp for this pair.
            await asyncio.to_thread(
                _update_latest_tool_output, mongo.collection, mongo_filter, clean,
            )
        except Exception as _ue:  # noqa: BLE001
            logger.warning("tool router: could not write tool_output: %s", _ue)

    async def _persist_rag_sources(
        self, *, conversation_id, user, sources=None,
    ):
        """Write the ``sources`` (RAG / SQL / web citations) of the DOCUMENTAL/general turn
        onto the doc already persisted in ``chatbot_memory``.

        The RAG/general path persists the Q&A via ``remember_interaction`` (legacy
        or ``_general_stream``), which does NOT accept arbitrary fields, so the
        sources emitted over SSE were lost on reload. Like
        ``_persist_tool_turn`` for the tool router, a follow-up update writes
        ``tool_output.sources`` onto the most recent doc of this
        conversation/user (by ``timestamp``). The history endpoint already
        exposes those fields, so the sources SURVIVE reloading. Never breaks
        the stream: any error is ignored."""
        clean: Dict[str, Any] = {}
        # Emitted sources (document citations, SQL, web URLs): same shape as the
        # SSE 'sources' event and the tool router, to rehydrate the chat.
        if sources:
            clean["sources"] = sources
        if not clean:
            return
        try:
            from framework.Database.MongoDb.MongoDb import MongoDBManager
            mongo = MongoDBManager("chatbot_memory", self.config)
            mongo_filter = {"conversation_id": conversation_id}
            if user:
                mongo_filter["usuario"] = user  # Mongo field name (stored data contract)
            await asyncio.to_thread(
                _update_latest_tool_output, mongo.collection, mongo_filter, clean,
            )
        except Exception as _ue:  # noqa: BLE001
            logger.warning("RAG/general: could not write sources: %s", _ue)

    async def _export_stream(
        self, *, fmt, content_hint, query, user_display, user, company,
        conversation_id, legacy, config_chatbot, timestamp, save_interaction,
        model, provider, agent_config, personality, tone,
    ) -> AsyncGenerator[str, None]:
        """Generate a downloadable file from the conversation and emit it over SSE.
        Does NOT run RAG. Defensive: on failure it emits a normal error message."""
        yield _sse({"type": "status", "message": "Generating your file…", "mode": "export", "source": "export"})

        # 1. Source content: the assistant's last answer.
        source = await self._get_last_answer(conversation_id, legacy)

        # 2. Does the request imply NEW content? (summary/comparison/…) → generate it.
        if wants_new_content((content_hint or "") + " " + (query or "")) or not source:
            source = await self._generate_export_content(
                query=query, content_hint=content_hint, source=source,
                provider=provider, model=model, agent_config=agent_config,
            )

        title = (content_hint or "").strip().capitalize() or "Document"

        # 3. Generate bytes with the matching generator.
        try:
            if fmt == "pptx":
                def _llm(prompt):
                    prov = get_provider(provider or (agent_config or {}).get("provider"), model)
                    return prov.complete(
                        [{"role": "system", "content": "Return ONLY valid JSON."},
                         {"role": "user", "content": prompt}],
                        model=model, temperature=0.3, max_tokens=1500,
                    )
                outline = await build_slide_outline(source, title, llm=_llm)
                data, filename = to_pptx(outline, title)
            elif fmt == "docx":
                data, filename = to_docx(source, title)
            elif fmt == "pdf":
                data, filename = to_pdf(source, title)
            elif fmt == "xlsx":
                data, filename = to_xlsx(source, title)
            elif fmt == "md":
                data, filename = to_md(source, title)
            else:
                raise ValueError(f"unsupported format: {fmt}")
        except Exception as exc:  # noqa: BLE001
            logger.error("export: file generation failed (%s)", exc)
            msg = "❌ I couldn't generate the file right now. Try again or ask for it a different way."
            yield _sse({"type": "content", "text": msg, "message": msg})
            yield _sse({"type": "done", "source": "export", "mode": "export"})
            return

        # 4. Store in GridFS (same storage as the chatbot's attachments).
        file_id = await asyncio.to_thread(self._store_export, data, filename, MIME_TYPES.get(fmt, "application/octet-stream"))
        if not file_id:
            msg = "❌ I generated the file but couldn't store it for download. Please try again."
            yield _sse({"type": "content", "text": msg, "message": msg})
            yield _sse({"type": "done", "source": "export", "mode": "export"})
            return

        # 5. Emit: short message + 'attachment' event + 'done'.
        short = f"📄 I generated your document **{filename}**. Tap to download."
        yield _sse({"type": "content", "text": short, "message": short})
        yield _sse({
            "type": "attachment",
            "filename": filename,
            "mime": MIME_TYPES.get(fmt, "application/octet-stream"),
            "size": len(data),
            "download_url": f"/virtualbot/chatbot/export/download/{file_id}",
            "file_id": file_id,
        })
        yield _sse({"type": "done", "source": "export", "mode": "export"})

        # 6. Persist the Q&A turn (with file_id if the schema allows it).
        if save_interaction and user:
            try:
                import datetime as _dt
                await legacy.custom.remember_interaction(
                    user=user,
                    user_message=((user_display or "").strip() or query),
                    bot_response=short,
                    conversation_id=conversation_id,
                    timestamp=timestamp or _dt.datetime.utcnow(),
                    config_chatbot=config_chatbot or "documental",
                    files_ids=[file_id],
                )
            except Exception as _se:  # noqa: BLE001
                logger.warning("export: could not save the interaction: %s", _se)

    async def stream(
        self,
        query: str,
        user: str,
        company: str,
        conversation_id: str,
        tone: str = "neutral",
        prompt_clean: str = None,
        timestamp=None,
        model: str = GPT_5_4_MINI,
        personality: str = "specialized legal assistant",
        doc_type: str = "general",
        config_chatbot: str = "documental",
        websearch: bool = False,
        context: str = None,
        llm_direct: bool = False,
        save_interaction: bool = True,
        response_mode: str = "hybrid",
        provider: str = None,
        agent_config: dict = None,
        files: list = None,
    ) -> AsyncGenerator[str, None]:
        """
        SSE streaming of the RAG pipeline.

        Flow:
        1. Export tools — if the user asks for a downloadable file, it is generated and we stop
        2. Tool router — if the LLM picks a tool, it emits its events and we stop
        3. Delegates to RAGPipeline.chattingBotStream() — all the optimized RAG logic
           (decomposer, batch extraction, cross-encoder rerank, synthesis)
        4. Pass-through of all SSE events to the client without modification
        5. Emits the turn's structured citations/sources
        """
        query_for_pipeline = query

        # ── Export tools: does the user ask to turn the answer into a file? ──
        # Intent detection (ES/EN regex), BEFORE the IntentRouter. If it applies,
        # we generate the downloadable file and do NOT run RAG this turn.
        export_enabled = os.getenv("CHATBOT_EXPORT_TOOLS", "true").lower() in ("1", "true", "yes")
        if export_enabled and not websearch:
            try:
                exp = detect_export(query)  # over the ORIGINAL text (keeps the hints)
            except Exception as _ee:  # noqa: BLE001
                logger.debug("detect_export failed (non-blocking): %s", _ee)
                exp = {"format": None}
            if exp.get("format"):
                logger.info("Export tools: format=%s hint=%r", exp["format"], exp.get("content_hint"))
                legacy = self._get_legacy_pipeline()
                async for ev in self._export_stream(
                    fmt=exp["format"], content_hint=exp.get("content_hint", ""),
                    query=query, user_display=(prompt_clean or query),
                    user=user, company=company,
                    conversation_id=conversation_id, legacy=legacy,
                    config_chatbot=config_chatbot, timestamp=timestamp,
                    save_interaction=save_interaction, model=model,
                    provider=provider, agent_config=agent_config,
                    personality=personality, tone=tone,
                ):
                    yield ev
                return

        # ── 2.7 Tool router (function-calling): the LLM decides which tool(s) to run ──
        # Orchestration layer that ties together export / data-analysis / diagrams /
        # deep-research. Runs AFTER the export regex fast-path (which already covers
        # "put it in word" deterministically) and BEFORE the IntentRouter/RAG.
        # If the model picks a tool → we emit its events and stop
        # (no RAG this turn). If it picks NONE → None → we continue
        # the normal pipeline, without breaking the regular chat.
        try:
            from framework.Tools.router import route_and_run, tool_router_enabled
            _tr_enabled = tool_router_enabled()
        except Exception as _ie:  # noqa: BLE001
            logger.debug("Tool router unavailable (non-blocking): %s", _ie)
            _tr_enabled = False
        if _tr_enabled and not websearch and not llm_direct:
            try:
                legacy = self._get_legacy_pipeline()
                tool_ctx = await self._build_tool_ctx(
                    user=user, company=company,
                    conversation_id=conversation_id, legacy=legacy,
                    model=model, provider=provider, agent_config=agent_config,
                    files=files,
                )
                tool_agen = await route_and_run(query, tool_ctx)
            except Exception as _te:  # noqa: BLE001
                logger.debug("Tool router failed (non-blocking, pipeline continues): %s", _te)
                tool_agen = None
            if tool_agen is not None:
                logger.info("Tool router: tool chosen → RAG skipped this turn.")
                # We accumulate the STRUCTURED outputs by parsing the SSE events on
                # the fly, to be able to PERSIST them later (they reappear when the
                # conversation is reopened). Before, only text and download survived.
                _tool_chart = None
                _tool_diagram = None
                _tool_report = None
                _tool_text_parts: List[str] = []
                _tool_file_ids: List[str] = []
                _tool_sources: List[Any] = []
                async for ev in tool_agen:
                    yield ev
                    try:
                        _kind, _payload = _parse_tool_sse(ev)
                        if _kind == "text" and _payload:
                            _tool_text_parts.append(_payload)
                        elif _kind == "chart" and _payload:
                            _tool_chart = _payload
                        elif _kind == "diagram" and _payload:
                            _tool_diagram = _payload
                        elif _kind == "report" and _payload:
                            _tool_report = _payload
                        elif _kind == "attachment" and _payload:
                            _tool_file_ids.append(str(_payload))
                        elif _kind == "sources" and _payload:
                            # Sources (e.g. SQL) are accumulated to persist them.
                            if isinstance(_payload, list):
                                _tool_sources.extend(_payload)
                            else:
                                _tool_sources.append(_payload)
                    except Exception:  # noqa: BLE001 — never break the stream
                        pass
                # Persist the tool router turn (Q&A + tool_output). Defensive.
                if save_interaction and user:
                    try:
                        await self._persist_tool_turn(
                            legacy=legacy,
                            user=user,
                            conversation_id=conversation_id,
                            user_message=((prompt_clean or query or "").strip() or query),
                            bot_response="".join(_tool_text_parts),
                            tool_output={
                                "chart": _tool_chart,
                                "diagram": _tool_diagram,
                                "report": _tool_report,
                            },
                            files_ids=_tool_file_ids,
                            sources=(_tool_sources or None),
                            config_chatbot=config_chatbot, timestamp=timestamp,
                        )
                    except Exception as _pe:  # noqa: BLE001
                        logger.warning("tool router: turn persistence failed (non-blocking): %s", _pe)
                return

        # ── 3. Intent + mode routing (hybrid chatbot) ────────────────────
        legacy = self._get_legacy_pipeline()
        response_chunks: List[str] = []
        mode = normalize_mode((agent_config or {}).get("response_mode") if agent_config else response_mode)
        hybrid_enabled = os.getenv("CHATBOT_HYBRID_ROUTER", "true").lower() in ("1", "true", "yes")

        # GENERAL coreference/context resolution (each turn): rewrites the
        # query into a SELF-CONTAINED form incorporating the thread's subject/topic ("what else",
        # "and that", "expand", "the second one", "give me the technical details"…). If it is already
        # self-contained or a new topic, it stays the same. Reuses MODEL_COREFERENCE.
        orig_q = (query_for_pipeline or "").strip()
        ctx_rewritten = False
        if hybrid_enabled and not websearch and not llm_direct:
            try:
                query_for_pipeline, ctx_rewritten = await self._resolve_context_query(
                    query_for_pipeline, conversation_id, legacy)
            except Exception as _fe:  # noqa: BLE001
                logger.debug("context resolve non-blocking: %s", _fe)

        decision = None
        if hybrid_enabled and not websearch and not llm_direct:
            try:
                decision = await IntentRouter(self.config).classify(query_for_pipeline)
                logger.info("IntentRouter: %s", decision.to_dict())
            except Exception as _re:
                logger.debug("IntentRouter failed (non-blocking): %s", _re)

        # Active documental thread + follow-up (query rewritten by context or
        # a short affirmation/continuation): keep RAG to bring MORE of the same
        # material, instead of falling back to general.
        # IMPROVEMENT 2: we anchor to the LAST real DOCUMENTAL mode of the conversation
        # (last_documental_mode), NOT to the immediately previous turn — so, if ONE
        # turn slipped to general, the next follow-up returns to documental
        # instead of staying contaminated (end of the cascade effect).
        is_followupish = bool(ctx_rewritten or _SHORT_FOLLOWUP.match(orig_q))
        _st_now = ConversationState.get(conversation_id)
        _anchor_mode = _st_now.get("last_documental_mode") or _st_now.get("last_mode")
        # Questions ABOUT the conversation ("what did I ask you?") are answered from the
        # history — never force them to RAG even if the thread is anchored to documental.
        if (decision is not None and getattr(decision, "category", None) != "memory"
                and _should_stay_documental(is_followupish, _anchor_mode)):
            if not decision.needs_rag:
                logger.info("Follow-up over documental thread: forcing needs_rag=True "
                            "(anchored to last documental subject: %.60s)",
                            _st_now.get("last_documental_subject") or "")
            decision.needs_rag = True

        # Answer directly with general knowledge (no RAG)?
        use_general = bool(
            decision is not None and mode != "enterprise_only"
            and (mode == "general" or not decision.needs_rag)
        )

        if use_general:
            async for ev in self._general_stream(
                query=query_for_pipeline, conversation_id=conversation_id, legacy=legacy,
                # This path does NOT inject documents (needs_rag=False). Use the
                # conversational "general" prompt, never the document-anchored "hybrid"/"enterprise"
                # one that demands citing nonexistent docs and left greetings/chit-chat empty
                # ("hola" returned nothing). Same as the generative fallback below.
                mode="general", personality=personality, tone=tone,
                provider=provider or (agent_config or {}).get("provider"),
                model=model, agent_config=agent_config,
                user=user, config_chatbot=config_chatbot,
                timestamp=timestamp, save_interaction=save_interaction,
                user_display=(prompt_clean or query),
            ):
                yield ev
                t = _sse_text(ev)
                if t:
                    response_chunks.append(t)
        else:
            # The legacy error events and the SENTINEL content ("Information not
            # found") are DEFERRED: if a generative fallback happens later
            # they are discarded (we don't show the error/"not found" before the answer).
            held_errors: List[str] = []
            held_sentinel: List[str] = []
            try:
                async for event in legacy.chattingBotStream(
                    query=query_for_pipeline,
                    user=user,
                    company=company,
                    conversation_id=conversation_id,
                    tone=tone,
                    prompt_clean=prompt_clean or query,
                    timestamp=timestamp,
                    model=model,
                    personality=personality,
                    doc_type=doc_type,
                    config_chatbot=config_chatbot,
                    websearch=websearch,
                    context=context,
                    llm_direct=llm_direct,
                    save_interaction=save_interaction,
                ):
                    if _is_error_event(event):
                        held_errors.append(event)
                        continue
                    t = _sse_text(event)
                    # Defer the canned "no evidence" block (it is not a real answer).
                    if t and _looks_like_not_found(t):
                        held_sentinel.append(event)
                        continue
                    yield event  # pass-through without modification
                    if t:
                        response_chunks.append(t)
            except Exception as e:
                logger.error("RagOrchestrator.stream: error in legacy pipeline: %s", e)

            # has_content = there was a REAL answer (the sentinel doesn't count).
            has_content = bool("".join(response_chunks).strip())
            needs_fallback = (not has_content and mode != "enterprise_only"
                              and (decision is None or decision.allow_general_llm))
            # If there is NO fallback, we emit the deferred ones (errors + "not found"),
            # e.g. enterprise_only without evidence must show the notice.
            if not needs_fallback:
                for e in held_errors:
                    yield e
                for e in held_sentinel:
                    yield e
                    t = _sse_text(e)
                    if t:
                        response_chunks.append(t)

            # Generative fallback: RAG with no evidence + the mode allows it.
            if needs_fallback:
                logger.info("Empty RAG → generative fallback (mode=%s)", mode)
                async for ev in self._general_stream(
                    query=query_for_pipeline, conversation_id=conversation_id, legacy=legacy,
                    mode="general", personality=personality, tone=tone,
                    provider=provider or (agent_config or {}).get("provider"),
                    model=model, agent_config=agent_config,
                    user=user, config_chatbot=config_chatbot,
                    timestamp=timestamp, save_interaction=save_interaction,
                    user_display=(prompt_clean or query),
                ):
                    yield ev
                    t = _sse_text(ev)
                    if t:
                        response_chunks.append(t)

        # Remember the turn's effective mode to resolve the next turn's follow-ups
        # ('documental' if there was a RAG answer; 'general' if it was general or fallback).
        # IMPROVEMENT 2: when the turn is documental, we also set last_documental_mode
        # and last_documental_subject (the self-contained query that brought the docs). That persists
        # across general detours and is what follow-ups re-anchor to.
        try:
            _eff_mode = "general" if use_general else ("documental" if has_content else "general")
            if _eff_mode == "documental":
                ConversationState.mark_documental(
                    conversation_id, subject=(query_for_pipeline or "").strip()[:300] or None)
            else:
                ConversationState.mark_general(conversation_id)
        except Exception:  # noqa: BLE001
            pass

        # ── 4. Structured citations (sources) ──────────────────────────────
        try:
            # Quick retrieval to build the citations with the real docs
            # (the user already received all chunks — this step is background).
            context_docs = []
            try:
                context_docs = await self._retrieve(
                    query=query_for_pipeline,
                    user=user,
                    company=company,
                )
            except Exception as _ve:
                logger.debug("Stream: retrieval for citations failed (non-blocking): %s", _ve)

            # Structured citations: only if there was documental evidence (not on the general path).
            citations_to_persist = None
            if not use_general and context_docs:
                try:
                    from chatbot.pipeline_v2.response.citation import CitationBuilder
                    citations = CitationBuilder().extract_citations(context_docs)
                    if citations:
                        citations_to_persist = citations
                        yield _sse({"type": "sources", "sources": citations})
                except Exception as _ce:
                    logger.debug("Stream: citations not emitted (non-blocking): %s", _ce)

            # RAG/general turn persistence: write the emitted sources onto the
            # already-saved doc (remember_interaction), so the "Sources (N)"
            # SURVIVE reloading — like the tool router via _persist_tool_turn.
            if save_interaction and user and citations_to_persist:
                try:
                    await self._persist_rag_sources(
                        conversation_id=conversation_id, user=user,
                        sources=citations_to_persist,
                    )
                except Exception as _pe:  # noqa: BLE001
                    logger.warning("Stream: source persistence failed (non-blocking): %s", _pe)
        except Exception as e:
            logger.warning("RagOrchestrator.stream: citation emission failed (non-blocking): %s", e)
