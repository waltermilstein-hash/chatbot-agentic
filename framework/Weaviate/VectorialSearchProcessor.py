"""
"""
import ast
import asyncio
import concurrent.futures
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple

from framework.Embeddings.EmbeddingProcessor import EmbeddingProcessor

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SearchResult:
    """Structure for a single search result."""
    text: str
    name: str
    metadata: Dict[str, Any]
    file_name: str
    doc_type: str
    score: Optional[float] = None
    chunk_id: str = ""
    source: str = "search_result"
    # New fields
    filename: str = ""  # Clean file name
    referencias: List[str] = field(default_factory=list)  # Detected references
    content_type: str = "text"  # Content type
    word_count: int = 0
    chunk_length: int = 0
    metadata_json: str = ""


@dataclass
class SearchConfig:
    """Configuration for vector searches."""
    limit: int = 100
    certainty: float = 0.4
    use_filters: bool = True
    remove_duplicates: bool = True
    max_chunks_per_file: int = 10
    enable_parallel: bool = False
    max_workers: int = 4


@dataclass
class QueryAnalysis:
    """LLM analysis of the user's query."""
    intent: str
    target_documents: List[str]
    content_type: Optional[str]
    section_reference: Optional[str]
    search_terms: List[str]
    action: str
    confidence: float
    needs_context: bool = False
    context_reference: Optional[str] = None


@dataclass
class ConversationContext:
    """Conversation context."""
    last_query: str = ""
    last_results: List[SearchResult] = field(default_factory=list)
    last_documents: List[str] = field(default_factory=list)
    last_intent: str = ""
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)

    def add_turn(self, query: str, results: List[SearchResult], intent: str):
        """Add a conversation turn."""
        self.last_query = query
        self.last_results = results
        self.last_intent = intent

        docs = list(set([r.file_name for r in results if hasattr(r, 'file_name')]))
        self.last_documents = docs

        self.conversation_history.append({
            "query": query,
            "intent": intent,
            "documents": docs,
            "result_count": len(results)
        })

        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]


# ============================================================================
# LLM INTENT ANALYZER
# ============================================================================

class LLMIntentAnalyzer:
    """LLM intent analyzer with conversation context."""

    ANALYSIS_PROMPT = """Analyze the following user question taking the conversation context into account and extract:

PREVIOUS CONTEXT:
{context}

CURRENT QUESTION: {query}

Respond in JSON with this EXACT structure:
{{
    "intent": "one of: buscar | relacionar | comparar | extraer_contenido | cumplimiento | listar_documentos | seguimiento | buscar_referencia",
    "target_documents": ["list of mentioned documents or []"],
    "content_type": "tabla | sección | cláusula | artículo | null",
    "section_reference": "section/article number or name if mentioned, or null",
    "search_terms": ["key terms to search for"],
    "action": "short description of what to do",
    "confidence": 0.0 to 1.0,
    "needs_context": true if the question refers to previous results,
    "context_reference": "description of what context is needed, or null"
}}

EXAMPLES:

Question: "¿Qué dice el artículo 27b?"
{{
    "intent": "buscar_referencia",
    "target_documents": [],
    "content_type": "artículo",
    "section_reference": "27b",
    "search_terms": ["artículo 27b"],
    "action": "Search for article 27b specifically",
    "confidence": 0.95,
    "needs_context": false,
    "context_reference": null
}}

Question: "¿Qué dice sobre ginecología?"
{{
    "intent": "buscar",
    "target_documents": [],
    "content_type": null,
    "section_reference": null,
    "search_terms": ["ginecología"],
    "action": "General search about gynecology",
    "confidence": 0.85,
    "needs_context": false,
    "context_reference": null
}}

Now analyze this question and respond with ONLY the JSON:"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    async def analyze_query(
            self,
            query: str,
            context: Optional[ConversationContext] = None
    ) -> QueryAnalysis:
        """Analyze the user's intent WITH conversation context."""

        if self.llm_client:
            try:
                context_str = self._format_context(context) if context else "No previous context"
                analysis_json = await self._analyze_with_llm(query, context_str)
                return self._parse_llm_response(analysis_json)
            except Exception as e:
                logger.warning(f"LLM error, using fallback: {e}")
                return self._fallback_analysis(query, context)
        else:
            return self._fallback_analysis(query, context)

    def _format_context(self, context: ConversationContext) -> str:
        """Format the context for the LLM."""
        if not context or not context.conversation_history:
            return "No conversation history"

        context_parts = []

        if context.last_query:
            context_parts.append(f"Last question: '{context.last_query}'")
            if context.last_documents:
                context_parts.append(f"Documents shown: {', '.join(context.last_documents[:3])}")
            context_parts.append(f"Results: {len(context.last_results)} chunks")

        if len(context.conversation_history) > 1:
            context_parts.append("\nPrevious turns:")
            for turn in context.conversation_history[-3:-1]:
                context_parts.append(f"  - '{turn['query']}' → {turn['result_count']} results")

        return "\n".join(context_parts)

    async def _analyze_with_llm(self, query: str, context_str: str) -> str:
        """Analyze with an LLM (implement according to your LLM)."""
        prompt = self.ANALYSIS_PROMPT.format(query=query, context=context_str)
        raise NotImplementedError("Configure your LLM client here")

    def _parse_llm_response(self, llm_response: str) -> QueryAnalysis:
        """Parse the LLM's JSON response."""
        try:
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                data = json.loads(llm_response)

            return QueryAnalysis(
                intent=data.get("intent", "buscar"),
                target_documents=data.get("target_documents", []),
                content_type=data.get("content_type"),
                section_reference=data.get("section_reference"),
                search_terms=data.get("search_terms", []),
                action=data.get("action", "General search"),
                confidence=data.get("confidence", 0.5),
                needs_context=data.get("needs_context", False),
                context_reference=data.get("context_reference")
            )
        except Exception as e:
            logger.error(f"Error parsing the LLM response: {e}")
            return self._fallback_analysis("", None)

    def _fallback_analysis(
            self,
            query: str,
            context: Optional[ConversationContext]
    ) -> QueryAnalysis:
        """Fallback analysis WITH context and reference detection."""
        query_lower = query.lower()

        # Reference detection (article, section, chapter)
        section_ref = self._extract_section_reference(query)
        if section_ref:
            return QueryAnalysis(
                intent="buscar_referencia",
                target_documents=[],
                content_type="artículo" if "artículo" in query_lower or "art" in query_lower else "sección",
                section_reference=section_ref,
                search_terms=[section_ref],
                action=f"Buscar {section_ref}",
                confidence=0.9,
                needs_context=False,
                context_reference=None
            )

        needs_context = any(ref in query_lower for ref in [
            'ese', 'esa', 'esos', 'esas', 'ahí', 'allí',
            'anterior', 'previo', 'último', 'mismo',
            'qué más', 'dime más', 'amplía', 'detalla'
        ])

        target_docs = []
        if needs_context and context and context.last_documents:
            target_docs = context.last_documents[:2]
        else:
            doc_patterns = [
                r'(?:ley|decreto|norma)\s+\d+[/-]?\d*',
                r'contrato\s+\w+',
                r'anexo\s+\d+',
            ]
            for pattern in doc_patterns:
                matches = re.findall(pattern, query_lower)
                target_docs.extend(matches)

        if 'relaci' in query_lower and len(target_docs) >= 2:
            intent = "relacionar"
            action = "Analyze the relationship between documents"
        elif 'compar' in query_lower:
            intent = "comparar"
            action = "Compare information"
        elif any(kw in query_lower for kw in ['tabla', 'cuadro']):
            intent = "extraer_contenido"
            action = "Extract structured content"
        elif needs_context:
            intent = "seguimiento"
            action = "Continue with previous context"
        else:
            intent = "buscar"
            action = "General search"

        stopwords = {'el', 'la', 'los', 'las', 'de', 'del', 'en', 'es', 'ese', 'esa'}
        words = re.findall(r'\b\w+\b', query_lower)
        search_terms = [w for w in words if w not in stopwords and len(w) > 3][:5]

        return QueryAnalysis(
            intent=intent,
            target_documents=target_docs,
            content_type="tabla" if "tabla" in query_lower else None,
            section_reference=section_ref,
            search_terms=search_terms,
            action=action,
            confidence=0.6,
            needs_context=needs_context,
            context_reference="contexto previo" if needs_context else None
        )

    def _extract_section_reference(self, query: str) -> Optional[str]:
        """
        Improved: extract specific references (e.g. artículo 27b, sección 3).
        """
        patterns = [
            (r'artículo\s+(\d+[a-z]?)', 'artículo'),
            (r'art\.?\s*(\d+[a-z]?)', 'artículo'),
            (r'sección\s+(\d+[a-z]?)', 'sección'),
            (r'sec\.?\s*(\d+[a-z]?)', 'sección'),
            (r'capítulo\s+([IVX]+|\d+)', 'capítulo'),
            (r'cap\.?\s*([IVX]+|\d+)', 'capítulo'),
        ]

        for pattern, tipo in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return f"{tipo} {match.group(1)}"

        return None


# ============================================================================
# MAIN VECTOR SEARCH PROCESSOR
# ============================================================================

class VectorialSearchProcessor:
    """
    Fixed version with:
    - Thread-safe parallel processing
    - No race conditions
    - No deadlocks
    - Robust fallbacks
    """

    def __init__(self, weaviate_client, embedding_processor=None, llm_client=None):
        self.client = weaviate_client
        self.embedding_processor = embedding_processor
        self.intent_analyzer = LLMIntentAnalyzer(llm_client)
        self.conversation_context = ConversationContext()

    # ========================================================================
    # NEW METHODS - SEARCH BY REFERENCES AND FILENAME
    # ========================================================================

    async def search_by_reference(
            self,
            user: str,
            company: str,
            class_name: str,
            reference: str,
            config: SearchConfig = None
    ) -> List[SearchResult]:
        """
        Specific search by legal/technical reference.
        """
        if config is None:
            config = SearchConfig()

        try:
            logger.info(f"🔍 Reference search: '{reference}'")

            ref_lower = reference.lower().strip()

            query_builder = self.client.query.get(
                class_name,
                [
                    "text", "name", "filename", "referencias",
                    "referencias_text", "content_type", "metadata_json",
                    "word_count", "chunk_length", "source", "doc_type"
                ]
            ).with_where({
                "operator": "And",
                "operands": [
                    {"path": ["user"], "operator": "Equal", "valueString": user},
                    {"path": ["company"], "operator": "Equal", "valueString": company},
                    {"path": ["referencias_text"], "operator": "Like", "valueText": f"*{ref_lower}*"}
                ]
            }).with_limit(config.limit)

            response = query_builder.do()

            if not response or not isinstance(response, dict):
                return []

            raw_results = response.get("data", {}).get("Get", {}).get(class_name, [])
            logger.info(f"✅ Found {len(raw_results)} chunks with '{reference}'")

            search_results = await self._process_raw_results(raw_results, config)

            return search_results[:config.limit]

        except Exception as e:
            logger.error(f"❌ Error in reference search: {e}", exc_info=True)
            return []

    async def search_by_filename(
            self,
            user: str,
            company: str,
            class_name: str,
            filename_query: str,
            config: SearchConfig = None
    ) -> List[SearchResult]:
        """
        Search by file name.
        """
        if config is None:
            config = SearchConfig()

        try:
            logger.info(f"📁 Filename search: '{filename_query}'")

            query_builder = self.client.query.get(
                class_name,
                [
                    "text", "name", "filename", "referencias",
                    "content_type", "metadata_json", "source"
                ]
            ).with_where({
                "operator": "And",
                "operands": [
                    {"path": ["user"], "operator": "Equal", "valueString": user},
                    {"path": ["company"], "operator": "Equal", "valueString": company},
                    {"path": ["filename"], "operator": "Like", "valueText": f"*{filename_query}*"}
                ]
            }).with_limit(config.limit)

            response = query_builder.do()

            if not response or not isinstance(response, dict):
                return []

            raw_results = response.get("data", {}).get("Get", {}).get(class_name, [])
            logger.info(f"✅ Found {len(raw_results)} file chunks with '{filename_query}'")

            search_results = await self._process_raw_results(raw_results, config)

            return search_results[:config.limit]

        except Exception as e:
            logger.error(f"❌ Error in filename search: {e}", exc_info=True)
            return []

    # ========================================================================
    # PROCESO PRINCIPAL - ACTUALIZADO
    # ========================================================================

    async def process_query(
            self,
            user: str,
            company: str,
            class_name: str,
            query: str,
            config: SearchConfig = None,
            maintain_context: bool = True
    ) -> Dict[str, Any]:
        """
        Process any query, detecting intent automatically.
        """
        logger.info(f"Processing query: '{query}'")

        intent_analysis = await self.intent_analyzer.analyze_query(
            query,
            self.conversation_context if maintain_context else None
        )

        logger.info(f"Intent: {intent_analysis.intent} (confidence: {intent_analysis.confidence:.2f})")
        if intent_analysis.needs_context:
            logger.info(f"Requiere contexto: {intent_analysis.context_reference}")

        if config is None:
            config = SearchConfig()

        if intent_analysis.intent == "buscar_referencia" and intent_analysis.section_reference:
            logger.info(f"🔍 Specific reference search: {intent_analysis.section_reference}")
            results = await self.search_by_reference(
                user, company, class_name,
                intent_analysis.section_reference,
                config
            )

            response = {
                "intent": "buscar_referencia",
                "action": intent_analysis.action,
                "section_reference": intent_analysis.section_reference,
                "results": results,
                "formatted_output": self._format_reference_results(results, intent_analysis.section_reference)
            }

            if maintain_context:
                self.conversation_context.add_turn(query, results, intent_analysis.intent)

            return response

        if intent_analysis.needs_context and intent_analysis.intent == "seguimiento":
            return await self._handle_context_followup(
                user, company, class_name, query,
                intent_analysis, config
            )

        if intent_analysis.intent == "buscar":
            results = await self._handle_search(
                user, company, class_name, query,
                intent_analysis.search_terms, config
            )
            response = {
                "intent": intent_analysis.intent,
                "action": intent_analysis.action,
                "results": results,
                "formatted_output": self._format_search_results(results)
            }

        elif intent_analysis.intent == "relacionar":
            docs = intent_analysis.target_documents
            if intent_analysis.needs_context and self.conversation_context.last_documents:
                docs = docs + self.conversation_context.last_documents

            response = await self._handle_document_relation(
                user, company, class_name, query, docs, config
            )

        elif intent_analysis.intent == "extraer_contenido":
            response = await self._handle_content_extraction(
                user, company, class_name, query,
                intent_analysis, config
            )

        elif intent_analysis.intent == "comparar":
            response = await self._handle_comparison(
                user, company, class_name, query,
                intent_analysis.target_documents, config
            )

        elif intent_analysis.intent == "cumplimiento":
            response = await self._handle_compliance(
                user, company, class_name, query,
                intent_analysis.target_documents, config
            )

        else:
            results = await self.search_similar_docs(
                user, company, class_name, query, config
            )
            response = {
                "intent": "buscar",
                "action": "General search",
                "results": results,
                "formatted_output": self._format_search_results(results)
            }

        if maintain_context:
            results = response.get("results", [])
            if isinstance(results, list):
                self.conversation_context.add_turn(query, results, intent_analysis.intent)

        return response

    def _format_reference_results(self, results: List[SearchResult], reference: str) -> str:
        """Format reference-search results."""
        if not results:
            return f"No chunks found mentioning '{reference}'"

        output = [
            f"\n{'=' * 70}",
            f"\nRESULTS FOR: {reference.upper()}",
            f"\n{'=' * 70}",
            f"\n\nEncontrados {len(results)} fragmentos:\n"
        ]

        for i, chunk in enumerate(results, 1):
            output.append(f"\n{'-' * 70}")
            output.append(f"\n{i}. File: {chunk.filename or chunk.file_name}")
            if chunk.referencias:
                output.append(f"\n   Referencias: {', '.join(chunk.referencias[:5])}")
            output.append(f"\n   Tipo: {chunk.content_type}")
            output.append(f"\n\n{chunk.text}\n")

        output.append(f"\n{'=' * 70}\n")

        return "".join(output)

    # ========================================================================
    # HANDLERS
    # ========================================================================

    async def _handle_context_followup(
            self,
            user: str,
            company: str,
            class_name: str,
            query: str,
            intent: QueryAnalysis,
            config: SearchConfig
    ) -> Dict[str, Any]:
        """Handle follow-up questions."""
        logger.info("Manejando seguimiento contextual")

        if not self.conversation_context.last_results:
            return {
                "intent": "seguimiento",
                "action": "Solicitar contexto",
                "results": [],
                "formatted_output": (
                    "I have no previous context in this conversation.\n"
                    "Could you be more specific about what information you need?"
                )
            }

        context_docs = self.conversation_context.last_documents
        context_chunks = self.conversation_context.last_results[:10]

        if context_docs:
            results = []
            for doc in context_docs:
                doc_chunks = await self._retrieve_document_chunks(
                    user, company, class_name, doc, config
                )
                results.extend(doc_chunks[:5])

            output = [
                f"\n{'=' * 70}",
                f"\nADDITIONAL CONTEXT INFORMATION",
                f"\n{'=' * 70}",
                f"\n\nBuscando en documentos previos: {', '.join(context_docs)}",
                f"\n\nResultados adicionales:\n"
            ]

            for i, chunk in enumerate(results[:5], 1):
                output.append(f"\n{i}. {chunk.text[:300]}...\n")

            output.append(f"\n{'=' * 70}\n")

            return {
                "intent": "seguimiento",
                "action": "Ampliar contexto previo",
                "results": results,
                "formatted_output": "".join(output),
                "context_used": context_docs
            }

        else:
            output = [
                f"\n{'=' * 70}",
                f"\nDETAILS OF PREVIOUS RESULTS",
                f"\n{'=' * 70}\n"
            ]

            for i, chunk in enumerate(context_chunks[:3], 1):
                output.append(f"\n{i}. File: {chunk.filename or chunk.file_name}")
                if chunk.referencias:
                    output.append(f"\n   Referencias: {', '.join(chunk.referencias)}")
                output.append(f"\n   {chunk.text}\n")

            output.append(f"\n{'=' * 70}\n")

            return {
                "intent": "seguimiento",
                "action": "Details of previous chunks",
                "results": context_chunks,
                "formatted_output": "".join(output)
            }

    def reset_conversation_context(self):
        """Reset the conversation context."""
        self.conversation_context = ConversationContext()
        logger.info("Contexto conversacional reseteado")

    async def _handle_search(
            self,
            user: str,
            company: str,
            class_name: str,
            query: str,
            search_terms: List[str],
            config: SearchConfig
    ) -> List[SearchResult]:
        return await self.search_similar_docs(
            user, company, class_name, query, config
        )

    async def _handle_document_relation(
            self,
            user: str,
            company: str,
            class_name: str,
            query: str,
            documents: List[str],
            config: SearchConfig
    ) -> Dict[str, Any]:
        logger.info(f"Analyzing relationship between: {documents}")

        doc_chunks = {}
        for doc_id in documents:
            chunks = await self._retrieve_document_chunks(
                user, company, class_name, doc_id, config
            )
            doc_chunks[doc_id] = chunks

        relations = []
        common_entities = set()

        if len(documents) >= 2:
            for i, doc1 in enumerate(documents):
                for doc2 in documents[i + 1:]:
                    chunks1 = doc_chunks.get(doc1, [])
                    chunks2 = doc_chunks.get(doc2, [])

                    if not chunks1 or not chunks2:
                        continue

                    cross_refs = self._find_cross_references(chunks1, chunks2, doc2)
                    if cross_refs:
                        relations.append(f"{doc1} menciona a {doc2}: {len(cross_refs)} referencias")

                    common = self._find_common_entities(chunks1, chunks2)
                    common_entities.update(common)

        output = [
            f"\n{'=' * 70}",
            f"\nRELATIONSHIP ANALYSIS BETWEEN DOCUMENTS",
            f"\n{'=' * 70}",
            f"\n\nDocumentos analizados:",
        ]

        for doc in documents:
            chunk_count = len(doc_chunks.get(doc, []))
            output.append(f"\n  - {doc} ({chunk_count} fragmentos)")

        if relations:
            output.append(f"\n\nRelaciones encontradas:")
            for rel in relations:
                output.append(f"\n  ✓ {rel}")

        if common_entities:
            output.append(f"\n\nEntidades comunes ({len(common_entities)}):")
            for entity in list(common_entities)[:10]:
                output.append(f"\n  • {entity}")

        if not relations and not common_entities:
            output.append(f"\n\n⚠️ No obvious direct relationships were found between the documents.")

        output.append(f"\n\n{'=' * 70}\n")

        return {
            "intent": "relacionar",
            "action": "Relationship analysis between documents",
            "documents": documents,
            "relations": relations,
            "common_entities": list(common_entities),
            "formatted_output": "".join(output)
        }

    async def _handle_content_extraction(
            self,
            user: str,
            company: str,
            class_name: str,
            query: str,
            intent: QueryAnalysis,
            config: SearchConfig
    ) -> Dict[str, Any]:
        logger.info(f"Extracting content: {intent.content_type} from section {intent.section_reference}")

        search_query = " ".join(intent.search_terms)
        results = await self.search_similar_docs(
            user, company, class_name, search_query, config
        )

        extracted = []
        for result in results:
            if intent.content_type == "tabla":
                tables = self._extract_tables(result.text)
                if tables:
                    extracted.extend(tables)
            else:
                if intent.section_reference:
                    if f"sección {intent.section_reference}" in result.text.lower():
                        extracted.append(result.text)
                else:
                    extracted.append(result.text)

        output = [
            f"\n{'=' * 70}",
            f"\nCONTENIDO EXTRAÍDO: {intent.content_type or 'general'}",
            f"\n{'=' * 70}",
        ]

        if intent.section_reference:
            output.append(f"\nSección: {intent.section_reference}")

        if extracted:
            output.append(f"\n\nSe encontraron {len(extracted)} elementos:\n")
            for i, content in enumerate(extracted[:5], 1):
                output.append(f"\n--- Elemento {i} ---\n{content[:500]}...\n")
        else:
            output.append(f"\n\n⚠️ No content of the requested type '{intent.content_type}' was found.")

        output.append(f"\n{'=' * 70}\n")

        return {
            "intent": "extraer_contenido",
            "action": intent.action,
            "content_type": intent.content_type,
            "extracted_items": len(extracted),
            "content": extracted,
            "formatted_output": "".join(output)
        }

    async def _handle_comparison(
            self,
            user: str,
            company: str,
            class_name: str,
            query: str,
            documents: List[str],
            config: SearchConfig
    ) -> Dict[str, Any]:
        logger.info(f"Comparando: {documents}")

        doc_data = {}
        for doc_id in documents:
            chunks = await self._retrieve_document_chunks(
                user, company, class_name, doc_id, config
            )
            doc_data[doc_id] = {
                "chunks": chunks,
                "text": " ".join([c.text for c in chunks[:10]])
            }

        output = [
            f"\n{'=' * 70}",
            f"\nCOMPARACIÓN ENTRE DOCUMENTOS",
            f"\n{'=' * 70}",
            f"\n\nDocumentos comparados:",
        ]

        for doc, data in doc_data.items():
            output.append(f"\n\n📄 {doc}:")
            output.append(f"\n   Fragmentos: {len(data['chunks'])}")
            output.append(f"\n   Vista previa: {data['text'][:200]}...")

        output.append(f"\n\n{'=' * 70}\n")

        return {
            "intent": "comparar",
            "action": "Comparison between documents",
            "documents": documents,
            "data": doc_data,
            "formatted_output": "".join(output)
        }

    async def _handle_compliance(
            self,
            user: str,
            company: str,
            class_name: str,
            query: str,
            documents: List[str],
            config: SearchConfig
    ) -> Dict[str, Any]:
        logger.info(f"Verificando cumplimiento: {documents}")
        return await self._handle_document_relation(
            user, company, class_name, query, documents, config
        )

    async def _retrieve_document_chunks(
            self,
            user: str,
            company: str,
            class_name: str,
            doc_identifier: str,
            config: SearchConfig
    ) -> List[SearchResult]:
        doc_config = SearchConfig(
            limit=300,
            certainty=0.25,
            remove_duplicates=False
        )

        query = f"{doc_identifier}"
        results = await self.search_similar_docs(
            user, company, class_name, query, doc_config
        )

        doc_lower = doc_identifier.lower()
        relevant = []

        for chunk in results:
            file_lower = (chunk.filename or chunk.file_name).lower()
            if (doc_lower in file_lower or
                    file_lower in doc_lower or
                    any(word in file_lower for word in doc_lower.split() if len(word) > 3)):
                relevant.append(chunk)

        logger.info(f"Retrieved {len(relevant)} chunks for '{doc_identifier}'")
        return relevant

    def _find_cross_references(
            self,
            chunks1: List[SearchResult],
            chunks2: List[SearchResult],
            doc2_name: str
    ) -> List[str]:
        references = []
        doc2_lower = doc2_name.lower()

        for chunk in chunks1:
            if doc2_lower in chunk.text.lower():
                context = chunk.text[:200]
                references.append(context)
                if len(references) >= 5:
                    break

        return references

    def _find_common_entities(
            self,
            chunks1: List[SearchResult],
            chunks2: List[SearchResult]
    ) -> Set[str]:
        def extract_entities(chunks):
            entities = set()
            for chunk in chunks:
                proper_nouns = re.findall(r'\b[A-Z][a-záéíóúñ]+(?:\s+[A-Z][a-záéíóúñ]+)*\b', chunk.text)
                entities.update([p for p in proper_nouns if len(p) > 3])

                numbers = re.findall(r'\b\d{4,}\b', chunk.text)
                entities.update(numbers)
            return entities

        entities1 = extract_entities(chunks1)
        entities2 = extract_entities(chunks2)

        common = entities1.intersection(entities2)
        stopwords = {'PARA', 'ESTE', 'ESTA', 'TODOS', 'TODO', 'COMO'}
        return {e for e in common if e.upper() not in stopwords}

    def _extract_tables(self, text: str) -> List[str]:
        tables = []
        lines = text.split('\n')
        current_table = []

        for line in lines:
            if '|' in line or re.search(r'\s{3,}', line):
                current_table.append(line)
            else:
                if len(current_table) > 2:
                    tables.append('\n'.join(current_table))
                current_table = []

        if len(current_table) > 2:
            tables.append('\n'.join(current_table))

        return tables

    def _format_search_results(self, results: List[SearchResult]) -> str:
        if not results:
            return "No results found"

        formatted_results = self.format_results_combined(results)
        return "\n".join(formatted_results)

    # ========================================================================
    # VECTOR SEARCH - FIXED
    # ========================================================================

    async def search_similar_docs(
            self,
            user: str,
            company: str,
            class_name: str,
            query: str,
            config: SearchConfig = None
    ) -> List[SearchResult]:
        """Main vector search."""
        if config is None:
            config = SearchConfig()

        logger.info(f"Starting vector search: query='{query}', limit={config.limit}")

        if not query or not query.strip():
            logger.warning("Empty query")
            return []

        try:
            query_vector = await self._generate_query_vector(query)
            if not query_vector:
                logger.error("Could not generate the query vector")
                return []

            raw_results = await self._execute_weaviate_search(
                class_name, query_vector, user, company, query, config
            )

            if not raw_results:
                logger.info("No results found in Weaviate")
                return []

            search_results = await self._process_raw_results(raw_results, config)

            if not search_results:
                logger.info("Could not process results")
                return []

            if config.remove_duplicates:
                search_results = self._remove_duplicates(search_results, config)

            final_results = search_results[:config.limit]
            logger.info(f"Search complete: {len(final_results)} results")

            return final_results

        except Exception as e:
            logger.error(f"Error in vector search: {e}", exc_info=True)
            return []

    async def _generate_query_vector(self, query: str) -> List[float]:
        """Generate the embedding vector for the query."""
        try:
            if self.embedding_processor:
                return await self.embedding_processor.embedding_query(query)
            else:
                processor = EmbeddingProcessor(max_workers=1, batch_size=1)
                return await processor.embedding_query(query)
        except Exception as e:
            logger.error(f"Error generating vector: {e}")
            return []

    async def _execute_weaviate_search(
            self,
            class_name: str,
            vector: List[float],
            user: str,
            company: str,
            query: str,
            config: SearchConfig
    ) -> List[Dict[str, Any]]:
        """Run a Weaviate search with enriched fields."""
        try:
            logger.debug(f"Running Weaviate search: vector dim={len(vector)}")

            # Base fields always present in every collection
            _base_fields = [
                "text", "name", "metadata_json", "filename",
                "referencias", "referencias_text", "content_type",
                "word_count", "chunk_length", "source", "doc_type",
                "uploaded_by", "uploaded_by_name", "created_date", "is_deleted",
                "user", "company",
            ]
            # Optional EWS fields — only present in collections indexed with doc_identity
            _ews_fields = ["doc_hash", "doc_version", "indexed_at", "superseded"]
            try:
                schema = self.client.schema.get(class_name)
                _existing = {p["name"] for p in schema.get("properties", [])}
                _ews_present = [f for f in _ews_fields if f in _existing]
            except Exception:
                _ews_present = []
            _fields = _base_fields + _ews_present

            query_builder = self.client.query.get(class_name, _fields)

            if query.strip():
                try:
                    query_builder = query_builder.with_hybrid(
                        query=query,
                        vector=vector,
                        alpha=0.75,
                        properties=["text", "referencias_text", "metadata_json"]
                    )
                    logger.debug("Using hybrid search")
                except Exception as e:
                    logger.debug(f"Hybrid unavailable, falling back to vector search: {e}")
                    query_builder = query_builder.with_near_vector({
                        "vector": vector,
                        "certainty": config.certainty
                    })
            else:
                query_builder = query_builder.with_near_vector({
                    "vector": vector,
                    "certainty": config.certainty
                })

            if config.use_filters:
                _operands = [
                    {"path": ["company"], "operator": "Equal", "valueString": company},
                    {"path": ["is_deleted"], "operator": "Equal", "valueBoolean": False},
                ]
                if user and user.strip():
                    _operands.insert(0, {"path": ["user"], "operator": "Equal", "valueString": user})
                query_builder = query_builder.with_where({
                    "operator": "And",
                    "operands": _operands,
                })
                # Note: the superseded filter is applied in post-processing (the field may not exist
                # in collections predating the doc_identity implementation)
            else:
                query_builder = query_builder.with_where({
                    "path": ["is_deleted"],
                    "operator": "Equal",
                    "valueBoolean": False,
                })

            query_builder = query_builder.with_limit(min(config.limit * 3, 1000))
            response = query_builder.do()

            if not response or not isinstance(response, dict):
                return []

            results = response.get("data", {}).get("Get", {}).get(class_name, [])
            results = [r for r in results if not r.get('is_deleted', False)]

            logger.debug(f"✅ Encontrados {len(results)} resultados (excluidos eliminados)")

            return results or []

        except Exception as e:
            logger.error(f"Error in Weaviate search: {e}", exc_info=True)
            return []

    # ========================================================================
    # RESULT PROCESSING - FIXED
    # ========================================================================

    async def _process_raw_results(
            self,
            raw_results: List[Dict[str, Any]],
            config: SearchConfig
    ) -> List[SearchResult]:
        """
        Fixed: process results with a robust fallback.
        """

        if not raw_results:
            return []

        try:
            # ✅ HABILITADO: Procesamiento paralelo
            if config.enable_parallel and len(raw_results) > 10:
                logger.debug(f"Using PARALLEL processing for {len(raw_results)} items")
                try:
                    result = await self._process_results_parallel(raw_results)
                    return result
                except Exception as e:
                    logger.warning(f"Error in parallel processing, falling back to sequential: {e}")
                    return self._process_results_sequential(raw_results)
            else:
                logger.debug(f"Using SEQUENTIAL processing for {len(raw_results)} items")
                docs =self._process_results_sequential(raw_results)
                return docs
        except Exception as e:
            logger.error(f"Error in _process_raw_results: {e}", exc_info=True)
            # Last fallback
            try:
                return self._process_results_sequential(raw_results)
            except Exception as fallback_error:
                logger.error(f"Fallback también falló: {fallback_error}")
                return []

    def _process_results_sequential(self, raw_results: List[Dict[str, Any]]) -> List[SearchResult]:
        """
        ✅ THREAD-SAFE: Procesamiento secuencial
        """
        search_results = []

        for item in raw_results:
            try:
                result = self._parse_single_result(item)
                if result:
                    search_results.append(result)
            except Exception as e:
                logger.debug(f"Error processing result: {e}")
                continue

        return search_results

    async def _process_results_parallel(self, raw_results: List[Dict[str, Any]]) -> List[SearchResult]:
        """
        ✅ CORREGIDO: Procesamiento paralelo sin deadlocks
        """

        logger.debug(f"Processing {len(raw_results)} items in parallel")

        import concurrent.futures

        def process_item(item: Dict[str, Any]) -> Optional[SearchResult]:
            """Thread-safe item processor"""
            try:
                return self._parse_single_result(item)
            except Exception as e:
                logger.debug(f"Error processing item: {e}")
                return None

        # Fix: without a with statement
        loop = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

        try:
            # Crear tareas
            tasks = [
                loop.run_in_executor(executor, process_item, item)
                for item in raw_results
            ]

            logger.debug(f"Ejecutando {len(tasks)} tareas en paralelo...")

            # Await with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=60.0
            )

            logger.debug(f"Tareas completadas: {len(results)} resultados")

        except asyncio.TimeoutError:
            logger.error("Timeout while processing results in parallel")
            return self._process_results_sequential(raw_results)

        except Exception as e:
            logger.error(f"Error in parallel processing: {e}", exc_info=True)
            return self._process_results_sequential(raw_results)

        finally:
            # Critical: close the executor AFTER gather
            executor.shutdown(wait=False)

        # Filter valid ones
        valid_results = [
            r for r in results
            if isinstance(r, SearchResult)
        ]

        logger.debug(f"Procesamiento: {len(valid_results)}/{len(raw_results)} válidos")

        return valid_results

    def _parse_single_result(self, item: Dict[str, Any]) -> Optional[SearchResult]:
        """
        Fixed: parse the result, extracting references from referencias_text.
        """
        try:
            # Extract fields with safe defaults
            raw_text = str(item.get("text", "")).strip()
            name = str(item.get("name", ""))
            metadata_str = str(item.get("metadata_json", ""))

            if not raw_text:
                return None

            # Extract nested content
            actual_text, extracted_metadata = self._extract_nested_content(raw_text)

            if not actual_text or not actual_text.strip():
                return None

            # ✅ THREAD-SAFE: Parsear metadata
            stored_metadata = {}
            if metadata_str and metadata_str.strip():
                try:
                    stored_metadata = json.loads(metadata_str)
                    if not isinstance(stored_metadata, dict):
                        stored_metadata = {}
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    logger.debug(f"Error parsing metadata_json: {e}")
                    stored_metadata = {}

            # Thread-safe: merge with a defensive copy
            final_metadata = {}
            final_metadata.update(stored_metadata)
            final_metadata.update(extracted_metadata)

            # Extract new fields with defaults
            filename = str(item.get("filename", ""))
            content_type = str(item.get("content_type", "text"))
            word_count = int(item.get("word_count", 0))
            chunk_length = int(item.get("chunk_length", 0))
            source = str(item.get("source", ""))
            doc_type = str(item.get("doc_type", ""))

            # Optimized fix: parse references with a clear priority order
            referencias = []

            # PRIORITY 1: referencias_text (dedicated field, most reliable)
            referencias_text = item.get("referencias_text", "")
            if referencias_text and referencias_text.strip():
                # Split by pipe "|" (the separator) or by comma as fallback
                if "|" in referencias_text:
                    referencias = [ref.strip() for ref in referencias_text.split('|') if ref.strip()]
                else:
                    referencias = [ref.strip() for ref in referencias_text.split(',') if ref.strip()]

            # PRIORITY 2: only if referencias_text is empty, try the referencias field
            if not referencias:
                referencias_raw = item.get("referencias")
                if isinstance(referencias_raw, list) and referencias_raw:
                    referencias = referencias_raw.copy()

            # PRIORITY 3: look in metadata_json only if there are still no references
            if not referencias and stored_metadata:
                meta_refs_text = stored_metadata.get("referencias_text", "")
                if meta_refs_text and meta_refs_text.strip():
                    if "|" in meta_refs_text:
                        referencias = [ref.strip() for ref in meta_refs_text.split('|') if ref.strip()]
                    else:
                        referencias = [ref.strip() for ref in meta_refs_text.split(',') if ref.strip()]

                # If still none, try the referencias field in metadata
                if not referencias:
                    meta_refs = stored_metadata.get("referencias", "")
                    if isinstance(meta_refs, list) and meta_refs:
                        referencias = meta_refs.copy()
                    elif meta_refs and isinstance(meta_refs, str) and meta_refs.strip():
                        referencias = [ref.strip() for ref in meta_refs.split(',') if ref.strip()]

            # PRIORITY 4: last resort - extract from text using regex
            if not referencias and actual_text:
                referencias = self._extract_referencias_from_text(actual_text)

            # Determinar filename final
            file_name = (
                    filename or
                    final_metadata.get("file_name", "") or
                    self._extract_filename_from_name(name)
            )

            doc_type = (
                    doc_type or
                    final_metadata.get("tipo_doc", "") or
                    "Document"
            )

            # ✅ Crear SearchResult
            return SearchResult(
                text=actual_text,
                name=name,
                metadata=final_metadata,
                file_name=file_name,
                doc_type=doc_type,
                chunk_id=name,
                source="search_result",
                filename=filename,
                referencias=referencias,  # Now with references from referencias_text first
                content_type=content_type,
                word_count=word_count,
                chunk_length=chunk_length,
                metadata_json=metadata_str  # Add this line
            )

        except Exception as e:
            logger.debug(f"Error parsing result: {e}")
            return None

    def _extract_referencias_from_text(self, text: str) -> List[str]:
        """
        New: extract references from the text when they are not in metadata.
        """
        referencias = []

        # Legal reference patterns
        patterns = [
            (r'artículo\s+\d+[a-z]?(?:\s+(?:bis|ter))?', 'artículo'),
            (r'art\.\s*\d+[a-z]?', 'artículo'),
            (r'Real Decreto[- ]ley\s+\d+/\d{4}', 'decreto'),
            (r'Real Decreto Legislativo\s+\d+/\d{4}', 'decreto'),
            (r'Ley\s+\d+/\d{4}', 'ley'),
            (r'Disposición\s+(?:Adicional|Transitoria|Final)\s+\w+', 'disposición'),
            (r'apartado\s+\d+', 'apartado'),
        ]

        text_lower = text.lower()

        for pattern, tipo in patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                ref = match.group(0)
                # Capitalizar apropiadamente
                if tipo == 'artículo':
                    ref = re.sub(r'art\.?\s*', 'Artículo ', ref, flags=re.IGNORECASE)
                elif tipo == 'decreto':
                    ref = ref.title()
                elif tipo == 'ley':
                    ref = ref.title()
                elif tipo == 'disposición':
                    ref = ref.title()

                if ref not in referencias:
                    referencias.append(ref)

        return referencias[:10]  # At most 10 references

    def _extract_nested_content(self, raw_text: str) -> Tuple[str, Dict[str, Any]]:
        """Extract nested content in a thread-safe way."""
        extracted_metadata = {}

        try:
            if '\n---\n' in raw_text:
                parts = raw_text.split('\n---\n', 1)
                if len(parts) == 2:
                    content = parts[0].strip()
                    for line in parts[1].split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            extracted_metadata[key.strip()] = value.strip()
                    return content, extracted_metadata

            if raw_text.strip().startswith(("{'text':", '{"text":')):
                try:
                    parsed = ast.literal_eval(raw_text) if raw_text.startswith("{'") else json.loads(raw_text)
                    if isinstance(parsed, dict):
                        text = parsed.get('text', '').replace('\\n', '\n')
                        meta_str = parsed.get('metadata_json', '')
                        if meta_str:
                            try:
                                extracted_metadata = json.loads(meta_str)
                            except:
                                pass
                        return text, extracted_metadata
                except:
                    pass

            return raw_text, extracted_metadata

        except Exception as e:
            logger.debug(f"Error extracting nested content: {e}")
            return raw_text, extracted_metadata

    def _extract_filename_from_name(self, name: str) -> str:
        """Extract the filename from name."""
        if '-' in name:
            parts = name.split('-')
            return parts[1] if len(parts) > 1 else name
        return name

    def _remove_duplicates(
            self,
            search_results: List[SearchResult],
            config: SearchConfig
    ) -> List[SearchResult]:
        """Remove duplicates, keeping the best chunks."""

        file_groups = defaultdict(list)
        for result in search_results:
            file_key = result.filename or result.file_name
            file_groups[file_key].append(result)

        deduplicated = []
        for file_name, results in file_groups.items():
            selected = results[:config.max_chunks_per_file]
            deduplicated.extend(selected)

            if len(deduplicated) >= config.limit:
                break

        logger.debug(f"Deduplication: {len(search_results)} -> {len(deduplicated)} chunks")
        return deduplicated

    # ========================================================================
    # FORMATOS
    # ========================================================================

    def format_results_combined(self, results: List[SearchResult]) -> List[str]:
        """Format results grouped by file."""
        file_groups = defaultdict(list)
        for result in results:
            file_key = result.filename or result.file_name
            file_groups[file_key].append(result)

        formatted = []
        for file_name, file_results in file_groups.items():
            combined_text = "\n\n--- FRAGMENTO ---\n".join([r.text for r in file_results])

            metadata = file_results[0].metadata
            doc_type = file_results[0].doc_type

            all_referencias = set()
            for r in file_results:
                if r.referencias:
                    all_referencias.update(r.referencias)

            metadata = "\n".join([
                f"{k}: {v}" for k, v in metadata.items()
                if k not in ["tipo_doc", "file_id", "user", "company"] and v
            ])

            formatted_text = (
                f"=== FILE: {file_name} ===\n"
                f"{combined_text}\n"
                f"---\n"
                f"Tipo doc: {doc_type}\n"
            )

            if all_referencias:
                formatted_text += f"Referencias: {', '.join(sorted(all_referencias))}\n"

            formatted_text += metadata

            formatted.append(formatted_text)

        return formatted

    def format_results_individual(self, results: List[SearchResult]) -> List[str]:
        """Formatea resultados individuales"""
        formatted = []
        for result in results:
            metadata = "\n".join([
                f"{k}: {v}" for k, v in result.metadata.items()
                if k not in ["tipo_doc", "file_id", "user", "company"] and v
            ])

            formatted_text = (
                f"=== FILE: {result.filename or result.file_name} ===\n"
                f"{result.text}\n"
                f"---\n"
                f"Tipo doc: {result.doc_type}\n"
                f"Chunk: {result.chunk_id}\n"
                f"Content type: {result.content_type}\n"
            )

            if result.referencias:
                formatted_text += f"Referencias: {', '.join(result.referencias)}\n"

            formatted_text += metadata

            formatted.append(formatted_text)

        return formatted

    def format_results_json(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Formatea resultados como JSON"""
        return [
            {
                "text": result.text,
                "file_name": result.file_name,
                "filename": result.filename,
                "tipo_doc": result.doc_type,
                "chunk_id": result.chunk_id,
                "referencias": result.referencias,
                "content_type": result.content_type,
                "word_count": result.word_count,
                "chunk_length": result.chunk_length,
                "metadata": result.metadata
            }
            for result in results
        ]

    # ========================================================================
    # COMPATIBILIDAD LEGACY
    # ========================================================================

    async def search_similar_docs_vectorial(
            self,
            user: str,
            company: str,
            class_name: str,
            query: str,
            limit: int = 100,
            use_filters: bool = True,
            remove_duplicates: bool = True
    ) -> List[str]:
        """Legacy method - full compatibility."""
        config = SearchConfig(
            limit=limit,
            use_filters=use_filters,
            remove_duplicates=remove_duplicates,
            max_chunks_per_file=10 if remove_duplicates else 1
        )

        results = await self.search_similar_docs(
            user, company, class_name, query, config
        )

        if remove_duplicates:
            return self.format_results_combined(results)
        else:
            return self.format_results_individual(results)


# ============================================================================
# FACTORY
# ============================================================================

class VectorSearchFactory:
    """Factory for creating processors."""

    @staticmethod
    def create_processor(weaviate_client, embedding_processor=None, llm_client=None):
        return VectorialSearchProcessor(weaviate_client, embedding_processor, llm_client)

    @staticmethod
    def create_with_embedding_processor(
            weaviate_client,
            embedding_processor=None,
            max_workers=4,
            llm_client=None
    ):
        if embedding_processor is None:
            embedding_processor = EmbeddingProcessor(
                max_workers=max_workers,
                batch_size=1
            )

        return VectorialSearchProcessor(weaviate_client, embedding_processor, llm_client)

    @staticmethod
    def create_legacy(weaviate_client):
        embedding_processor = EmbeddingProcessor(
            max_workers=4,
            batch_size=1
        )

        return VectorialSearchProcessor(weaviate_client, embedding_processor, llm_client=None)