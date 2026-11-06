"""
"""
import asyncio
import datetime
import json
import logging
import re
import time
import traceback
import uuid
from cmath import log
from collections import Counter, defaultdict
from typing import AsyncGenerator, Dict, Any, List
from typing import Tuple, Callable

import anthropic
import openai
import tiktoken
import weaviate
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

import config
from config.Constants import KEY_OPEN_AI, GPT_5_NANO, GPT_5_1, GPT_5_4_MINI, GEMINI_3_1_PRO, MODEL_COREFERENCE, MODEL_ROUTER,\
    CLAUDE_SONNET, MODEL_DECOMPOSER, MODEL_REFORMULATION
from framework import Redis
from framework.Rankers.CrossEncoderReranker import rerank_async as _cross_rerank
from chatbot.QueryCache import QueryCache as _QueryCache
from framework.ConfigurationManager.ConfigurationManagerSelected import ConfigurationManagerSelected
from framework.Database.MongoDb.MongoDb import MongoDBManager
from framework.Embeddings.EmbeddingProcessor import EmbeddingProcessor
from framework.Rankers.ReRankers import FastOpenAIReranker
from framework.Searchers.SearchInternet import WebSearchClient
from framework.Weaviate.VectorialSearchProcessor import SearchConfig, VectorSearchFactory
from chatbot.CustomLLMRetrieverByCelery import CustomLLMRetrieverByCelery

# Alias for asyncio.TimeoutError
AsyncioTimeoutError = asyncio.TimeoutError

import os

# META-CONVERSATION markers: tokens that EXPLICITLY refer to the dialogue itself
# (what the user asked/said, what the assistant answered, what was talked about…).
# A query without any of these is NOT a memory question, even if it shares
# vocabulary with the topic (e.g. "which of the candidates works in Seville?" looks
# similar by embeddings to "what did you tell me about the candidates?", but it is NOT
# memory: it asks for a NEW fact from the documents). Used to downgrade MEMORY→DOCS
# (false positive).
_META_CONV_MARKERS = re.compile(
    r"(\bpregunt[eéê]\b|\bpregunt[eé]\w*|\bpreguntaste\b|\bpreguntas?\s+(te\s+)?(hice|he\s+hecho)\b|"
    r"\b(mi|primera|[uú]ltima|anterior)\s+pregunta\b|\bqu[eé]\s+te\s+(dije|pregunt\w*)\b|"
    r"\bdij[ei]\b|\bdijiste\b|\brespond[ií]\b|\brespondiste\b|\bme\s+(dijiste|contaste|explicaste|mencionaste|respondiste)\b|"
    r"\bcontaste\b|\bexplicaste\b|\bmencionaste\b|"
    r"\bhablamos\b|\bhabl[aá]bamos\b|\bven[ií]amos\s+hablando\b|\bde\s+qu[eé]\s+habl\w+\b|"
    r"\bconversaci[oó]n\b|\bcharla\b|\bhistorial\b|\bhasta\s+ahora\b|"
    r"\blo\s+que\s+(dijiste|me\s+dijiste|explicaste|mencionaste|contaste|respondiste|hablamos)\b|"
    r"\bresum[ií]\w*\s+(la\s+)?(conversaci[oó]n|charla|lo\s+que\s+hablamos)\b)",
    re.IGNORECASE,
)

# SELECTION question over an EXPLICIT set ("which of the candidates…?",
# "which of the documents…?"). It is SELF-CONTAINED: it covers the WHOLE collection,
# not the entities just mentioned. Coreference rewriting CONTAMINATES it (re-anchors to
# the previous subjects → narrows the universe and leaves out the right one). NOT rewritten.
# Anaphoric collectives are excluded ("the two", "the mentioned ones", "the previous ones").
_COLLECTIVE_SELECTION_RP = re.compile(
    r"^[¿¡\s]*"
    r"(qui[eé]n(es)?|cu[aá]l(es)?|cu[aá]nt[oa]s?|algun[oa]s?|cada\s+un[oa])\s+"
    r"de\s+(los|las)\s+"
    r"(?!dos\b|tres\b|cuatro\b|cinco\b|ambos\b|ambas\b|mism\w+|anteriores\b|"
    r"mencionad\w+|citad\w+|dich[oa]s\b|referid\w+|nombrad\w+|primer[oa]s?\b|"
    r"[uú]ltim[oa]s?\b|restantes\b)"
    r"[a-záéíóúñ]{3,}",
    re.IGNORECASE,
)


class GhostAttentionConfig:
    """Configuration for Ghost Attention"""
    HISTORY_LIMIT = 5
    CONTEXT_MAX_CHARS = 8000
    RESPONSE_MIN_LENGTH = 50
    MESSAGE_MAX_LENGTH = 3000
    INJECTION_METHOD = "prefix"
    ENABLE_VALIDATION = True
    DEBUG_MODE = True

_AUGMENTATION_CACHE = {}
_CACHE_MAX_SIZE = 100

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RAGPipeline:
    """
    Pipeline RAG 100% genérico - Sistema completo de procesamiento de consultas
    """

    def __init__(self, config, redis_client=Redis,max_tokens_context: int = 50000):
        """
        Inicializa el pipeline RAG.

        Args:
            openai_api_key: Clave API de OpenAI
        """
        self.custom = CustomLLMRetrieverByCelery(config)
        self.config = config  # Almacenar la configuración para OpenAI y Milvus
        self.memory = {}
        self.collection = MongoDBManager("chatbot_memory", config)
        self.weaviate_url = self.config['weaviate']['url']
        self.openai_api_key = self.config['openai_key']['key']
        self.asyncOpenAI = AsyncOpenAI(api_key=self.openai_api_key)

        self.tokenizer = tiktoken.get_encoding("o200k_base")
        self._call_count = 0
        self._last_reset = time.time()
        self.max_concurrent_requests = 3
        self.redis_client = redis_client
        self.client_async_openai = AsyncOpenAI(api_key=KEY_OPEN_AI)


        self.max_tokens_context = max_tokens_context
        self.encoding = tiktoken.get_encoding("cl100k_base")  # Para GPT-4
        self.batch_size = 32
        additional_headers = {
            "X-Openai-Api-Key": self.openai_api_key,
        }
        self.semaphore = asyncio.Semaphore(100)  # Limitar a 60 solicitudes simultáneas
        # Crear cliente de Weaviate con los headers personalizados
        self.client = weaviate.Client(
            self.weaviate_url,
            startup_period=30,  # Startup wait increased to 30 seconds
            additional_headers=additional_headers  # Pass the headers with the OpenAI API key
        )
        self.configSearch = SearchConfig(
                limit=50,               # fix3: 500→50, reduces noise and reranking overhead
                certainty=0.65,         # fix3: 0.25→0.65, only truly relevant chunks
                remove_duplicates=True,
                max_chunks_per_file=20,
                enable_parallel=True
            )

        self.embedding_processor = EmbeddingProcessor(
            max_workers=4,
            batch_size=160
        )
        self.search_processor = VectorSearchFactory.create_with_embedding_processor(
            weaviate_client=self.client,
            embedding_processor=self.embedding_processor,  # ← AGREGAR
            max_workers=4
        )

    async def _handle_web_search(
            self,
            query: str,
            user: str,
            conversation_id: str,
            timestamp,
            config_chatbot: str
    ) -> str:
        """
        Handles web search with robust error handling.

        Raises:
            ValidationError: If the query is empty
            InfrastructureError: If the search client fails
            TimeoutError: If the search exceeds the time limit
        """


        if not query or not query.strip():
            print(f"[WEB_SEARCH] ❌ Empty query for web search")
            raise ValueError("Empty query for web search")

        print(f"[WEB_SEARCH] 🌐 Starting web search: {query[:50]}...")

        try:
            try:
                buscador = WebSearchClient()
            except Exception as e:
                print(f"[WEB_SEARCH] ❌ Error creating search client: {e}")
                raise RuntimeError(f"Error initializing internet search client: {str(e)}")

            try:
                print(f"[WEB_SEARCH] ⏱️ Timeout: 60s")

                # FIX: search_and_answer_web is async → direct await
                # (before: asyncio.to_thread() which only works for sync functions)
                response_text = await asyncio.wait_for(
                    buscador.search_and_answer_web(
                        query,
                        context_message="search the internet and return the sources"
                    ),
                    timeout=260.0
                )

                print(f"[WEB_SEARCH] ✅ Search completed: {len(response_text)} chars")

            except TimeoutError:
                print(f"[WEB_SEARCH] ⏱️ TIMEOUT in web search")
                raise TimeoutError("The web search exceeded the time limit")

            except Exception as e:
                print(f"[WEB_SEARCH] ❌ Search error: {e}")

                error_str = str(e).lower()
                if "rate limit" in error_str or "quota" in error_str:
                    raise RuntimeError(f"Search limit exceeded: {str(e)}")
                else:
                    raise RuntimeError(f"Web search error: {str(e)}")

            if not response_text or not response_text.strip():
                print(f"[WEB_SEARCH] ⚠️ Empty response from the search client")
                response_text = "<p>No relevant results were found in the web search.</p>"

            return response_text

        except (ValueError, RuntimeError, TimeoutError):
            raise

        except Exception as e:
            print(f"[WEB_SEARCH] ❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()


            raise RuntimeError(f"Unexpected error in web search: {str(e)}")
    def _detect_query_ambiguity(self, query: str) -> dict:
        """
        Fully generic ambiguity detector with error handling.
        Requires no domain configuration.
        Applies to any document content.

        Returns:
            dict: {
                "is_ambiguous": bool,
                "type": str | None,
                "response": str | None
            }
        """

        # Validación básica
        if not query:
            return {
                "is_ambiguous": False,
                "type": None,
                "response": None
            }

        try:
            query_lower = query.lower().strip()
            query_words = [w for w in query.split() if len(w) > 2]

            # ==========================================
            # 1. PRONOMBRES SIN REFERENTE
            # ==========================================
            try:
                deictic_pattern = r'\b(eso|esto|aquello|esa|esta|aquella|ese|este|aquel)\b'
                if re.search(deictic_pattern, query_lower):
                    # Check whether it has immediate context
                    has_context = re.search(r'(sobre|de|del|en el|en la)\s+\w+', query_lower)
                    if not has_context:
                        pronoun = re.search(deictic_pattern, query_lower).group()
                        return {
                            "is_ambiguous": True,
                            "type": "pronoun_without_referent",
                            "response": f"""
    <div class="ambiguity-warning">
        <h2>⚠️ I need more information</h2>
        <p>Your question uses "<strong>{pronoun}</strong>"
        but does not specify what you are referring to.</p>
        <p>Could you rephrase your question to be more specific?</p>
        <p><strong>Example:</strong> Instead of "What does it say about that?", ask
        "What does it say about [specific topic]?"</p>
    </div>
    """
                        }
            except Exception as e:
                print(f"[AMBIGUITY] ⚠️ Error in pronoun detection: {e}")
                # Continue with other validations

            # ==========================================
            # 2. ULTRA-VAGUE QUESTIONS (< 3 meaningful words)
            # ==========================================
            try:
                if len(query_words) < 3:
                    ultra_vague = [
                        r'^¿?cómo funciona (todo|esto)\??$',
                        r'^¿?qué (dice|es|hay)\??$',
                        r'^¿?explica(me)?\??$',
                        r'^¿?(cuál|qué|cuándo|dónde)\??$'
                    ]
                    for pattern in ultra_vague:
                        if re.search(pattern, query_lower):
                            return {
                                "is_ambiguous": True,
                                "type": "too_vague",
                                "response": """
    <div class="ambiguity-warning">
        <h2>⚠️ Question too general</h2>
        <p>Your question is very broad. To give you a useful answer, I need more details.</p>
        <p><strong>Suggestions:</strong></p>
        <ul>
            <li>Which specific topic do you have doubts about?</li>
            <li>Is there a particular document or section you are interested in?</li>
            <li>Are you looking for information about a specific process, requirement or procedure?</li>
        </ul>
    </div>
    """
                            }
            except Exception as e:
                print(f"[AMBIGUITY] ⚠️ Error in vague-question detection: {e}")

            # ==========================================
            # 3. INCOMPLETE COMPARISONS
            # ==========================================
            try:
                comparison_patterns = [
                    r'^¿?(cuál|qué)\s+(tiene|es|está)\s+(más|menos|mejor|peor)',
                    r'^¿?(cuál|qué)\s+(tiene|da|permite)\s+(más|menos)\s+\w+\??$'
                ]
                for pattern in comparison_patterns:
                    if re.search(pattern, query_lower) and len(query_words) <= 5:
                        return {
                            "is_ambiguous": True,
                            "type": "incomplete_comparison",
                            "response": """
    <div class="ambiguity-warning">
        <h2>⚠️ Incomplete comparison</h2>
        <p>Your question requests a comparison but does not specify which elements to compare.</p>
        <p><strong>Please indicate:</strong></p>
        <ul>
            <li>Which concepts, categories or elements do you want to compare?</li>
            <li>In which specific aspect (days, months, amount, requirements, etc.)?</li>
        </ul>
        <p><strong>Example:</strong> "Which one has more days?" → "Which probation period has more days,
        group 1 or group 3?"</p>
    </div>
    """
                        }
            except Exception as e:
                print(f"[AMBIGUITY] ⚠️ Error in comparison detection: {e}")

            # ==========================================
            # 4. ACTION VERBS WITHOUT OBJECT/CONTEXT
            # ==========================================
            try:
                action_patterns = [
                    (r'^¿?(cuándo|cómo)\s+(tengo que|debo|hay que)\s+\w+\??$', 'acción'),
                    (r'^¿?(puedo|debo|necesito)\s+\w+\??$', 'permiso'),
                    (r'^¿?(qué|cuánto)\s+(cobro|recibo|pago|me dan)\??$', 'dinero')
                ]

                for pattern, tipo in action_patterns:
                    if re.search(pattern, query_lower) and len(query_words) <= 5:
                        if tipo == 'acción':
                            msg = "Notify/do/submit WHAT?"
                        elif tipo == 'permiso':
                            msg = "Do WHAT specifically?"
                        else:
                            msg = "Charge/pay for WHICH concept?"

                        return {
                            "is_ambiguous": True,
                            "type": "action_without_context",
                            "response": f"""
    <div class="ambiguity-warning">
        <h2>⚠️ {msg}</h2>
        <p>Your question needs more context to be answered correctly.</p>
        <p>Please specify the topic, process or concept you are asking about.</p>
    </div>
    """
                        }
            except Exception as e:
                print(f"[AMBIGUITY] ⚠️ Error in action detection: {e}")

            # ==========================================
            # 5. TOTALIZING WORDS WITHOUT SCOPE
            # ==========================================
            try:
                if re.search(r'\b(todo|toda|todos|todas)\b', query_lower) and len(query_words) < 8:
                    return {
                        "is_ambiguous": True,
                        "type": "totalizer_without_scope",
                        "response": """
    <div class="ambiguity-warning">
        <h2>⚠️ Question too broad</h2>
        <p>Your question covers a very wide scope.</p>
        <p>Could you focus on a specific aspect or topic?</p>
    </div>
    """
                    }
            except Exception as e:
                print(f"[AMBIGUITY] ⚠️ Error in totalizer detection: {e}")

            # ==========================================
            # 6. QUESTIONS WITH ARTICLES ONLY
            # ==========================================
            try:
                article_pattern = r'^¿?(el|la|los|las)\s+\w+\s+(de|del|en)\s+\w+\??$'
                if re.search(article_pattern, query_lower) and 'artículo' not in query_lower:
                    if len(query_words) <= 4:
                        return {
                            "is_ambiguous": True,
                            "type": "unclear_subject",
                            "response": """
    <div class="ambiguity-warning">
        <h2>⚠️ I can't identify the topic</h2>
        <p>Your question does not clearly specify what you are asking about.</p>
        <p>Could you include the specific topic or concept in your question?</p>
    </div>
    """
                        }
            except Exception as e:
                print(f"[AMBIGUITY] ⚠️ Error in article detection: {e}")

            # Not ambiguous
            return {
                "is_ambiguous": False,
                "type": None,
                "response": None
            }

        except Exception as e:
            # Unexpected error in ambiguity detection
            print(f"[AMBIGUITY] ❌ Critical error in detection: {e}")
            import traceback
            traceback.print_exc()

            # Do not block the flow; return not ambiguous
            return {
                "is_ambiguous": False,
                "type": None,
                "response": None
            }

    def _create_executors(
            self,
            model: str,
            ghost_context,
            use_ghost_attention: bool,
            user: str,
            company: str,
            collection_name: str
    ):
        """
        Crea ejecutores de modelo y búsqueda con manejo de errores robusto.

        Returns:
            tuple: (simple_model_executor, vector_search_func)
        """
        import copy

        # ════════════════════════════════════════════════════════════════
        # FUNCIÓN INLINE 1: simple_model_executor
        # ════════════════════════════════════════════════════════════════
        async def simple_model_executor(
                prompt: str,
                model_override: str = None
        ) -> str:
            """
            Ejecutor simple del modelo con manejo de errores.

            Args:
                prompt: Prompt a ejecutar
                model_override: Modelo específico a usar (opcional)

            Returns:
                str: Respuesta del modelo (vacía si falla)
            """

            # Crear contexto de error

            try:
                # Inyectar ghost attention si está habilitado
                if use_ghost_attention and ghost_context:  # ← Variables del scope exterior
                    try:
                        prompt = self._inject_ghost_attention(prompt, ghost_context,
                                                              "prefix")  # ← self del scope exterior
                    except Exception as e:
                        print(f"[EXECUTOR] ⚠️ Error en ghost attention: {e}")
                        # Continuar sin ghost attention

                # Usar modelo específico o default
                # fix: GPT_5_NANO → GPT_5_4_MINI (más rápido y estable)
                model_to_use = model_override or GPT_5_4_MINI

                print(f"[EXECUTOR] 🤖 Ejecutando {model_to_use}...")

                try:
                    response = await asyncio.wait_for(
                        self.custom.run_model_with_default(prompt, model=model_to_use),
                        # ← self del scope exterior
                        timeout=60.0
                    )

                    print(f"[EXECUTOR] ✅ Respuesta: {len(response)} chars")
                    return response

                except TimeoutError:
                    print(f"[EXECUTOR] ⏱️ TIMEOUT en ejecución de modelo")
                    return ""

                except Exception as e:
                    print(f"[EXECUTOR] ❌ Error en modelo: {e}")
                    return ""

            except Exception as e:
                print(f"[EXECUTOR] ❌ Error inesperado: {e}")
                import traceback
                traceback.print_exc()
                return ""

        # ════════════════════════════════════════════════════════════════
        # FUNCIÓN INLINE 2: vector_search_func
        # ════════════════════════════════════════════════════════════════
        async def vector_search_func(
                search_query: str,
                limit: int = None
        ):
            """
            Función de búsqueda vectorial con manejo de errores.

            Args:
                search_query: Query a buscar
                limit: Límite de resultados (opcional)

            Returns:
                list: Documentos encontrados

            Raises:
                WeaviateError: Si hay error en búsqueda vectorial
                ValidationError: Si search_query está vacía
            """

            # Crear contexto de error

            # Validar query
            if not search_query or not search_query.strip():
                print(f"[SEARCH] ❌ Query vacía para búsqueda vectorial")
                raise ValueError("Query vacía para búsqueda vectorial")

            try:
                # Configurar límite
                if limit is None:
                    config = self.configSearch  # ← self del scope exterior
                else:
                    config = copy.deepcopy(self.configSearch)  # ← self del scope exterior
                    config.limit = limit

                print(f"[SEARCH] 🔍 Buscando en {collection_name} (limit: {config.limit})...")

                # Realizar búsqueda con timeout
                try:
                    results = await asyncio.wait_for(
                        self.search_processor.search_similar_docs(  # ← self del scope exterior
                            user=user,  # ← Variable del scope exterior
                            company=company,  # ← Variable del scope exterior
                            class_name=collection_name,  # ← Variable del scope exterior
                            query=search_query,
                            config=config
                        ),
                        timeout=10.0
                    )

                    print(f"[SEARCH] ✅ Encontrados: {len(results)} documentos")
                    return results

                except TimeoutError:
                    print(f"[SEARCH] ⏱️ TIMEOUT en búsqueda vectorial")
                    raise TimeoutError("La búsqueda vectorial excedió el tiempo límite")

                except Exception as e:
                    print(f"[SEARCH] ❌ Error en búsqueda: {e}")

                    # Determinar tipo de error
                    error_str = str(e).lower()
                    if "not found" in error_str or "no existe" in error_str:
                        raise RuntimeError(f"Colección '{collection_name}' no encontrada")
                    elif "connection" in error_str or "timeout" in error_str:
                        raise RuntimeError(f"Error de conexión con Weaviate: {str(e)}")
                    else:
                        raise RuntimeError(f"Error en búsqueda vectorial: {str(e)}")

            except (ValueError, RuntimeError, TimeoutError):
                # Re-lanzar errores ya manejados
                raise

            except Exception as e:
                print(f"[SEARCH] ❌ Error inesperado: {e}")
                import traceback
                traceback.print_exc()


                raise RuntimeError(f"Error inesperado en búsqueda vectorial: {str(e)}")

        # ════════════════════════════════════════════════════════════════
        # RETORNAR LAS DOS FUNCIONES COMO TUPLA
        # ════════════════════════════════════════════════════════════════
        return simple_model_executor, vector_search_func

    async def _build_derived_context_with_cache(
            self,
            query: str,
            vector_search_func: Callable,
            max_variants: int = 5,
            include_corpus_context: bool = False,
            use_cache: bool = True,
            simple_model_executor=None
    ) -> Dict[str, Any]:
        """
        Versión con cache para queries repetidas con manejo de errores.

        Uso:
            # Primera vez: tarda 6-7s
            result1 = await _build_derived_context_with_cache(...)

            # Segunda vez (misma query): < 0.1s
            result2 = await _build_derived_context_with_cache(...)

        Returns:
            Dict con resultado o fallback en caso de error
        """

        # Crear contexto de error

        try:
            # Validar query
            if not query or not query.strip():
                print(f"[Augmentation] ⚠️ Query vacía, retornando fallback")
                return {
                    'error': 'empty_query',
                    'fallback': True,
                    'derived_queries': [],
                    'time_taken': 0.0
                }

            # Normalizar query para cache
            cache_key = query.strip().lower()

            # Verificar cache
            if use_cache and cache_key in _AUGMENTATION_CACHE:
                try:
                    cached = _AUGMENTATION_CACHE[cache_key]
                    print(f"[Augmentation] ✅ Cache HIT - {cached.get('time_taken', 0):.2f}s ahorrados")
                    return cached
                except Exception as e:
                    print(f"[Augmentation] ⚠️ Error leyendo cache: {e}")
                    # Continuar sin cache

            # Ejecutar augmentation normal
            try:
                print(f"[Augmentation] 🔄 Ejecutando augmentation...")

                result = await asyncio.wait_for(
                    self._build_derived_context(
                        query,
                        vector_search_func,
                        max_variants,
                        include_corpus_context,
                        simple_model_executor
                    ),
                    timeout=30.0
                )

                print(f"[Augmentation] ✅ Augmentation completado")

            except TimeoutError:
                print(f"[Augmentation] ⏱️ TIMEOUT en augmentation")
                return {
                    'error': 'timeout',
                    'fallback': True,
                    'derived_queries': [],
                    'time_taken': 30.0
                }

            except Exception as e:
                print(f"[Augmentation] ❌ Error en augmentation: {e}")
                return {
                    'error': str(e),
                    'fallback': True,
                    'derived_queries': [],
                    'time_taken': 0.0
                }

            # Guardar en cache (si no hubo error)
            if use_cache and not result.get('error') and not result.get('fallback'):
                try:
                    # Limitar tamaño del cache
                    if len(_AUGMENTATION_CACHE) >= _CACHE_MAX_SIZE:
                        # Eliminar entrada más antigua
                        first_key = next(iter(_AUGMENTATION_CACHE))
                        del _AUGMENTATION_CACHE[first_key]

                    _AUGMENTATION_CACHE[cache_key] = result
                    print(f"[Augmentation] 💾 Guardado en cache ({len(_AUGMENTATION_CACHE)}/{_CACHE_MAX_SIZE})")

                except Exception as e:
                    print(f"[Augmentation] ⚠️ Error guardando en cache: {e}")
                    # No fallar por error de cache

            return result

        except Exception as e:
            # Error inesperado
            print(f"[Augmentation] ❌ Error inesperado: {e}")
            import traceback
            traceback.print_exc()

            return {
                'error': str(e),
                'fallback': True,
                'derived_queries': [],
                'time_taken': 0.0
            }

    def _get_doc_id(self, doc) -> str:
        """
        Extrae ID único de un documento (dict o SearchResult).
        """
        try:
            # Convertir a dict si es necesario
            if not isinstance(doc, dict):
                doc = self._convert_to_dict(doc)

            if not doc:
                return str(id(doc))

            # Ahora doc es un dict, aplicar la lógica anterior
            doc_name = doc.get('name', '')
            if doc_name:
                return str(doc_name)

            doc_id = doc.get('id', '') or doc.get('_id', '') or doc.get('uuid', '')
            if doc_id:
                return str(doc_id)

            file_id = doc.get('file_id', '')
            chunk_index = doc.get('chunk_index')

            if file_id and chunk_index is not None:
                return f"{file_id}_{chunk_index}"

            text = doc.get('text', '')
            if text:
                import hashlib
                return hashlib.md5(text.encode('utf-8')).hexdigest()

            import json
            import hashlib
            doc_str = json.dumps(doc, sort_keys=True, default=str)
            return hashlib.md5(doc_str.encode('utf-8')).hexdigest()

        except Exception as e:
            print(f"[_get_doc_id] ⚠️ Error: {e}")
            return str(id(doc))

    async def _multi_query_search(
                self,
                original_query: str,
                vector_search_func,
                simple_model_executor,
                doc_type: str = "documentos",
                max_queries: int = 6
        ) -> List:
            """
            Multi-Query Search optimizado: Genera variaciones y busca con todas.
            v3.1: Shortcircuit de corpus saturado en dedup loop.
            """


            print(f"\n{'─' * 70}")
            print(f"[Multi-Query] 🔄 Generando {max_queries} variaciones...")
            print(f"[Multi-Query] Query original: {original_query[:60]}...")
            print(f"{'─' * 70}")

            if not original_query or not original_query.strip():
                print(f"[Multi-Query] ❌ Query vacía")
                return []

            # ═══════════════════════════════════════════════════════════════
            # GENERAR VARIACIONES
            # ═══════════════════════════════════════════════════════════════

            all_queries = [original_query]

            try:
                prompt = f"""Genera {max_queries - 1} variaciones diferentes para búsqueda semántica.

    Pregunta: "{original_query}"
    Contexto: {doc_type}

    ═══════════════════════════════════════════════════════════════════
    ESTRATEGIA
    ═══════════════════════════════════════════════════════════════════

    ESPECÍFICA (menciona "Grupo 3", "Artículo 15"):
      40% variaciones: mantén el elemento
      60% variaciones: elimina elemento, busca contexto completo

    GENÉRICA (pide "todos los grupos", "requisitos"):
      100% variaciones: reformula con sinónimos técnicos

    COMPARATIVA ("diferencia entre X e Y"):
      50% variaciones: mantén ambos elementos
      50% variaciones: contexto general

    ═══════════════════════════════════════════════════════════════════
    EJEMPLOS
    ═══════════════════════════════════════════════════════════════════

    ESPECÍFICA: "¿Periodo de prueba Grupo 3?"
    1. duración fase probatoria Grupo 3
    2. tiempo evaluación Grupo 3
    3. periodos prueba clasificación grupos
    4. tabla duración periodo prueba todos grupos

    GENÉRICA: "Resumen promoción trabajo"
    1. promoción profesional ascenso requisitos
    2. proceso mejora categoría profesional
    3. criterios progresión carrera
    4. artículos ascenso clasificación

    COMPARATIVA: "Diferencia Grupo 1 y Grupo 7"
    1. duración periodo prueba Grupo 1 versus Grupo 7
    2. comparación evaluación Grupo 1 Grupo 7
    3. periodos prueba todos grupos diferencias
    4. clasificación duración fase probatoria

    ═══════════════════════════════════════════════════════════════════
    REGLAS
    ═══════════════════════════════════════════════════════════════════

    ✓ Estructuralmente diferentes
    ✓ Sinónimos técnicos del dominio
    ✓ Mantén términos técnicos (TPM, TPC, LPRL)
    ✓ Variaciones genéricas capturan tablas completas

    Genera {max_queries - 1} variaciones (una por línea):"""

                try:
                    response_text = await asyncio.wait_for(
                        simple_model_executor(prompt),
                        timeout=40.0
                    )

                    variations = []
                    for line in response_text.strip().split('\n'):
                        line = line.strip()

                        if not line or len(line) < 15:
                            continue

                        skip_keywords = ['genera', 'variaciones:', 'output:', 'input:', '===', '───', '━━━']
                        if any(kw in line.lower() for kw in skip_keywords):
                            continue

                        if line[0].isdigit() and len(line) > 2 and line[1:3] in ['. ', ') ', '- ']:
                            line = line[3:].strip()
                        elif line[0:2] in ['- ', '• ', '* ']:
                            line = line[2:].strip()

                        if '[' in line and ']' in line:
                            bracket_pos = line.rfind('[')
                            if bracket_pos > 0:
                                line = line[:bracket_pos].strip()

                        if len(line) > 15:
                            variations.append(line)
                            if len(variations) >= max_queries - 1:
                                break

                    if len(variations) == 0:
                        print(f"[Multi-Query] ⚠️ No se generaron variaciones válidas, usando solo query original")

                    all_queries = [original_query] + variations

                    print(f"\n[Multi-Query] ✅ {len(all_queries)} queries generadas:")
                    for i, q in enumerate(all_queries, 1):
                        emoji = "🎯" if i == 1 else "🔄"
                        print(f"  {emoji} {i}. {q[:65]}{'...' if len(q) > 65 else ''}")

                except TimeoutError:
                    print(f"[Multi-Query] ⏱️ TIMEOUT generando variaciones, usando solo query original")
                    all_queries = [original_query]

                except Exception as e:
                    print(f"[Multi-Query] ⚠️ Error generando variaciones: {e}")
                    all_queries = [original_query]

            except Exception as e:
                print(f"[Multi-Query] ❌ Error crítico en generación: {e}")
                all_queries = [original_query]

            # ═══════════════════════════════════════════════════════════════
            # BÚSQUEDA PARALELA con todas las queries
            # ═══════════════════════════════════════════════════════════════

            print(f"\n[Multi-Query] 🔍 Ejecutando {len(all_queries)} búsquedas en paralelo...")

            try:
                search_tasks = [
                    asyncio.wait_for(vector_search_func(q), timeout=10.0)
                    for q in all_queries
                ]
                all_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            except Exception as e:
                print(f"[Multi-Query] ❌ Error en búsquedas paralelas: {e}")
                print(f"[Multi-Query] 🔄 Fallback: búsqueda secuencial...")
                all_results = []
                for i, q in enumerate(all_queries, 1):
                    try:
                        result = await asyncio.wait_for(vector_search_func(q), timeout=10.0)
                        all_results.append(result)
                        print(f"[Multi-Query]   ✓ Query {i}: exitosa")
                    except Exception as e:
                        print(f"[Multi-Query]   ✗ Query {i}: {e}")
                        all_results.append(Exception(str(e)))

            # ═══════════════════════════════════════════════════════════════
            # DEDUPLICACIÓN + SHORTCIRCUIT de corpus saturado
            # ═══════════════════════════════════════════════════════════════

            seen_ids = set()
            unique_docs = []
            successful_searches = 0
            _first_query_result_size = 0  # para detección de corpus saturado

            for i, results in enumerate(all_results, 1):
                if isinstance(results, Exception):
                    print(f"[Multi-Query]   ⚠️ Query {i} falló: {str(results)[:50]}")
                    continue

                if not isinstance(results, list):
                    print(f"[Multi-Query]   ⚠️ Query {i}: resultado no es lista")
                    continue

                successful_searches += 1
                docs_added = 0

                try:
                    for doc in results:
                        try:
                            if not isinstance(doc, dict):
                                if hasattr(doc, 'properties'):
                                    doc = doc.properties
                                elif hasattr(doc, '__dict__'):
                                    doc_dict = doc.__dict__.copy()
                                    if hasattr(doc, 'referencias') and 'referencias' not in doc_dict:
                                        doc_dict['referencias'] = doc.referencias
                                    doc = doc_dict
                                else:
                                    doc = {}

                            if not doc:
                                continue

                            doc_id = self._get_doc_id(doc)

                            if doc_id and doc_id not in seen_ids:
                                seen_ids.add(doc_id)
                                unique_docs.append(doc)
                                docs_added += 1

                        except Exception as e:
                            print(f"[Multi-Query]      ⚠️ Error procesando doc: {e}")
                            continue

                    print(f"[Multi-Query]   ✓ Query {i}: {len(results)} docs → {docs_added} nuevos")

                    # Shortcircuit: corpus saturado → query 2 devuelve 0 nuevos con mismo tamaño
                    if i == 1:
                        _first_query_result_size = len(results)
                    elif i == 2 and docs_added == 0 and len(
                            results) == _first_query_result_size and _first_query_result_size > 0:
                        remaining = len(all_results) - i
                        if remaining > 0:
                            print(
                                f"[Multi-Query] ⚡ Corpus saturado ({_first_query_result_size} docs total) — omitiendo {remaining} resultados redundantes")
                        break

                except Exception as e:
                    print(f"[Multi-Query]   ⚠️ Error procesando resultados de query {i}: {e}")
                    continue

            print(f"\n[Multi-Query] ✅ Búsquedas exitosas: {successful_searches}/{len(all_queries)}")
            print(f"[Multi-Query] ✅ Total: {len(unique_docs)} documentos únicos recuperados")

            if unique_docs:
                unique_filenames = set()
                for doc in unique_docs:
                    filename = doc.get('filename', '')
                    if filename:
                        unique_filenames.add(filename)

                if unique_filenames:
                    print(f"[Multi-Query] 📄 Archivos encontrados: {len(unique_filenames)}")
                    for filename in sorted(unique_filenames):
                        chunks = sum(1 for d in unique_docs if d.get('filename') == filename)
                        print(f"[Multi-Query]    • {filename} ({chunks} chunks)")

            print(f"{'─' * 70}\n")

            if len(unique_docs) == 0:
                print(f"[Multi-Query] ⚠️ ADVERTENCIA: No se recuperaron documentos")

            return unique_docs

    async def _execute_smart_search(
            self,
            query: str,
            doc_type: str,
            simple_model_executor,
            vector_search_func,
            use_multi_query: bool = True,
            max_queries: int = 6
    ) -> List:
        """
        Ejecuta búsqueda con o sin multi-query, con manejo de errores robusto.
        v2.0: max_queries como parámetro, timeout optimizado, mantiene compatibilidad.

        Args:
            query: Query del usuario
            tipo_doc: Tipo de documento
            simple_model_executor: Ejecutor del modelo
            vector_search_func: Función de búsqueda vectorial
            use_multi_query: Si usar multi-query search
            max_queries: Número de variaciones (default 5)

        Returns:
            List: Documentos encontrados (vacío si falla)
        """

        # Crear contexto de error

        # Validar query
        if not query or not query.strip():
            print(f"[Smart Search] ❌ Query vacía")
            return []

        try:
            # ═══════════════════════════════════════════════════════════
            # MULTI-QUERY SEARCH
            # ═══════════════════════════════════════════════════════════
            if use_multi_query:
                print(f"[Smart Search] 🚀 Modo: Multi-Query ({max_queries} variaciones)")

                try:
                    # 🔥 TIMEOUT OPTIMIZADO: 30s (antes 60s)
                    # 🔥 max_queries DINÁMICO (antes hardcodeado a 5)
                    docs = await asyncio.wait_for(
                        self._multi_query_search(
                            original_query=query,
                            vector_search_func=vector_search_func,
                            simple_model_executor=simple_model_executor,
                            doc_type=doc_type,
                            max_queries=max_queries
                        ),
                        timeout=45.0
                    )

                    print(f"[Smart Search] ✅ Multi-query completado: {len(docs)} docs")
                    return docs

                except TimeoutError:
                    print(f"[Smart Search] ⏱️ TIMEOUT en multi-query (30s), fallback a búsqueda simple")
                    # Fallback a búsqueda simple

                except Exception as e:
                    print(f"[Smart Search] ❌ Error en multi-query: {e}, fallback a búsqueda simple")
                    # Fallback a búsqueda simple

            # ═══════════════════════════════════════════════════════════
            # BÚSQUEDA SIMPLE (o fallback)
            # ═══════════════════════════════════════════════════════════
            print(f"[Smart Search] 🔍 Modo: Búsqueda simple")

            try:
                docs = await asyncio.wait_for(
                    vector_search_func(query),
                    timeout=10.0
                )

                print(f"[Smart Search] ✅ Búsqueda simple completada: {len(docs)} docs")
                return docs

            except TimeoutError:
                print(f"[Smart Search] ⏱️ TIMEOUT en búsqueda simple (10s)")
                raise TimeoutError("La búsqueda simple excedió el tiempo límite")

            except Exception as e:
                print(f"[Smart Search] ❌ Error en búsqueda simple: {e}")
                raise

        except (TimeoutError, RuntimeError, ValueError):
            # Re-lanzar errores ya manejados
            raise

        except Exception as e:
            print(f"[Smart Search] ❌ Error inesperado: {e}")
            import traceback
            traceback.print_exc()


            # Retornar lista vacía como fallback
            print(f"[Smart Search] ⚠️ Retornando lista vacía como fallback")
            return []


    from typing import Dict, Any, List, Callable


    async def _fallback_search(
            self,
            sub_query: str,
            vector_search_func,
            step_idx: int
    ) -> List:
        """
        Búsqueda de fallback simple con manejo de errores robusto.

        Args:
            sub_query: Query a buscar
            vector_search_func: Función de búsqueda vectorial
            step_idx: Índice del paso (para logging)

        Returns:
            List: Documentos encontrados (vacío si falla todo)
        """

        print(f"[QD Paso {step_idx}] 🔄 Iniciando búsqueda de fallback...")

        # Crear contexto de error

        # Validar sub_query
        if not sub_query or not sub_query.strip():
            print(f"[QD Paso {step_idx}] ⚠️ Query vacía en fallback")
            return []

        try:
            # Búsqueda con timeout
            try:
                print(f"[QD Paso {step_idx}] 🔍 Buscando (timeout: 5s)...")

                docs = await asyncio.wait_for(
                    vector_search_func(sub_query),
                    timeout=5.0
                )

                print(f"[QD Paso {step_idx}] ✅ Fallback búsqueda: {len(docs)} docs encontrados")

            except TimeoutError:
                print(f"[QD Paso {step_idx}] ⏱️ TIMEOUT en búsqueda fallback")
                return []

            except Exception as e:
                print(f"[QD Paso {step_idx}] ❌ Error en búsqueda fallback: {e}")
                return []

            # Validar respuesta
            if not docs or not isinstance(docs, list):
                print(f"[QD Paso {step_idx}] ⚠️ Respuesta inválida de búsqueda")
                return []

            # Rerank con timeout
            try:
                print(f"[QD Paso {step_idx}] 🎯 Reranking {min(len(docs), 20)} docs...")

                reranked = await asyncio.wait_for(
                    self._rerank_and_diversify(sub_query, docs[:20]),
                    timeout=5.0
                )

                print(f"[QD Paso {step_idx}] ✅ Fallback completado: {len(reranked)} docs finales")
                return reranked

            except TimeoutError:
                print(f"[QD Paso {step_idx}] ⏱️ TIMEOUT en rerank, retornando docs sin rerank")
                return docs[:20]

            except Exception as e:
                print(f"[QD Paso {step_idx}] ⚠️ Error en rerank: {e}, retornando docs sin rerank")
                return docs[:20]

        except Exception as e:
            print(f"[QD Paso {step_idx}] ❌ Error inesperado en fallback: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def _execute_multi_query_search(
            self,
            sub_query: str,
            augmented_step: Dict[str, Any],
            doc_type: str,
            simple_model_executor,
            vector_search_func,
            global_docs: List,
            step_idx: int
    ) -> List:
        """
        Ejecuta Multi-Query Search con fallback y manejo de errores robusto.

        Args:
            sub_query: Query del paso
            augmented_step: Resultado de augmentation del paso
            tipo_doc: Tipo de documento
            simple_model_executor: Ejecutor del modelo
            vector_search_func: Función de búsqueda vectorial
            global_docs: Documentos globales precargados
            step_idx: Índice del paso

        Returns:
            List: Documentos relevantes (vacío si falla todo)
        """

        # Crear contexto de error

        # Validar sub_query
        if not sub_query or not sub_query.strip():
            print(f"[QD Paso {step_idx}] ⚠️ Query vacía")
            return []

        try:
            # Usar docs globales si disponibles
            if global_docs and len(global_docs) > 0:
                print(f"[QD Paso {step_idx}] 📚 Usando {len(global_docs)} docs globales")

                try:
                    reranked = await asyncio.wait_for(
                        self._rerank_and_diversify(sub_query, global_docs[:25]),
                        timeout=5.0
                    )
                    print(f"[QD Paso {step_idx}] ✅ Docs globales reranked: {len(reranked)}")
                    return reranked

                except TimeoutError:
                    print(f"[QD Paso {step_idx}] ⏱️ TIMEOUT en rerank, usando docs sin rerank")
                    return global_docs[:25]

                except Exception as e:
                    print(f"[QD Paso {step_idx}] ⚠️ Error en rerank: {e}, usando docs sin rerank")
                    return global_docs[:25]

            # Multi-Query Search
            print(f"[QD Paso {step_idx}] 🚀 Ejecutando Multi-Query Search...")

            try:
                docs = await asyncio.wait_for(
                    self._execute_smart_search(
                        sub_query,
                        doc_type,
                        simple_model_executor,
                        vector_search_func,
                        augmented_result=augmented_step,
                        use_multi_query=True
                    ),
                    timeout=30.0
                )

                # Validar respuesta
                if isinstance(docs, dict):
                    if docs.get("error"):
                        print(f"[QD Paso {step_idx}] ❌ Error en smart search: {docs.get('mensaje', 'Unknown')}")
                        return await self._fallback_search(sub_query, vector_search_func, step_idx)
                    else:
                        print(f"[QD Paso {step_idx}] ⚠️ Respuesta dict inesperada")
                        return await self._fallback_search(sub_query, vector_search_func, step_idx)

                if not isinstance(docs, list):
                    print(f"[QD Paso {step_idx}] ⚠️ Respuesta no es lista: {type(docs)}")
                    return await self._fallback_search(sub_query, vector_search_func, step_idx)

                # Rerank con timeout
                try:
                    relevant_docs = await asyncio.wait_for(
                        self._rerank_and_diversify(sub_query, docs[:30]),
                        timeout=5.0
                    )
                    print(f"[QD Paso {step_idx}] ✅ Multi-Query completado: {len(relevant_docs)} docs")
                    return relevant_docs

                except TimeoutError:
                    print(f"[QD Paso {step_idx}] ⏱️ TIMEOUT en rerank, usando docs sin rerank")
                    return docs[:30]

                except Exception as e:
                    print(f"[QD Paso {step_idx}] ⚠️ Error en rerank: {e}, usando docs sin rerank")
                    return docs[:30]

            except TimeoutError:
                print(f"[QD Paso {step_idx}] ⏱️ TIMEOUT en Multi-Query - usando fallback")
                return await self._fallback_search(sub_query, vector_search_func, step_idx)

            except Exception as e:
                print(f"[QD Paso {step_idx}] ❌ Error en Multi-Query: {e} - usando fallback")
                return await self._fallback_search(sub_query, vector_search_func, step_idx)

        except Exception as e:
            print(f"[QD Paso {step_idx}] ❌ Error inesperado: {e}")
            import traceback
            traceback.print_exc()

            # Último fallback
            try:
                return await self._fallback_search(sub_query, vector_search_func, step_idx)
            except Exception as e2:
                print(f"[QD Paso {step_idx}] ❌ Fallback también falló: {e2}")
                return []

    async def _build_derived_context(
            self,
            query: str,
            vector_search_func: Callable,
            max_variants: int = 5,
            include_corpus_context: bool = False,
            timeout: float = 7.0,
            simple_model_executor=None
    ) -> Dict[str, Any]:
        """
        Query Augmentation optimizado con manejo de errores robusto.

        Optimizaciones:
        - Genera variantes + términos en UNA sola llamada LLM
        - Ejecuta búsquedas en paralelo
        - Timeout interno de 7s
        - Fallback rápido si falla

        Args:
            query: Query original del usuario
            vector_search_func: Función de búsqueda vectorial
            max_variants: Número máximo de variantes
            include_corpus_context: Si incluir contexto del corpus
            timeout: Timeout en segundos
            simple_model_executor: Ejecutor del modelo

        Returns:
            Dict: {
                "context": str,
                "search_queries": List[str],
                "key_terms": List[str],
                "variants": List[str],
                "time_taken": float,
                "fallback": bool (opcional),
                "error": bool (opcional)
            }
        """

        start = time.time()

        # Crear contexto de error

        print(f"[Augmentation] 🔄 Iniciando query augmentation...")
        print(f"[Augmentation] ⏱️ Timeout: {timeout}s")

        # Validar query
        if not query or not query.strip():
            print(f"[Augmentation] ⚠️ Query vacía, retornando fallback")
            return {
                "context": "",
                "search_queries": [],
                "key_terms": [],
                "variants": [],
                "time_taken": 0.0,
                "fallback": True,
                "error": "empty_query"
            }

        try:
            # Ejecutar augmentation con timeout
            try:
                derived = await asyncio.wait_for(
                    self._build_derived_context_internal(
                        query,
                        vector_search_func,
                        max_variants,
                        include_corpus_context,
                        simple_model_executor
                    ),
                    timeout=timeout
                )

                elapsed = time.time() - start
                derived['time_taken'] = elapsed

                print(f"[Augmentation] ✅ Completado en {elapsed:.2f}s")
                print(f"[Augmentation]    - Variantes: {len(derived.get('variants', []))}")
                print(f"[Augmentation]    - Términos clave: {len(derived.get('key_terms', []))}")

                return derived

            except TimeoutError:
                elapsed = time.time() - start
                print(f"[Augmentation] ⏱️ TIMEOUT interno ({elapsed:.2f}s) - devolviendo fallback")

                return {
                    "context": "",
                    "search_queries": [query],
                    "key_terms": [],
                    "variants": [],
                    "time_taken": elapsed,
                    "fallback": True,
                    "error": "timeout"
                }

            except Exception as e:
                elapsed = time.time() - start
                print(f"[Augmentation] ❌ Error en augmentation: {e}")

                # Log detallado del error
                error_str = str(e).lower()
                if "rate limit" in error_str or "quota" in error_str:
                    print(f"[Augmentation] ⚠️ Rate limit excedido")
                    error_type = "rate_limit"
                elif "connection" in error_str or "timeout" in error_str:
                    print(f"[Augmentation] ⚠️ Error de conexión")
                    error_type = "connection"
                else:
                    print(f"[Augmentation] ⚠️ Error genérico: {type(e).__name__}")
                    error_type = type(e).__name__

                return {
                    "context": "",
                    "search_queries": [query],
                    "key_terms": [],
                    "variants": [],
                    "time_taken": elapsed,
                    "fallback": True,
                    "error": error_type
                }

        except Exception as e:
            # Error inesperado fuera del try interno
            elapsed = time.time() - start
            print(f"[Augmentation] ❌ Error crítico inesperado: {e}")
            import traceback
            traceback.print_exc()


            return {
                "context": "",
                "search_queries": [query],
                "key_terms": [],
                "variants": [],
                "time_taken": elapsed,
                "fallback": True,
                "error": "critical"
            }

    from typing import Any, Optional

    def _extract_text(self, doc) -> str:
        """
        Extrae texto de diferentes formatos de documento con manejo de errores.

        Args:
            doc: Documento en cualquier formato

        Returns:
            str: Texto extraído (vacío si falla)
        """
        try:
            # Caso 1: Ya es string
            if isinstance(doc, str):
                return doc

            # Caso 2: Tiene atributo text
            elif hasattr(doc, 'text'):
                text = doc.text
                return str(text) if text is not None else ""

            # Caso 3: Es dict con clave 'text'
            elif isinstance(doc, dict):
                text = doc.get("text", "")
                return str(text) if text is not None else ""

            # Caso 4: Tiene atributo page_content (Document de LangChain)
            elif hasattr(doc, 'page_content'):
                content = doc.page_content
                return str(content) if content is not None else ""

            # Caso 5: Convertir a string
            else:
                return str(doc)

        except Exception as e:
            print(f"[_extract_text] ⚠️ Error extrayendo texto: {e}")
            # Intentar fallback con str()
            try:
                return str(doc)
            except:
                return ""


    async def _process_single_chunk(
                self,
                idx: int,
                doc: Any,
                sub_query: str,
                simple_model_executor,
                step_idx: int,
                timeout: float = 10.0
        ) -> Optional[Dict[str, Any]]:

            print(f"\n{'=' * 70}")
            print(f"🔵 [CHUNK {idx + 1}] Paso {step_idx} - Iniciando procesamiento")
            print(f"{'=' * 70}")


            chunk_metadata = {
                'chunk_idx': idx,
                'chunk_id': '',
                'filename': '',
                'source': '',
                'metadata_json': ''
            }

            try:
                if isinstance(doc, dict):
                    chunk_text = doc.get('text', '')
                    chunk_metadata['metadata_json'] = doc.get('metadata_json', '')
                    chunk_metadata['filename'] = doc.get('filename', '')

                elif hasattr(doc, 'metadata'):
                    metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
                    chunk_metadata['chunk_id'] = doc.get('chunk_id', doc.get('name', ''))
                    chunk_metadata['filename'] = metadata.get('filename', metadata.get('file_name', ''))
                    chunk_metadata['source'] = doc.get('source', '')
                    chunk_metadata['metadata_json'] = doc.get('metadata_json', '')

            except Exception as e:
                print(f"🔴 [METADATA] Error: {e}")
                return None

            filename_display = chunk_metadata['filename'][:40] if chunk_metadata['filename'] else 'Sin nombre'
            print(f"📄 [METADATA] Archivo: {filename_display}...")

            try:
                # ═══════════════════════════════════════════════════════════════
                # EXTRACCIÓN DE TEXTO
                # ═══════════════════════════════════════════════════════════════

                try:
                    chunk_text = self._extract_text(doc)
                    print(f"📝 [TEXTO] Extraído: {len(chunk_text)} caracteres")
                    text_preview = ' '.join(chunk_text[:150].split())
                    print(f"👁️  [PREVIEW] {text_preview}...")

                except Exception as e:
                    print(f"🔴 [TEXTO] Error extrayendo: {e}\n")
                    return None

                if not chunk_text or not chunk_text.strip():
                    print(f"⚠️  [VALIDACIÓN] Texto vacío - Descartado\n")
                    return None

                if len(chunk_text) < 50:
                    print(f"⚠️  [VALIDACIÓN] Texto muy corto ({len(chunk_text)} chars) - Descartado\n")
                    return None

                # ═══════════════════════════════════════════════════════════════
                # ANÁLISIS DE CONTENIDO
                # ═══════════════════════════════════════════════════════════════

                import re

                num_words = len(chunk_text.split())
                num_numbers = len(re.findall(r'\d+', chunk_text))
                num_mayusculas = len(re.findall(r'\b[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ]+\b', chunk_text))
                tiene_estructura = bool(
                    re.search(r'^\s*[-•*]\s', chunk_text, re.MULTILINE) or
                    re.search(r'^\s*\d+[\.)]\s', chunk_text, re.MULTILINE) or
                    re.search(r':\s*$', chunk_text, re.MULTILINE)
                )

                print(f"\n🔍 [ANÁLISIS] Características del contenido:")
                print(f"   • Palabras: {num_words}")
                print(f"   • Números: {num_numbers}")
                print(f"   • Términos destacados: {num_mayusculas}")
                print(f"   • Estructura: {'Sí' if tiene_estructura else 'No'}")

                # ═══════════════════════════════════════════════════════════════
                # CONSTRUCCIÓN DEL PROMPT
                # ═══════════════════════════════════════════════════════════════

                try:
                    prompt = f"""Eres un extractor de información preciso.

    PREGUNTA:
    {sub_query}

    DOCUMENTO:
    {chunk_text[:11000]}

    INSTRUCCIONES:

    Si el documento NO contiene información relevante para responder la pregunta, responde ÚNICAMENTE con la palabra: SIN_INFO

    Si SÍ contiene información relevante, extrae el contenido en HTML:
    - Usa <p><strong>Título:</strong> contenido</p>
    - Incluye datos específicos, referencias normativas, requisitos
    - Sé detallado pero NO inventes información que no está en el documento

    IMPORTANTE: Solo responde SIN_INFO si el documento claramente NO contiene NINGUNA información relacionada con la pregunta.
    Si hay información parcialmente relacionada, aunque sea implícita o expresada con otras palabras (por ejemplo, "trabajo en equipo" es relevante para "actitud frente a compañeros", "responsabilidad" es relevante para "competencias blandas"), extráela.
    Ante la duda, extrae el contenido.

    RESPONDE AHORA:"""

                    print(f"📋 [PROMPT] Construido: {len(prompt):,} caracteres")

                except Exception as e:
                    print(f"🔴 [PROMPT] Error: {e}\n")
                    return None

                # ═══════════════════════════════════════════════════════════════
                # LLAMADA AL LLM — GPT_5_1 directo (más capaz para inferencia semántica)
                # ═══════════════════════════════════════════════════════════════

                print(f"🤖 [LLM] Llamando modelo {GPT_5_1} (timeout: {timeout}s)...")

                response = None
                try:
                    extractor_client = openai.AsyncOpenAI(api_key=KEY_OPEN_AI)
                    extractor_response = await asyncio.wait_for(
                        extractor_client.chat.completions.create(
                            model=GPT_5_1,
                            messages=[{"role": "user", "content": prompt}],
                            max_completion_tokens=3500,
                        ),
                        timeout=timeout
                    )
                    response = extractor_response.choices[0].message.content or ""
                    print(f"✅ [LLM] Respuesta recibida ({GPT_5_1}): {len(response)} caracteres")

                except TimeoutError:
                    print(
                        f"⏱️  [LLM] Timeout GPT_5_1 después de {timeout}s, intentando fallback con executor original...")
                    try:
                        response = await asyncio.wait_for(
                            simple_model_executor(prompt),
                            timeout=timeout
                        )
                        print(f"✅ [LLM] Fallback executor: {len(response) if response else 0} caracteres")
                    except Exception as fe:
                        print(f"🔴 [LLM] Fallback también falló: {fe}\n")
                        return None

                except Exception as e:
                    print(f"🔴 [LLM] Error GPT_5_1: {e}, intentando fallback con executor original...")
                    try:
                        response = await asyncio.wait_for(
                            simple_model_executor(prompt),
                            timeout=timeout
                        )
                        print(f"✅ [LLM] Fallback executor: {len(response) if response else 0} caracteres")
                    except Exception as fe:
                        print(f"🔴 [LLM] Fallback también falló: {fe}\n")
                        return None

                # ═══════════════════════════════════════════════════════════════
                # VALIDACIÓN DE RESPUESTA
                # ═══════════════════════════════════════════════════════════════

                if not response:
                    print(f"⚠️  [RESPUESTA] Vacía del LLM\n")
                    return None

                try:
                    response_stripped = response.strip()

                    response_preview = ' '.join(response_stripped[:200].split())
                    print(f"\n📤 [RESPUESTA] Preview:")
                    print(f"   {response_preview}...")
                    print(f"   Longitud total: {len(response_stripped)} caracteres")

                except Exception as e:
                    print(f"🔴 [RESPUESTA] Error procesando: {e}\n")
                    return None

                try:
                    if len(response_stripped) < 10:
                        print(f"⚠️  [VALIDACIÓN] Respuesta muy corta ({len(response_stripped)} chars) - Descartado\n")
                        return None

                    lower_response = response_stripped.lower().strip()

                    # Rechazar SIN_INFO explícito
                    if lower_response == "sin_info":
                        print(f"⚠️  [VALIDACIÓN] SIN_INFO - Descartado\n")
                        return None

                    # Rechazar frases negativas + respuesta corta (< 300 chars)
                    soft_irrelevant = [
                        "no se encontró información",
                        "no contiene información",
                        "no hay información",
                        "documento no menciona",
                        "no se menciona"
                    ]

                    has_soft_irrelevant = any(phrase in lower_response for phrase in soft_irrelevant)

                    if has_soft_irrelevant and len(response_stripped) < 300:
                        print(
                            f"⚠️  [VALIDACIÓN] Frase negativa + respuesta corta ({len(response_stripped)} chars) - Descartado\n")
                        return None
                    elif has_soft_irrelevant:
                        print(f"✅ [VALIDACIÓN] Frase negativa pero largo ({len(response_stripped)} chars) - Aceptado")
                    else:
                        print(f"✅ [VALIDACIÓN] Contenido relevante ({len(response_stripped)} chars)")

                    # ═══════════════════════════════════════════════════════════════
                    # CONSTRUCCIÓN DE RESULTADO
                    # ═══════════════════════════════════════════════════════════════

                    result = {
                        'response': response_stripped,
                        **chunk_metadata
                    }

                    print(f"\n✅ [RESULTADO] Chunk procesado exitosamente")
                    print(f"   Filename: {result.get('filename', 'N/A')[:40]}")
                    print(f"   Response length: {len(result.get('response', ''))} chars")
                    print(f"{'=' * 70}\n")

                    return result

                except Exception as e:
                    print(f"🔴 [VALIDACIÓN] Error: {e}")
                    if 'response_stripped' in locals() and response_stripped and len(response_stripped) > 15:
                        result = {
                            'response': response_stripped,
                            **chunk_metadata
                        }
                        print(f"⚠️  [RECOVERY] Retornando con recovery ({len(response_stripped)} chars)\n")
                        return result
                    print(f"🔴 [RECOVERY] Fallo completo - Return None\n")
                    return None

            except Exception as e:
                print(f"🔴 [CRITICAL] Error crítico: {e}")
                import traceback
                traceback.print_exc()
                print(f"{'=' * 70}\n")
                return None


    def _build_step_prompt_with_cot(
            self,
            sub_query: str,
            chunks_text: str,
            context_from_previous: Optional[str] = None
    ) -> str:
        """
        Construye prompt para paso individual CON Chain-of-Thought.
        Con manejo de errores en construcción del prompt.

        Args:
            sub_query: Query del paso
            chunks_text: Texto de los chunks
            context_from_previous: Contexto de pasos anteriores (opcional)

        Returns:
            str: Prompt construido (con fallback si falla)
        """

        try:
            # Validar inputs
            if not sub_query:
                sub_query = "Pregunta no especificada"

            if not chunks_text:
                chunks_text = "(Sin información disponible)"

            # Truncar chunks_text si es muy largo
            max_chunks_length = 50000
            if len(chunks_text) > max_chunks_length:
                print(f"[_build_step_prompt] ⚠️ Truncando chunks: {len(chunks_text)} → {max_chunks_length} chars")
                chunks_text = chunks_text[:max_chunks_length] + "\n\n... (truncado)"

            few_shot_method = """
    ═══════════════════════════════════════════════════════════════════
    📖 DEFINICIONES CRÍTICAS
    ═══════════════════════════════════════════════════════════════════

    IMPORTANTE: NO confundir estos términos:

    "Más cualificado" o "Mayor cualificación" significa:
      → Tiene más formación académica previa
      → Tiene más experiencia previa
      → Tiene certificaciones previas
      → Está MÁS preparado al inicio

    "Menos cualificado" o "Menor cualificación" significa:
      → Tiene menos formación académica previa
      → Tiene menos experiencia previa
      → No tiene certificaciones previas
      → Está MENOS preparado al inicio

    Si aplicas LÓGICA INVERSA (periodo largo = menos preparado):
      → Entonces periodo LARGO = MENOR cualificación inicial
      → Y periodo CORTO = MAYOR cualificación inicial

    NO digas: "Los más cualificados tienen periodos largos y están menos preparados"
    Eso es contradictorio porque "cualificado" = "preparado"

    ═══════════════════════════════════════════════════════════════════
    🧠 MÉTODO DE RAZONAMIENTO (APLICAR SIEMPRE)
    ═══════════════════════════════════════════════════════════════════

    Cuando analices valores numéricos de categorías:

    REGLA 1: Identificar qué mide el valor
      - Si mide "tiempo de formación/adaptación/prueba necesario"
        → LÓGICA INVERSA: más tiempo = menos preparado inicialmente

      - Si mide "experiencia/certificación requerida"
        → LÓGICA DIRECTA: más requisitos = mayor nivel exigido

    REGLA 2: Aplicar consistentemente
      - Si usas lógica INVERSA:
        → Categoría con valor ALTO = menos preparada = MENOR cualificación
        → Categoría con valor BAJO = más preparada = MAYOR cualificación

      - Si usas lógica DIRECTA:
        → Categoría con valor ALTO = mayor nivel = MAYOR cualificación
        → Categoría con valor BAJO = menor nivel = MENOR cualificación

    REGLA 3: Buscar palabras clave en el contexto
      - "periodo de prueba" → generalmente INVERSA
      - "formación necesaria" → INVERSA
      - "experiencia mínima" → DIRECTA
      - "certificación requerida" → DIRECTA

    🚨 REGLA 4: VERIFICAR TÉRMINOS
      - "Cualificado" = "preparado" = POSITIVO
      - Si dices "lógica inversa: periodo largo = menos preparado"
      - Entonces DEBES decir "periodo largo = MENOR cualificación"
      - NO puedes decir "periodo largo = MÁS cualificado"
      - Porque estarías diciendo "más preparado y menos preparado" al mismo tiempo

    EJEMPLO CORRECTO:
    Datos: "Grupo 1: 6 meses, Grupo 7: 15 días"
    Palabra clave: "periodo de prueba"
    Lógica: INVERSA (periodo = tiempo de formación)
    Razonamiento: Más periodo = más formación necesaria = menos preparado = MENOR cualificación
    Conclusión: Grupo 1 (6 meses) = MENOR cualificación inicial
               Grupo 7 (15 días) = MAYOR cualificación inicial

    EJEMPLO INCORRECTO:
    Lógica: INVERSA (más tiempo = menos preparado)
    Conclusión: "Grupos con más tiempo son más cualificados" ❌
    Esto es contradictorio: no pueden estar "menos preparados" y ser "más cualificados"

    ═══════════════════════════════════════════════════════════════════
    """

            context_section = ""
            if context_from_previous:
                try:
                    # Truncar contexto previo si es muy largo
                    max_context_length = 10000
                    if len(context_from_previous) > max_context_length:
                        context_from_previous = context_from_previous[:max_context_length] + "\n... (truncado)"

                    context_section = f"\nContexto de pasos anteriores:\n{context_from_previous}"
                except Exception as e:
                    print(f"[_build_step_prompt] ⚠️ Error procesando contexto previo: {e}")
                    context_section = ""

            prompt = f"""{few_shot_method}

    ══════════════════════════════════════════════════════════════════════
    📊 INFORMACIÓN DISPONIBLE
    ══════════════════════════════════════════════════════════════════════

    {chunks_text}
    {context_section}

    ══════════════════════════════════════════════════════════════════════
    🎯 TU TAREA
    ══════════════════════════════════════════════════════════════════════

    {sub_query}

    🚨 OBLIGATORIO: Razona explícitamente antes de responder

    <reasoning>

    PASO 1: ¿Qué categorías y valores aparecen?
    [Listar exactamente como aparecen en el contexto]

    PASO 2: ¿Qué miden esos valores según el contexto?
    [Buscar palabras clave: "periodo", "formación", "experiencia", etc.]

    PASO 3: ¿Qué lógica aplico?
    [INVERSA si mide necesidad, DIRECTA si mide requisito]

    PASO 4: Aplicar lógica y determinar cualificación
    [Si INVERSA: valor alto = menos preparado = MENOR cualificación]
    [Si DIRECTA: valor alto = más exigencia = MAYOR cualificación]

    PASO 5: VERIFICACIÓN DE TÉRMINOS
    □ Apliqué lógica: [INVERSA/DIRECTA]
    □ Por tanto, valores altos significan: [menos/más] cualificación
    □ ¿Voy a usar los términos correctamente?
      - Si INVERSA + valor alto → debo decir "MENOR cualificación"
      - Si DIRECTA + valor alto → debo decir "MAYOR cualificación"

    PASO 6: Conclusión (términos verificados)
    [Usar términos consistentes: cualificado = preparado]

    </reasoning>

    ══════════════════════════════════════════════════════════════════════
    🎯 GENERACIÓN DE RESPUESTA
    ══════════════════════════════════════════════════════════════════════

    Genera tu respuesta HTML.

    🚨 VERIFICACIÓN FINAL:
    - Si dijiste "lógica inversa" y "menos preparado"
    - Entonces DEBES decir "MENOR cualificación" (NO "más cualificados")
    - Porque "cualificado" = "preparado"

    Ahora genera tu respuesta HTML aplicando correctamente los términos."""

            return prompt

        except Exception as e:
            print(f"[_build_step_prompt] ❌ Error crítico construyendo prompt: {e}")
            import traceback
            traceback.print_exc()

            # Fallback: prompt mínimo
            return f"""Analiza la siguiente información y responde la pregunta:

    INFORMACIÓN:
    {chunks_text[:10000] if chunks_text else "(Sin información)"}

    PREGUNTA:
    {sub_query if sub_query else "Pregunta no especificada"}

    Responde en formato HTML."""

    from typing import Dict, Any, List, Callable, Optional

    async def _synthesize_step_response(
            self,
            valid_results: List[Dict[str, Any]],  # ✅ CAMBIADO: Dict en lugar de str
            sub_query: str,
            simple_model_executor,
            step_idx: int,
            timeout: float = 20.0
    ) -> str:
        """
        Sintetiza respuesta de un paso CON Chain-of-Thought y manejo de errores robusto.

        Args:
            valid_results: Lista de chunks procesados (dicts con 'response' y metadata)
            sub_query: Query del paso
            simple_model_executor: Ejecutor del modelo
            step_idx: Índice del paso
            timeout: Timeout en segundos

        Returns:
            str: Respuesta HTML sintetizada (con fallback si falla)
        """

        # Crear contexto de error

        # Validar inputs
        if not valid_results or len(valid_results) == 0:
            print(f"[QD Paso {step_idx}] ⚠️ Sin chunks válidos para sintetizar")
            return "<p><em>No se encontró información suficiente.</em></p>"

        if not sub_query or not sub_query.strip():
            print(f"[QD Paso {step_idx}] ⚠️ Query vacía para síntesis")
            return "<p><em>Error: consulta no especificada.</em></p>"

        # ═══════════════════════════════════════════════════════════════
        # ✅ NUEVO: EXTRAER RESPONSES Y METADATA DE LOS DICTS
        # ═══════════════════════════════════════════════════════════════

        chunk_responses = []
        filenames = []

        for result in valid_results:
            if isinstance(result, dict):
                response = result.get('response', '')
                filename = result.get('filename', '')

                if response:
                    chunk_responses.append(response)
                if filename:
                    filenames.append(filename)
            elif isinstance(result, str):
                # Fallback por si algún chunk todavía es string
                chunk_responses.append(result)

        if not chunk_responses:
            print(f"[QD Paso {step_idx}] ⚠️ Chunks sin respuestas válidas")
            return "<p><em>No se encontró información suficiente.</em></p>"

        # Información de fuentes
        unique_filenames = list(set([f for f in filenames if f]))

        if unique_filenames:
            print(f"[QD Paso {step_idx}] 📁 Fuentes: {len(unique_filenames)} archivos únicos")
            print(f"[QD Paso {step_idx}] 📄 Ejemplos: {unique_filenames[:3]}")

        print(f"[QD Paso {step_idx}] 🔄 Sintetizando {len(chunk_responses)} chunks...")

        try:
            # Combinar chunks
            try:
                chunks_to_combine = chunk_responses[:10]  # Limitar a 10
                combined_context = "\n\n---\n\n".join([
                    f"Fuente {i + 1}:\n{chunk[:5000]}"  # Truncar cada chunk a 5K chars
                    for i, chunk in enumerate(chunks_to_combine)
                ])

                # Verificar longitud total
                if len(combined_context) > 50000:
                    print(f"[QD Paso {step_idx}] ⚠️ Contexto muy largo, truncando: {len(combined_context)} chars")
                    combined_context = combined_context[:50000] + "\n\n... (truncado)"

            except Exception as e:
                print(f"[QD Paso {step_idx}] ❌ Error combinando chunks: {e}")
                # Fallback: usar solo primer chunk
                combined_context = f"Fuente 1:\n{chunk_responses[0][:10000]}"

            # Construir prompt CON few-shot CoT
            try:
                synthesis_prompt = self._build_step_prompt_with_cot(
                    sub_query=sub_query,
                    chunks_text=combined_context,
                    context_from_previous=None
                )
            except Exception as e:
                print(f"[QD Paso {step_idx}] ❌ Error construyendo prompt: {e}")
                return "<p>⚠️ Error interno al construir el prompt.</p>"

            # Llamar a GPT-4O con razonamiento
            try:
                from openai import AsyncOpenAI
                from config.Constants import KEY_OPEN_AI

                client = AsyncOpenAI(api_key=KEY_OPEN_AI)
            except Exception as e:
                print(f"[QD Paso {step_idx}] ❌ Error inicializando cliente OpenAI: {e}")
                return "<p>⚠️ Error configurando el servicio de IA.</p>"

            try:
                print(f"[QD Paso {step_idx}] 🤖 Llamando a GPT-4O (timeout: {timeout}s)...")

                response_obj = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=GPT_5_4_MINI,  # ✅ CORREGIDO: "gpt-4.1" → "gpt-5.6"
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Eres un asistente de análisis que aplica razonamiento lógico riguroso. "
                                    "Determinas qué miden los valores ANTES de interpretarlos. "
                                    "Aplicas lógica INVERSA cuando valores miden necesidad/formación, "
                                    "y lógica DIRECTA cuando miden requisitos de entrada. "
                                    "SIEMPRE razonas con <reasoning> tags antes de responder."
                                )
                            },
                            {
                                "role": "user",
                                "content": synthesis_prompt
                            }
                        ],
                        #temperature=0.1,
                        max_completion_tokens=7000,
                        reasoning_effort="low"
                    ),
                    timeout=timeout
                )

                response = response_obj.choices[0].message.content

                if not response or not response.strip():
                    print(f"[QD Paso {step_idx}] ⚠️ Respuesta vacía de GPT-4O")
                    return "<p><em>No se pudo generar una respuesta. Intenta reformular la consulta.</em></p>"

                # Extraer y loguear razonamiento
                try:
                    if "<reasoning>" in response:
                        reasoning_match = re.search(
                            r'<reasoning>(.*?)</reasoning>',
                            response,
                            re.DOTALL
                        )
                        if reasoning_match:
                            reasoning_text = reasoning_match.group(1).strip()
                            print(f"[QD Paso {step_idx}] 🧠 Razonamiento generado:")

                            # Mostrar primeras líneas
                            reasoning_lines = reasoning_text.split('\n')[:8]
                            for line in reasoning_lines:
                                if line.strip():
                                    print(f"  {line.strip()[:70]}")

                        # Remover reasoning del HTML
                        response = re.sub(
                            r'<reasoning>.*?</reasoning>',
                            '',
                            response,
                            flags=re.DOTALL
                        ).strip()
                    else:
                        print(f"[QD Paso {step_idx}] ⚠️ No se generó razonamiento explícito")
                except Exception as e:
                    print(f"[QD Paso {step_idx}] ⚠️ Error procesando razonamiento: {e}")
                    # Continuar con la respuesta completa

                print(f"[QD Paso {step_idx}] ✅ Síntesis generada ({len(response)} chars)")

                # ✅ OPCIONAL: Agregar nota de fuentes al final
                if unique_filenames and len(unique_filenames) <= 5:
                    sources_note = "<p><small><em>Fuentes: " + ", ".join(unique_filenames[:5]) + "</em></small></p>"
                    response = response + "\n\n" + sources_note

                return response

            except TimeoutError:
                print(f"[QD Paso {step_idx}] ⏱️ TIMEOUT en síntesis ({timeout}s)")

                # ✅ Fallback: Retornar chunks combinados sin síntesis
                print(f"[QD Paso {step_idx}] 🔄 Fallback: Retornando chunks sin síntesis")
                fallback_response = "<div>" + "\n\n".join(chunk_responses[:5]) + "</div>"
                return fallback_response

            except Exception as e:
                print(f"[QD Paso {step_idx}] ❌ Error en llamada a GPT-4O: {e}")

                error_str = str(e).lower()
                if "rate limit" in error_str or "quota" in error_str:
                    return "<p>⚠️ Límite de uso excedido. Por favor intenta nuevamente en unos momentos.</p>"
                elif "api key" in error_str:
                    return "<p>⚠️ Error de configuración del servicio. Contacta al administrador.</p>"
                else:
                    # ✅ Fallback con chunks combinados
                    print(f"[QD Paso {step_idx}] 🔄 Fallback por error: Retornando chunks sin síntesis")
                    fallback_response = "<div>" + "\n\n".join(chunk_responses[:5]) + "</div>"
                    return fallback_response

        except Exception as e:
            print(f"[QD Paso {step_idx}] ❌ Error crítico en síntesis: {e}")
            import traceback
            traceback.print_exc()


            # ✅ Último fallback
            if chunk_responses:
                print(f"[QD Paso {step_idx}] 🔄 Fallback crítico: Retornando primer chunk")
                return f"<div>{chunk_responses[0][:3000]}</div>"

            return "<p>⚠️ Error crítico al procesar la respuesta. Por favor intenta nuevamente.</p>"

    async def _build_derived_context_internal(
            self,
            query: str,
            vector_search_func: Callable,
            max_variants: int,
            include_corpus_context: bool,
            simple_model_executor
    ) -> Dict[str, Any]:
        """
        Lógica interna de augmentation con manejo de errores robusto.
        (sin timeout, controlado por wrapper)

        Returns:
            Dict: Siempre retorna dict válido con fallback
        """

        start = time.time()

        # Crear contexto de error

        # Validación crítica
        if simple_model_executor is None:
            print(f"[Augmentation] ⚠️ simple_model_executor es None - retornando fallback")
            return {
                "context": "",
                "search_queries": [query],
                "key_terms": [],
                "variants": [],
                "time_taken": time.time() - start,
                "error": "no_executor"
            }

        if not query or not query.strip():
            print(f"[Augmentation] ⚠️ Query vacía")
            return {
                "context": "",
                "search_queries": [],
                "key_terms": [],
                "variants": [],
                "time_taken": time.time() - start,
                "error": "empty_query"
            }

        try:
            # ============================================================
            # OPTIMIZACIÓN 1: Una sola llamada LLM para todo
            # ============================================================

            augmentation_prompt = f"""Analiza esta consulta y genera variantes semánticas + términos clave.

    **CONSULTA ORIGINAL:**
    {query}

    **TAREA:**
    Genera EXACTAMENTE {max_variants} variantes semánticas que:
    - Mantengan el significado original
    - Usen sinónimos y reformulaciones
    - Sean útiles para búsqueda vectorial

    También extrae los términos clave más importantes (conceptos, entidades, fechas, números).

    **FORMATO DE RESPUESTA (JSON estricto):**
    {{
      "variants": [
        "variante 1 aquí",
        "variante 2 aquí",
        "variante 3 aquí"
      ],
      "key_terms": ["término1", "término2", "término3"]
    }}

    **RESPONDE SOLO CON EL JSON, SIN EXPLICACIONES:**"""

            # Llamada LLM con timeout
            try:
                print(f"[Augmentation] 🤖 Generando {max_variants} variantes (timeout: 4s)...")

                llm_response = await asyncio.wait_for(
                    simple_model_executor(augmentation_prompt),
                    timeout=10.0
                )

                # Parsear JSON
                try:
                    llm_response_clean = llm_response.strip()

                    # Limpiar markdown
                    if '```json' in llm_response_clean:
                        llm_response_clean = llm_response_clean.split('```json')[1].split('```')[0].strip()
                    elif '```' in llm_response_clean:
                        llm_response_clean = llm_response_clean.split('```')[1].split('```')[0].strip()

                    parsed = json.loads(llm_response_clean)

                    variants = parsed.get('variants', [])[:max_variants]
                    key_terms = parsed.get('key_terms', [])[:8]

                    # Validar variantes
                    if not variants or len(variants) == 0:
                        print(f"[Augmentation] ⚠️ Sin variantes válidas, usando query original")
                        variants = [query]
                    else:
                        # Filtrar variantes vacías
                        variants = [v.strip() for v in variants if v and v.strip()]
                        if not variants:
                            variants = [query]

                    print(f"[Augmentation] ✅ Generadas {len(variants)} variantes + {len(key_terms)} términos")

                except json.JSONDecodeError as e:
                    print(f"[Augmentation] ⚠️ Error parsing JSON: {e}")
                    print(f"[Augmentation]    Response: {llm_response[:200]}...")
                    variants = [query]
                    key_terms = []

                except Exception as e:
                    print(f"[Augmentation] ⚠️ Error procesando respuesta LLM: {e}")
                    variants = [query]
                    key_terms = []

            except TimeoutError:
                print(f"[Augmentation] ⏱️ TIMEOUT LLM (4s) - usando fallback")
                variants = [query]
                key_terms = []

            except Exception as e:
                print(f"[Augmentation] ❌ Error en llamada LLM: {e}")
                variants = [query]
                key_terms = []

            # ============================================================
            # OPTIMIZACIÓN 2: Búsquedas en paralelo (si están habilitadas)
            # ============================================================

            search_queries = [query] + variants[:4]  # Máximo 5 queries totales
            corpus_context = ""

            if include_corpus_context:
                print(f"[Augmentation] 🔍 Ejecutando {len(search_queries)} búsquedas en paralelo...")

                try:
                    # Búsquedas paralelas con timeout
                    search_tasks = [
                        asyncio.wait_for(vector_search_func(q), timeout=2.0)
                        for q in search_queries
                    ]

                    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

                    # Filtrar errores
                    valid_results = [
                        r for r in search_results
                        if r and not isinstance(r, Exception) and isinstance(r, list)
                    ]

                    print(f"[Augmentation]    ✓ {len(valid_results)}/{len(search_queries)} búsquedas exitosas")

                    # Extraer contexto
                    context_parts = []
                    seen_doc_ids = set()

                    for result_list in valid_results:
                        try:
                            for doc in result_list[:3]:  # Top 3 por búsqueda
                                try:
                                    doc_id = self._get_doc_id(doc)

                                    if doc_id and doc_id not in seen_doc_ids:
                                        seen_doc_ids.add(doc_id)

                                        # Extraer texto
                                        doc_text = self._extract_text(doc)
                                        if doc_text and len(doc_text) > 50:
                                            context_parts.append(doc_text[:500])

                                    if len(context_parts) >= 5:
                                        break
                                except Exception as e:
                                    print(f"[Augmentation]      ⚠️ Error procesando doc: {e}")
                                    continue

                            if len(context_parts) >= 5:
                                break
                        except Exception as e:
                            print(f"[Augmentation]    ⚠️ Error procesando result_list: {e}")
                            continue

                    corpus_context = "\n\n".join(context_parts)
                    print(f"[Augmentation] ✅ Contexto: {len(corpus_context)} chars de {len(context_parts)} docs")

                except Exception as e:
                    print(f"[Augmentation] ❌ Error en búsquedas paralelas: {e}")
                    corpus_context = ""
            else:
                print(f"[Augmentation] ⏭️ Skip corpus_context (include_corpus_context=False)")

            # ============================================================
            # RESULTADO FINAL
            # ============================================================

            elapsed = time.time() - start

            result = {
                "context": corpus_context,
                "search_queries": search_queries,
                "key_terms": key_terms,
                "variants": variants,
                "time_taken": elapsed
            }

            print(f"[Augmentation] ✅ Completado en {elapsed:.2f}s")

            return result

        except Exception as e:
            print(f"[Augmentation] ❌ Error crítico: {e}")
            import traceback
            traceback.print_exc()

            elapsed = time.time() - start

            return {
                "context": "",
                "search_queries": [query],
                "key_terms": [],
                "variants": [],
                "time_taken": elapsed,
                "error": type(e).__name__
            }

    async def _safe_augmentation(
            self,
            sub_query: str,
            vector_search_func,
            step_idx: int,
            timeout: float = 8.0,
            use_cache: bool = False,
            simple_model_executor=None,
            max_variants: int = 3
    ) -> Dict[str, Any]:
        """
        Query Augmentation con timeout, configuración optimizada y manejo de errores robusto.

        Args:
            sub_query: Query del paso
            vector_search_func: Función de búsqueda vectorial
            step_idx: Índice del paso
            timeout: Timeout en segundos
            use_cache: Si usar cache
            simple_model_executor: Ejecutor del modelo
            max_variants: Número de variantes a generar
                         - 0: Skip augmentation (usar _safe_augmentation_light)
                         - 2-3: Modo rápido para QueryDecomposer
                         - 5: Modo completo para pipeline estándar

        Returns:
            Dict: Siempre retorna dict válido con fallback
        """

        # Crear contexto de error

        # Validar query
        if not sub_query or not sub_query.strip():
            print(f"[Aug Paso {step_idx}] ⚠️ Query vacía")
            return {
                "context": "",
                "search_queries": [],
                "key_terms": [],
                "variants": [],
                "error": "empty_query"
            }

        try:
            # ✅ OPTIMIZACIÓN: Si max_variants=0, retornar mínimo
            if max_variants == 0:
                print(f"[Aug Paso {step_idx}] ⏭️ Skip augmentation (max_variants=0)")
                return {
                    "context": "",
                    "search_queries": [sub_query],
                    "key_terms": [],
                    "variants": []
                }

            aug_start = time.time()

            # Ejecutar augmentation con timeout
            try:
                if use_cache:
                    print(f"[Aug Paso {step_idx}] 🔄 Modo: Con cache")
                    result = await asyncio.wait_for(
                        self._build_derived_context_with_cache(
                            sub_query,
                            vector_search_func,
                            max_variants=max_variants,
                            include_corpus_context=False,
                            use_cache=True,
                            simple_model_executor=simple_model_executor
                        ),
                        timeout=timeout
                    )
                else:
                    print(f"[Aug Paso {step_idx}] 🔄 Modo: Sin cache")
                    result = await asyncio.wait_for(
                        self._build_derived_context(
                            sub_query,
                            vector_search_func,
                            max_variants=max_variants,
                            include_corpus_context=False,
                            simple_model_executor=simple_model_executor
                        ),
                        timeout=timeout
                    )

                aug_time = time.time() - aug_start
                print(f"[Aug Paso {step_idx}] ✅ Completado en {aug_time:.2f}s ({max_variants} variantes)")

                return result

            except TimeoutError:
                aug_time = time.time() - aug_start
                print(f"[Aug Paso {step_idx}] ⏱️ TIMEOUT ({aug_time:.2f}s)")
                return {
                    "context": "",
                    "search_queries": [sub_query],
                    "key_terms": [],
                    "variants": [],
                    "error": "timeout"
                }

            except Exception as e:
                aug_time = time.time() - aug_start
                print(f"[Aug Paso {step_idx}] ❌ Error en augmentation: {e}")

                error_str = str(e).lower()
                if "rate limit" in error_str:
                    error_type = "rate_limit"
                elif "connection" in error_str:
                    error_type = "connection"
                else:
                    error_type = type(e).__name__

                return {
                    "context": "",
                    "search_queries": [sub_query],
                    "key_terms": [],
                    "variants": [],
                    "error": error_type
                }

        except Exception as e:
            print(f"[Aug Paso {step_idx}] ❌ Error crítico: {e}")
            import traceback
            traceback.print_exc()


            return {
                "context": "",
                "search_queries": [sub_query],
                "key_terms": [],
                "variants": [],
                "error": "critical"
            }

    def _build_universal_few_shot_examples(self) -> str:
        """
        Ejemplos few-shot universales que enseñan CÓMO razonar con valores numéricos.
        100% genérico - funciona para CUALQUIER dominio.

        Returns:
            str: String con ejemplos (nunca falla)
        """
        try:
            return """
    ═══════════════════════════════════════════════════════════════════
    📚 MÉTODO DE RAZONAMIENTO UNIVERSAL
    ═══════════════════════════════════════════════════════════════════

    REGLA DE ORO: El SIGNIFICADO de un valor determina su interpretación.

    ═══════════════════════════════════════════════════════════════════
    EJEMPLO 1: Valores que representan RECURSOS NECESARIOS
    ═══════════════════════════════════════════════════════════════════

    Datos:
    - Item A: necesita 100 unidades de recurso X
    - Item B: necesita 10 unidades de recurso X

    Pregunta: ¿Cuál es más avanzado/preparado?

    ❌ ERROR: "A tiene valor mayor → A es superior"

    ✅ CORRECTO:
    PASO 1: ¿Qué es X? → tiempo de entrenamiento
    PASO 2: Lógica inversa
      - Más entrenamiento necesario = menos preparado inicialmente
      - Menos entrenamiento necesario = más preparado inicialmente
    PASO 3: Conclusión
      - A (100 unidades) = MENOS preparado
      - B (10 unidades) = MÁS preparado

    PATRÓN: Cuando el valor mide "necesidad", más valor = menos capacidad previa

    ═══════════════════════════════════════════════════════════════════
    EJEMPLO 2: Valores que representan REQUISITOS MÍNIMOS
    ═══════════════════════════════════════════════════════════════════

    Datos:
    - Posición A: requiere mínimo 100 unidades de experiencia
    - Posición B: requiere mínimo 10 unidades de experiencia

    Pregunta: ¿Cuál requiere más nivel?

    ✅ CORRECTO:
    PASO 1: ¿Qué es la experiencia aquí? → requisito de entrada
    PASO 2: Lógica directa
      - Más requisitos = mayor nivel exigido
      - Menos requisitos = menor nivel exigido
    PASO 3: Conclusión
      - A (100) = MAYOR nivel requerido
      - B (10) = MENOR nivel requerido

    PATRÓN: Cuando el valor mide "requisito", más valor = más nivel exigido

    ═══════════════════════════════════════════════════════════════════
    MÉTODO UNIVERSAL DE ANÁLISIS
    ═══════════════════════════════════════════════════════════════════

    Para CUALQUIER par de valores (Categoría X: valor alto, Categoría Y: valor bajo):

    PASO 1: IDENTIFICAR QUÉ MIDE EL VALOR
    □ ¿Es un recurso/tiempo NECESARIO para preparar?
    □ ¿Es un requisito/nivel MÍNIMO para acceder?
    □ ¿Es ambiguo? → Buscar descripciones en el contexto

    PASO 2: APLICAR LÓGICA CORRECTA
    Si mide NECESIDAD:
      → Más valor = necesita más preparación = menos preparado inicialmente

    Si mide REQUISITO:
      → Más valor = se exige más = mayor nivel de entrada

    PASO 3: VERIFICAR CON CONTEXTO
    □ ¿Hay descripciones que confirmen mi interpretación?
    □ ¿Hay palabras clave como "requiere", "necesita", "debe tener"?
    □ Si hay ambigüedad, reconocerla explícitamente

    ═══════════════════════════════════════════════════════════════════
    """
        except Exception as e:
            print(f"[_build_universal_few_shot_examples] ⚠️ Error: {e}")
            return "# Error generando ejemplos few-shot\n"

    async def _check_aggregation_needed(
            self,
            query: str,
            simple_model_executor
    ) -> bool:
        """
        Verifica si la query requiere agregación de datos con manejo de errores robusto.

        Returns:
            bool: True si requiere agregación, False en caso contrario o error
        """

        # Crear contexto de error

        # Validar inputs
        if not query or not query.strip():
            print(f"[PHASE 3] ⚠️ Query vacía, asumiendo False")
            return False

        if simple_model_executor is None:
            print(f"[PHASE 3] ⚠️ Executor es None, asumiendo False")
            return False

        try:
            print(f"[PHASE 3] 🧮 Verificando necesidad de agregación...")

            agg_prompt = f"""Analiza si esta query requiere cálculos o agregación de múltiples elementos.

    Query: "{query}"

    Responde SOLO con este JSON exacto (sin markdown, sin explicaciones):
    {{"needs_aggregation": true}}
    o
    {{"needs_aggregation": false}}

    JSON:"""

            try:
                agg_resp = await asyncio.wait_for(
                    simple_model_executor(agg_prompt),
                    timeout=5.0
                )
            except TimeoutError:
                print(f"[PHASE 3] ⏱️ TIMEOUT (5s), asumiendo False")
                return False
            except Exception as e:
                print(f"[PHASE 3] ❌ Error en LLM: {e}, asumiendo False")
                return False

            # Validar respuesta
            if not agg_resp or not agg_resp.strip():
                print(f"[PHASE 3] ⚠️ Respuesta vacía, asumiendo False")
                return False

            # Limpiar respuesta
            try:
                agg_clean = str(agg_resp).strip()

                # Remover markdown
                agg_clean = agg_clean.replace('```json', '').replace('```', '').strip()

                # Extraer JSON con regex
                json_match = re.search(
                    r'\{[^}]*"needs_aggregation"[^}]*\}',
                    agg_clean,
                    re.IGNORECASE
                )

                if json_match:
                    agg_clean = json_match.group(0)

                # Intentar parsear JSON
                try:
                    parsed = json.loads(agg_clean)
                    result = bool(parsed.get('needs_aggregation', False))
                    print(f"[PHASE 3] {'✅' if result else '❌'} Agregación: {result}")
                    return result

                except json.JSONDecodeError as je:
                    # Fallback: buscar palabras clave
                    print(f"[PHASE 3] ⚠️ JSON inválido: {je}")
                    print(f"[PHASE 3] 📄 Respuesta: {agg_clean[:150]}")

                    # Buscar indicadores
                    agg_lower = agg_clean.lower()

                    # Búsqueda estricta
                    if '"needs_aggregation": true' in agg_lower or '"needs_aggregation":true' in agg_lower:
                        print(f"[PHASE 3] ✅ Detectado 'true' explícito → True")
                        return True

                    # Búsqueda flexible
                    if 'true' in agg_lower and 'aggregation' in agg_lower:
                        print(f"[PHASE 3] ✅ Detectado indicadores → True")
                        return True

                    print(f"[PHASE 3] ❌ No detectado True → False")
                    return False

            except Exception as e:
                print(f"[PHASE 3] ⚠️ Error procesando respuesta: {e}, asumiendo False")
                return False

        except Exception as e:
            print(f"[PHASE 3] ❌ Error crítico: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False

    from typing import List, Tuple, Callable

    async def _do_multi_query_search(
            self,
            query: str,
            vector_search_func: Callable,
            simple_model_executor,
            doc_type: str
    ) -> List:
        """
        Ejecuta multi-query search con fallback robusto a búsqueda simple.

        Returns:
            List: Documentos recuperados (vacío si todo falla)
        """

        # Crear contexto de error

        # Validar query
        if not query or not query.strip():
            print(f"[PHASE 3] ⚠️ Query vacía para multi-query search")
            return []

        if not vector_search_func:
            print(f"[PHASE 3] ⚠️ vector_search_func no disponible")
            return []

        try:
            print(f"[PHASE 3] 🔍 Multi-query search (5 variaciones)...")
            print(f"[PHASE 3] ⏱️ Timeout: 10s")

            try:
                docs = await asyncio.wait_for(
                    self._multi_query_search(
                        original_query=query,
                        vector_search_func=vector_search_func,
                        simple_model_executor=simple_model_executor,
                        doc_type=doc_type,
                        max_queries=5
                    ),
                    timeout=10.0
                )

                # Validar resultado
                if not isinstance(docs, list):
                    print(f"[PHASE 3] ⚠️ Multi-query retornó tipo inválido: {type(docs)}")
                    docs = []

                print(f"[PHASE 3] ✅ Multi-query: {len(docs)} documentos recuperados")
                return docs

            except TimeoutError:
                print(f"[PHASE 3] ⏱️ TIMEOUT en multi-query → Fallback a búsqueda simple")

            except Exception as e:
                print(f"[PHASE 3] ❌ Error en multi-query: {e}")
                import traceback
                traceback.print_exc()

            # Fallback: búsqueda simple
            try:
                print(f"[PHASE 3] 🔄 Ejecutando fallback búsqueda simple...")

                docs = await asyncio.wait_for(
                    vector_search_func(query),
                    timeout=5.0
                )

                if not isinstance(docs, list):
                    print(f"[PHASE 3] ⚠️ Fallback retornó tipo inválido: {type(docs)}")
                    return []

                print(f"[PHASE 3] ✅ Búsqueda simple: {len(docs)} documentos")
                return docs

            except TimeoutError:
                print(f"[PHASE 3] ⏱️ TIMEOUT en fallback búsqueda simple")
                return []

            except Exception as e:
                print(f"[PHASE 3] ❌ Error en fallback: {e}")
                return []

        except Exception as e:
            print(f"[PHASE 3] ❌ Error crítico: {e}")
            import traceback
            traceback.print_exc()


            return []

    async def _rerank_and_diversify(
            self,
            query: str,
            relevant_docs: List
    ) -> List:


        if not relevant_docs:
            print(f"[Rerank] ⚠️ Lista de docs vacía")
            return []

        if not isinstance(relevant_docs, list):
            print(f"[Rerank] ⚠️ relevant_docs no es lista: {type(relevant_docs)}")
            return []

        if not query or not query.strip():
            print(f"[Rerank] ⚠️ Query vacía, retornando docs sin rerank")
            return relevant_docs

        if len(relevant_docs) < 8:
            print(f"[Rerank] ⏭️ Skip - solo {len(relevant_docs)} docs")
            return relevant_docs

        try:
            print(f"[Rerank] 🎯 Iniciando rerank de {len(relevant_docs)} docs...")

            try:
                reranker = FastOpenAIReranker()
            except ImportError as e:
                print(f"[Rerank] ⚠️ No se pudo importar FastOpenAIReranker: {e}")
                return relevant_docs[:20]
            except Exception as e:
                print(f"[Rerank] ⚠️ Error inicializando reranker: {e}")
                return relevant_docs[:20]

            try:
                reranked_docs = await asyncio.wait_for(
                    reranker.rerank_documents(query, relevant_docs, top_k=25),
                    timeout=60.0
                )

                if not reranked_docs or not isinstance(reranked_docs, list):
                    print(f"[Rerank] ⚠️ Resultado inválido: {type(reranked_docs)}")
                    return relevant_docs[:20]

                print(f"[Rerank] ✅ Aplicado → {len(reranked_docs)} docs (top_k=25)")
                return reranked_docs

            except TimeoutError:
                print(f"[Rerank] ⏱️ TIMEOUT (60s), retornando docs sin rerank")
                return relevant_docs[:20]

            except Exception as e:
                print(f"[Rerank] ❌ Error en reranking: {e}")
                return relevant_docs[:20]

        except Exception as e:
            print(f"[Rerank] ❌ Error crítico: {e}")
            import traceback
            traceback.print_exc()
            return relevant_docs[:20] if relevant_docs else []


    def _consolidate_group_chunks(self, chunks: List) -> List:
        """
        Consolida chunks que contienen el mismo grupo en un solo chunk.
        Con manejo de errores robusto.

        Args:
            chunks: Lista de chunks a consolidar

        Returns:
            List: Chunks consolidados (original si falla)
        """

        # Crear contexto de error

        # Validar input
        if not chunks:
            print(f"[CONSOLIDATE] ⚠️ Lista de chunks vacía")
            return []

        if not isinstance(chunks, list):
            print(f"[CONSOLIDATE] ⚠️ chunks no es lista: {type(chunks)}")
            return chunks if chunks else []

        try:
            print(f"[CONSOLIDATE] 🔄 Consolidando {len(chunks)} chunks con grupos duplicados...")

            # Diccionario para agrupar chunks
            group_map = defaultdict(list)
            non_group_chunks = []

            for i, chunk in enumerate(chunks):
                try:
                    # Extraer contenido
                    content = ""

                    if hasattr(chunk, 'page_content'):
                        content = str(chunk.page_content)
                    elif isinstance(chunk, dict):
                        content = str(chunk.get('text', chunk.get('content', chunk.get('page_content', ''))))  # ✅ Agregar 'text'
                    else:
                        content = str(chunk)

                    if not content or not content.strip():
                        print(f"[CONSOLIDATE]   ⚠️ Chunk {i} vacío")
                        continue

                    # Buscar grupos en el contenido
                    try:
                        grupos_encontrados = re.findall(r'Grupo\s+(\d+)', content, re.IGNORECASE)
                    except Exception as e:
                        print(f"[CONSOLIDATE]   ⚠️ Error buscando grupos en chunk {i}: {e}")
                        grupos_encontrados = []

                    if grupos_encontrados:
                        # Normalizar grupos
                        grupos_unicos = sorted(set(grupos_encontrados))
                        group_key = grupos_unicos[0]

                        group_map[group_key].append({
                            'chunk': chunk,
                            'content': content,
                            'grupos': grupos_unicos,
                            'index': i
                        })
                    else:
                        # Chunks sin grupos
                        non_group_chunks.append(chunk)

                except Exception as e:
                    print(f"[CONSOLIDATE]   ⚠️ Error procesando chunk {i}: {e}")
                    non_group_chunks.append(chunk)
                    continue

            # Consolidar chunks por grupo
            consolidated_chunks = []

            try:
                for group_key in sorted(group_map.keys(), key=lambda x: int(x)):
                    try:
                        chunks_del_grupo = group_map[group_key]

                        if len(chunks_del_grupo) == 1:
                            # Solo un chunk
                            consolidated_chunks.append(chunks_del_grupo[0]['chunk'])
                            print(f"[CONSOLIDATE]   Grupo {group_key}: 1 chunk (sin consolidar)")
                        else:
                            # Múltiples chunks, consolidar
                            print(f"[CONSOLIDATE]   Grupo {group_key}: {len(chunks_del_grupo)} chunks → consolidando")

                            # Combinar contenidos
                            try:
                                combined_content = "\n\n".join([
                                    f"<!-- SEGMENTO {idx + 1} -->\n{item['content']}"
                                    for idx, item in enumerate(chunks_del_grupo)
                                ])
                            except Exception as e:
                                print(f"[CONSOLIDATE]     ⚠️ Error combinando contenidos: {e}")
                                # Fallback: usar solo el primer chunk
                                consolidated_chunks.append(chunks_del_grupo[0]['chunk'])
                                continue

                            # Crear chunk consolidado
                            try:
                                first_chunk = chunks_del_grupo[0]['chunk']

                                if hasattr(first_chunk, 'page_content'):
                                    # Document de LangChain
                                    consolidated_chunk = type(first_chunk)(
                                        page_content=combined_content,
                                        metadata={
                                            **(first_chunk.metadata if hasattr(first_chunk, 'metadata') else {}),
                                            'consolidated': True,
                                            'grupos': [item['grupos'] for item in chunks_del_grupo],
                                            'original_chunks': len(chunks_del_grupo)
                                        }
                                    )
                                elif isinstance(first_chunk, dict):
                                    # Diccionario
                                    consolidated_chunk = {
                                        **first_chunk,
                                        'content': combined_content,
                                        'page_content': combined_content,
                                        'consolidated': True,
                                        'grupos': [item['grupos'] for item in chunks_del_grupo],
                                        'original_chunks': len(chunks_del_grupo)
                                    }
                                else:
                                    # Otro tipo
                                    consolidated_chunk = combined_content

                                consolidated_chunks.append(consolidated_chunk)

                            except Exception as e:
                                print(f"[CONSOLIDATE]     ⚠️ Error creando chunk consolidado: {e}")
                                # Fallback: agregar chunks individuales
                                for item in chunks_del_grupo:
                                    consolidated_chunks.append(item['chunk'])

                    except Exception as e:
                        print(f"[CONSOLIDATE]   ⚠️ Error consolidando grupo {group_key}: {e}")
                        # Agregar chunks individuales del grupo
                        if chunks_del_grupo:
                            for item in chunks_del_grupo:
                                consolidated_chunks.append(item['chunk'])
                        continue

            except Exception as e:
                print(f"[CONSOLIDATE] ❌ Error en loop de consolidación: {e}")
                # Fallback: usar chunks originales
                return chunks

            # Agregar chunks sin grupos
            consolidated_chunks.extend(non_group_chunks)

            print(f"[CONSOLIDATE] ✅ {len(chunks)} chunks → {len(consolidated_chunks)} chunks consolidados")
            print(f"[CONSOLIDATE]    Grupos consolidados: {len(group_map)}")
            print(f"[CONSOLIDATE]    Chunks sin grupos: {len(non_group_chunks)}\n")

            return consolidated_chunks

        except Exception as e:
            print(f"[CONSOLIDATE] ❌ Error crítico: {e}")
            import traceback
            traceback.print_exc()


            # Fallback: retornar chunks originales
            return chunks

    from typing import List, Dict, Any, Union

    async def filter_chunks_reflexive(
            self,
            chunks: List,
            query: str,
            simple_model_executor
    ) -> List:
        """
        Reranking de chunks usando cross-encoder local (sin llamadas LLM).

        ANTES: 1 llamada LLM por chunk → O(n) × ~0.5s = hasta 25s con 50 chunks.
        AHORA: cross-encoder local → O(n) en un solo forward pass ≈ 200ms total.

        Firma idéntica a la versión original — backward compatible.
        simple_model_executor se mantiene por compatibilidad pero ya no se usa.
        """
        if not chunks:
            return []
        if not query or not query.strip():
            return chunks

        try:
            result = await asyncio.wait_for(
                _cross_rerank(query, chunks, top_k=min(20, len(chunks))),
                timeout=10.0,
            )
            if result:
                print(f"[RERANK] ✅ {len(chunks)}→{len(result)} chunks | cross-encoder local")
                return result
        except Exception as e:
            print(f"[RERANK] ⚠️ Cross-encoder falló ({e}), retornando top-20 por score")

        # Fallback: top-20 por score vectorial existente
        def _score(c):
            if isinstance(c, dict):
                return float(c.get("certainty") or c.get("score") or c.get("similitud") or 0.0)
            return 0.0

        return sorted(chunks, key=_score, reverse=True)[:20]

    async def _evaluate_single_chunk(
            self,
            chunk: Any,
            query: str,
            simple_model_executor,
            idx: int
    ) -> bool:
        """
        Evalúa UN chunk individual con manejo de errores.

        Args:
            chunk: Chunk a evaluar
            query: Query del usuario
            simple_model_executor: Ejecutor del modelo
            idx: Índice del chunk (para logging)

        Returns:
            bool: True si es relevante, False si no o si hay error
        """

        # Crear contexto de error

        try:
            # Extraer texto con manejo de errores
            try:
                chunk_text = self._extract_text(chunk)

                if not chunk_text or not chunk_text.strip():
                    print(f"[REFLEXIVE] Chunk {idx + 1}: ⚠️ Texto vacío")
                    return False

                # Truncar a 1000 chars
                chunk_text = chunk_text[:1000]

            except Exception as e:
                print(f"[REFLEXIVE] Chunk {idx + 1}: ❌ Error extrayendo texto: {e}")
                return False

            # Construir prompt
            try:
                prompt = f"""¿Este fragmento es relevante para responder la pregunta?

    Fragmento: {chunk_text}
    Pregunta: {query}

    Responde solo: SI o NO"""
            except Exception as e:
                print(f"[REFLEXIVE] Chunk {idx + 1}: ❌ Error construyendo prompt: {e}")
                return False

            # Ejecutar modelo con timeout
            try:
                response = await asyncio.wait_for(
                    simple_model_executor(prompt),
                    timeout=3.0
                )

                if not response or not response.strip():
                    print(f"[REFLEXIVE] Chunk {idx + 1}: ⚠️ Respuesta vacía")
                    return False

                # Evaluar respuesta
                is_relevant = "si" in response.lower().strip()

                status = "✅" if is_relevant else "❌"
                print(f"[REFLEXIVE] Chunk {idx + 1}: {status}")

                return is_relevant

            except TimeoutError:
                print(f"[REFLEXIVE] Chunk {idx + 1}: ⏱️ TIMEOUT (3s)")
                return False

            except Exception as e:
                print(f"[REFLEXIVE] Chunk {idx + 1}: ❌ Error en modelo: {e}")
                return False

        except Exception as e:
            print(f"[REFLEXIVE] Chunk {idx + 1}: ❌ Error crítico: {e}")
            return False

    def _extract_chunk_metadata(
            self,
            chunk: Union[str, Any],
            max_sample_length: int = 200
    ) -> Dict[str, str]:
        """
        Extrae metadatos y contenido de diferentes tipos de chunks con manejo de errores robusto.

        Maneja múltiples formatos de entrada:
        - Objetos SearchResult con atributos .text, .file_name, .tipo_doc
        - Strings con formato específico (=== ARCHIVO: ===, file_name:, etc.)
        - Objetos con texto JSON anidado

        Args:
            chunk: Chunk a procesar
            max_sample_length: Longitud máxima para content_sample

        Returns:
            Dict[str, str]: Diccionario con metadatos extraídos:
                - file_name: Nombre del archivo
                - doc_type: Tipo de documento
                - content_sample: Muestra del contenido
        """

        # Crear contexto de error

        # Metadata por defecto
        metadata = {
            'file_name': "archivo_desconocido",
            'doc_type': "desconocido",
            'content_sample': ""
        }

        # Validación
        if chunk is None:
            return metadata

        try:
            # CASO 1: Objeto SearchResult o similar con atributos
            if hasattr(chunk, 'text') and hasattr(chunk, 'file_name'):
                try:
                    metadata['file_name'] = str(getattr(chunk, 'file_name', 'archivo_desconocido'))
                    metadata['doc_type'] = str(getattr(chunk, 'tipo_doc', 'desconocido'))

                    # Extraer texto del atributo .text
                    chunk_text = chunk.text

                    if isinstance(chunk_text, str):
                        try:
                            # Intentar parsear como JSON si tiene formato de diccionario
                            if chunk_text.strip().startswith(("{'text':", '{"text":')):
                                try:
                                    # Intentar con json.loads primero (más seguro)
                                    if chunk_text.strip().startswith('{"'):
                                        text_data = json.loads(chunk_text)
                                    else:
                                        # Fallback a eval para comillas simples (con precaución)
                                        import ast
                                        text_data = ast.literal_eval(chunk_text)

                                    actual_text = text_data.get('text', chunk_text)
                                except (json.JSONDecodeError, SyntaxError, ValueError) as e:
                                    # Si falla el parsing, usar el texto tal como está
                                    actual_text = chunk_text
                            else:
                                actual_text = chunk_text
                        except Exception as e:
                            actual_text = chunk_text
                    else:
                        actual_text = str(chunk_text)

                    metadata['content_sample'] = actual_text[:max_sample_length]
                    return metadata

                except Exception as e:
                    print(f"[_extract_chunk_metadata] ⚠️ Error en CASO 1: {e}")
                    # Continuar con CASO 2

            # CASO 2: String o objeto convertible a string
            try:
                chunk_str = str(chunk)
            except Exception as e:
                print(f"[_extract_chunk_metadata] ⚠️ Error convirtiendo a string: {e}")
                metadata['content_sample'] = "(error convirtiendo chunk)"
                return metadata

            # Buscar patrones para extraer nombre de archivo
            try:
                file_name_patterns = [
                    ("=== ARCHIVO:", lambda s: (
                        s.split("=== ARCHIVO:")[1].split("===")[0].strip()
                        if "===" in s.split("=== ARCHIVO:")[1]
                        else s.split("=== ARCHIVO:")[1].split("\n")[0].strip()
                    )),
                    ("file_name:", lambda s: next(
                        (line.split('file_name:')[1].strip()
                         for line in s.split('\n')
                         if 'file_name:' in line),
                        None
                    )),
                    ("archivo:", lambda s: next(
                        (line.split(':')[1].strip()
                         for line in s.split('\n')
                         if 'archivo:' in line.lower()),
                        None
                    ))
                ]

                for pattern, extractor in file_name_patterns:
                    if pattern in chunk_str:
                        try:
                            extracted = extractor(chunk_str)
                            if extracted:
                                metadata['file_name'] = extracted
                                break
                        except Exception as e:
                            print(f"[_extract_chunk_metadata]   ⚠️ Error extrayendo file_name con {pattern}: {e}")
                            continue
            except Exception as e:
                print(f"[_extract_chunk_metadata] ⚠️ Error en extracción de file_name: {e}")

            # Buscar patrones para extraer tipo de documento
            try:
                doc_type_patterns = [
                    ("Tipo doc:", lambda s: next(
                        (line.split("Tipo doc:")[1].strip()
                         for line in s.split('\n')
                         if 'Tipo doc:' in line),
                        None
                    )),
                    ("tipo_doc:", lambda s: next(
                        (line.split("tipo_doc:")[1].strip()
                         for line in s.split('\n')
                         if 'tipo_doc:' in line),
                        None
                    )),
                    ("tipo:", lambda s: next(
                        (line.split(":")[1].strip()
                         for line in s.split('\n')
                         if 'tipo:' in line.lower()),
                        None
                    ))
                ]

                for pattern, extractor in doc_type_patterns:
                    if pattern in chunk_str:
                        try:
                            extracted = extractor(chunk_str)
                            if extracted:
                                metadata['doc_type'] = extracted
                                break
                        except Exception as e:
                            print(f"[_extract_chunk_metadata]   ⚠️ Error extrayendo doc_type con {pattern}: {e}")
                            continue
            except Exception as e:
                print(f"[_extract_chunk_metadata] ⚠️ Error en extracción de doc_type: {e}")

            # Extraer contenido representativo
            try:
                if "=== ARCHIVO:" in chunk_str:
                    parts = chunk_str.split("===")
                    if len(parts) >= 3:
                        metadata['content_sample'] = parts[2].strip()[:max_sample_length]
                    else:
                        metadata['content_sample'] = chunk_str.strip()[:max_sample_length]
                else:
                    metadata['content_sample'] = chunk_str.strip()[:max_sample_length]
            except Exception as e:
                print(f"[_extract_chunk_metadata] ⚠️ Error extrayendo content_sample: {e}")
                try:
                    metadata['content_sample'] = chunk_str[:max_sample_length]
                except:
                    metadata['content_sample'] = "(error extrayendo contenido)"

        except Exception as e:
            print(f"[_extract_chunk_metadata] ❌ Error crítico: {e}")
            import traceback
            traceback.print_exc()


            # Intentar último fallback
            try:
                metadata['content_sample'] = str(chunk)[:max_sample_length]
            except:
                metadata['content_sample'] = "(error procesando chunk)"

        return metadata

    async def _generate_standard_response(
            self,
            chunks: List,
            query: str,
            personality: str,
            tone: str,
            model_executor,
            simple_model_executor=None,
            faq_data=None,
            domain_config=None,
            conversation_history=None,
            skip_reflexive: bool = True
    ) -> str:
        """
        Genera respuesta estándar con manejo de errores robusto.

        Args:
            chunks: Lista de chunks a procesar
            query: Query del usuario
            personalidad: Personalidad del bot
            tono: Tono de respuesta
            model_executor: Ejecutor del modelo principal
            simple_model_executor: Ejecutor del modelo simple (opcional)
            faq_data: Datos FAQ (opcional)
            domain_config: Configuración del dominio (opcional)
            conversation_history: Historial de conversación (opcional)
            skip_reflexive: Si saltar filtro reflexivo

        Returns:
            str: Respuesta HTML (con mensaje de error si falla)
        """

        # Crear contexto de error

        try:
            print(f"[DEBUG] 📦 Chunks recibidos: {len(chunks) if chunks else 0}")

            # Validar chunks
            if not chunks:
                print(f"[DEBUG] ⚠️ Sin chunks para procesar")
                return "❌ No se proporcionaron fragmentos de texto para procesar."

            # Normalizar chunks a lista
            try:
                if isinstance(chunks, str):
                    chunks = [chunks]
                elif not isinstance(chunks, list):
                    chunks = [chunks]
            except Exception as e:
                print(f"[DEBUG] ⚠️ Error normalizando chunks: {e}")
                return "❌ Error procesando los fragmentos de texto."

            # Limpiar chunks
            clean_chunks = []
            try:
                for c in chunks:
                    try:
                        chunk_str = str(c).strip()
                        if chunk_str:
                            clean_chunks.append(chunk_str)
                    except Exception as e:
                        print(f"[DEBUG]   ⚠️ Error limpiando chunk: {e}")
                        continue

                print(f"[DEBUG] ✅ Chunks después de limpieza: {len(clean_chunks)}")
            except Exception as e:
                print(f"[DEBUG] ⚠️ Error en limpieza de chunks: {e}")
                return "❌ Error al procesar los fragmentos de texto."

            if not clean_chunks:
                print(f"[DEBUG] ⚠️ Lista de chunks vacía después de limpieza")
                return "❌ Error: la lista de fragmentos está vacía."

            # Filtro reflexivo (opcional)
            if simple_model_executor and not skip_reflexive:
                try:
                    print(f"[REFLEXIVE] 🔄 Aplicando filtro reflexivo...")

                    filtered_chunks = await asyncio.wait_for(
                        self.filter_chunks_reflexive(
                            clean_chunks,
                            query,
                            simple_model_executor
                        ),
                        timeout=30.0
                    )

                    # Validar resultado
                    if not filtered_chunks or not isinstance(filtered_chunks, list):
                        print(f"[REFLEXIVE] ⚠️ Resultado inválido, usando chunks originales")
                        filtered_chunks = clean_chunks

                    clean_chunks = filtered_chunks
                    print(f"[REFLEXIVE] ✅ Chunks después del filtro: {len(clean_chunks)}")

                except TimeoutError:
                    print(f"[REFLEXIVE] ⏱️ TIMEOUT (30s), usando chunks sin filtrar")
                except Exception as e:
                    print(f"[REFLEXIVE] ⚠️ Error en filtro: {e}, usando chunks sin filtrar")
            elif skip_reflexive:
                print(f"[REFLEXIVE] ⏭️ SKIP - usando batch extraction para {len(clean_chunks)} chunks")

            # fix: usar batch extraction (1 LLM call) en vez de _process_chunks_simple
            # _process_chunks_simple hacía 1 llamada por chunk — reemplazado por batch
            try:
                batch_results = await asyncio.wait_for(
                    self._process_chunks_batch_extraction(
                        clean_chunks, query, step_idx=0, top_k=8
                    ),
                    timeout=40.0,
                )

                if batch_results:
                    # Combinar los fragmentos relevantes extraídos en un contexto único
                    combined_context = "\n\n".join(
                        r.get("response", "") for r in batch_results if r.get("response")
                    )
                    if combined_context.strip():
                        # Usar model_executor para la síntesis final
                        retorno = await asyncio.wait_for(
                            model_executor(
                                f"Con base en la siguiente información, responde de forma concisa y directa:\n\n"
                                f"{combined_context[:8000]}\n\n"
                                f"Pregunta: {query}\n\nRespuesta en HTML:"
                            ),
                            timeout=45.0,
                        )
                        if retorno and retorno.strip():
                            return retorno

                # Fallback si batch no produjo resultado
                retorno = await asyncio.wait_for(
                    self._process_chunks_simple(
                        clean_chunks, query, faq_data, personality, tone,
                        model_executor, domain_config, conversation_history
                    ),
                    timeout=60.0,
                )

                if not retorno or not retorno.strip():
                    print(f"[DEBUG] ⚠️ Respuesta vacía")
                    return "❌ No se pudo generar una respuesta. Por favor intenta reformular tu consulta."

                return retorno

            except TimeoutError:
                print(f"[DEBUG] ⏱️ TIMEOUT en _process_chunks_simple (60s)")
                return "❌ El procesamiento tardó demasiado tiempo. Por favor intenta con una consulta más específica."

            except Exception as e:
                print(f"[DEBUG] ❌ Error en _process_chunks_simple: {e}")
                import traceback
                traceback.print_exc()
                return f"❌ Error generando respuesta: {str(e)[:100]}"

        except Exception as e:
            print(f"[ERROR] ❌ Error crítico en _generate_standard_response: {str(e)}")
            import traceback
            traceback.print_exc()


            return f"❌ Error inesperado: {str(e)[:100]}"

    def _prepare_faq_content(self, faq_data: Any) -> str:
        """
        Prepara contenido FAQ de forma eficiente con manejo de errores.

        Args:
            faq_data: Datos FAQ (list, dict, o None)

        Returns:
            str: Contenido FAQ formateado (vacío si falla o no hay datos)
        """

        # Crear contexto de error

        # Validar input
        if not faq_data:
            return ""

        try:
            faq_items = []

            # CASO 1: Lista de FAQs
            if isinstance(faq_data, list):
                try:
                    # Limitar a 3 FAQs más relevantes
                    for i, faq in enumerate(faq_data[:3]):
                        try:
                            if not isinstance(faq, dict):
                                print(f"[FAQ] ⚠️ Item {i} no es dict: {type(faq)}")
                                continue

                            q = str(faq.get('pregunta', '')).strip()
                            a = str(faq.get('respuesta', '')).strip()

                            if q and a:
                                faq_items.append(f"P: {q}\nR: {a}")
                            else:
                                print(f"[FAQ] ⚠️ Item {i} sin pregunta o respuesta")

                        except Exception as e:
                            print(f"[FAQ] ⚠️ Error procesando FAQ {i}: {e}")
                            continue

                except Exception as e:
                    print(f"[FAQ] ⚠️ Error procesando lista de FAQs: {e}")

            # CASO 2: FAQ individual (dict)
            elif isinstance(faq_data, dict):
                try:
                    q = str(faq_data.get('pregunta', '')).strip()
                    a = str(faq_data.get('respuesta', '')).strip()

                    if q and a:
                        faq_items.append(f"P: {q}\nR: {a}")
                    else:
                        print(f"[FAQ] ⚠️ FAQ dict sin pregunta o respuesta")

                except Exception as e:
                    print(f"[FAQ] ⚠️ Error procesando FAQ dict: {e}")

            # CASO 3: Otro tipo
            else:
                print(f"[FAQ] ⚠️ faq_data tipo no soportado: {type(faq_data)}")

            # Construir resultado
            if not faq_items:
                return ""

            try:
                result = f"\n=== INFORMACIÓN ADICIONAL ===\n" + "\n\n".join(faq_items)
                return result
            except Exception as e:
                print(f"[FAQ] ⚠️ Error construyendo resultado: {e}")
                return ""

        except Exception as e:
            print(f"[FAQ] ❌ Error crítico: {e}")
            import traceback
            traceback.print_exc()


            return ""

    def _build_simple_rag_prompt(
            self,
            query: str,
            combined_text: str,
            faq_content: str,
            personality: str,
            tone: str,
            domain_config: Optional[Dict],
            conversation_history: Optional[List] = None
    ) -> str:
        """
        Construye prompt RAG simple mejorado con manejo de errores robusto.

        Args:
            query: Query del usuario
            combined_text: Texto combinado de chunks
            faq_content: Contenido FAQ
            personalidad: Personalidad del bot
            tono: Tono de respuesta
            domain_config: Configuración del dominio
            conversation_history: Historial de conversación

        Returns:
            str: Prompt construido (con fallback si falla)
        """

        # Crear contexto de error

        try:
            # Validar inputs básicos
            if not query:
                query = "Pregunta no especificada"

            if not combined_text:
                combined_text = "(Sin información disponible)"

            if not faq_content:
                faq_content = ""

            if not personality:
                personality = "profesional"

            if not tone:
                tone = "formal"

            # Extraer rol
            role = "asistente experto"
            try:
                if domain_config and isinstance(domain_config, dict):
                    role = domain_config.get("role_description", "asistente experto")
            except Exception as e:
                print(f"[PROMPT] ⚠️ Error extrayendo role: {e}")

            # Mapear tono
            tono_map = {
                "formal": "profesional",
                "casual": "amigable",
                "filosófico": "reflexivo"
            }
            tono_desc = tono_map.get(tone, "claro")

            # Construir sección de historial
            historial_section = ""

            try:
                if conversation_history and len(conversation_history) > 0:
                    valid_questions = []

                    for i, item in enumerate(conversation_history):
                        try:
                            if not isinstance(item, dict):
                                continue

                            if "user_message" not in item:
                                continue

                            user_msg = str(item["user_message"])

                            # Limpiar sufijos de formato
                            try:
                                clean_q = re.sub(
                                    r"\s+que\s+la\s+salida\s+sea\s+html.*$",
                                    "",
                                    user_msg,
                                    flags=re.IGNORECASE
                                ).strip()
                            except Exception as e:
                                print(f"[PROMPT]   ⚠️ Error limpiando pregunta {i}: {e}")
                                clean_q = user_msg.strip()

                            if clean_q and len(clean_q) > 5:
                                valid_questions.append(clean_q)

                        except Exception as e:
                            print(f"[PROMPT]   ⚠️ Error procesando item historial {i}: {e}")
                            continue

                    if valid_questions:
                        try:
                            historial_section = f"""
    === HISTORIAL DE PREGUNTAS ANTERIORES ===

    Si el usuario pregunta sobre sus consultas previas (ej: "qué te pregunté", "que te pregunte", 
    "cuáles fueron mis preguntas", "mis preguntas anteriores", etc.), responde ÚNICAMENTE 
    listando estas preguntas en orden:

    """
                            for i, q in enumerate(valid_questions, 1):
                                historial_section += f"{i}) {q}\n"

                            historial_section += """
    IMPORTANTE: 
    - Si pregunta sobre el historial: lista SOLO las preguntas de arriba, NO las respuestas
    - Si pregunta sobre datos/información: ignora esta sección y usa los DOCUMENTOS
    - NO inventes preguntas que no estén en esta lista
    - NO incluyas variaciones con "formato html" o similares

    """
                        except Exception as e:
                            print(f"[PROMPT] ⚠️ Error construyendo sección historial: {e}")
                            historial_section = ""

            except Exception as e:
                print(f"[PROMPT] ⚠️ Error procesando conversation_history: {e}")
                historial_section = ""

            # Construir prompt completo
            try:
                prompt = f"""Eres un {role} especializado con tono {tono_desc}. Combina información de documentos con conocimiento complementario:

    CONSULTA: {query}

    {historial_section}
    === INFORMACIÓN DE DOCUMENTOS ===
    {combined_text}{faq_content}

    === ESTRATEGIA RAG INTELIGENTE ===

    Proporciona respuestas completas y valiosas:

    **INFORMACIÓN DE DOCUMENTOS (Fuente principal):**
    - Prioriza datos específicos de documentos: cifras, fechas, códigos, nombres
    - Esta información es la más confiable

    **CONOCIMIENTO COMPLEMENTARIO (Enriquecimiento):**
    - Añade contexto sobre conceptos técnicos, normativas, procedimientos
    - Explica significados, implicaciones y relevancia
    - Conecta con mejores prácticas del sector

    **COMBINACIÓN INTELIGENTE:**
    - Para datos específicos: usa SOLO información de documentos
    - Para explicaciones: combina documentos + conocimiento técnico
    - Si falta información específica: indica "no tengo información específica sobre X"

    === REGLAS CRÍTICAS DE VERACIDAD ===

    **PROHIBIDO TERMINANTEMENTE:**
    - NO inventes datos específicos que no aparezcan en los documentos
    - NO completes cifras, códigos, fechas o nombres faltantes con suposiciones
    - NO generes información que no esté presente explícitamente
    - NO asumas detalles no mencionados en el texto

    **COMPORTAMIENTO OBLIGATORIO:**
    - Si no encuentras datos específicos: declara "la información no se especifica en los documentos"
    - Para información parcial: reporta solo lo que está presente
    - Para datos faltantes: usa "no se menciona" o "información no disponible"
    - Distingue claramente entre hechos documentados y conocimiento general

    **FRASES REQUERIDAS PARA LIMITACIONES:**
    - "según los documentos proporcionados"
    - "la información disponible no incluye"
    - "no se especifica en el contenido"
    - "basándose únicamente en los documentos"

    === FORMATO DE RESPUESTA ===

    - Responde SOLO en HTML limpio
    - NO uses style, clases CSS, ni estilos inline
    - Solo etiquetas básicas: h1, h2, h3, p, ul, li, strong, em, table, tr, td, th
    - Estructura tablas simple: <table><tr><th>Header</th></tr><tr><td>Data</td></tr></table>

    === TRACKING DE FUENTES MEJORADO ===

    Incluye comentarios HTML para rastrear origen:
    - <!-- FUENTE: DOCUMENTOS --> antes de info extraída de documentos
    - <!-- FUENTE: CONOCIMIENTO --> antes de contexto complementario
    - <!-- FUENTE: SÍNTESIS --> antes de análisis combinado
    - <!-- FUENTE: LIMITACIÓN --> cuando admitas falta de información
    - <!-- FUENTE: HISTORIAL --> cuando listes preguntas anteriores

    === CONFIGURACIÓN PERSONALIZADA ===

    Personalidad: {personality}
    Actúa como: {role}
    Tono: {tono_desc}

    RESPONDE COMBINANDO FUENTES CON MÁXIMA VERACIDAD Y SIN INVENTAR DATOS:"""

                return prompt

            except Exception as e:
                print(f"[PROMPT] ❌ Error construyendo prompt completo: {e}")
                # Fallback: prompt mínimo
                return f"""Eres un asistente experto.

    CONSULTA: {query}

    INFORMACIÓN:
    {combined_text[:5000] if combined_text else "(Sin información)"}

    Responde en HTML limpio usando solo etiquetas básicas (h1, h2, p, ul, li, strong).
    """

        except Exception as e:
            print(f"[PROMPT] ❌ Error crítico: {e}")
            import traceback
            traceback.print_exc()


            # Fallback ultra-básico
            return f"""Responde la siguiente consulta usando la información proporcionada:

    CONSULTA: {query if query else "Pregunta no especificada"}

    INFORMACIÓN: {combined_text[:3000] if combined_text else "(Sin información)"}

    Responde en HTML simple."""

    async def _process_chunks_simple(
            self,
            chunks: List,
            query: str,
            faq_data: Any,
            personality: str,
            tone: str,
            model_executor,
            domain_config: Optional[Dict],
            conversation_history: Optional[List]
    ) -> str:
        """
        Procesa todos los chunks juntos con manejo de errores robusto.

        Args:
            chunks: Lista de chunks a procesar
            query: Query del usuario
            faq_data: Datos FAQ
            personalidad: Personalidad del bot
            tono: Tono de respuesta
            model_executor: Ejecutor del modelo
            domain_config: Configuración del dominio
            conversation_history: Historial de conversación

        Returns:
            str: Respuesta HTML (con mensaje de error si falla)
        """

        # Crear contexto de error

        # Validaciones
        if not chunks:
            print(f"[_process_chunks_simple] ⚠️ Sin chunks para procesar")
            return "❌ No se proporcionaron fragmentos para procesar."

        if not isinstance(chunks, list):
            print(f"[_process_chunks_simple] ⚠️ chunks no es lista: {type(chunks)}")
            chunks = [chunks]

        if not query:
            print(f"[_process_chunks_simple] ⚠️ Query vacía")
            query = "Pregunta no especificada"

        try:
            # Preparar texto combinado
            combined_text = "\n\n=== DOCUMENTOS DISPONIBLES ===\n\n"

            try:
                for i, chunk in enumerate(chunks, 1):
                    try:
                        chunk_str = str(chunk)
                        combined_text += f"DOCUMENTO {i}:\n{chunk_str}\n\n"
                    except Exception as e:
                        print(f"[_process_chunks_simple]   ⚠️ Error procesando chunk {i}: {e}")
                        continue
            except Exception as e:
                print(f"[_process_chunks_simple] ⚠️ Error construyendo combined_text: {e}")
                combined_text = "\n\n=== DOCUMENTOS DISPONIBLES ===\n\n(Error procesando chunks)\n\n"

            # Preparar FAQ
            try:
                faq_content = self._prepare_faq_content(faq_data)
            except Exception as e:
                print(f"[_process_chunks_simple] ⚠️ Error preparando FAQ: {e}")
                faq_content = ""

            # Construir prompt
            try:
                prompt = self._build_simple_rag_prompt(
                    query,
                    combined_text,
                    faq_content,
                    personality,
                    tone,
                    domain_config,
                    conversation_history
                )
            except Exception as e:
                print(f"[_process_chunks_simple] ❌ Error construyendo prompt: {e}")
                return "❌ Error interno al construir el prompt de respuesta."

            # Ejecutar modelo con timeout
            try:
                print(f"[_process_chunks_simple] 🤖 Ejecutando modelo (timeout: 45s)...")

                response = await asyncio.wait_for(
                    model_executor(prompt),
                    timeout=45.0
                )

                if not response or not response.strip():
                    print(f"[_process_chunks_simple] ⚠️ Respuesta vacía del modelo")
                    return "❌ No se pudo generar una respuesta. Por favor intenta reformular tu consulta."

                print(f"[_process_chunks_simple] ✅ Respuesta generada: {len(response)} chars")
                return response

            except TimeoutError:
                print(f"[_process_chunks_simple] ⏱️ TIMEOUT (45s)")
                return "❌ El procesamiento tardó demasiado. Por favor intenta con una consulta más específica."

            except Exception as e:
                print(f"[_process_chunks_simple] ❌ Error ejecutando modelo: {e}")
                return f"❌ Error generando respuesta: {str(e)[:100]}"

        except Exception as e:
            print(f"[_process_chunks_simple] ❌ Error crítico: {e}")
            import traceback
            traceback.print_exc()


            return f"❌ Error inesperado: {str(e)[:100]}"


    def _detect_patterns_in_text(
            self,
            text: str,
            max_length: int = 50000
    ) -> Dict[str, Any]:
        """
        Detecta automáticamente patrones estructurados en el texto con manejo de errores.
        NO usa patrones predefinidos, los descubre dinámicamente.

        Returns:
            Dict: Patrones detectados y su frecuencia (dict vacío si falla)
        """

        # Crear contexto de error

        # Resultado por defecto
        default_result = {
            'enumerated_items': [],
            'numeric_values': [],
            'structured_sections': [],
            'has_critical_info': False,
            'complexity_score': 0
        }

        # Validar input
        if not text or not text.strip():
            return default_result

        try:
            # Limitar longitud
            try:
                sample_text = text[:max_length]
            except Exception as e:
                print(f"[_detect_patterns] ⚠️ Error truncando texto: {e}")
                sample_text = text

            detected = {
                'enumerated_items': [],
                'numeric_values': [],
                'structured_sections': [],
                'has_critical_info': False,
                'complexity_score': 0
            }

            # 1. DETECTAR ITEMS ENUMERADOS GENÉRICAMENTE
            try:
                enumerated_pattern = r'\b([A-ZÁ-Ú][a-záéíóúüñ]+)\s+(\d+|[A-Z]|[IVX]+)\b'
                enumerated_matches = re.findall(enumerated_pattern, sample_text)

                if enumerated_matches:
                    # Agrupar por tipo
                    enumerated_counter = Counter([match[0] for match in enumerated_matches])
                    detected['enumerated_items'] = [
                        {'type': item_type, 'count': count}
                        for item_type, count in enumerated_counter.most_common(10)
                        if count >= 2
                    ]
            except Exception as e:
                print(f"[_detect_patterns] ⚠️ Error detectando items enumerados: {e}")

            # 2. DETECTAR VALORES NUMÉRICOS CON UNIDADES
            try:
                numeric_units_pattern = r'(\d+(?:[.,]\d+)?)\s*(días?|meses?|años?|€|euros?|horas?|minutos?|%|porcentaje)'
                numeric_matches = re.findall(numeric_units_pattern, sample_text, re.IGNORECASE)

                if numeric_matches:
                    numeric_counter = Counter([match[1].lower() for match in numeric_matches])
                    detected['numeric_values'] = [
                        {'unit': unit, 'count': count}
                        for unit, count in numeric_counter.most_common(5)
                    ]
            except Exception as e:
                print(f"[_detect_patterns] ⚠️ Error detectando valores numéricos: {e}")

            # 3. DETECTAR SECCIONES ESTRUCTURADAS
            try:
                section_patterns = [
                    (r'^[A-ZÁÉÍÓÚÜ\s]{5,}$', 'UPPERCASE_TITLE'),
                    (r'^\d+\.\s+', 'NUMBERED_LIST'),
                    (r'^[-•]\s+', 'BULLET_LIST'),
                    (r'^\w+\s*:\s*', 'KEY_VALUE')
                ]

                lines = sample_text.split('\n')
                structure_counts = Counter()

                for line in lines[:500]:  # Primeras 500 líneas
                    try:
                        line = line.strip()
                        if not line:
                            continue

                        for pattern, label in section_patterns:
                            try:
                                if re.match(pattern, line, re.MULTILINE):
                                    structure_counts[label] += 1
                            except Exception as e:
                                print(f"[_detect_patterns]     ⚠️ Error con pattern {label}: {e}")
                                continue
                    except Exception as e:
                        print(f"[_detect_patterns]   ⚠️ Error procesando línea: {e}")
                        continue

                detected['structured_sections'] = [
                    {'type': struct_type, 'count': count}
                    for struct_type, count in structure_counts.most_common()
                ]
            except Exception as e:
                print(f"[_detect_patterns] ⚠️ Error detectando secciones: {e}")

            # 4. CALCULAR SCORE DE COMPLEJIDAD
            try:
                complexity = 0

                if detected['enumerated_items']:
                    complexity += len(detected['enumerated_items']) * 2

                if detected['numeric_values']:
                    complexity += len(detected['numeric_values'])

                if detected['structured_sections']:
                    complexity += sum(item.get('count', 0) for item in detected['structured_sections']) // 10

                detected['complexity_score'] = min(complexity, 100)

                detected['has_critical_info'] = bool(
                    detected['enumerated_items'] or
                    detected['numeric_values'] or
                    len(detected['structured_sections']) > 2
                )
            except Exception as e:
                print(f"[_detect_patterns] ⚠️ Error calculando complejidad: {e}")

            return detected

        except Exception as e:
            print(f"[_detect_patterns] ❌ Error crítico: {e}")
            import traceback
            traceback.print_exc()


            return default_result

    def _inject_ghost_attention(
            self,
            base_prompt: str,
            ghost_context: str,
            injection_method: str = None
    ) -> str:
        """
        Inyecta Ghost Attention en el prompt con manejo de errores robusto.

        Args:
            base_prompt: Prompt base
            ghost_context: Contexto persistente
            injection_method: Método de inyección (prefix/suffix/sandwich)

        Returns:
            str: Prompt con ghost attention inyectado (o base_prompt si falla)
        """

        # Crear contexto de error

        # Validar inputs
        if not base_prompt:
            print(f"[_inject_ghost_attention] ⚠️ base_prompt vacío")
            return ""

        if not ghost_context or not ghost_context.strip():
            return base_prompt

        try:
            # Determinar método
            try:

                method = injection_method or GhostAttentionConfig.INJECTION_METHOD
            except (ImportError, AttributeError) as e:
                print(f"[_inject_ghost_attention] ⚠️ Error cargando config: {e}, usando 'prefix'")
                method = injection_method or "prefix"

            # Crear separador
            try:
                separator = f"\n\n{'▼' * 30}\n\n"
            except Exception as e:
                print(f"[_inject_ghost_attention] ⚠️ Error creando separador: {e}")
                separator = "\n\n" + ("=" * 30) + "\n\n"

            # Aplicar método de inyección
            try:
                if method == "prefix":
                    result = f"{ghost_context}{separator}{base_prompt}"

                elif method == "suffix":
                    result = f"{base_prompt}{separator}{ghost_context}"

                elif method == "sandwich":
                    try:
                        reminder = """
    ╔════════════════════════════════════════════╗
    ║  RECORDATORIO: Respeta las instrucciones  ║
    ║  persistentes y el historial de arriba    ║
    ╚════════════════════════════════════════════╝
    """
                    except Exception as e:
                        print(f"[_inject_ghost_attention] ⚠️ Error creando reminder: {e}")
                        reminder = "\n--- RECORDATORIO: Respeta instrucciones persistentes ---\n"

                    result = f"{ghost_context}{separator}{base_prompt}{separator}{reminder}"

                else:
                    print(f"[_inject_ghost_attention] ⚠️ Método desconocido: {method}, usando base_prompt")
                    result = base_prompt

                return result

            except Exception as e:
                print(f"[_inject_ghost_attention] ⚠️ Error aplicando método {method}: {e}")
                return base_prompt

        except Exception as e:
            print(f"[_inject_ghost_attention] ❌ Error crítico: {e}")
            traceback.print_exc()


            return base_prompt

    async def _handle_error(
            self,
            e: Exception,
            query: str,
            personality: str,
            tone: str,
            ghost_context: Optional[str],
            use_ghost_attention: bool
    ) -> str:
        """
        Maneja errores críticos con fallback robusto.

        Args:
            e: Excepción original
            query: Query del usuario
            personalidad: Personalidad del bot
            tono: Tono de respuesta
            ghost_context: Contexto ghost (opcional)
            use_ghost_attention: Si usar ghost attention

        Returns:
            str: Respuesta de fallback (siempre retorna algo)
        """

        # Crear contexto de error

        print(f"[Error Crítico] ❌ RAG falló: {e}")
        traceback.print_exc()

        try:
            # Obtener modelo de fallback
            try:
                fallback_model = self.custom.get_fallback_model()
                print(f"[Emergency Fallback] 🆘 Usando {fallback_model}")
            except Exception as e2:
                print(f"[Emergency Fallback] ⚠️ Error obteniendo fallback model: {e2}")
                fallback_model = "gpt-5.6"  # Fallback hardcoded

            # Validar inputs
            if not query or not query.strip():
                query = "consulta no especificada"

            if not personality:
                personality = "profesional"

            if not tone:
                tone = "formal"

            # Construir prompt simple
            try:
                simple_prompt = (
                    f"Pregunta: {query}\n\n"
                    f"Responde como {personality} con tono {tone}."
                )

                # Inyectar ghost attention si está habilitado
                if use_ghost_attention and ghost_context:
                    try:
                        simple_prompt = self._inject_ghost_attention(
                            simple_prompt,
                            ghost_context,
                            "prefix"
                        )
                    except Exception as e3:
                        print(f"[Emergency Fallback] ⚠️ Error inyectando ghost: {e3}")

            except Exception as e4:
                print(f"[Emergency Fallback] ⚠️ Error construyendo prompt: {e4}")
                simple_prompt = f"Responde esta pregunta: {query}"

            # Ejecutar modelo de emergencia
            try:
                emergency_response = await asyncio.wait_for(
                    self.custom.run_model_with_default(
                        simple_prompt,
                        model=fallback_model
                    ),
                    timeout=30.0
                )

                if not emergency_response or not emergency_response.strip():
                    print(f"[Emergency Fallback] ⚠️ Respuesta vacía")
                    emergency_response = "No se pudo generar una respuesta en este momento."

                return (
                    f"⚠️ Problema técnico. Respuesta basada en conocimiento general:\n\n"
                    f"{emergency_response}\n\n"
                    f"💡 Recomendación: intentar nuevamente o contactar soporte."
                )

            except TimeoutError:
                print(f"[Emergency Fallback] ⏱️ TIMEOUT (30s)")
                return (
                    f"⚠️ Problema técnico temporal.\n\n"
                    f"❓ Tu pregunta: {query[:200]}\n\n"
                    f"💡 Por favor, intenta reformular tu consulta de forma más específica "
                    f"o contacta con soporte técnico."
                )

            except Exception as emergency_error:
                print(f"[Error Final] ❌ Emergency fallback falló: {emergency_error}")
                traceback.print_exc()

                return (
                    f"⚠️ Error técnico crítico.\n\n"
                    f"❓ Tu pregunta: {query[:200]}\n\n"
                    f"🔧 Error: {str(emergency_error)[:100]}\n\n"
                    f"💡 Por favor, contacta con soporte técnico citando este mensaje."
                )

        except Exception as final_error:
            print(f"[Error Final] ❌ Todo falló: {final_error}")
            traceback.print_exc()


            # Último recurso: mensaje estático
            return (
                f"❌ Error crítico del sistema: {str(e)[:100]}\n\n"
                f"💡 Contactar soporte técnico inmediatamente."
            )

    """
    Fix para extractive_trim_answer_aware - Manejo robusto de MMR
    ==============================================================

    Parche para evitar errores con números complejos en MMR

    Versión: 1.0
    Fecha: 20/10/2025
    """

    from typing import Optional


    def extractive_trim_answer_aware(
            self,
            text: str,
            query: str,
            max_chars: int = 1200,
            max_tokens: Optional[int] = None,
            window: int = 1,
            head_units: int = 3,
            reserve_chars_head: int = 180,
            lambda_div: float = 0.3,
    ) -> str:
        """
        Recorte extractivo con manejo de errores robusto y fix para MMR.

        🔧 FIX: Convierte números complejos a float en cálculos MMR
        """


        if not text or not text.strip():
            return text if text is not None else ""

        try:
            # 1) Normalización
            try:
                s = re.sub(r"[ \t]+", " ", text)
                s = re.sub(r"\r\n?", "\n", s).strip()
            except Exception as e:
                print(f"[extractive_trim] ⚠️ Error normalizando: {e}")
                s = text.strip()

            # 2) Segmentación
            units = []
            try:
                rough = re.split(r"(?<=[\.\?\!])\s+|\n+", s)
                for seg in rough:
                    try:
                        seg = (seg or "").strip()
                        if not seg:
                            continue
                        parts = re.split(r"(?:^|\s)(?:•|-|\*)\s+", seg)
                        for p in parts:
                            p = (p or "").strip()
                            if p:
                                units.append(p)
                    except Exception as e:
                        print(f"[extractive_trim]   ⚠️ Error procesando segmento: {e}")
                        continue
            except Exception as e:
                print(f"[extractive_trim] ⚠️ Error segmentando: {e}")
                return s[:max_chars]

            if not units:
                return s[:max_chars]

            # 3) Preparativos
            try:
                q_terms = set(re.findall(r"\w{3,}", (query or "").lower()))
                token_pat = re.compile(r"\w{3,}")
                num_pat = re.compile(r"\d{2,}")
                year_pat = re.compile(r"\b(20\d{2}|19\d{2})\b")
                date_pat = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-](?:20\d{2}|19\d{2})\b")
            except Exception as e:
                print(f"[extractive_trim] ⚠️ Error compilando patterns: {e}")
                return s[:max_chars]

            # IDF intradocumental
            term_df = {}
            try:
                for t in q_terms:
                    df = sum(1 for sent in units if t in token_pat.findall(sent.lower()))
                    term_df[t] = max(1, df)
                N = len(units)
            except Exception as e:
                print(f"[extractive_trim] ⚠️ Error calculando IDF: {e}")
                term_df = {t: 1 for t in q_terms}
                N = len(units)

            # 🔧 HELPER: Sanitizar números a float
            def safe_float(value, default=0.0):
                """Convierte cualquier número a float, manejando complejos"""
                try:
                    # Si es complejo, tomar parte real
                    if isinstance(value, complex):
                        value = value.real

                    # Convertir a float
                    result = float(value)

                    # Verificar que no es NaN o infinito
                    if not (-1e308 < result < 1e308):  # Rango válido para float
                        return default

                    return result
                except:
                    return default

            # Funciones auxiliares
            def uniq_ratio(tokens):
                try:
                    return safe_float(len(set(tokens)) / max(1, len(tokens)))
                except:
                    return 0.0

            def char_entropy_approx(text_segment):
                try:
                    chars = set(text_segment)
                    return safe_float(len(chars) / max(1, len(text_segment)))
                except:
                    return 0.0

            def info_score(u: str, toks: list) -> float:
                try:
                    sc = 0.0
                    sc += 0.8 * uniq_ratio(toks)
                    sc += 0.6 * char_entropy_approx(u)
                    sc += 0.6 * bool(num_pat.search(u))
                    sc += 0.6 * bool(year_pat.search(u))
                    sc += 0.5 * bool(date_pat.search(u))
                    sc -= 0.0008 * max(0, len(u) - 400)
                    return safe_float(sc)
                except:
                    return 0.0

            def rel_score(u: str, toks_set: set) -> float:
                try:
                    if not q_terms:
                        return 0.0
                    overlap = q_terms & toks_set
                    if not overlap:
                        return 0.0
                    idf_sum = sum(log(N / term_df[t]) for t in overlap)
                    prop = len(overlap) / max(1, len(q_terms))
                    result = 1.4 * idf_sum + 1.0 * prop
                    return safe_float(result)
                except:
                    return 0.0

            # 4) Scoring
            try:
                toks_list = [token_pat.findall(u.lower()) for u in units]
                toks_sets = [set(t) for t in toks_list]

                base_scores = []
                for u, toks, tset in zip(units, toks_list, toks_sets):
                    try:
                        sc = rel_score(u, tset) + info_score(u, toks)
                        base_scores.append(safe_float(sc))
                    except Exception as e:
                        print(f"[extractive_trim]     ⚠️ Error scoring unidad: {e}")
                        base_scores.append(0.0)
            except Exception as e:
                print(f"[extractive_trim] ⚠️ Error en scoring: {e}")
                return s[:max_chars]

            # 5) 🔧 SELECCIÓN MMR CON FIX
            try:
                def jaccard(i, j):
                    try:
                        a, b = toks_sets[i], toks_sets[j]
                        if not a or not b:
                            return 0.0
                        inter = len(a & b)
                        uni = len(a | b) or 1
                        return safe_float(inter / uni)
                    except:
                        return 0.0

                idxs = list(range(len(units)))
                pos_prior = [safe_float(max(0.0, 1.0 - (i / max(1, len(units) - 1))) * 0.1) for i in idxs]
                scores = [safe_float(s + p) for s, p in zip(base_scores, pos_prior)]

                if not idxs:
                    return s[:max_chars]

                first = max(idxs, key=lambda i: scores[i])
                selected = [first]
                remaining = set(idxs)
                remaining.discard(first)

                # 🔧 GREEDY MMR CON SANITIZACIÓN
                while remaining:
                    try:
                        best_i, best_val = None, -1e18

                        for i in remaining:
                            try:
                                # 🔧 Sanitizar relevance
                                rel = safe_float(scores[i])

                                # 🔧 Sanitizar redundancy
                                if selected:
                                    red_values = [jaccard(i, j) for j in selected]
                                    red = safe_float(max(red_values) if red_values else 0.0)
                                else:
                                    red = 0.0

                                # 🔧 Calcular MMR con valores sanitizados
                                mmr = safe_float(lambda_div * rel - (1.0 - lambda_div) * red)

                                if mmr > best_val:
                                    best_val, best_i = mmr, i

                            except Exception as e:
                                print(f"[extractive_trim]     ⚠️ Error calculando MMR para idx {i}: {e}")
                                continue

                        if best_i is None:
                            break

                        selected.append(best_i)
                        remaining.remove(best_i)

                        if sum(len(units[k]) for k in selected) > 3 * max_chars:
                            break

                    except Exception as e:
                        print(f"[extractive_trim]   ⚠️ Error en MMR loop: {e}")
                        break

            except Exception as e:
                print(f"[extractive_trim] ⚠️ Error en MMR: {e}")
                selected = list(range(min(10, len(units))))

            # 6) Añadir contexto
            try:
                head_range = list(range(0, min(head_units, len(units))))
                prelim = set(selected)

                for idx in list(selected):
                    for j in range(max(0, idx - window), min(len(units), idx + window + 1)):
                        prelim.add(j)

                for h in head_range:
                    prelim.add(h)
            except Exception as e:
                print(f"[extractive_trim] ⚠️ Error añadiendo contexto: {e}")
                prelim = set(selected)

            # 7) Ensamblado
            try:
                def fingerprint(t: str) -> str:
                    try:
                        t = t.lower()
                        t = re.sub(r"\s+", " ", t)
                        t = re.sub(r"[^a-z0-9áéíóúñ ]", "", t)
                        return t.strip()
                    except:
                        return t

                chosen = sorted(prelim)
                seen = set()
                pieces = []
                total_chars = 0
                total_tokens = 0

                head_budget = max(0, min(reserve_chars_head, max_chars // 3))

                for i in head_range:
                    if i not in chosen:
                        chosen.insert(0, i)

                for i in chosen:
                    try:
                        u = units[i]
                        fp = fingerprint(u)

                        if not fp or fp in seen:
                            continue

                        length = len(u) + 1
                        fits_chars = (total_chars + length <= max_chars)
                        fits_tokens = True

                        if max_tokens and hasattr(self, "count_tokens") and callable(self.count_tokens):
                            try:
                                tk = self.count_tokens(u)
                                fits_tokens = (total_tokens + tk <= max_tokens)
                            except:
                                fits_tokens = True

                        if not (fits_chars and fits_tokens):
                            continue

                        if i in head_range and head_budget > 0:
                            if head_budget - length >= 0:
                                head_budget -= length

                        pieces.append(u)
                        seen.add(fp)
                        total_chars += length

                        if max_tokens and hasattr(self, "count_tokens") and callable(self.count_tokens):
                            try:
                                total_tokens += self.count_tokens(u)
                            except:
                                pass

                        if total_chars >= max_chars:
                            break
                        if max_tokens and total_tokens >= max_tokens:
                            break

                    except Exception as e:
                        print(f"[extractive_trim]     ⚠️ Error procesando unidad {i}: {e}")
                        continue

                if not pieces:
                    return s[:max_chars]

                trimmed = " ".join(pieces).strip()

                if len(trimmed) > max_chars:
                    trimmed = trimmed[:max_chars].rstrip()

                if max_tokens and hasattr(self, "count_tokens") and callable(self.count_tokens):
                    try:
                        tk = self.count_tokens(trimmed)
                        if tk > max_tokens:
                            approx = max(200, int(max_tokens * 4))
                            trimmed = trimmed[:approx].rstrip()
                    except:
                        pass

                return trimmed

            except Exception as e:
                print(f"[extractive_trim] ⚠️ Error en ensamblado: {e}")
                return s[:max_chars]

        except Exception as e:
            print(f"[extractive_trim] ❌ Error crítico: {e}")
            traceback.print_exc()


            try:
                return text[:max_chars].rstrip()
            except:
                return ""

    def _process_documents(
            self,
            relevant_docs: List,
            query: str,
            derived_context_block: Optional[str]
    ) -> List[Dict[str, Any]]:  # ✅ CAMBIAR A Dict
        """
        Procesa y recorta documentos con manejo de errores robusto.
        v3.0: Preserva metadata_json
        """


        processed_chunks = []

        if not relevant_docs:
            print(f"[Processing] ⚠️ Sin documentos para procesar")
            if derived_context_block:
                return [{'text': derived_context_block, 'metadata_json': ''}]
            return []

        if not isinstance(relevant_docs, list):
            print(f"[Processing] ⚠️ relevant_docs no es lista: {type(relevant_docs)}")
            relevant_docs = [relevant_docs]

        try:
            print(f"[Processing] 🔄 Procesando {len(relevant_docs)} documentos...")

            MAX_CHARS = 6000
            MAX_TOKENS = 1500

            is_structure_query = bool(re.search(
                r'\b(grupo|tabla|clasificación|categoría|todos|completo)\b',
                query,
                re.IGNORECASE
            ))

            for i, d in enumerate(relevant_docs):
                try:
                    # ═══════════════════════════════════════════════════════════
                    # ✅ EXTRAER TEXTO Y METADATA
                    # ═══════════════════════════════════════════════════════════

                    txt = ''
                    metadata_json = ''
                    filename = ''
                    source = ''
                    chunk_id = ''

                    if isinstance(d, str):
                        txt = d
                    elif isinstance(d, dict):
                        txt = d.get("text", d.get("page_content", d.get("content", str(d))))
                        metadata_json = d.get("metadata_json", "")
                        filename = d.get("filename", d.get("file_name", ""))
                        source = d.get("source", "")
                        chunk_id = d.get("chunk_id", d.get("name", ""))
                    elif hasattr(d, 'text'):
                        txt = d.text
                        metadata_json = getattr(d, 'metadata_json', '')
                        filename = getattr(d, 'filename', '') or getattr(d, 'file_name', '')
                        source = getattr(d, 'source', '')
                        chunk_id = getattr(d, 'chunk_id', '') or getattr(d, 'name', '')
                    elif hasattr(d, 'page_content'):
                        txt = d.page_content
                        if hasattr(d, 'metadata'):
                            metadata_json = d.metadata.get('metadata_json', '')
                            filename = d.metadata.get('filename', d.metadata.get('file_name', ''))
                            source = d.metadata.get('source', '')
                            chunk_id = d.metadata.get('chunk_id', d.metadata.get('name', ''))
                    else:
                        txt = str(d)

                    if not txt or not txt.strip():
                        print(f"[Processing]   ⚠️ Doc {i}: Texto vacío")
                        continue

                    # Detección de grupos
                    grupos_originales = re.findall(r'Grupo\s+(\d+)', txt)
                    num_grupos_originales = len(set(grupos_originales))

                    # Aplicar trimming
                    try:
                        if is_structure_query and num_grupos_originales >= 3:
                            max_chars_adjusted = int(MAX_CHARS * 1.3)
                            print(
                                f"[Processing]   📊 Doc {i}: Query estructural, {num_grupos_originales} grupos → max_chars={max_chars_adjusted}")
                        else:
                            max_chars_adjusted = MAX_CHARS

                        trimmed = self.extractive_trim_answer_aware(
                            txt,
                            query,
                            max_chars=max_chars_adjusted
                        )

                    except Exception as e:
                        print(f"[Processing]   ⚠️ Doc {i}: Error en trim: {e}, usando truncado")
                        trimmed = txt[:MAX_CHARS]

                    # Verificar tokens
                    if hasattr(self, "count_tokens") and callable(self.count_tokens):
                        try:
                            tks = self.count_tokens(trimmed)
                            if tks > MAX_TOKENS:
                                print(f"[Processing]   ℹ️ Doc {i}: Ajustando tokens {tks} → {MAX_TOKENS}")
                                while tks > MAX_TOKENS and len(trimmed) > 100:
                                    trimmed = trimmed[:int(len(trimmed) * 0.9)]
                                    tks = self.count_tokens(trimmed)
                                print(f"[Processing]   ✓ Doc {i}: Ajuste final: {tks} tokens")
                        except Exception as e:
                            print(f"[Processing]   ⚠️ Doc {i}: Error contando tokens: {e}")

                    if not trimmed or not trimmed.strip():
                        print(f"[Processing]   ⚠️ Doc {i}: Resultado vacío después de trim")
                        continue

                    # Verificar pérdida de información
                    grupos_finales = re.findall(r'Grupo\s+(\d+)', trimmed)
                    num_grupos_finales = len(set(grupos_finales))

                    if num_grupos_originales > 0:
                        if num_grupos_finales == num_grupos_originales:
                            print(
                                f"[Processing]   ✅ Doc {i}: Grupos preservados: {num_grupos_finales}/{num_grupos_originales}")
                        else:
                            grupos_perdidos = set(grupos_originales) - set(grupos_finales)
                            print(
                                f"[Processing]   ⚠️ Doc {i}: Grupos preservados: {num_grupos_finales}/{num_grupos_originales}")
                            print(f"[Processing]      → Perdidos: {sorted(grupos_perdidos)}")

                    # ═══════════════════════════════════════════════════════════
                    # ✅ AGREGAR COMO DICT CON METADATA
                    # ═══════════════════════════════════════════════════════════

                    processed_chunks.append({
                        'text': trimmed.strip(),
                        'metadata_json': metadata_json,
                        'filename': filename,
                        'source': source,
                        'chunk_id': chunk_id
                    })

                except Exception as e:
                    print(f"[Processing]   ❌ Doc {i}: Error crítico: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            print(f"[Processing] ✅ {len(processed_chunks)} chunks procesados de {len(relevant_docs)} docs")

        except Exception as e:
            print(f"[Processing] ❌ Error crítico en loop: {e}")
            import traceback
            traceback.print_exc()

        # Añadir contexto derivado al inicio
        if derived_context_block:
            try:
                processed_chunks.insert(0, {
                    'text': derived_context_block,
                    'metadata_json': '',
                    'filename': '',
                    'source': '',
                    'chunk_id': ''
                })
                print(f"[Processing] ✓ Contexto derivado añadido al inicio")
            except Exception as e:
                print(f"[Processing] ⚠️ Error añadiendo contexto derivado: {e}")

        return processed_chunks


    def _validate_response_coherence(
            self,
            response: str,
            expected_tone: str,
            expected_role: str
    ) -> Tuple[bool, List[str]]:
        """
        Valida coherencia de respuesta con instrucciones con manejo de errores robusto.

        Args:
            response: Respuesta a validar
            expected_tone: Tono esperado
            expected_role: Rol esperado

        Returns:
            Tuple[bool, List[str]]: (es_válida, lista_de_problemas)
        """

        # Crear contexto de error

        problems = []

        # Validación de inputs
        if not response or not response.strip():
            print(f"[_validate_response] ⚠️ Respuesta vacía")
            return True, []  # Consideramos válida una respuesta vacía

        if not expected_tone:
            expected_tone = ""

        if not expected_role:
            expected_role = ""

        try:
            # Convertir a minúsculas de forma segura
            try:
                response_lower = response.lower()
            except Exception as e:
                print(f"[_validate_response] ⚠️ Error convirtiendo a lower: {e}")
                response_lower = str(response).lower()

            # ✅ Validar tono formal
            try:
                if "formal" in expected_tone.lower():
                    informal_markers = [
                        "che", "tipo", "re ", "jaja", "jeje", "buenísimo",
                        "tío", "tía", "chaval", "mola", "guay"
                    ]

                    found_informal = []
                    for marker in informal_markers:
                        try:
                            if marker in response_lower:
                                found_informal.append(marker)
                        except Exception as e:
                            print(f"[_validate_response]   ⚠️ Error buscando '{marker}': {e}")
                            continue

                    if found_informal:
                        problems.append(f"Marcadores informales: {found_informal}")
            except Exception as e:
                print(f"[_validate_response] ⚠️ Error validando tono: {e}")

            # ✅ Validar rol
            try:
                role_terms = ["legal", "abogado", "jurídico"]
                if any(term in expected_role.lower() for term in role_terms):
                    off_domain = [
                        "diagnóstico médico", "receta médica", "tratamiento farmacológico",
                        "prescripción", "tomar medicamento"
                    ]

                    found_off_domain = []
                    for term in off_domain:
                        try:
                            if term in response_lower:
                                found_off_domain.append(term)
                        except Exception as e:
                            print(f"[_validate_response]   ⚠️ Error buscando '{term}': {e}")
                            continue

                    if found_off_domain:
                        problems.append(f"Contenido fuera del dominio legal: {found_off_domain}")
            except Exception as e:
                print(f"[_validate_response] ⚠️ Error validando rol: {e}")

            # ✅ Detectar alucinaciones
            try:
                hallucination_signals = [
                    "según mi base de datos interna",
                    "he verificado en mis registros",
                    "consultando mi información privada",
                    "en mi memoria interna"
                ]

                found_hallucinations = []
                for signal in hallucination_signals:
                    try:
                        if signal in response_lower:
                            found_hallucinations.append(signal)
                    except Exception as e:
                        print(f"[_validate_response]   ⚠️ Error buscando '{signal}': {e}")
                        continue

                if found_hallucinations:
                    problems.append(f"Posibles alucinaciones: {found_hallucinations}")
            except Exception as e:
                print(f"[_validate_response] ⚠️ Error detectando alucinaciones: {e}")

            # Calcular validez
            is_valid = len(problems) == 0

            if not is_valid:
                print(f"[_validate_response] ⚠️ {len(problems)} problemas detectados")
            else:
                print(f"[_validate_response] ✅ Respuesta válida, sin problemas")

            return is_valid, problems

        except Exception as e:
            print(f"[_validate_response] ❌ Error crítico: {e}")
            import traceback
            traceback.print_exc()


            # En caso de error, consideramos la respuesta válida para no bloquear
            return True, []

    def _create_main_executor(
            self,
            model: str,
            ghost_context: Optional[str],
            use_ghost_attention: bool,
            tone: str,
            personality: str
    ):
        """
        Crea ejecutor principal con Ghost Attention y manejo de errores robusto.

        Args:
            model: Modelo a usar
            ghost_context: Contexto ghost (opcional)
            use_ghost_attention: Si usar ghost attention
            tono: Tono de respuesta
            personalidad: Personalidad del bot

        Returns:
            Callable: Función ejecutora async
        """

        # Crear contexto de error

        # Validaciones
        if not model:
            print(f"[_create_main_executor] ⚠️ Modelo no especificado, usando default")
            model = "gpt-5.6"

        if not tone:
            tone = "formal"

        if not personality:
            personality = "profesional"

        async def model_executor(prompt: str) -> str:
            """
            Ejecutor del modelo con Ghost Attention y validación.

            Args:
                prompt: Prompt a ejecutar

            Returns:
                str: Respuesta del modelo (con mensaje de error si falla)
            """

            # Validar prompt
            if not prompt or not prompt.strip():
                print(f"[model_executor] ⚠️ Prompt vacío")
                return "❌ Error: prompt vacío."

            try:
                # ✅ Preparar prompt
                try:
                    final_prompt = prompt.strip()
                except Exception as e:
                    print(f"[model_executor] ⚠️ Error preparando prompt: {e}")
                    final_prompt = str(prompt)

                # ✅ Inyectar Ghost Attention
                if use_ghost_attention and ghost_context:
                    # Determinar método de inyección
                    try:
                        injection_method = GhostAttentionConfig.INJECTION_METHOD
                        print(f"[model_executor] ℹ️ Usando injection_method: {injection_method}")
                    except (AttributeError, NameError) as e:
                        print(f"[model_executor] ⚠️ Error accediendo a INJECTION_METHOD: {e}, usando 'prefix'")
                        injection_method = "prefix"

                    # Aplicar inyección
                    try:
                        final_prompt = self._inject_ghost_attention(
                            final_prompt,
                            ghost_context,
                            injection_method
                        )
                        print(f"[model_executor] ✅ Ghost attention inyectado")
                    except Exception as e:
                        print(f"[model_executor] ⚠️ Error inyectando ghost: {e}")

                # ✅ Añadir recordatorio
                try:
                    final_prompt += "\n\n[RECORDATORIO] Si no tenés información: 'no tengo información'."
                except Exception as e:
                    print(f"[model_executor] ⚠️ Error añadiendo recordatorio: {e}")

                # ✅ Ejecutar modelo principal
                try:
                    print(f"[model_executor] 🤖 Ejecutando modelo {model} (timeout: 60s)...")

                    response = await asyncio.wait_for(
                        self.custom.run_model_with_default(final_prompt, model=model),
                        timeout=60.0
                    )

                    if not response or not response.strip():
                        print(f"[model_executor] ⚠️ Respuesta vacía del modelo")
                        return "❌ No se pudo generar una respuesta."

                    print(f"[model_executor] ✅ Respuesta recibida: {len(response)} chars")

                except TimeoutError:
                    print(f"[model_executor] ⏱️ TIMEOUT en modelo principal (60s)")
                    raise  # Re-lanzar para que el fallback lo maneje

                except Exception as e:
                    print(f"[model_executor] ❌ Error en modelo {model}: {e}")
                    raise  # Re-lanzar para que el fallback lo maneje

                # ✅ Validar respuesta (si Ghost Attention está habilitado)
                if use_ghost_attention:
                    try:
                        enable_validation = GhostAttentionConfig.ENABLE_VALIDATION
                        print(
                            f"[model_executor] ℹ️ Validación: {'HABILITADA' if enable_validation else 'DESHABILITADA'}")
                    except (AttributeError, NameError) as e:
                        print(f"[model_executor] ⚠️ Error accediendo a ENABLE_VALIDATION: {e}")
                        enable_validation = False

                    if enable_validation:
                        try:
                            print(f"[GhostAttention] 🔍 Validando coherencia...")

                            is_valid, problems = self._validate_response_coherence(
                                response, tone, personality
                            )

                            if not is_valid and problems:
                                print(f"[GhostAttention] ⚠️ Problemas detectados: {problems}")
                                print("[GhostAttention] 🔄 Reintentando con refuerzo...")

                                # Construir prompt de retry
                                try:
                                    problems_text = '\n'.join(f'  - {p}' for p in problems)
                                    retry_prompt = f"""
        {ghost_context}

        {'=' * 60}
        🔴 CRÍTICO - CORREGIR PROBLEMAS DETECTADOS:
        {problems_text}
        {'=' * 60}

        {final_prompt}
        """
                                except Exception as e:
                                    print(f"[GhostAttention] ⚠️ Error construyendo retry_prompt: {e}")
                                    retry_prompt = final_prompt

                                # Reintentar
                                try:
                                    response = await asyncio.wait_for(
                                        self.custom.run_model_with_default(retry_prompt, model=model),
                                        timeout=60.0
                                    )

                                    if response and response.strip():
                                        print(f"[GhostAttention] ✅ Reintento exitoso: {len(response)} chars")
                                    else:
                                        print(f"[GhostAttention] ⚠️ Reintento retornó vacío, usando original")

                                except TimeoutError:
                                    print(f"[GhostAttention] ⏱️ TIMEOUT en reintento, usando original")
                                except Exception as e:
                                    print(f"[GhostAttention] ⚠️ Error en reintento: {e}, usando original")

                        except Exception as e:
                            print(f"[GhostAttention] ⚠️ Error en validación: {e}")

                return response

            except Exception as e:
                print(f"[model_executor] ❌ Error en ejecución principal: {e}")
                import traceback
                traceback.print_exc()

                # ✅ Fallback a modelo alternativo
                try:
                    fallback_model = self.custom.get_fallback_model()
                    print(f"[Fallback] 🆘 Usando modelo alternativo: {fallback_model}")

                    try:
                        fallback_response = await asyncio.wait_for(
                            self.custom.run_model_with_default(final_prompt, model=fallback_model),
                            timeout=30.0
                        )

                        if fallback_response and fallback_response.strip():
                            print(f"[Fallback] ✅ Respuesta exitosa: {len(fallback_response)} chars")
                            return fallback_response
                        else:
                            print(f"[Fallback] ⚠️ Respuesta vacía")
                            return f"❌ Error: {str(e)[:100]}"

                    except TimeoutError:
                        print(f"[Fallback] ⏱️ TIMEOUT (30s)")
                        return f"❌ Timeout. Error original: {str(e)[:100]}"

                    except Exception as fe:
                        print(f"[Fallback] ❌ Error en fallback: {fe}")
                        return f"❌ Error: {str(e)[:100]}"

                except Exception as fe:
                    print(f"[Error Crítico] ❌ Fallback falló: {fe}")
                    traceback.print_exc()
                    return f"❌ Error crítico: {str(e)[:100]}"

        return model_executor
    # ════════════════════════════════════════════════════════════════════
    # MÉTODO PRINCIPAL CON MANEJO DE ERRORES COMPLETO
    # ════════════════════════════════════════════════════════════════════
    def _extract_conclusion_with_fallbacks(self, text: str) -> str:
        """
        Extrae conclusiones con múltiples estrategias de fallback

        Returns:
            Texto de la conclusión o cadena vacía
        """
        if not text:
            return ""

        # Limpiar HTML primero
        clean_text = re.sub(r'<[^>]+>', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # NIVEL 1: Búsqueda con múltiples patrones
        patrones = [
            r'(?:Conclusión|conclusión)[:\s]*(.{50,500})',
            r'(?:En resumen|en resumen)[:\s,]*(.{50,500})',
            r'(?:Por tanto|por tanto)[:\s,]*(.{50,500})',
            r'(?:Respuesta final|respuesta final)[:\s:]*(.{50,500})',
            r'(?:Resultado|resultado)[:\s:]*(.{50,500})',
            r'(?:En definitiva|en definitiva)[:\s,]*(.{50,500})',
            r'(?:Finalmente|finalmente)[:\s,]*(.{50,500})',
        ]

        for patron in patrones:
            match = re.search(patron, clean_text, re.IGNORECASE | re.DOTALL)
            if match:
                conclusion = match.group(1).strip()
                conclusion = re.sub(r'\s+', ' ', conclusion)

                # Umbral más bajo: 30 caracteres en lugar de 50
                if len(conclusion) >= 30:
                    return conclusion[:400]

        # NIVEL 2: Último párrafo significativo
        paragraphs = [p.strip() for p in clean_text.split('\n\n') if p.strip()]
        if paragraphs:
            last_para = paragraphs[-1]
            if 40 <= len(last_para) <= 500:
                # Verificar que no sea solo metadata
                if not re.match(r'^(Fuente|Documento|Artículo|Referencia|Paso \d+):', last_para, re.IGNORECASE):
                    return last_para[:400]

        # NIVEL 3: Cualquier oración con indicadores de cierre
        oraciones = re.split(r'[.!?]\s+', clean_text)
        for oracion in reversed(oraciones[-5:]):  # Últimas 5 oraciones
            indicadores = ['por lo tanto', 'así', 'finalmente', 'en definitiva', 'por ende', 'en consecuencia']
            if any(kw in oracion.lower() for kw in indicadores):
                if 30 <= len(oracion) <= 500:
                    return oracion.strip()[:400]

        # NIVEL 4: Si el texto es corto, usar todo
        if 50 <= len(clean_text) <= 400:
            return clean_text

        # Sin conclusión encontrada
        return ""

    def _extract_filenames_from_response(self, response: str) -> set:
        """
        Extrae filenames de una respuesta (texto o HTML)

        Returns:
            Set de filenames únicos
        """
        if not response:
            return set()

        filenames = set()

        # Patrón 1: Referencias explícitas a archivos
        # Ejemplo: "según el archivo convenio_metal.pdf"
        pattern1 = r'(?:archivo|documento|fichero|file|doc)\s+([a-zA-Z0-9_\-áéíóúñÁÉÍÓÚÑ\s]+\.(?:pdf|docx|xlsx|txt|csv|doc))'
        matches1 = re.findall(pattern1, response, re.IGNORECASE)
        filenames.update(m.strip() for m in matches1)

        # Patrón 2: Extensiones de archivo comunes (más permisivo)
        pattern2 = r'([a-zA-Z0-9_\-áéíóúñÁÉÍÓÚÑ\s]{3,50}\.(?:pdf|docx|xlsx|txt|csv|doc))'
        matches2 = re.findall(pattern2, response, re.IGNORECASE)
        filenames.update(m.strip() for m in matches2)

        # Patrón 3: Metadata tags si existen
        pattern3 = r'<meta\s+name=["\']?(?:filename|file|source)["\']?\s+content=["\']([^"\']+)["\']'
        matches3 = re.findall(pattern3, response, re.IGNORECASE)
        filenames.update(matches3)

        # Patrón 4: Atributo data-filename o similares
        pattern4 = r'data-(?:filename|file|source)=["\']([^"\']+\.(?:pdf|docx|xlsx|txt|csv|doc))["\']'
        matches4 = re.findall(pattern4, response, re.IGNORECASE)
        filenames.update(matches4)

        # Filtrar falsos positivos comunes
        filtered = set()
        falsos_positivos = ['documento.pdf', 'archivo.pdf', 'file.pdf', 'ejemplo.pdf', 'test.pdf', 'doc.pdf']

        for fname in filenames:
            fname_clean = fname.strip()
            fname_lower = fname_clean.lower()

            # Excluir nombres genéricos
            if fname_lower in falsos_positivos:
                continue

            # Excluir si es muy corto
            if len(fname_clean) < 5:
                continue

            # Excluir si tiene caracteres extraños
            if re.search(r'[<>{}()\[\]]', fname_clean):
                continue

            filtered.add(fname_clean)

        return filtered

    async def _synthesize_responses(
            self,
            query: str,
            responses: list,
            model: str = GPT_5_NANO,  # 🎯 CAMBIO 1: gpt-5.6 → gpt-5.6
            timeout: float = 80.0,  # 🎯 CAMBIO 2: 90s → 40s
            include_filenames: bool = True,
            request_id: str = None
    ) -> str:
        """
        Síntesis unificada para ambas ramas (Decomposer y Estándar)

        Args:
            query: Pregunta original
            responses: Lista de respuestas a sintetizar (dict con 'step', 'query', 'response')
            model: Modelo a usar
            timeout: Timeout en segundos
            include_filenames: Si extraer y mostrar filenames
            request_id: ID de la petición para logging

        Returns:
            HTML sintetizado con razonamiento
        """
        from openai import AsyncOpenAI
        from config.Constants import KEY_OPEN_AI

        if not responses:
            print(f"[SYNTHESIS] ⚠️ Sin respuestas para sintetizar")
            return "<p>No se pudo generar respuesta.</p>"

        # Filtrar respuestas válidas
        valid_responses = [r for r in responses if not r.get('error') and r.get('response')]
        if not valid_responses:
            print(f"[SYNTHESIS] ⚠️ Todas las respuestas tienen errores")
            return "<p>No se encontró información relevante.</p>"

        print(f"[SYNTHESIS] 📊 Procesando {len(valid_responses)} respuestas válidas...")

        # Preparar contexto
        steps_context = []
        key_conclusions = []
        filenames_unicos = set()

        for sr in valid_responses:
            paso_num = sr.get('step', sr.get('index', 0))
            paso_query = sr.get('query', '')
            paso_response = sr.get('response', '')

            # Agregar al contexto
            steps_context.append({
                'paso': paso_num,
                'query': paso_query,
                'respuesta': paso_response
            })

            # 🔥 EXTRACCIÓN DE CONCLUSIONES MEJORADA
            conclusion_text = self._extract_conclusion_with_fallbacks(paso_response)
            if conclusion_text:
                key_conclusions.append({
                    'paso': paso_num,
                    'conclusion': conclusion_text
                })
                print(f"[SYNTHESIS] 📝 Conclusión Paso {paso_num}: {conclusion_text[:80]}...")

            # 🔥 EXTRACCIÓN DE FILENAMES
            if include_filenames:
                filenames = self._extract_filenames_from_response(paso_response)
                if filenames:
                    filenames_unicos.update(filenames)
                    print(f"[SYNTHESIS] 📁 Paso {paso_num}: {len(filenames)} filename(s) encontrado(s)")

        # Preparar texto del contexto
        context_text = []
        for item in steps_context:
            text = re.sub(r'<[^>]+>', ' ', item['respuesta'])
            text = re.sub(r'\s+', ' ', text).strip()

            # Detectar si tiene estructura compleja
            tiene_estructura = bool(
                re.search(
                    r'Grupo \d+|Artículo \d+|lista|categoría|requisito|tabla|periodo|sección|apartado',
                    text,
                    re.IGNORECASE
                )
            )

            # 🎯 CAMBIO 3: Reducir contexto (12000→8000, 8000→5000)
            max_len = 8000 if tiene_estructura else 5000
            context_text.append(f"PASO {item['paso']}: {item['query']}\n{text[:max_len]}")

        # Preparar conclusiones
        if key_conclusions:
            conclusiones_text = "\n".join([
                f"• Paso {c['paso']}: {c['conclusion']}"
                for c in key_conclusions
            ])
            print(f"[SYNTHESIS] 📊 Conclusiones extraídas: {len(key_conclusions)}")
        else:
            conclusiones_text = "(No se encontraron conclusiones explícitas en los pasos anteriores)"
            print(f"[SYNTHESIS] ⚠️ Sin conclusiones explícitas")

        # Preparar filenames
        filenames_text = ""
        filenames_html = ""

        if include_filenames and filenames_unicos:
            filenames_sorted = sorted(filenames_unicos)

            filenames_text = f"""

    ══════════════════════════════════════════════════════════════════════
    📄 DOCUMENTOS CONSULTADOS ({len(filenames_sorted)})
    ══════════════════════════════════════════════════════════════════════

    {chr(10).join(f"• {fname}" for fname in filenames_sorted)}
    """

            filenames_html = f"""
    <h4>📄 Documentos consultados</h4>
    <ul>
    {chr(10).join(f"  <li>{fname}</li>" for fname in filenames_sorted)}
    </ul>
    """

            print(f"[SYNTHESIS] 📁 Total filenames únicos: {len(filenames_sorted)}")
            for fname in filenames_sorted[:3]:
                print(f"[SYNTHESIS]    • {fname}")
            if len(filenames_sorted) > 3:
                print(f"[SYNTHESIS]    ... y {len(filenames_sorted) - 3} más")

        # 🎯 CAMBIO 4: Prompt simplificado pero manteniendo funcionalidad
        synthesis_prompt = f"""Pregunta original: {query}

                {self._build_universal_few_shot_examples() if hasattr(self, '_build_universal_few_shot_examples') else ''}

                ══════════════════════════════════════════════════════════════════════
                📝 CONCLUSIONES DE PASOS ANTERIORES
                ══════════════════════════════════════════════════════════════════════

                {conclusiones_text}

                ══════════════════════════════════════════════════════════════════════
                📊 CONTEXTO DE {len(context_text)} PASOS
                ══════════════════════════════════════════════════════════════════════

                {chr(10).join(context_text)}

                {filenames_text}

                ══════════════════════════════════════════════════════════════════════
                🚨 REGLAS CRÍTICAS DE FIDELIDAD ABSOLUTA AL CONTEXTO
                ══════════════════════════════════════════════════════════════════════

                PRINCIPIO FUNDAMENTAL: Tu respuesta debe ser un ESPEJO EXACTO del contexto.

                OBLIGACIONES ABSOLUTAS:

                1. FIDELIDAD TEXTUAL:
                   ✅ SOLO usa información presente en el contexto
                   ✅ Reproduce valores numéricos exactamente como aparecen
                   ✅ Respeta el nivel de especificidad del contexto

                2. CITACIÓN EXHAUSTIVA:
                   ✅ CADA afirmación DEBE ir con: [Documento: X | Artículo: Y]
                   ✅ Para listas: cita CADA elemento con su fuente

                3. INFORMACIÓN AUSENTE:
                   ✅ Si algo NO está, declara: "⚠️ INFORMACIÓN NO ENCONTRADA: [tema]"

                4. TRANSPARENCIA:
                   ✅ Indica nivel de confianza (ALTA/MEDIA/BAJA)
                   ✅ Marca explícitamente limitaciones

                5. NO INVENTES:
                   ❌ NO uses conocimiento externo
                   ❌ NO hagas inferencias sin evidencia
                   ❌ NO generalices sin justificación

                ══════════════════════════════════════════════════════════════════════
                📋 ESTRUCTURA HTML REQUERIDA
                ══════════════════════════════════════════════════════════════════════

                <div class="respuesta-completa">

                <h4>1. Respuesta Principal</h4>
                [Contenido basado en contexto con citas]

                <h4>2. Información Detallada</h4>
                [Detalles específicos con citas para cada elemento]

                <h4>3. Limitaciones Detectadas</h4>
                <div class="warning-section">
                <p><strong>⚠️ Información NO encontrada:</strong></p>
                <ul>
                <li>[Temas ausentes con descripción]</li>
                </ul>
                </div>

                <h4>4. Nivel de Confianza</h4>
                <ul>
                <li><strong>ALTA:</strong> [aspectos verificados]</li>
                <li><strong>MEDIA:</strong> [aspectos parciales]</li>
                <li><strong>BAJA:</strong> [aspectos insuficientes]</li>
                </ul>

                <h4>5. Documentos Consultados</h4>
                {filenames_html if filenames_html else ""}

                </div>

                ══════════════════════════════════════════════════════════════════════
                🚨 VERIFICACIÓN FINAL
                ══════════════════════════════════════════════════════════════════════

                ✓ Cada afirmación tiene cita [Documento | Artículo]
                ✓ He marcado información ausente
                ✓ He indicado nivel de confianza
                ✓ He incluido sección numerada para cada sub-pregunta

                FORMATO: 
                1. <reasoning> completo (opcional pero recomendado)
                2. HTML verboso y transparente (sin estilos inline)

                GENERA AHORA:"""

        # Llamar al modelo
        client = AsyncOpenAI(api_key=KEY_OPEN_AI)

        print(f"[SYNTHESIS] 🤖 Modelo: {model}")
        print(f"[SYNTHESIS] ⏱️ Timeout: {timeout}s")
        print(f"[SYNTHESIS] 📦 Prompt size: {len(synthesis_prompt)} chars")

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Eres un asistente de análisis con razonamiento riguroso. "
                                "Determinas qué miden los valores ANTES de interpretarlos. "
                                "Aplicas lógica INVERSA cuando valores miden necesidad/formación "
                                "(más valor = menos preparado), "
                                "y lógica DIRECTA cuando miden requisitos de entrada "
                                "(más valor = más nivel exigido). "
                                "CRÍTICO: NO te contradices. "
                                "SIEMPRE incluyes la sección '📄 Documentos consultados' al final. "
                                "SIEMPRE usas <reasoning> tags para mostrar tu proceso."
                            )
                        },
                        {"role": "user", "content": synthesis_prompt}
                    ],
                    #temperature=0,
                    max_completion_tokens=5000,  # 🎯 CAMBIO 5: 16000 → 5000
                    reasoning_effort="low"
                ),
                timeout=timeout
            )

            synthesis_response = response.choices[0].message.content

            # Extraer y loguear razonamiento
            if "<reasoning>" in synthesis_response:
                reasoning_match = re.search(
                    r'<reasoning>(.*?)</reasoning>',
                    synthesis_response,
                    re.DOTALL
                )
                if reasoning_match:
                    reasoning_text = reasoning_match.group(1).strip()
                    print(f"\n[SYNTHESIS] 🧠 RAZONAMIENTO GENERADO ({len(reasoning_text)} chars):")
                    print(f"{'═' * 70}")

                    # Mostrar primeras líneas
                    for i, line in enumerate(reasoning_text.split('\n')[:10]):
                        if line.strip():
                            print(f"  {line.strip()[:70]}")

                    if len(reasoning_text.split('\n')) > 10:
                        print(f"  ... (truncado)")

                    print(f"{'═' * 70}\n")

                # Remover reasoning del HTML final
                synthesis_response = re.sub(
                    r'<reasoning>.*?</reasoning>',
                    '',
                    synthesis_response,
                    flags=re.DOTALL
                ).strip()
            else:
                print(f"[SYNTHESIS] ⚠️ No se generó razonamiento explícito")

            # Verificar que incluya filenames si existen
            if filenames_unicos and "Documentos consultados" not in synthesis_response:
                print(f"[SYNTHESIS] ⚠️ Agregando sección de documentos (LLM la omitió)")
                synthesis_response += filenames_html

            # 🆕 GARANTIZAR que SIEMPRE se muestren los documentos si hay filenames
            if filenames_unicos:
                # Verificar si está presente de alguna forma
                tiene_seccion_docs = any(x in synthesis_response.lower() for x in [
                    "documentos consultados",
                    "fuentes consultadas",
                    "📄",
                    "documentos:"
                ])

                if not tiene_seccion_docs:
                    print(f"[SYNTHESIS] 🔧 Forzando inclusión de {len(filenames_unicos)} documento(s)")
                    synthesis_response += f"\n\n{filenames_html}"

            print(f"[SYNTHESIS] ✅ Respuesta generada: {len(synthesis_response)} chars\n")

            return synthesis_response

        except TimeoutError:
            print(f"[SYNTHESIS] ⏱️ TIMEOUT después de {timeout}s")
            raise
        except Exception as e:
            print(f"[SYNTHESIS] ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

    def synthesize_answers_no_mix(self,steps_results: list, query_original: str) -> str:


        print(f"\n[SÍNTESIS] 🎯 Sintetizando {len(steps_results)} respuestas SIN mezcla...")

        # ══════════════════════════════════════════════════════════════
        # AGRUPAR RESPUESTAS POR PASO
        # ══════════════════════════════════════════════════════════════

        html_parts = []

        html_parts.append("<div style='font-family: Arial, sans-serif;'>")
        html_parts.append(f"<h2>📋 Respuesta a: {query_original}</h2>")
        html_parts.append("<hr>")

        for i, paso in enumerate(steps_results, 1):
            query_paso = paso.get('query', '')
            response_paso = paso.get('response', '')
            chunks_used = paso.get('chunks_used', 0)

            # Limpiar response si tiene tags HTML mal cerrados
            response_limpia = response_paso.strip()

            # Agregar sección del paso
            html_parts.append(
                f"<div style='margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; background: #f8f9fa;'>")
            html_parts.append(f"<h3 style='margin-top: 0; color: #007bff;'>🔍 Pregunta {i}: {query_paso}</h3>")
            html_parts.append(response_limpia)
            html_parts.append(
                f"<p style='margin-top: 10px; font-size: 0.9em; color: #6c757d;'>📊 Basado en {chunks_used} documento(s)</p>")
            html_parts.append("</div>")

        html_parts.append("</div>")

        result = "\n".join(html_parts)

        print(f"[SÍNTESIS] ✅ Síntesis completada: {len(result)} chars")

        return result

    # ══════════════════════════════════════════════════════════════
    # ALTERNATIVA: Síntesis con GPT (SI QUIERES UNIFICAR)
    # ══════════════════════════════════════════════════════════════

    async def synthesize_with_gpt_no_mix(self,steps_results: list, query_original: str, openai_client) -> str:
        """
        Usa GPT para sintetizar MANTENIENDO separación clara.
        """

        # Preparar contexto para GPT
        steps_context = []
        for i, paso in enumerate(steps_results, 1):
            steps_context.append(f"""
    PREGUNTA {i}: {paso['query']}
    RESPUESTA {i}: {paso['response']}
    ---""")

        prompt = f"""Tienes {len(steps_results)} respuestas a diferentes aspectos de una pregunta.

    Query original del usuario: "{query_original}"

    Respuestas individuales:
    {chr(10).join(steps_context)}

    INSTRUCCIONES CRÍTICAS:
    1. Mantén las respuestas CLARAMENTE SEPARADAS
    2. NO mezcles información entre preguntas
    3. Si una pregunta es sobre "Grupo 1", NO menciones "Grupo 7" en esa respuesta
    4. Si una pregunta NO tiene respuesta, di "No se encontró información" solo para ESA pregunta
    5. Usa formato HTML con secciones separadas

    Genera HTML estructurado con cada respuesta en su propia sección."""

        try:
            response = await openai_client.chat.completions.create(
                model=GPT_5_4_MINI,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un asistente que organiza respuestas manteniendo separación clara entre diferentes aspectos de una pregunta."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                #temperature=0,
                max_completion_tokens=2000,
                reasoning_effort="low"
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"[SÍNTESIS] ❌ Error en GPT: {e}")
            # Fallback a síntesis simple
            #return self.synthesize_answers_no_mix(pasos_resultados, query_original)

    async def _process_chunks_batch_extraction(
            self,
            chunks: list,
            sub_query: str,
            step_idx: int,
            top_k: int = 8,
    ) -> list:
        """
        fix1+2: reemplaza N llamadas LLM individuales por 1 sola llamada batch.

        Antes: 15 chunks × GPT-5.1 individual = 15 llamadas ≈ 22-25s
        Ahora:
          1. Cross-encoder pre-filter → top_k chunks relevantes (~200ms, sin API)
          2. 1 llamada GPT-5.4-mini con todos los chunks → parse respuesta

        Resultado: ~3s total vs 22-25s anterior.
        """
        if not chunks:
            return []

        # ── Fix 2: cross-encoder pre-filter ─────────────────────────────
        try:
            ranked = await asyncio.wait_for(
                _cross_rerank(sub_query, chunks, top_k=min(top_k, len(chunks))),
                timeout=8.0,
            )
            print(f"[BATCH_EXTRACT] ✅ Cross-encoder: {len(chunks)}→{len(ranked)} chunks (paso {step_idx})")
        except Exception as e:
            print(f"[BATCH_EXTRACT] ⚠️ Cross-encoder falló ({e}), usando top-{top_k} por score")
            def _s(c):
                if isinstance(c, dict):
                    return float(c.get("certainty") or c.get("score") or 0.0)
                return 0.0
            ranked = sorted(chunks, key=_s, reverse=True)[:top_k]

        # ── Fix 1: batch extraction con 1 sola llamada LLM ─────────────
        fragmentos = []
        metadatos = []
        for i, doc in enumerate(ranked):
            text = self._extract_text(doc)[:1500]
            name = ""
            if isinstance(doc, dict):
                name = doc.get("filename") or doc.get("nombre") or f"Doc{i+1}"
            fragmentos.append(f"[{i+1}] {name}\n{text}")
            metadatos.append({
                "idx": i,
                "filename": name,
                "original_doc": doc,
            })

        prompt = f"""Analiza los siguientes fragmentos y extrae la información relevante para esta pregunta.

PREGUNTA: {sub_query}

FRAGMENTOS:
{chr(10).join(fragmentos)}

INSTRUCCIONES:
Para cada fragmento, responde con exactamente este formato:
FRAG_1: <contenido relevante en HTML con <p> y <strong>> O bien: SIN_INFO
FRAG_2: <contenido relevante en HTML> O bien: SIN_INFO
...y así hasta FRAG_{len(ranked)}

Responde SOLO si el fragmento contiene información directamente relevante para la pregunta.
Si un fragmento es relevante aunque sea parcialmente, extrae esa parte.
"""
        try:
            client = openai.AsyncOpenAI(api_key=KEY_OPEN_AI)
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=GPT_5_4_MINI,
                    messages=[
                        {"role": "system", "content": "Eres un extractor de información preciso. Responde SOLO con el formato FRAG_N: indicado."},
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=4000,
                ),
                timeout=30.0,
            )
            raw = resp.choices[0].message.content or ""
            print(f"[BATCH_EXTRACT] ✅ LLM respondió: {len(raw)} chars (1 llamada para {len(ranked)} chunks)")
        except Exception as e:
            print(f"[BATCH_EXTRACT] ❌ Error LLM batch: {e}")
            return []

        # ── Parsear respuesta ────────────────────────────────────────────
        results = []
        import re as _re
        for i, meta in enumerate(metadatos):
            pattern = rf"FRAG_{i+1}:\s*(.*?)(?=FRAG_\d+:|$)"
            match = _re.search(pattern, raw, _re.DOTALL | _re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content and content.upper() != "SIN_INFO" and len(content) > 20:
                    results.append({
                        "response": content,
                        "filename": meta["filename"],
                        "chunk_idx": meta["idx"],
                    })

        print(f"[BATCH_EXTRACT] ✅ {len(results)}/{len(ranked)} fragmentos relevantes (paso {step_idx})")
        return results

    async def _process_chunks_in_batches(
            self,
            chunks: List[Any],
            query: str,
            executor: Callable,
            batch_size: int = 15,
            timeout_per_chunk: float = 45.0
    ) -> List[Dict[str, Any]]:
        """
        Procesa chunks en batches paralelos de forma optimizada.
        """

        if not chunks:
            return []

        total_chunks = len(chunks)
        num_batches = (total_chunks + batch_size - 1) // batch_size

        print(f"[PROCESSING] 🚀 Iniciando procesamiento paralelo en batches...")
        print(f"[PROCESSING] 📦 Total batches: {num_batches} (tamaño: {batch_size})")

        all_results = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_chunks)
            batch_chunks = chunks[start_idx:end_idx]

            print(f"\n{'─' * 80}")
            print(f"[BATCH {batch_idx + 1}/{num_batches}] 🚀 Procesando {len(batch_chunks)} chunks...")
            print(f"{'─' * 80}\n")

            # Crear tareas para el batch
            tasks = []
            for idx, chunk in enumerate(batch_chunks, start=start_idx + 1):
                task = asyncio.wait_for(
                    self._process_single_chunk(idx, chunk, query, executor, 0, timeout_per_chunk),
                    timeout=120
                )
                tasks.append(task)

            # Ejecutar batch en paralelo
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filtrar resultados válidos
            valid_count = 0
            timeout_count = 0
            error_count = 0

            for result_idx, result in enumerate(batch_results):
                chunk_num = start_idx + result_idx + 1
                original_chunk = batch_chunks[result_idx]  # ← OBTENER CHUNK ORIGINAL

                # Manejar timeouts
                if isinstance(result, TimeoutError):
                    timeout_count += 1
                    print(f"[BATCH {batch_idx + 1}] ⏱️ Chunk {chunk_num} - TIMEOUT")
                    continue

                # Manejar otros errores
                if isinstance(result, Exception):
                    error_count += 1
                    print(f"[BATCH {batch_idx + 1}] ❌ Chunk {chunk_num} - ERROR: {str(result)[:100]}")
                    continue

                # ✅ VALIDACIÓN CORREGIDA: Acepta dict o string Y preserva metadata
                if result:
                    # Caso 1: Ya es un dict con 'response'
                    if isinstance(result, dict) and result.get('response'):
                        # ✅ Agregar metadata del chunk original si no existe
                        if 'filename' not in result and hasattr(original_chunk, 'metadata'):
                            result['filename'] = original_chunk.metadata.get('file_name',
                                                                             original_chunk.metadata.get('filename',
                                                                                                         ''))
                        elif 'filename' not in result and isinstance(original_chunk, dict):
                            result['filename'] = original_chunk.get('filename',
                                                                    original_chunk.get('file_name', ''))

                        all_results.append(result)
                        valid_count += 1
                        print(
                            f"[BATCH {batch_idx + 1}] ✅ Chunk {chunk_num} - Válido (dict, {len(result.get('response', ''))} chars)")

                    # Caso 2: Es un string válido (lo que retorna _process_single_chunk)
                    elif isinstance(result, str) and len(result.strip()) > 0:
                        # ✅ Extraer metadata del chunk original
                        filename = ''

                        # Intentar obtener filename del chunk original
                        if hasattr(original_chunk, 'metadata'):
                            metadata = original_chunk.metadata
                            filename = metadata.get('file_name', metadata.get('filename', ''))
                        elif isinstance(original_chunk, dict):
                            filename = original_chunk.get('filename', original_chunk.get('file_name', ''))

                        # Convertir string a dict CON metadata preservada
                        chunk_result = {
                            'response': result.strip(),
                            'chunk_idx': chunk_num,
                            'filename': filename,  # ← PRESERVAR FILENAME
                            # ✅ Agregar más metadata si es necesaria
                            'source': original_chunk.get('source', '') if isinstance(original_chunk, dict) else '',
                            'chunk_id': original_chunk.get('chunk_id', '') if isinstance(original_chunk, dict) else ''
                        }

                        all_results.append(chunk_result)
                        valid_count += 1
                        print(
                            f"[BATCH {batch_idx + 1}] ✅ Chunk {chunk_num} - Válido (string, {len(result)} chars, file: {filename[:30]}...)")

            print(f"\n[BATCH {batch_idx + 1}/{num_batches}] 📊 RESUMEN:")
            print(f"   ✅ Válidos: {valid_count}/{len(batch_chunks)}")
            print(f"   ⏱️ Timeouts: {timeout_count}")
            print(f"   ❌ Errores: {error_count}")
            print(f"   📈 Total acumulado: {len(all_results)}\n")

        print(f"{'═' * 80}")
        print(f"[PROCESSING] ✅ PROCESAMIENTO COMPLETADO")
        print(f"{'═' * 80}")
        print(f"[PROCESSING] 📊 Chunks procesados: {len(all_results)}/{total_chunks}")
        if total_chunks > 0:
            print(f"[PROCESSING] 📊 Tasa éxito: {len(all_results) / total_chunks * 100:.1f}%")
        print(f"{'═' * 80}\n")

        return all_results



    """
    Método synthesize_answers MEJORADO v3.0
    - Razonamiento profundo centralizado
    - Chain-of-Thought explícito
    - Mayor calidad en respuestas complejas

    REEMPLAZA el método actual synthesize_answers en tu clase
    """

    from typing import List, Dict, Any, Callable

    def _extract_referencias_from_html(self, html_text: str) -> List[str]:
        """
        ✅ Extrae referencias del HTML de forma GENÉRICA
        Detecta cualquier texto en <strong> seguido de ":" al inicio de párrafos
        """
        referencias = set()

        # Patrón genérico: <strong>Cualquier texto:</strong>
        # Captura todo lo que esté en negrita seguido de ":"
        pattern = r'<strong>([^<]+?):</strong>'

        matches = re.finditer(pattern, html_text, re.IGNORECASE)

        for match in matches:
            ref = match.group(1).strip()

            # Filtrar referencias muy cortas (probablemente no son referencias)
            if len(ref) < 5:
                continue

            # Filtrar palabras comunes que no son referencias
            skip_words = ['nota', 'importante', 'ejemplo', 'resumen', 'conclusión']
            if any(word in ref.lower() for word in skip_words):
                continue

            # Normalizar capitalización (primera letra mayúscula)
            if ref and ref[0].islower():
                ref = ref[0].upper() + ref[1:]

            referencias.add(ref)

        return list(referencias)

    async def synthesize_answers(
                self,
                all_results: List[Dict[str, Any]],
                query: str,
                model_executor: Callable,
                simple_model_executor: Callable,
                query_type: str,
                requires_aggregation: bool,
                user: str,
                company: str,
                conversation_id: str,
                config_chatbot: str,
                decomposed_steps_info: Optional[List[Dict]] = None,
                total_steps: Optional[int] = None,
                chunks_by_step: Optional[Dict[int, List]] = None,
                conversation_history: Optional[List[Dict]] = None,  # FIX: contexto BOTH
                fallback_chunks: Optional[List[Any]] = None,  # recall: chunks crudos recuperados
        ) -> str:
            """
            Sintetiza múltiples respuestas con FEW-SHOT LEARNING v5.2
            ✅ INCLUYE METADATA ESTRUCTURADA DE DOCUMENTOS
            ✅ SYNTHESIS: gemini-3.1-pro-preview | SUMMARY: GPT_5_NANO
            ✅ BOTH: inyecta historial de conversación en el prompt de síntesis
            """
            synthesis_start = time.time()

            if not all_results:
                # Recall: la extracción por-chunk no produjo respuestas (p.ej. el doc
                # solo NOMBRA la entidad sin definirla). Antes de rendirse, intentar
                # sintetizar del TEXTO CRUDO recuperado.
                if not self._sources_from_raw_chunks(fallback_chunks):
                    return self._generate_no_info_response(query)
                all_results = []  # sigue; sources_content se llena del fallback abajo

            # ═══════════════════════════════════════════════════════════════
            # EXTRACCIÓN PREVIA: Referencias + Datos Usuario
            # ═══════════════════════════════════════════════════════════════

            all_referencias = set()
            for result in all_results:
                response_html = result.get('response', '')
                if response_html:
                    refs = self._extract_referencias_from_html(response_html)
                    all_referencias.update(refs)

            # fix2: _extract_user_data corre en paralelo con el análisis de query
            # Antes: bloqueaba ~5s antes de siquiera arrancar la síntesis
            user_data_task = asyncio.ensure_future(
                self._extract_user_data_from_query(query)
            )

            print(f"\n{'🧠' * 40}")
            print(f"[SYNTHESIS v5.2] 📋 Referencias: {len(all_referencias)}")
            print(f"[SYNTHESIS v5.2] 🤖 Modelo síntesis: gpt-5.1 (analítico) / gpt-5.4-mini (conversacional)")
            print(f"[SYNTHESIS v5.2] 💬 Historial: {len(conversation_history or [])} turns")

            # ═══════════════════════════════════════════════════════════════
            # ANÁLISIS DE QUERY
            # ═══════════════════════════════════════════════════════════════

            is_decomposed = query_type == "MULTI_PARTE_DECOMPOSED" or (decomposed_steps_info is not None)
            query_lower = query.lower()

            needs_table = any(kw in query_lower for kw in ['compara', 'comparación', 'versus', 'vs', 'diferencia'])
            needs_ranking = any(kw in query_lower for kw in ['mejor', 'ranking', 'recomienda', 'cuál es'])
            needs_steps = sum(1 for kw in ['primero', 'luego', 'después', 'paso'] if kw in query_lower) >= 2
            needs_legal_refs = any(kw in query_lower for kw in ['artículo', 'ley', 'decreto', 'convenio', 'normativa'])

            # ═══════════════════════════════════════════════════════════════
            # ✅ CONSTRUCCIÓN DE CONTEXTO (CON METADATA)
            # ═══════════════════════════════════════════════════════════════

            sources_content = []

            if is_decomposed and chunks_by_step:
                for step_num in sorted(chunks_by_step.keys()):
                    step_chunks = chunks_by_step[step_num]
                    step_info = next((s for s in (decomposed_steps_info or []) if s['step'] == step_num), None)
                    step_query = step_info['query'] if step_info else f"Paso {step_num}"

                    for chunk in step_chunks:
                        response_text = chunk.get('response', '')
                        if response_text and len(response_text) > 20:
                            metadata_str = self._get_metadata_context_(chunk)

                            sources_content.append({
                                'index': len(sources_content) + 1,
                                'content': response_text,
                                'metadata': metadata_str,
                                'step_number': step_num,
                                'step_query': step_query
                            })
            else:
                for idx, result in enumerate(all_results, 1):
                    response_text = result.get('response', '')
                    if response_text and len(response_text) > 20:
                        metadata_str = self._get_metadata_context_(result)

                        sources_content.append({
                            'index': idx,
                            'content': response_text,
                            'metadata': metadata_str
                        })

            if not sources_content:
                # Recall fallback: sintetizar del texto crudo recuperado en lugar de
                # devolver "no encontré información" (name-drops, chunks sin extracción).
                sources_content = self._sources_from_raw_chunks(fallback_chunks)
                if sources_content:
                    logger.info("synthesize_answers: usando %d chunk(s) crudos como "
                                "fallback de recall (extracción por-chunk vacía)", len(sources_content))
                else:
                    return self._generate_no_info_response(query)

            # ═══════════════════════════════════════════════════════════════
            # ✅ CONSTRUIR sources_text CON METADATA
            # ═══════════════════════════════════════════════════════════════

            sources_text_parts = []

            for src in sources_content:
                parts = [f"═══ FUENTE {src['index']} ═══"]

                if src.get('metadata'):
                    parts.append(f"📋 METADATA ESTRUCTURADA:\n{src['metadata']}")

                if src.get('step_query'):
                    parts.append(f"🔍 PASO {src['step_number']}: {src['step_query']}")

                parts.append(f"📄 CONTENIDO:\n{src['content']}")

                sources_text_parts.append("\n".join(parts))

            sources_text = "\n\n" + ("─" * 50 + "\n\n").join(sources_text_parts)

            # Gemini 3.1 soporta 1M tokens — contexto ampliado
            max_context = 80000 if is_decomposed else 60000
            if len(sources_text) > max_context:
                sources_text = sources_text[:max_context] + "\n... (truncado)"

            # ═══════════════════════════════════════════════════════════════
            # FIX: CONTEXTO DE CONVERSACIÓN PREVIA (solo modo BOTH)
            # ═══════════════════════════════════════════════════════════════

            history_context_block = ""
            if conversation_history:
                lines = []
                for turn in conversation_history[-8:]:  # últimos 8 turns
                    role_label = "Usuario" if turn.get("role") == "user" else "Assistant"
                    content = turn.get("content", "")[:400]
                    lines.append(f"[{role_label}]: {content}")

                history_context_block = (
                        "\n\nCONTEXTO DE CONVERSACIÓN PREVIA:\n"
                        "═══════════════════════════════════════════════════════════════\n"
                        + "\n".join(lines)
                        + "\n═══════════════════════════════════════════════════════════════\n"
                          "INSTRUCCIÓN ESPECIAL: La respuesta debe tener en cuenta el contexto "
                          "previo mostrado arriba. Si el usuario hace referencia a algo ya discutido, "
                          "reconócelo explícitamente y amplía o corrige según los documentos recuperados.\n"
                          "═══════════════════════════════════════════════════════════════\n"
                )

            # ═══════════════════════════════════════════════════════════════
            # FEW-SHOT EXAMPLES
            # ═══════════════════════════════════════════════════════════════

            # fix2: resolver user_data aquí (ya corrió en paralelo con el análisis)
            try:
                user_data = await asyncio.wait_for(user_data_task, timeout=8.0)
            except Exception:
                user_data = ""
            print(f"[SYNTHESIS v5.2] 👤 Datos usuario: {len(user_data)} chars")

            few_shot_examples = self._build_few_shot_examples(
                needs_legal_refs=needs_legal_refs,
                needs_ranking=needs_ranking,
                needs_table=needs_table,
                needs_steps=needs_steps,
                all_referencias=all_referencias,
                query=query,
                user_data=user_data
            )

            # ═══════════════════════════════════════════════════════════════
            # ✅ PROMPT DE SÍNTESIS
            # ═══════════════════════════════════════════════════════════════

            # ── Detectar si la query es conversacional (corta, informal) ──────
            _query_words = len(query.split())
            _is_conversational = (
                _query_words <= 15
                and not needs_table
                and not needs_legal_refs
                and not is_decomposed
            )

            if _is_conversational:
                # Modo conversacional: tono natural, sin HTML pesado, GPT-5-nano (rápido)
                reasoning_prompt = f"""Eres un asistente inteligente y directo. Responde de forma natural y conversacional.

PREGUNTA: {query}
{history_context_block}
INFORMACIÓN DISPONIBLE:
{sources_text[:20000]}

INSTRUCCIONES:
- Responde directamente a la pregunta, como lo haría un asistente amigable
- Usa HTML mínimo: solo <p> y <strong> cuando sea necesario
- Sé conciso: 2-4 párrafos máximo para preguntas simples
- Si la información está en los documentos, úsala; si no, dilo claramente
- Habla en primera persona cuando sea natural ("En los documentos encuentro...", "Según lo que tengo...")
- NUNCA inventes ni estimes precios, productos, cifras, fechas ni fuentes que no estén en la información disponible. Si piden comparar con datos externos (mercado, competencia, precios de otros lugares) y no hay fuente real, DECILO ("no tengo datos de mercado en esta base") y ofrecé usar la herramienta de investigación web (Investigar); no armes comparativas ni "precios aproximados" sin fuente.

<respuesta_html>
[Respuesta conversacional y directa]
</respuesta_html>"""
                _synthesis_model = GPT_5_4_MINI  # rápido y estable para respuestas conversacionales
            else:
                # Modo analítico: respuesta completa con HTML estructurado, Gemini (potente)
                reasoning_prompt = f"""Eres un experto analista. Responde siguiendo los ejemplos.

    PREGUNTA DEL USUARIO:
    {query}
    {history_context_block}
    INFORMACIÓN DISPONIBLE ({len(sources_content)} fuentes):
    {sources_text}

    {few_shot_examples}

    ⚠️ INSTRUCCIONES CRÍTICAS:
    1. Cada fuente puede incluir "📋 METADATA ESTRUCTURADA" con información en formato JSON
    2. La metadata contiene datos estructurados del documento (nombres, títulos, fechas, etc.)
    3. USA la información de la metadata para dar respuestas precisas y específicas
    4. NUNCA digas "no se especifica" o "no se proporciona" si el dato está en la metadata
    5. Combina la información de METADATA + CONTENIDO para respuestas completas

    INSTRUCCIONES:
    1. Analiza TODA la información disponible (metadata + contenido)
    2. USA referencias específicas de las fuentes
    3. USA los datos del usuario en tu análisis
    4. Estructura con HTML: <h4>, <p>, <ul>, <li>, <strong>
    5. SÉ ESPECÍFICO, nunca vago

    ⛔ GUARDRAIL ANTI-FABRICACIÓN (OBLIGATORIO):
    - Respondé SOLO con datos que provengan de la información disponible arriba (fuentes/base de datos) o de una herramienta real de esta sesión. NUNCA inventes ni estimes precios, productos, cifras, fechas, porcentajes ni fuentes que no aparezcan en esa información.
    - Si el usuario pide comparar con datos externos (mercado, competencia, precios de otros lugares/comercios, valores actuales) y no hay una fuente real disponible, DECILO explícitamente (por ejemplo: "no tengo datos de mercado ni de competencia en esta base") y ofrecé usar la herramienta de investigación web (Investigar) en lugar de inventar. NO armes tablas de "precios aproximados" ni comparativas sin fuente real.
    - Si sólo tenés un valor de la base, mostrá ese valor y aclará que no hay datos de comparación disponibles.

    {self._get_format_instructions(needs_ranking, needs_table, needs_legal_refs, all_referencias)}

    FORMATO DE RESPUESTA:

    <respuesta_html>
    [Tu respuesta en HTML bien formateado]
    </respuesta_html>

    COMIENZA:"""
                # fix4: GPT-5.1 en lugar de Gemini para síntesis analítica
                # Gemini: ~26-38s | GPT-5.1: ~5-7s | Calidad comparable para docs jurídicos
                _synthesis_model = GPT_5_1

            # ═══════════════════════════════════════════════════════════════
            # EJECUCIÓN — modelo seleccionado por complejidad de query
            # conversacional (≤15 palabras) → GPT-5-nano  (~3s)
            # analítico   (>15 palabras)    → Gemini 3.1 (~18s)
            # ═══════════════════════════════════════════════════════════════

            try:
                print(f"[SYNTHESIS v5.2] 🚀 Modelo: {'GPT-5.4-mini (conversacional)' if _is_conversational else 'GPT-5.1 (analítico)'}")

                if _is_conversational:
                    # Modo conversacional: usa run_model_with_default (GPT-5.4-mini)
                    deep_response = await asyncio.wait_for(
                        self.custom.run_model_with_default(reasoning_prompt, model=_synthesis_model),
                        timeout=60.0,
                    )
                else:
                    # Modo analítico: llamada directa con parámetros controlados
                    # reasoning_effort="low" + max_tokens=4000 → ~6-8s para queries normales
                    # Timeout 90s para queries complejas con muchos turnos de historial
                    _synthesis_client = openai.AsyncOpenAI(api_key=KEY_OPEN_AI)
                    _synthesis_resp = await asyncio.wait_for(
                        _synthesis_client.chat.completions.create(
                            model=_synthesis_model,
                            messages=[{"role": "user", "content": reasoning_prompt}],
                            max_completion_tokens=4000,
                            reasoning_effort="low",
                        ),
                        timeout=90.0,
                    )
                    deep_response = _synthesis_resp.choices[0].message.content or ""

                response_match = re.search(r'<respuesta_html>(.*?)</respuesta_html>', deep_response,
                                           re.DOTALL | re.IGNORECASE)

                if response_match:
                    synthesis_response = response_match.group(1).strip()
                else:
                    synthesis_response = deep_response.strip()

                if len(synthesis_response) < 100:
                    print(f"[SYNTHESIS v5.2] ⚠️ Respuesta corta, intentando fallback gpt-5.2...")
                    synthesis_response = await self._synthesis_fallback_gpt52(reasoning_prompt, all_results, query)

                synthesis_response = self._improve_html_spacing(synthesis_response)

                print(f"[SYNTHESIS v5.2] ✅ Completado: {len(synthesis_response)} chars")

            except TimeoutError:
                print(f"[SYNTHESIS v5.2] ⏱️ Timeout GPT-5.1 (90s), fallback → gpt-5.4-mini")
                synthesis_response = await self._synthesis_fallback_gpt52(reasoning_prompt, all_results, query)

            except Exception as e:
                print(f"[SYNTHESIS v5.2] ❌ Error GPT-5.1: {e} → fallback gpt-5.4-mini")
                synthesis_response = await self._synthesis_fallback_gpt52(reasoning_prompt, all_results, query)

            # ═══════════════════════════════════════════════════════════════
            # Assistant SUMMARY — GPT_5_NANO
            # ═══════════════════════════════════════════════════════════════

            brief_instructions = "Resume en 3-4 frases lo más importante."
            if needs_ranking:
                brief_instructions += " Menciona el TOP 1."

            brief_prompt = (
                f"Resume: {query}\n\n"
                f"RESPUESTA:\n{synthesis_response[:2000]}\n\n"
                f"{brief_instructions}\n\nRESUMEN:"
            )

            try:
                print(f"[SYNTHESIS v5.2] 📝 Assistant summary con gpt-5.4-mini...")
                brief_summary = await asyncio.wait_for(
                    self.custom.run_model_with_default(brief_prompt, model=GPT_5_4_MINI),
                    timeout=20.0
                )
                if not brief_summary or len(brief_summary) < 20:
                    brief_summary = "Consulta la respuesta detallada arriba."
            except:
                brief_summary = "Consulta la respuesta detallada arriba."

            # ═══════════════════════════════════════════════════════════════
            # HTML FINAL
            # ═══════════════════════════════════════════════════════════════

            response_html = self._build_final_response(
                query=query,
                synthesis=synthesis_response,
                brief_summary=brief_summary,
                # usar sources_content (refleja el fallback de recall) y no all_results,
                # que puede quedar vacío cuando se sintetiza del texto crudo.
                sources_count=(len(sources_content) or len(all_results)),
                query_type=query_type
            )

            total_elapsed = time.time() - synthesis_start
            print(f"[SYNTHESIS v5.2] ⏱️ Total: {total_elapsed:.2f}s\n")
            html = self._improve_html_spacing(response_html)
            return html

    async def _synthesis_fallback_gpt52(
            self,
            reasoning_prompt: str,
            all_results: List[Dict[str, Any]],
            query: str
    ) -> str:
        """Fallback de síntesis: gpt-5.4-mini rápido → _fallback_synthesis si también falla."""
        try:
            print(f"[SYNTHESIS v5.2] 🔄 Fallback ejecutando gpt-5.4-mini...")
            response = await asyncio.wait_for(
                self.custom.run_model_with_default(reasoning_prompt, model=GPT_5_4_MINI),
                timeout=60.0
            )
            match = re.search(r'<respuesta_html>(.*?)</respuesta_html>', response, re.DOTALL | re.IGNORECASE)
            result = match.group(1).strip() if match else response.strip()
            if len(result) < 100:
                raise ValueError("Respuesta fallback también demasiado corta")
            print(f"[SYNTHESIS v5.2] ✅ Fallback gpt-5.4-mini OK: {len(result)} chars")
            return result
        except Exception as e:
            print(f"[SYNTHESIS v5.2] ❌ Fallback gpt-5.4-mini falló: {e} → _fallback_synthesis")
            return self._fallback_synthesis(all_results, query)

    def _get_metadata_context_(self, chunk: Dict[str, Any]) -> str:
        """
        Extrae y formatea metadata_json para incluir en el contexto del LLM.
        ✅ GENÉRICO: Funciona para cualquier tipo de documento.
        """
        metadata_json = chunk.get('metadata_json', '')

        if not metadata_json:
            return ""

        try:
            if isinstance(metadata_json, str):
                metadata = json.loads(metadata_json)
            elif isinstance(metadata_json, dict):
                metadata = metadata_json
            else:
                return ""
        except json.JSONDecodeError:
            return ""

        # ═══════════════════════════════════════════════════════════════
        # EXCLUIR SOLO CAMPOS TÉCNICOS/INTERNOS
        # ═══════════════════════════════════════════════════════════════

        exclude_fields = {
            # Campos técnicos de sistema
            'file_id', 'chunk_id', 'chunk_index', 'chunk_length',
            'embedding', 'vector', 'text_hash', 'embedding_model',
            'schema_version', 'processing_time',
            # Campos de usuario/company (redundantes)
            'usuario', 'company', 'user',
            # Campos de archivo (ya están en otro lugar)
            'file_size', 'upload_ip', 'uploaded_by', 'uploaded_by_name',
            # Campos de procesamiento
            'fecha_procesamiento', 'longitud_texto', 'word_count',
            'has_numbers', 'digit_density', 'has_urls', 'has_emails',
            'content_type', 'language'
        }

        # Incluir todo lo que NO esté en exclude_fields y tenga valor
        relevant = {k: v for k, v in metadata.items()
                    if k not in exclude_fields and v}

        if not relevant:
            return ""

        return json.dumps(relevant, ensure_ascii=False, indent=2)

    def _improve_html_spacing(self, html: str) -> str:
        """✅ HTML ultra-compacto - mínimo espaciado"""

        # Remover TODO el whitespace innecesario
        html = re.sub(r'\s+', ' ', html)  # Todos los espacios → 1 espacio

        # Agregar saltos SOLO donde sean necesarios
        html = re.sub(r'</h4>', '</h4>\n', html)
        html = re.sub(r'</p>', '</p>\n', html)
        html = re.sub(r'</ul>', '</ul>\n', html)
        html = re.sub(r'</table>', '</table>\n', html)

        # Salto antes de cada medalla
        html = re.sub(r'([🥇🥈🥉])', r'\n\1', html)

        # Limpiar múltiples saltos
        html = re.sub(r'\n{2,}', '\n', html)

        return html.strip()

    def _get_format_instructions(self, needs_ranking: bool, needs_table: bool, needs_legal_refs: bool,
                                 all_referencias: set) -> str:
        """Genera instrucciones de formato específicas"""
        instructions = ""

        if needs_ranking:
            instructions += "\n⚠️ RANKING: Usa 🥇🥈🥉 con puntuaciones (X/100)"

        if needs_table:
            instructions += "\n⚠️ TABLA: Usa <table> HTML con datos específicos"

        if needs_legal_refs and all_referencias:
            refs_str = ', '.join(list(all_referencias)[:8])
            instructions += f"\n⚠️ REFERENCIAS DISPONIBLES: {refs_str}"
            instructions += "\nDEBES citar estas referencias específicamente"

        return instructions

    def _build_few_shot_examples(
            self,
            needs_legal_refs: bool,
            needs_ranking: bool,
            needs_table: bool,
            needs_steps: bool,
            all_referencias: set,
            query: str = "",
            user_data: str = ""
    ) -> str:
        """
        ✅ Construye ejemplos few-shot dinámicos + datos del usuario
        """
        if not (needs_legal_refs or needs_ranking or needs_table or needs_steps):
            return ""

        examples = f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║ DATOS DEL USUARIO (EXTRAÍDOS DE LA PREGUNTA)              ║
    ╚═══════════════════════════════════════════════════════════╝

    {user_data if user_data else "ℹ️ No se detectaron data específicos del user"}

    ⚠️ CRÍTICO: Si hay datos arriba, DEBES usarlos TODOS en tu análisis.
    Incluye: edad, cotización, códigos, categorías, cálculos, etc.
    Tu respuesta debe ser PERSONALIZADA con TODOS estos datos específicos.

    ╔═══════════════════════════════════════════════════════════╗
    ║ EJEMPLOS DE RESPUESTAS CORRECTAS (FEW-SHOT)              ║
    ╚═══════════════════════════════════════════════════════════╝
    """

        # ✅ EJEMPLO 1: Referencias
        if needs_legal_refs:
            ref_examples = list(all_referencias)[:3] if all_referencias else ["Artículo 27", "Real Decreto-ley 20/2018"]
            examples += f"""
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    📋 EJEMPLO 1: Cómo citar referencias (ESPECÍFICAS)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ❌ INCORRECTO (vago):
    "Según las fuentes, existe normativa..."
    "La documentación indica que..."
    "Fuentes 2, 3, 4 mencionan..."

    ✅ CORRECTO (específico):
    "El <strong>{ref_examples[0]}</strong> establece que...
    Adicionalmente, <strong>{ref_examples[1] if len(ref_examples) > 1 else 'Real Decreto-ley correspondiente'}</strong> 
    indica que..."

    ⚠️ SIEMPRE menciona el nombre COMPLETO y EXACTO de la referencia.

    """

        # ✅ EJEMPLO 2: Rankings CON ORDEN CORRECTO
        if needs_ranking:
            examples += """
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    🏆 EJEMPLO 2: Rankings personalizados (ORDEN CORRECTO)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ⚠️ CRÍTICO - ORDEN DEL RANKING:
    🥇 = MEJOR opción (puntuación más alta, cumple MEJOR el objetivo)
    🥈 = Segunda mejor
    🥉 = Tercera mejor

    La opción que MEJOR responde a la pregunta del usuario va PRIMERO con 🥇

    <h4>🥇 Opción C: Contrato de relevo (92/100)</h4>

    <p><strong>Marco de referencia:</strong></p>
    <ul>
      <li>Artículo 27 del IV Convenio Colectivo del Metal</li>
      <li>Disposición Transitoria Cuarta, apartado 6</li>
      <li>Real Decreto-ley 5/2013, artículo 8</li>
    </ul>

    <p><strong>Análisis personalizado para su situación:</strong></p>
    <p>Con <strong>64 años</strong>, <strong>Grupo 3 (oficial de primera)</strong>, 
    y <strong>36 años y 9 meses cotizados</strong>, <strong>le faltan 9 meses</strong> 
    para alcanzar los 37a6m requeridos por la Disposición Transitoria Cuarta 
    para jubilación obligatoria a 65 años.</p>

    <p>Su company <strong>CNAE 3030</strong> (fabricación aeronaves) está cubierta 
    por el apartado 6. Realizando <strong>tareas de montaje con esfuerzo físico 
    relevante</strong>, cumple requisitos del Artículo 27. Esta opción le permite 
    trabajar hasta los 67 años.</p>

    <p><strong>Desglose de puntuación (92/100):</strong></p>
    <ul>
      <li><strong>Permite trabajar hasta 67 años:</strong> 25/25 - Cumple objetivo principal</li>
      <li><strong>Se adapta a su situación:</strong> 23/25 - Considera todos sus datos</li>
      <li><strong>Estabilidad laboral:</strong> 22/25 - Contrato más estable</li>
      <li><strong>Aplicabilidad sector:</strong> 22/25 - CNAE 3030 cubierto</li>
    </ul>

    <h4>🥈 Opción A: Renovación eventual (78/100)</h4>

    <p><strong>Marco de referencia:</strong></p>
    <ul>
      <li>Artículo 15 del Estatuto de Trabajadores</li>
    </ul>

    <p><strong>Análisis personalizado:</strong></p>
    <p>Con sus <strong>36a9m cotizados (le faltan 9 meses para 37a6m)</strong>, 
    NO pueden obligarle a jubilarse a los 65. Sin embargo, el contrato eventual 
    ofrece menor estabilidad que el contrato de relevo.</p>

    <p><strong>Desglose (78/100):</strong></p>
    <ul>
      <li><strong>Flexibilidad:</strong> 20/25</li>
      <li><strong>Estabilidad:</strong> 18/25 - Eventual menos estable</li>
      <li><strong>Permite trabajar:</strong> 20/25</li>
      <li><strong>Condiciones:</strong> 20/25</li>
    </ul>

    ⚠️ Observa cómo:
    1. Se usan TODOS los datos del usuario (edad, cotización, CNAE, Grupo, etc.)
    2. Se CALCULAN valores derivados ("le faltan 9 meses")
    3. El ORDEN es correcto: mejor opción = 🥇
    4. Cada punto usa datos ESPECÍFICOS del usuario

    """

        # ✅ EJEMPLO 3: Tablas comparativas
        if needs_table:
            examples += """
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    📊 EJEMPLO 3: Tablas comparativas personalizadas
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    <table border="1">
      <thead>
        <tr>
          <th>Criterio</th>
          <th>Opción A</th>
          <th>Opción B</th>
          <th>Opción C</th>
          <th>Su Situación</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Referencia legal</strong></td>
          <td>Art. 15 ET</td>
          <td>RD 2720/1998</td>
          <td>Art. 27 Convenio</td>
          <td>-</td>
        </tr>
        <tr>
          <td><strong>Cotización requerida</strong></td>
          <td>No aplica</td>
          <td>37a6m</td>
          <td>No obligatoria</td>
          <td>36a9m ❌ (faltan 9m)</td>
        </tr>
        <tr>
          <td><strong>CNAE aplicable</strong></td>
          <td>Todos</td>
          <td>Todos</td>
          <td>CNAE 3030 ✅</td>
          <td>CNAE 3030 ✅</td>
        </tr>
        <tr>
          <td><strong>Edad máxima trabajo</strong></td>
          <td>66a2m</td>
          <td>65 años</td>
          <td>67 años ✅</td>
          <td>Objetivo: 67</td>
        </tr>
        <tr>
          <td><strong>Grupo profesional</strong></td>
          <td>Todos</td>
          <td>Todos</td>
          <td>Grupo 3 ✅</td>
          <td>Grupo 3 ✅</td>
        </tr>
      </tbody>
    </table>

    <p><strong>Interpretación para su caso específico:</strong></p>
    <p>Con <strong>64 años, Grupo 3, 36a9m cotizados, CNAE 3030</strong>, 
    la Opción C es la única que le permite alcanzar su objetivo de trabajar 
    hasta los 67 años.</p>

    ⚠️ Incluye columna "Su Situación" con los datos REALES del usuario.

    """

        # ✅ EJEMPLO 4: Procesos por pasos
        if needs_steps:
            examples += """
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    📝 EJEMPLO 4: Procesos personalizados
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    <h4>Paso 1: Verificar requisitos actuales</h4>
    <p>Según el <strong>Artículo 27 del IV Convenio</strong>, con sus datos específicos:</p>
    <ul>
      <li>✅ <strong>Edad:</strong> 64 años (cumple requisito >60 años)</li>
      <li>❌ <strong>Cotización:</strong> 36a9m (faltan 9 meses para 37a6m requeridos)</li>
      <li>✅ <strong>CNAE:</strong> 3030 (cubierto por Disposición Transitoria Cuarta)</li>
      <li>✅ <strong>Grupo profesional:</strong> 3 - Oficial de 1ª (elegible)</li>
      <li>⚠️ <strong>Esfuerzo físico:</strong> Relevante en montaje (favorece anticipación)</li>
    </ul>

    <h4>Paso 2: Acciones necesarias</h4>
    <p>Para su caso (<strong>64 años, Grupo 3, 36a9m, CNAE 3030</strong>), debe:</p>
    <ol>
      <li><strong>Completar cotización:</strong> Trabajar 9 meses más para alcanzar 37a6m (lo logrará a los 64 años y 9 meses)</li>
      <li><strong>Solicitar evaluación:</strong> Esfuerzo físico puede reducir edad según Art. 27</li>
      <li><strong>Documentación:</strong> Certificado vida laboral + Clasificación CNAE company</li>
    </ol>

    <h4>Paso 3: Plazos y proyección</h4>
    <ul>
      <li>📅 <strong>Hoy:</strong> 64 años con 36a9m</li>
      <li>📅 <strong>En 9 meses:</strong> 64a9m con 37a6m (cumple requisito jubilación 65)</li>
      <li>📅 <strong>A los 65:</strong> Puede optar por jubilación obligatoria o continuar</li>
      <li>📅 <strong>Hasta los 67:</strong> Posible con contrato de relevo</li>
    </ul>

    ⚠️ Personaliza CADA paso con datos ESPECÍFICOS del usuario.

    """

        examples += """
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ⚠️ REGLAS CRÍTICAS OBLIGATORIAS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    1. ✅ CITA referencias COMPLETAS y EXACTAS
       ❌ "Fuente 2", "las fuentes", "según normativa"
       ✅ "Artículo 27 del IV Convenio Colectivo del Metal"

    2. ✅ USA TODOS los datos del usuario extraídos arriba
       ❌ "Si cumple requisitos de cotización..."
       ✅ "Con 64 años, Grupo 3, y 36a9m cotizados..."

    3. ✅ INCLUYE cálculos y datos derivados
       ❌ "No cumple requisitos temporales"
       ✅ "Le faltan 9 meses para alcanzar 37a6m"

    4. ✅ ORDEN CORRECTO en rankings
       🥇 = MEJOR opción (mayor puntuación, cumple mejor el objetivo)
       🥈 = Segunda mejor
       🥉 = Tercera mejor

    5. ✅ ESTRUCTURA con HTML limpio
       Usa: <h4>, <p>, <ul>, <li>, <strong>, <table>
       Evita: Texto plano, formato inconsistente

    6. ✅ SÉ ESPECÍFICO, nunca vago
       ❌ "varios", "algunos", "múltiples", "diversas normativas"
       ✅ "3 opciones", "Artículo 27 y Disposición Cuarta", "9 meses"

    7. ✅ PERSONALIZA cada análisis
       Cada opción/criterio debe explicar cómo aplica a los datos ESPECÍFICOS del usuario

    """

        return examples
    async def _extract_user_data_from_query(self, query: str) -> str:
        """
        ✅ Extrae datos del usuario usando LLM con CÁLCULOS DETALLADOS
        Funciona para cualquier dominio: medicina, derecho, finanzas, ingeniería, etc.
        """

        extraction_prompt = f"""Analiza esta pregunta y extrae TODOS los datos específicos mencionados sobre la persona o situación.

    PREGUNTA:
    {query}

    INSTRUCCIONES CRÍTICAS:
    1. Identifica CUALQUIER dato cuantificable, medible o clasificable
    2. Normaliza valores numéricos (ej: "treinta y cinco" → "35")
    3. **CALCULA valores derivados útiles** (diferencias, plazos, meses faltantes, etc.)
    4. Interpreta abreviaturas técnicas del contexto
    5. Agrupa datos relacionados
    6. **INCLUYE TODOS los datos mencionados** (edad, categorías, códigos, antigüedad, etc.)

    EXTRAE (si existen):
    - Edad actual
    - Antigüedad/experiencia/cotización
    - Códigos identificadores (CNAE, grupos, categorías)
    - Clasificaciones profesionales
    - Medidas/métricas (peso, IMC, capital, etc.)
    - Estados/diagnósticos/situaciones
    - Fechas/plazos/duraciones

    FORMATO DE SALIDA:
    Texto descriptivo en español con emojis apropiados, organizado por categorías.

    REGLAS OBLIGATORIAS:
    - USA emojis contextuales (📅 edad, 💼 trabajo, 💰 dinero, 🏥 salud, 📊 métricas)
    - **CALCULA todo lo que sea útil** y márcalo con "📐 Calculado:"
    - **INCLUYE TODOS los datos del usuario sin omitir ninguno**
    - Si NO hay datos específicos: "ℹ️ No se mencionan datos específicos del usuario"

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    EJEMPLOS DE EXTRACCIÓN CORRECTA CON CÁLCULOS:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Entrada: "Tengo 64 años, 36 años y 9 meses cotizados, company CNAE 3030, soy oficial de 1ª Grupo 3, contrato eventual que termina en 3 meses, esfuerzo físico relevante en montaje"

    Salida:
    📅 Edad actual: 64 años
    💼 Cotización acumulada: 36 años y 9 meses (441 meses totales)
    🏢 Sector company: CNAE 3030 (Fabricación de aeronaves y naves espaciales)
    👔 Categoría profesional: Grupo 3 - Oficial de primera
    📄 Tipo de contrato: Eventual (vence en 3 meses)
    💪 Condiciones laborales: Esfuerzo físico relevante en tareas de montaje
    📐 Calculado: Requisito jubilación obligatoria 65 años = 37 años y 6 meses (450 meses)
    📐 Calculado: Le faltan 9 meses de cotización para alcanzar el requisito (450 - 441 = 9)
    📐 Calculado: Alcanzará 37a6m a los 64 años y 9 meses (en 9 meses)

    ---

    Entrada: "45 años, diabetes tipo 2 desde hace 5 años, última HbA1c 8.5%, peso 95 kg, mido 1.70m, tensión 145/92"

    Salida:
    📅 Edad: 45 años
    🩺 Diagnóstico: Diabetes tipo 2 (antigüedad: 5 años)
    💉 HbA1c: 8.5% ⚠️ Elevado (objetivo <7.0%, exceso: +1.5%)
    ⚖️ Peso: 95 kg
    📏 Talla: 1.70 m
    🩺 Tensión arterial: 145/92 mmHg ⚠️ Hipertensión estadio 1 (objetivo <140/90)
    📐 Calculado: IMC = 32.9 (Obesidad grado I, rango normal: 18.5-24.9)
    📐 Calculado: Peso objetivo para IMC normal: 70-75 kg (debe perder 20-25 kg)
    📐 Calculado: Reducción HbA1c necesaria: 1.5 puntos porcentuales

    ---

    Entrada: "Tengo 50.000€ ahorrados, perfil conservador, quiero invertir a largo plazo por 20 años, tengo 40 años"

    Salida:
    📅 Edad actual: 40 años
    💰 Capital disponible: 50,000€
    📊 Perfil de riesgo: Conservador (baja tolerancia a volatilidad)
    ⏳ Horizonte temporal: 20 años (largo plazo)
    📐 Calculado: Edad al vencimiento: 60 años
    📐 Calculado: Con retorno conservador 4% anual → Capital final estimado: ~109,500€

    ---

    Entrada: "Edificio 15 plantas, luces entre pilares 8 metros, carga de diseño 600 kg/m², viento 140 km/h, zona sísmica C"

    Salida:
    🏗️ Estructura: Edificio de 15 plantas
    📏 Luz de vano: 8 metros entre pilares
    ⚖️ Carga de diseño: 600 kg/m²
    💨 Viento de diseño: 140 km/h
    🌍 Zona sísmica: C (exposición alta)
    📐 Calculado: Altura aproximada edificio: 45-50 metros (15 plantas × 3m)
    📐 Calculado: Requiere diseño especial para viento >120 km/h

    ---

    Entrada: "¿Qué es la fotosíntesis?"

    Salida:
    ℹ️ No se mencionan datos específicos del usuario

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ⚠️ RECORDATORIO FINAL:
    - Extrae TODOS los datos mencionados sin omitir ninguno
    - Calcula SIEMPRE que sea útil (diferencias, faltantes, proyecciones)
    - Marca cálculos con "📐 Calculado:"
    - Usa emojis apropiados al dominio

    AHORA EXTRAE Y CALCULA LOS DATOS DE LA PREGUNTA:"""

        try:
            print(f"[EXTRACT] 🧠 Extrayendo datos del usuario con LLM...")

            response = await asyncio.wait_for(
                self.custom.run_model_with_default(
                    extraction_prompt,
                    model=GPT_5_1
                ),
                timeout=30.0
            )

            # Limpiar respuesta
            response_clean = response.strip()

            # Remover markdown si existe
            if response_clean.startswith("```"):
                lines = response_clean.split('\n')
                response_clean = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_clean

            response_clean = response_clean.strip()

            # Validar que hay contenido útil
            if not response_clean or len(response_clean) < 10:
                print(f"[EXTRACT] ⚠️ Respuesta vacía o muy corta")
                return "ℹ️ No se detectaron datos específicos del usuario"

            # Si dice que no hay datos, respetar
            if "no se mencionan datos" in response_clean.lower() or "no hay datos" in response_clean.lower():
                print(f"[EXTRACT] ℹ️ LLM reporta: sin datos específicos")
                return "ℹ️ No se mencionan datos específicos del usuario"

            print(f"[EXTRACT] ✅ Datos extraídos ({len(response_clean)} chars)")

            # Limitar longitud para no saturar el prompt (mantener razonable)
            if len(response_clean) > 700:
                response_clean = response_clean[:700] + "\n..."

            return response_clean

        except TimeoutError:
            print(f"[EXTRACT] ⏱️ Timeout en extracción (15s)")
            return "ℹ️ No se pudieron extraer datos (timeout)"

        except Exception as e:
            print(f"[EXTRACT] ❌ Error: {type(e).__name__}: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            return "ℹ️ No se pudieron extraer datos del usuario"

    @staticmethod
    def _md_to_html(text: str) -> str:
        """Convierte Markdown básico a HTML. El front renderiza HTML (no Markdown),
        así que sin esto la negrita ``**x**`` se ve con los asteriscos literales.
        Cubre negrita, títulos (#) y viñetas. Idempotente y defensivo: si ya es
        HTML o no hay nada que convertir, devuelve el texto tal cual."""
        if not text or not isinstance(text, str):
            return text or ""
        t = text
        # Negrita **x** -> <strong>x</strong> (lo que emite el LLM).
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t, flags=re.S)
        # Títulos Markdown (#, ##, ###…) al inicio de línea -> negrita en su línea.
        t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*(.+?)\s*$", r"<strong>\1</strong>", t)
        # Viñetas "- " / "* " al inicio de línea -> • (evita el guion/asterisco crudo).
        t = re.sub(r"(?m)^\s*[-*]\s+", "• ", t)
        return t

    def _build_final_response(
            self,
            query: str,
            synthesis: str,
            brief_summary: str,
            sources_count: int,
            query_type: str
    ) -> str:
        """
        Construye la respuesta HTML final con formato profesional.

        MEJORA v4.0: NO incluye la query en el HTML (evita repetición)
        """

        # UX: respuesta conversacional de Assistant como PRIMARIA (concisa) + el análisis
        # detallado colapsado tras un desplegable. Antes se mostraba el volcado largo
        # Y el resumen Assistant (duplicado); ahora Assistant es la respuesta y el detalle es
        # opcional. Si no hay resumen Assistant, el detallado pasa a ser la respuesta.
        summary = self._md_to_html((brief_summary or "").strip())
        detalle = self._md_to_html((synthesis or "").strip())
        base_badge = f"<p><em>Basada en {sources_count} documento(s).</em></p>"
        aviso = "<p><em>⚠️ IA - Verifica información crítica</em></p>"

        if summary:
            detalle_block = (
                f"""<details>
            <summary>Ver análisis detallado</summary>
            <div>{detalle}</div>
        </details>"""
                if detalle else ""
            )
            return f"""<div>
        <h2>💡 Assistant</h2>
        {base_badge}
        <p>{summary}</p>
        {detalle_block}
        {aviso}
    </div>"""

        # Sin resumen Assistant: mostrar el detallado como respuesta (no colapsado).
        return f"""<div>
        <h2>💡 Assistant</h2>
        {base_badge}
        <div>{detalle}</div>
        {aviso}
    </div>"""


    def _fallback_synthesis(
            self,
            all_results: List[Dict[str, Any]],
            query: str
    ) -> str:
        """
        Síntesis de respaldo cuando falla la síntesis principal.
        """

        responses_html = []
        for idx, result in enumerate(all_results, 1):
            response = result.get('response', '')
            if response and len(response) > 20:
                responses_html.append(f"""
    <div>
        <h4>Fuente {idx}</h4>
        <p>{response}</p>
    </div>
    """)

        return f"""<div>
        <h2>💡 Respuesta generada por IA</h2>
        <p><strong>{query}</strong></p>

        {''.join(responses_html)}

        <p><em>⚠️ IA - Verifica información crítica</em></p>
    </div>"""

    def _sources_from_raw_chunks(self, chunks: Optional[List[Any]], limit: int = 8) -> List[Dict[str, Any]]:
        """Recall fallback: construye sources_content a partir del TEXTO CRUDO de los
        chunks recuperados, para cuando la extracción map-reduce por-chunk quedó vacía
        (p.ej. el documento solo NOMBRA una entidad sin definirla). Así la síntesis
        responde con el contexto real disponible en vez de 'no encontré información'."""
        sources: List[Dict[str, Any]] = []
        for ch in (chunks or []):
            if len(sources) >= limit:
                break
            try:
                text = (self._extract_text(ch) or "").strip()
            except Exception:
                text = ""
            if len(text) < 20:
                continue
            try:
                meta = self._get_metadata_context_(ch) if isinstance(ch, dict) else ""
            except Exception:
                meta = ""
            sources.append({"index": len(sources) + 1, "content": text[:1500], "metadata": meta})
        return sources

    def _generate_no_info_response(self, query: str) -> str:
        """
        Genera respuesta cuando no se encuentra información.
        """

        return f"""<div>
        <h2>⚠️ Información no encontrada</h2>
        <p><strong>{query}</strong></p>

        <p>Lo siento, no pude encontrar información relevante en los documentos disponibles para responder tu consulta.</p>

        <p><strong>Sugerencias:</strong></p>
        <ul>
            <li>Reformula tu pregunta con otros términos</li>
            <li>Verifica que la información esté en los documentos cargados</li>
            <li>Intenta una consulta más específica o más general</li>
        </ul>

        <p><em>⚠️ Si crees que esta información debería estar disponible, contacta al administrador.</em></p>
    </div>"""

    async def _classify_query_intent(self, query: str, conversation_history: list) -> str:
        """
        Clasifica intención con GPT-5 Nano antes de decidir pipeline.

        Returns:
            "MEMORY"  → solo historial, no necesita Weaviate
            "BOTH"    → historial + documentos
            "DOCS"    → solo documentos (default RAG)
        """
        # Sin historial → siempre DOCS, no tiene sentido preguntar a Nano
        if not conversation_history:
            return "DOCS"

        # Muestra compacta del historial para no inflar el prompt
        history_preview = ""
        for turn in conversation_history[-6:]:
            role = "U" if turn['role'] == 'user' else "A"
            history_preview += f"[{role}]: {turn['content'][:150]}\n"

        system = (
            "Clasifica la intención de la query del usuario en UNA de estas tres categorías:\n"
            "MEMORY  → el usuario pregunta sobre el historial de esta misma conversación "
            "(ej: qué preguntas hice, de qué hablamos, qué respondiste, resumen de la charla)\n"
            "BOTH    → el usuario hace referencia al historial Y necesita información de documentos "
            "(ej: lo que dijiste sobre X, antes mencionaste Y, amplía lo que explicaste)\n"
            "DOCS    → el usuario consulta información de documentos sin referencia al historial\n\n"
            "Responde ÚNICAMENTE con una de estas palabras: MEMORY, BOTH, DOCS"
        )

        user_prompt = (
            f"HISTORIAL RECIENTE:\n{history_preview}\n\n"
            f"QUERY: {query}\n\n"
            f"Clasificación:"
        )

        try:
            client = openai.AsyncOpenAI(api_key=config.Constants.KEY_OPEN_AI)
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=GPT_5_4_MINI,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=200,
                    reasoning_effort="low",
                ),
                timeout=16.0
            )
            result = response.choices[0].message.content.strip().upper()
            if result not in ("MEMORY", "BOTH", "DOCS"):
                return "DOCS"
            return result

        except Exception as e:
            print(f"[ROUTER] ⚠️ Clasificador falló, default DOCS: {e}")
            return "DOCS"

    async def _handle_memory_query(
            self,
            query: str,
            conversation_history: list,
            user: str,
            company: str,
            tone: str,
            personality: str,
    ) -> str:
        """
        Responde queries sobre el historial de conversación sin tocar Weaviate.
        Llama a GPT-5.1 con el historial como contexto.
        """
        if not conversation_history:
            return (
                "<div><p>No tengo historial de conversación previo en esta sesión. "
                "Es la primera interacción registrada.</p></div>"
            )

        # Formatear historial para el prompt
        history_text = ""
        for i, turn in enumerate(conversation_history, 1):
            role_label = "Usuario" if turn['role'] == 'user' else "Assistant"
            content_preview = turn['content'][:500]
            history_text += f"[{i}] {role_label}: {content_preview}\n\n"

        system_prompt = (
            f"Eres {personality}. "
            f"El usuario te pregunta sobre el historial de esta conversación. "
            f"Responde en HTML. Sé preciso y directo. "
            f"NO inventes preguntas ni respuestas que no estén en el historial. "
            f"Si el historial está vacío o no contiene lo que busca, dilo claramente."
        )

        user_prompt = (
            f"HISTORIAL DE CONVERSACIÓN:\n\n{history_text}\n\n"
            f"PREGUNTA DEL USUARIO: {query}\n\n"
            f"Responde basándote ÚNICAMENTE en el historial mostrado arriba."
        )

        try:
            client = openai.AsyncOpenAI(api_key=config.Constants.KEY_OPEN_AI)
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=GPT_5_1,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=1500,
                    reasoning_effort='low'
                ),
                timeout=20.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[MEMORY_HANDLER] ⚠️ Error: {e}")
            # Fallback: respuesta estructurada manual
            turns_user = [t for t in conversation_history if t['role'] == 'user']
            items = "".join(f"<li>{t['content'][:200]}</li>" for t in turns_user)
            return (
                f"<div><h3>Preguntas registradas en esta sesión</h3>"
                f"<ul>{items}</ul></div>"
            )

    """
    chattingBotStream - VERSIÓN CON VALIDACIÓN DE DOCUMENTOS VACÍOS
    =================================================================

    Cambios implementados:
    - ✅ Validación cuando initial_docs está vacío
    - ✅ Mensaje de error claro al usuario
    - ✅ Sugerencias de diagnóstico
    - ✅ Manejo de errores simplificado
    """

    import re
    import os
    import time
    import uuid
    import json
    import asyncio
    import datetime

    import openai
    import anthropic
    from anthropic import AsyncAnthropic
    from typing import AsyncGenerator, Dict, Any

    # ─── Model constants ────────────────────────────────────────────────────────
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_1 = "gpt-5-1"

    # ─── Correct Anthropic model strings ────────────────────────────────────────
    CLAUDE_OPUS = "claude-opus-4-6"
    CLAUDE_SONNET = "claude-sonnet-4-6"
    CLAUDE_HAIKU = "claude-haiku-4-5-20251001"

    _GPT_TO_CLAUDE = {
        "gpt-4": CLAUDE_OPUS,
        "gpt-4-turbo": CLAUDE_OPUS,
        "gpt-5.6": CLAUDE_OPUS,
        "gpt-5.6": CLAUDE_SONNET,
        "gpt-3.5-turbo": CLAUDE_HAIKU,
    }


    # ────────────────────────────────────────────────────────────────────────────

    async def globalSearchStream(
            self,
            query: str,
            user: str,
            company: str,
            conversation_id: str = None,
            tone: str = "formal y preciso",
            prompt_html: str = None,
            prompt_clean: str = None,
            timestamp=None,
            model: str = "gpt-5-nano",
            personality: str = "asistente legal especializado, meticuloso y riguroso",
            doc_type: str = "contratos",
            config_chatbot: str = "documental",
            websearch: bool = False,
            context: str = None,
            llm_direct: bool = False,
            save_interaction: bool = True,
            date_from: float = None,   # Unix timestamp — filtro por indexed_at
            date_to: float = None,   # Unix timestamp — filtro por indexed_at
    ) -> AsyncGenerator[str, None]:
        """
        Buscador global sobre GENAI_BUSQUEDA_GLOBAL.
        Firma idéntica a chattingBotStream para compatibilidad con el caller.
        Usa el mismo pipeline RAG estándar pero apunta a la colección global.
        Sin router MEMORY/BOTH, sin decomposer, sin websearch/llm_direct.
        Siempre filtra por usuario + company internamente via _create_executors.
        """

        COLLECTION = "GENAI_BUSQUEDA_GLOBAL"

        start_time_total = time.time()
        conversation_id = conversation_id or str(uuid.uuid4())
        request_id = f"req_{uuid.uuid4().hex[:12]}"

        # ── helper emit — idéntico a chattingBotStream ────────────────────
        def emit(event_type: str, data: Dict[str, Any]) -> str:
            payload = {"type": event_type, "timestamp": time.time()}

            if event_type == "status":
                payload["message"] = data.get("message", "")
                if "phase" in data: payload["phase"] = data["phase"]
                if "details" in data: payload["details"] = data["details"]
                if "steps" in data: payload["steps"] = data["steps"]

            elif event_type == "chunk":
                payload["type"] = "content"
                payload["message"] = data.get("message", data.get("text", ""))
                if "step" in data: payload["step"] = data["step"]

            elif event_type == "complete":
                payload["message"] = data.get("message", "")
                time_elapsed = data.get("time", 0)
                steps = data.get("steps", 0)
                if not payload["message"]:
                    payload["message"] = (
                        f"✅ Completado en {time_elapsed}s ({steps} pasos)"
                        if steps > 0 else f"✅ Completado en {time_elapsed}s"
                    )
                payload["time"] = time_elapsed
                if steps > 0: payload["steps"] = steps

            elif event_type == "metadata":
                payload["message"] = data.get("message", "Metadata")
                payload.update({k: v for k, v in data.items() if k != "message"})

            elif event_type == "error":
                payload["message"] = data.get("message", "Error")
                payload.update({k: v for k, v in data.items() if k != "message"})

            else:
                payload["message"] = data.get("message", str(data))
                payload.update({k: v for k, v in data.items() if k != "message"})

            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        # ── helper guardar interacción ────────────────────────────────────
        # Lo que se GUARDA/MUESTRA como pregunta = texto ORIGINAL del usuario
        # (prompt_clean trae lo que tipeó, sin la reescritura de correferencia).
        # La query reescrita se usa SOLO para retrieval/síntesis, no para mostrar.
        query_to_save = (prompt_clean or "").strip() or (query or "").strip()

        async def _save(user_msg: str, bot_msg: str) -> None:
            if not save_interaction:
                return
            try:
                await self.custom.remember_interaction(
                    user=user,
                    user_message=(query_to_save or user_msg),
                    bot_response=bot_msg,
                    conversation_id=conversation_id,
                    timestamp=timestamp or datetime.datetime.utcnow(),
                    config_chatbot=config_chatbot
                )
            except Exception as _e:
                print(f"[SAVE] ⚠️ Error guardando interacción: {_e}")



        try:
            # ═══════════════════════════════════════════════════════════
            # FASE 0: VALIDACIÓN
            # ═══════════════════════════════════════════════════════════

            yield emit("status", {"message": "🔍 Validando consulta...", "phase": 0})

            query = (query or "").strip()
            if not query:
                raise ValueError("La consulta está vacía")

            query_original_user = query
            query_for_retrieval = query

            yield emit("status", {"message": f"✅ Query válida ({len(query)} caracteres)", "phase": 0})

            # ═══════════════════════════════════════════════════════════
            # HISTORIAL + COREFERENCE
            # ═══════════════════════════════════════════════════════════

            conversation_history = []
            try:
                raw_history = await self.custom.get_conversation_history(
                    conversation_id=conversation_id, limit=10
                )
                if raw_history:
                    for turn in raw_history:
                        if isinstance(turn, dict):
                            role = turn.get("role") or ("user" if turn.get("user_message") else "assistant")
                            content = turn.get("content") or turn.get("user_message") or turn.get("bot_response", "")
                            if content:
                                conversation_history.append({"role": role, "content": content})
                        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
                            conversation_history.append({"role": turn[0], "content": turn[1]})

                if conversation_history:
                    yield emit("status", {
                        "message": f"📜 Historial cargado: {len(conversation_history)} mensajes",
                        "phase": 0
                    })
            except Exception as hist_err:
                print(f"[HISTORY] ⚠️ No se pudo cargar historial: {hist_err}")
                conversation_history = []

            if conversation_history and _COLLECTIVE_SELECTION_RP.match(query):
                # Selección sobre el conjunto completo: NO reescribir (evita reanclar a
                # los candidatos recién mencionados y excluir al correcto). Bug Sevilla.
                print(f"[COREFERENCE] ↩️ omitida (selección de conjunto): '{query[:80]}'")
            elif conversation_history:
                try:
                    REWRITER_SYSTEM = (
                        "Dado el historial de conversación y la última pregunta del usuario, "
                        "reescribe la pregunta como una consulta autónoma que no necesite contexto previo. "
                        "Resuelve pronombres y referencias implícitas. "
                        "Si la pregunta ya es autónoma y clara, devuélvela tal cual. "
                        "NUNCA restrinjas una pregunta sobre un conjunto general ('los candidatos', "
                        "'las facturas') a las entidades ya mencionadas: es sobre TODO el conjunto. "
                        "Responde SOLO con la pregunta reescrita, sin explicaciones ni comillas."
                    )
                    rewrite_messages = conversation_history[-10:] + [{"role": "user", "content": query}]
                    rewrite_client = openai.AsyncOpenAI(api_key=config.Constants.KEY_OPEN_AI)

                    rewrite_response = await asyncio.wait_for(
                        rewrite_client.chat.completions.create(
                            model=MODEL_COREFERENCE,
                            messages=[
                                {"role": "system", "content": REWRITER_SYSTEM},
                                *rewrite_messages
                            ],
                            max_completion_tokens=600,
                        ),
                        timeout=12.0
                    )

                    rewritten = rewrite_response.choices[0].message.content.strip()
                    if rewritten and rewritten != query:
                        query_for_retrieval = rewritten
                        print(f"[COREFERENCE] ✏️ '{query}' → '{rewritten}'")
                        yield emit("status", {"message": "✏️ Query contextualizada", "phase": 0})

                except Exception as rewrite_err:
                    print(f"[COREFERENCE] ⚠️ Rewriter falló: {rewrite_err}")
                    query_for_retrieval = query

            timestamp = timestamp or datetime.datetime.utcnow()

            # ═══════════════════════════════════════════════════════════
            # FASE 1: CONFIGURACIÓN — colección global fija
            # ═══════════════════════════════════════════════════════════

            yield emit("status", {"message": "⚙️ Configurando búsqueda global...", "phase": 1})

            try:
                if not model:
                    model = MODEL_DECOMPOSER

                validation_task = await self.custom.exists_weaviate_class(COLLECTION)
                if asyncio.iscoroutine(validation_task) or asyncio.isfuture(validation_task):
                    class_exists = await validation_task
                else:
                    class_exists = validation_task

                if not class_exists:
                    raise RuntimeError(f"No se encontró la colección global '{COLLECTION}'")

                yield emit("status", {"message": f"✅ Colección {COLLECTION} verificada", "phase": 1})

                simple_model_executor, vector_search_func = self._create_executors(
                    model, None, False, user, company, COLLECTION
                )
                model_executor = self._create_main_executor(model, None, False, tone, personality)

                yield emit("status", {"message": "✅ Sistema RAG configurado", "phase": 1})

            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"Error en configuración: {str(e)}")

            # ═══════════════════════════════════════════════════════════
            # FASE 2: GENERACIÓN DE QUERIES EXHAUSTIVAS
            # LLM genera N variaciones semánticamente distintas para
            # maximizar cobertura del corpus — cada una cubre ángulos
            # distintos del espacio vectorial
            # ═══════════════════════════════════════════════════════════

            yield emit("status", {"message": "🧠 Generando queries exhaustivas...", "phase": 2})

            try:
                # Variantes proporcionales a complejidad:
                # query corta (<= 6 palabras) → 3 variantes (rápido)
                # query larga (> 6 palabras)  → 8 variantes (cobertura máxima)
                _word_count = len(query_for_retrieval.split())
                _n_variants = 3 if _word_count <= 6 else 8

                QUERY_GEN_SYSTEM = (
                    "Eres un experto en recuperación de información. "
                    f"Dado un pedido del usuario, generá exactamente {_n_variants} queries de búsqueda "
                    "semánticamente DISTINTAS que juntas cubran el máximo espacio posible del corpus. "
                    "Incluí: la query original, variantes con sinónimos, queries genéricas de cobertura, "
                    "queries por tipo de documento, queries por contenido probable. "
                    f"Respondé SOLO con un JSON array de {_n_variants} strings, sin explicaciones ni backticks. "
                    "Ejemplo: [query1, query2, ...]"
                )
                qgen_client = openai.AsyncOpenAI(api_key=config.Constants.KEY_OPEN_AI)
                qgen_response = await asyncio.wait_for(
                    qgen_client.chat.completions.create(
                        model=MODEL_DECOMPOSER,
                        messages=[
                            {"role": "system", "content": QUERY_GEN_SYSTEM},
                            {"role": "user", "content": f"Pedido del usuario: {query_for_retrieval}"}
                        ],
                        max_completion_tokens=1000 if _n_variants <= 3 else 4000,
                    ),
                    timeout=10.0 if _n_variants <= 3 else 15.0
                )

                raw = qgen_response.choices[0].message.content.strip()
                if not raw or not raw.strip():
                    raise ValueError("empty")
                raw = raw.strip()

                match = re.search(r'\[.*\]', raw, re.DOTALL)
                if match:
                    raw = match.group()

                query_variants = json.loads(raw)
                if not isinstance(query_variants, list):
                    raise ValueError("No es lista")
                # Asegurar que la original esté primera, limitar a _n_variants
                query_variants = [query_for_retrieval] + [q for q in query_variants if q != query_for_retrieval]
                query_variants = query_variants[:_n_variants]
            except Exception as qgen_err:
                print(f"[QUERY_GEN] ⚠️ Falló generación de queries: {qgen_err}")
                # Fallback: queries manuales de cobertura
                query_variants = [
                    query_for_retrieval,
                    f"documento archivo información",
                    f"contenido datos registros",
                    f"texto descripción detalle",
                ]

            yield emit("status", {
                "message": f"✅ {len(query_variants)} queries generadas",
                "phase": 2
            })

            # ═══════════════════════════════════════════════════════════
            # FASE 3: BÚSQUEDA EXHAUSTIVA EN PARALELO
            # Corre todas las queries simultáneamente y mergea resultados
            # ═══════════════════════════════════════════════════════════

            yield emit("status", {"message": "🔍 Búsqueda exhaustiva en paralelo...", "phase": 3})

            try:
                # check aggregation separado para no mezclar con búsquedas
                requires_aggregation = False
                try:
                    requires_aggregation = await asyncio.wait_for(
                        self._check_aggregation_needed(query_for_retrieval, simple_model_executor),
                        timeout=8.0
                    )
                except Exception:
                    pass

                # Correr queries en batches de 3 para no sobrecargar el event loop
                BATCH_SIZE = 3
                search_results = []

                for batch_start in range(0, len(query_variants), BATCH_SIZE):
                    batch = query_variants[batch_start:batch_start + BATCH_SIZE]
                    batch_tasks = [
                        self._do_multi_query_search(q, vector_search_func, simple_model_executor, None)
                        for q in batch
                    ]
                    try:
                        batch_results = await asyncio.wait_for(
                            asyncio.gather(*batch_tasks, return_exceptions=True),
                            timeout=30.0
                        )
                        search_results.extend(batch_results)
                    except TimeoutError:
                        print(f"[EXHAUSTIVE] ⚠️ Timeout en batch {batch_start // BATCH_SIZE + 1}, continuando...")
                    except asyncio.CancelledError:
                        print(
                            f"[EXHAUSTIVE] ⚠️ CancelledError en batch {batch_start // BATCH_SIZE + 1}, continuando...")
                    except Exception as batch_err:
                        print(f"[EXHAUSTIVE] ⚠️ Error en batch: {batch_err}, continuando...")

                # Merge deduplicado — orden preserva relevancia (primera query = más relevante)
                # SearchResult es dataclass — acceso por atributo, no por .get()
                seen_ids = set()
                initial_docs = []

                for i, result in enumerate(search_results):
                    if isinstance(result, Exception):
                        print(f"[EXHAUSTIVE] ⚠️ Query {i} falló: {result}")
                        continue
                    docs = result if isinstance(result, list) else []
                    for doc in docs:
                        # SearchResult dataclass: usar atributos directos
                        if hasattr(doc, 'filename') or hasattr(doc, 'file_name'):
                            uid = (getattr(doc, 'filename', '') or
                                   getattr(doc, 'file_name', '') or
                                   getattr(doc, 'chunk_id', ''))
                        else:
                            # fallback si es dict
                            uid = doc.get("file_id") or doc.get("filename") or doc.get("_id", "")
                        if uid and uid not in seen_ids:
                            seen_ids.add(uid)
                            initial_docs.append(doc)
                        elif not uid:
                            initial_docs.append(doc)

                # ── Filtro por fecha (indexed_at) ──────────────────────────
                if date_from or date_to:
                    before = len(initial_docs)
                    filtered = []
                    for _d in initial_docs:
                        _ia = (getattr(_d, "indexed_at", None) or
                               (_d.get("indexed_at") if isinstance(_d, dict) else None))
                        if _ia is None:
                            filtered.append(_d)  # sin indexed_at → no filtrar
                            continue
                        try:
                            _ia = float(_ia)
                            if date_from and _ia < date_from:
                                continue
                            if date_to and _ia > date_to:
                                continue
                            filtered.append(_d)
                        except (TypeError, ValueError):
                            filtered.append(_d)
                    initial_docs = filtered
                    if before != len(initial_docs):
                        yield emit("status", {
                            "message": f"📅 Filtro fecha: {len(initial_docs)}/{before} docs en rango",
                            "phase": 3
                        })

                doc_count = len(initial_docs)

                if doc_count == 0:
                    await _save(query_original_user, "")
                    yield emit("error", {
                        "code": "NO_DOCUMENTS_FOUND",
                        "message": "No se encontraron documentos en la colección global",
                        "details": {
                            "collection": COLLECTION,
                            "query_preview": query_for_retrieval[:100],
                            "usuario": user,
                            "company": company,
                        },
                        "suggestions": [
                            "Verificá que los documentos estén indexados en la colección global",
                            f"Consulta: curl \"http://localhost:8087/v1/objects?class={COLLECTION}&limit=1\""
                        ]
                    })
                    return

                yield emit("status", {
                    "message": f"✅ {doc_count} docs únicos de {len(query_variants)} queries exhaustivas",
                    "phase": 3
                })

                # ── Normalización de tipo_doc por filename ─────────────────
                # Weaviate puede tener tipo_doc mal clasificado en la indexación.
                # Se corrige heurísticamente por nombre de archivo antes del rerank.
                _TIPO_DOC_RULES = [
                    (["factura", "invoice", "fac_", "_fac"], "factura"),
                    (["receta", "recipe", "rx_", "_rx"], "receta"),
                    (["contrato", "contract", "acuerdo"], "contrato"),
                    (["nomina", "nomina", "payroll", "sueldo"], "nomina"),
                    (["cv", "curriculum", "curriculo", "resume",
                      "vitae", "candidato"], "cv"),
                    (["boleto", "ticket", "billet"], "boleto"),
                    (["csv", ".csv"], "dataset"),
                ]

                corrected = 0
                for doc in initial_docs:
                    fname = (getattr(doc, "filename", "") or getattr(doc, "file_name", "")).lower()
                    if not fname:
                        continue
                    current_tipo = getattr(doc, "tipo_doc", "")
                    inferred = current_tipo
                    for keywords, tipo in _TIPO_DOC_RULES:
                        if any(kw in fname for kw in keywords):
                            inferred = tipo
                            break
                    if inferred != current_tipo:
                        try:
                            doc.doc_type = inferred
                            corrected += 1
                        except Exception:
                            pass

                if corrected > 0:
                    yield emit("status", {
                        "message": f"🏷️ {corrected} tipo_doc corregidos por filename",
                        "phase": 3
                    })

            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"Error en búsqueda exhaustiva: {str(e)}")

            query_type = "UNKNOWN"
            num_partes = 0
            if isinstance(initial_docs, list) and initial_docs:
                first_doc = initial_docs[0]
                if isinstance(first_doc, dict):
                    query_type = first_doc.get("query_type", "UNKNOWN")
                    num_partes = first_doc.get("num_partes", 0)

            # ═══════════════════════════════════════════════════════════
            # FASE 4: RERANK
            # ═══════════════════════════════════════════════════════════

            yield emit("status", {"message": "🎯 Ordenando por relevancia...", "phase": 4})

            try:
                relevant_docs = await asyncio.wait_for(
                    self._rerank_and_diversify(query_for_retrieval, initial_docs), timeout=5.0
                )
                yield emit("status", {"message": f"✅ {len(relevant_docs)} docs seleccionados", "phase": 4})
            except TimeoutError:
                relevant_docs = initial_docs[:10]
                yield emit("status", {"message": "⏱️ Timeout rerank - usando top 10", "phase": 4})
            except Exception:
                relevant_docs = initial_docs[:10]
                yield emit("status", {"message": "⚠️ Error rerank - usando top 10", "phase": 4})

            # ── Emitir fuentes para que el endpoint las exponga al cliente ──
            try:
                _sources = []
                _seen_src = set()
                for _doc in relevant_docs:
                    _fname = (getattr(_doc, "filename", "") or
                              getattr(_doc, "file_name", "") or
                              (_doc.get("filename", "") if isinstance(_doc, dict) else ""))
                    _score = (getattr(_doc, "certainty", None) or
                              getattr(_doc, "score", None) or
                              (_doc.get("certainty", _doc.get("score", 0.0)) if isinstance(_doc, dict) else 0.0))
                    _preview = (getattr(_doc, "text", "") or
                                (_doc.get("text", "") if isinstance(_doc, dict) else ""))
                    if _fname and _fname not in _seen_src:
                        _seen_src.add(_fname)
                        _sources.append({
                            "filename": _fname,
                            "score": round(float(_score or 0), 4),
                            "fragment": (_preview[:200] + "...") if len(_preview) > 200 else _preview,
                        })
                if _sources:
                    yield emit("sources", {"sources": _sources[:10]})
            except Exception:
                pass

            # ═══════════════════════════════════════════════════════════
            # FASE 5: PROCESAMIENTO DE CHUNKS
            # ═══════════════════════════════════════════════════════════

            yield emit("status", {"message": "📝 Procesando documentos...", "phase": 4})

            try:
                processed_chunks = self._process_documents(relevant_docs, query_for_retrieval, "")
                processed_chunks = self._consolidate_group_chunks(processed_chunks)
            except Exception:
                processed_chunks = relevant_docs

            CHUNKS_CONFIG = {"simple": 30, "multi": 15, "default": 30}
            is_simple_query = query_type in ("SIMPLE_EXACTA", "SIMPLE_CONCEPTUAL") and num_partes <= 2
            use_fast_model = is_simple_query or (query_type == "MULTI_PARTE" and num_partes <= 3)

            if is_simple_query:
                max_chunks = CHUNKS_CONFIG["simple"]
            elif query_type == "MULTI_PARTE":
                max_chunks = CHUNKS_CONFIG["multi"]
            else:
                max_chunks = min(len(processed_chunks), CHUNKS_CONFIG["default"])

            processed_chunks = processed_chunks[:max_chunks]
            chunk_executor = simple_model_executor if use_fast_model else model_executor

            yield emit("status", {"message": f"⚙️ Procesando {len(processed_chunks)} chunks...", "phase": 5})

            try:
                all_results = await self._process_chunks_in_batches(
                    processed_chunks, query_for_retrieval, chunk_executor
                )
                yield emit("status", {"message": f"✅ {len(all_results)} chunks procesados", "phase": 5})
            except Exception as e:
                raise RuntimeError(f"Error procesando chunks: {str(e)}")

            # ═══════════════════════════════════════════════════════════
            # FASE 6: SÍNTESIS
            # ═══════════════════════════════════════════════════════════

            yield emit("status", {"message": "🤖 Generando respuesta final...", "phase": 6})

            response_text = None
            try:
                response_text = await asyncio.wait_for(
                    self.synthesize_answers(
                        all_results, query_original_user, model_executor, simple_model_executor,
                        query_type, requires_aggregation, user, company,
                        conversation_id, config_chatbot,
                        conversation_history=conversation_history,
                        fallback_chunks=processed_chunks,
                    ),
                    timeout=60.0
                )

                for i in range(0, len(response_text), 100):
                    yield emit("chunk", {"message": response_text[i:i + 100]})
                    await asyncio.sleep(0.01)

            except Exception as e:
                yield emit("status", {"message": f"⚠️ Error en síntesis: {str(e)[:100]}", "phase": 6})
                response_text = (
                    f"<div><h2>Resultados encontrados</h2>"
                    f"<p><strong>Consulta:</strong> {query_original_user}</p>"
                    f"<p><em>⚠️ Se encontraron {len(all_results)} fragmentos relevantes "
                    f"pero ocurrió un error en la síntesis final.</em></p></div>"
                )
                for i in range(0, len(response_text), 100):
                    yield emit("chunk", {"message": response_text[i:i + 100]})
                    await asyncio.sleep(0.01)

            await _save(query_original_user, response_text or "")

            elapsed_total = time.time() - start_time_total
            yield emit("complete", {"time": round(elapsed_total, 2)})

        # ═══════════════════════════════════════════════════════════════
        # MANEJO DE ERRORES — idéntico a chattingBotStream
        # ═══════════════════════════════════════════════════════════════

        except Exception as e:
            elapsed = time.time() - start_time_total


            error_str = str(e).lower()
            error_type_name = type(e).__name__

            if isinstance(e, TimeoutError) or "timeout" in error_str:
                error_code = "TIMEOUT"
                error_message = "La operación excedió el tiempo límite permitido"
            elif isinstance(e, ValueError) or "validation" in error_type_name.lower():
                error_code = "VALIDATION_ERROR"
                error_message = getattr(e, "message", str(e))
            elif any(x in error_str for x in ["openai", "anthropic", "claude", "llm", "model"]):
                error_code = "LLM_ERROR"
                error_message = f"Error en el modelo de lenguaje: {str(e)}"
            elif any(x in error_str for x in ["weaviate", "vector"]):
                error_code = "VECTOR_DB_ERROR"
                error_message = f"Error en base de datos vectorial: {str(e)}"
            elif any(x in error_str for x in ["database", "mongo", "postgres", "mysql"]):
                error_code = "DATABASE_ERROR"
                error_message = f"Error en base de datos: {str(e)}"
            elif any(x in error_str for x in ["connection", "network", "socket"]):
                error_code = "CONNECTION_ERROR"
                error_message = f"Error de conexión: {str(e)}"
            elif any(x in error_str for x in ["permission", "unauthorized", "forbidden"]):
                error_code = "PERMISSION_ERROR"
                error_message = f"Error de permisos: {str(e)}"
            else:
                error_code = "UNEXPECTED_ERROR"
                error_message = f"Error inesperado: {str(e)}"

            logger.error(
                f"[{error_code}] {error_message}",
                extra={
                    "request_id": request_id,
                    "usuario": user,
                    "company": company,
                    "collection": COLLECTION,
                    "elapsed_ms": round(elapsed * 1000, 2),
                    "error_type": error_type_name
                }
            )

            error_details = {
                "code": error_code,
                "message": error_message,
                "type": error_type_name,
                "timestamp": time.time(),
                "request_id": request_id,
                "elapsed_ms": round(elapsed * 1000, 2)
            }

            if os.getenv("ENVIRONMENT", "production") != "production":
                error_details["debug"] = {
                    "usuario": user,
                    "company": company,
                    "collection": COLLECTION,
                    "query_preview": query[:100],
                    "traceback": ""
                }

            yield emit("error", error_details)

    async def chattingBotStream(
            self,
            query: str,
            user: str,
            company: str,
            conversation_id: str = None,
            tone: str = "formal y preciso",
            prompt_html: str = None,
            prompt_clean: str = None,
            timestamp=None,
            model: str = "gpt-5-nano",
            personality="asistente legal especializado, meticuloso y riguroso",
            doc_type: str = "contratos",
            config_chatbot="documental",
            websearch: bool = False,
            context: str = None,
            llm_direct: bool = False,
            save_interaction=True
    ) -> AsyncGenerator[str, None]:
        """
        Pipeline inteligente con 4 modos:
        1. websearch=True  → Búsqueda en web
        2. llm_direct=True   → Claude directo (sin RAG)
        3. MEMORY           → historial de conversación (sin Weaviate)
        4. DOCS / BOTH      → Pipeline RAG completo

        HISTORIAL:   carga los últimos 10 mensajes del conversation_id desde PG.
        COREFERENCE: reescribe la query con contexto previo antes del retrieval.
        ROUTER:      clasifica intención (MEMORY/BOTH/DOCS) con GPT-5 Nano.
        """

        start_time_total = time.time()
        conversation_id = conversation_id or str(uuid.uuid4())
        request_id = f"req_{uuid.uuid4().hex[:12]}"

        # ── helper ────────────────────────────────────────────────────────────
        def emit(event_type: str, data: Dict[str, Any]) -> str:
            payload = {"type": event_type, "timestamp": time.time()}

            if event_type == "status":
                payload["message"] = data.get("message", "")
                if "phase" in data: payload["phase"] = data["phase"]
                if "details" in data: payload["details"] = data["details"]
                if "steps" in data: payload["steps"] = data["steps"]

            elif event_type == "chunk":
                payload["type"] = "content"
                payload["message"] = data.get("message", data.get("text", ""))
                if "step" in data: payload["step"] = data["step"]

            elif event_type == "complete":
                payload["message"] = data.get("message", "")
                time_elapsed = data.get("time", 0)
                steps = data.get("steps", 0)
                if not payload["message"]:
                    payload["message"] = (
                        f"✅ Completado en {time_elapsed}s ({steps} pasos)"
                        if steps > 0 else f"✅ Completado en {time_elapsed}s"
                    )
                payload["time"] = time_elapsed
                if steps > 0: payload["steps"] = steps

            elif event_type == "metadata":
                payload["message"] = data.get("message", "Metadata")
                payload.update({k: v for k, v in data.items() if k != "message"})

            elif event_type == "error":
                payload["message"] = data.get("message", "Error")
                payload.update({k: v for k, v in data.items() if k != "message"})

            else:
                payload["message"] = data.get("message", str(data))
                payload.update({k: v for k, v in data.items() if k != "message"})

            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        # ── helper: guardar interacción con timestamp siempre válido ──────────
        # Lo que se GUARDA/MUESTRA como pregunta = texto ORIGINAL del usuario
        # (prompt_clean trae lo que tipeó, sin la reescritura de correferencia).
        # La query reescrita se usa SOLO para retrieval/síntesis y para la cache.
        query_to_save = (prompt_clean or "").strip() or (query or "").strip()

        async def _save(user_msg: str, bot_msg: str) -> None:
            """Wrapper centralizado para remember_interaction — nunca lanza."""
            if not save_interaction:
                return
            try:
                await self.custom.remember_interaction(
                    user=user,
                    user_message=(query_to_save or user_msg),
                    bot_response=bot_msg,
                    conversation_id=conversation_id,
                    timestamp=timestamp or datetime.datetime.utcnow(),  # FIX: nunca None
                    config_chatbot=config_chatbot
                )
                # Guardar en cache si hay respuesta y no es websearch
                # (cache keyed por la query de PROCESO, no por el texto a mostrar)
                if bot_msg and not websearch:
                    await _qcache.set(company, user_msg, bot_msg)
            except Exception as _e:
                print(f"[SAVE] ⚠️ Error guardando interacción: {_e}")

        try:
            # ═══════════════════════════════════════════════════════════
            # VALIDACIÓN INICIAL DE QUERY
            # ═══════════════════════════════════════════════════════════

            yield emit("status", {"message": "🔍 Validando consulta...", "phase": 0})

            query = (query or "").strip()
            if not query:
                raise ValueError("La consulta está vacía")

            yield emit("status", {"message": f"✅ Query válida ({len(query)} caracteres)", "phase": 0})

            # Asignar aquí para que estén disponibles antes de CHITCHAT y COREFERENCE
            query_original_user = query
            query_for_retrieval = query

            # ═══════════════════════════════════════════════════════════
            # CACHE CHECK — respuesta inmediata si ya procesamos esta query
            # Solo activo para consultas sin websearch (resultados estables)
            # ═══════════════════════════════════════════════════════════
            _qcache = _QueryCache(self.redis_client)
            if not websearch:
                _cached = await _qcache.get(company, query)
                if _cached:
                    yield emit("status", {"message": "⚡ Respuesta desde caché", "phase": 0})
                    yield emit("chunk", {"text": _cached})
                    yield emit("complete", {"message": "✅ Completado (caché)", "steps": 0})
                    await self.custom.remember_interaction(
                        conversation_id=conversation_id, user=user,
                        timestamp=timestamp or datetime.datetime.utcnow(),
                        config_chatbot=config_chatbot,
                    )
                    return

            # ═══════════════════════════════════════════════════════════
            # HISTORIAL DE CONVERSACIÓN
            # ═══════════════════════════════════════════════════════════

            conversation_history = []
            try:
                raw_history = await self.custom.get_conversation_history(
                    conversation_id=conversation_id, limit=10
                )
                if raw_history:
                    for turn in raw_history:
                        if isinstance(turn, dict):
                            role = turn.get("role") or ("user" if turn.get("user_message") else "assistant")
                            content = turn.get("content") or turn.get("user_message") or turn.get("bot_response", "")
                            if content:
                                conversation_history.append({"role": role, "content": content})
                        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
                            conversation_history.append({"role": turn[0], "content": turn[1]})

                if conversation_history:
                    yield emit("status", {
                        "message": f"📜 Historial cargado: {len(conversation_history)} mensajes",
                        "phase": 0
                    })
            except Exception as hist_err:
                print(f"[HISTORY] ⚠️ No se pudo cargar historial: {hist_err}")
                conversation_history = []

            # ═══════════════════════════════════════════════════════════
            # COREFERENCE RESOLUTION (query rewriting)
            # ═══════════════════════════════════════════════════════════

            if conversation_history and _COLLECTIVE_SELECTION_RP.match(query):
                # Selección sobre el conjunto completo: NO reescribir (bug Sevilla).
                print(f"[COREFERENCE] ↩️ omitida (selección de conjunto): '{query[:80]}'")
            elif conversation_history:
                try:
                    REWRITER_SYSTEM = (
                        "Dado el historial de conversación y la última pregunta del usuario, "
                        "reescribe la pregunta como una consulta autónoma que no necesite contexto previo. "
                        "Resuelve pronombres y referencias implícitas. "
                        "Si la pregunta lista alternativas del mismo concepto separadas por comas o 'o', "
                        "consolídalas en una expresión genérica que las represente. "
                        "NUNCA restrinjas una pregunta sobre un conjunto general ('los candidatos', "
                        "'las facturas') a las entidades ya mencionadas: es sobre TODO el conjunto. "
                        "Si la pregunta ya es autónoma y clara, devuélvela tal cual. "
                        "Responde SOLO con la pregunta reescrita, sin explicaciones ni comillas.\n\n"

                        "EJEMPLOS DE CONSOLIDACIÓN DE ALTERNATIVAS:\n"
                        "- 'experiencia en X, Y o Z' → 'experiencia en [categoría que agrupa X, Y, Z]'\n"
                        "- 'conoce A, B o C' → 'conoce [categoría que agrupa A, B, C]'\n"
                        "- 'usó herramienta1, herramienta2 o herramienta3' → 'usó [tipo de herramienta]'\n"
                        "- 'trabaja con formato1, formato2 o formato3' → 'trabaja con [tipo de formato]'\n\n"

                        "EJEMPLOS DE RESOLUCIÓN DE COREFERENCES:\n"
                        "[U]: Hablamos del contrato de servicios\n"
                        "[A]: El contrato establece...\n"
                        "Query: '¿y las penalidades?' → '¿Cuáles son las penalidades en el contrato de servicios?'\n\n"

                        "[U]: Dame el resumen del documento de compliance\n"
                        "[A]: El documento cubre...\n"
                        "Query: '¿quién lo firmó?' → '¿Quién firmó el documento de compliance?'\n\n"

                        "EJEMPLOS SIN CAMBIO (ya autónoma):\n"
                        "- '¿Cuál es el plazo máximo establecido en el contrato?' → igual\n"
                        "- '¿Qué obligaciones tiene el proveedor?' → igual\n"
                    )
                    rewrite_messages = conversation_history[-10:] + [{"role": "user", "content": query}]
                    rewrite_client = openai.AsyncOpenAI(api_key=config.Constants.KEY_OPEN_AI)

                    rewrite_response = await asyncio.wait_for(
                        rewrite_client.chat.completions.create(
                            model=MODEL_COREFERENCE,
                            messages=[
                                {"role": "system", "content": REWRITER_SYSTEM},
                                *rewrite_messages
                            ],
                            max_completion_tokens=300,
                        ),
                        timeout=12.0
                    )

                    rewritten = rewrite_response.choices[0].message.content.strip()
                    if rewritten and rewritten != query:
                        query_for_retrieval = rewritten
                        print(f"[COREFERENCE] ✏️ Query reescrita: '{query}' → '{rewritten}'")
                        yield emit("status", {
                            "message": "✏️ Query contextualizada para búsqueda",
                            "phase": 0
                        })

                except Exception as rewrite_err:
                    print(f"[COREFERENCE] ⚠️ Rewriter falló, usando query original: {rewrite_err}")
                    query_for_retrieval = query

            # ═══════════════════════════════════════════════════════════
            # ROUTER MEMORY / BOTH / DOCS
            # Lógica en 3 capas:
            # 1. Keyword pre-check  → MEMORY directo sin LLM (0 latencia)
            # 2. Overlap detection  → ¿la query toca contenido del historial?
            #                         SÍ+necesita docs → BOTH
            #                         SÍ solo historial → MEMORY
            # 3. LLM fallback       → casos ambiguos restantes
            # Sin historial → siempre DOCS.
            # ═══════════════════════════════════════════════════════════

            query_intent = "DOCS"

            if conversation_history:
                yield emit("status", {"message": "🔀 Clasificando intención...", "phase": 0})

                import time as _time
                _router_t0 = _time.time()
                query_lower = query.lower()

                # ════════════════════════════════════════════════════════
                # ROUTER — 3 capas
                # Capa 0: regex BOTH signals (ms, sin red)
                # Capa 1: embeddings semánticos MEMORY / BOTH (text-embedding-3-small)
                # Capa 2: LLM fallback max_tokens=20
                # ════════════════════════════════════════════════════════

                # ── Capa 0: BOTH signals regex — prioridad máxima ────────
                _BOTH_SIGNALS = [
                    "amplía lo que", "amplia lo que",
                    "más detalle del documento", "mas detalle del documento",
                    "más detalle sobre lo que", "mas detalle sobre lo que",
                    "profundiza en lo que", "profundiza sobre lo que",
                    "detalla más lo que", "detalla mas lo que",
                    "lo que dijiste sobre", "lo que explicaste sobre",
                    "lo que mencionaste sobre", "lo que respondiste sobre",
                    "según lo que dijiste", "segun lo que dijiste",
                    "basándote en lo que dijiste", "basandote en lo que dijiste",
                    "como dijiste antes", "como mencionaste antes",
                    "amplía la respuesta", "amplia la respuesta",
                    "amplía con más detalle", "amplia con mas detalle",
                    "profundiza la respuesta", "expande la respuesta",
                ]
                if any(s in query_lower for s in _BOTH_SIGNALS):
                    query_intent = "BOTH"
                    print(
                        f"[ROUTER] intent=BOTH (capa 0 regex) | {_time.time() - _router_t0:.2f}s | query='{query[:80]}'")

                # ── Capa 1: embeddings semánticos ─────────────────────────
                if query_intent == "DOCS":
                    try:
                        _MEMORY_PROTOS = [
                            "¿qué preguntas te hice hoy?",
                            "¿cuáles fueron todas mis preguntas?",
                            "¿de qué hablamos en esta conversación?",
                            "resume nuestra conversación",
                            "¿qué respondiste antes?",
                            "¿qué te pregunté hasta ahora?",
                            "¿cuántas preguntas te hice?",
                            "¿qué dijiste sobre este tema?",
                            "¿qué temas tocamos hoy?",
                            "historial de lo que hablamos",
                            "¿qué me explicaste antes?",
                            "¿qué me contaste sobre los candidatos?",
                        ]
                        _BOTH_PROTOS = [
                            "amplía lo que dijiste con más detalle del documento",
                            "profundiza en la respuesta anterior con info del CV",
                            "basándote en lo que me explicaste, dame más detalle",
                            "retomando lo que mencionaste, ¿puedes ampliar con el documento?",
                            "completa lo que dijiste con información adicional del archivo",
                            "expande la respuesta que diste con más datos del texto",
                            "según lo que me contaste antes, ahora necesito más detalle técnico",
                        ]

                        _emb_client = openai.AsyncOpenAI(api_key=config.Constants.KEY_OPEN_AI)

                        # embed query + todos los prototipos en una sola llamada
                        _all_texts = [query] + _MEMORY_PROTOS + _BOTH_PROTOS
                        _emb_resp = await asyncio.wait_for(
                            _emb_client.embeddings.create(
                                model="text-embedding-3-small",
                                input=_all_texts,
                            ),
                            timeout=5.0
                        )
                        _vecs = [e.embedding for e in _emb_resp.data]

                        def _cosine(a, b):
                            dot = sum(x * y for x, y in zip(a, b))
                            na = sum(x * x for x in a) ** 0.5
                            nb = sum(x * x for x in b) ** 0.5
                            return dot / (na * nb + 1e-9)

                        _q_vec = _vecs[0]
                        _mem_vecs = _vecs[1: 1 + len(_MEMORY_PROTOS)]
                        _both_vecs = _vecs[1 + len(_MEMORY_PROTOS):]

                        _sim_mem = max(_cosine(_q_vec, v) for v in _mem_vecs)
                        _sim_both = max(_cosine(_q_vec, v) for v in _both_vecs)

                        _THRESH_MEMORY = 0.72
                        _THRESH_BOTH = 0.72

                        print(
                            f"[ROUTER] 🧠 emb sim | MEMORY={_sim_mem:.3f} BOTH={_sim_both:.3f} | thresh={_THRESH_MEMORY}")

                        if _sim_mem >= _THRESH_BOTH and _sim_mem >= _sim_both:
                            query_intent = "MEMORY"
                            print(
                                f"[ROUTER] intent=MEMORY (capa 1 emb) | {_time.time() - _router_t0:.2f}s | query='{query[:80]}'")
                        elif _sim_both >= _THRESH_BOTH:
                            query_intent = "BOTH"
                            print(
                                f"[ROUTER] intent=BOTH (capa 1 emb) | {_time.time() - _router_t0:.2f}s | query='{query[:80]}'")

                    except TimeoutError:
                        print(f"[ROUTER] ⏱️ Timeout embeddings capa 1 → fallback LLM")
                    except Exception as _emb_err:
                        print(f"[ROUTER] ⚠️ Error embeddings capa 1: {_emb_err} → fallback LLM")

                # ── Capa 2: LLM fallback para casos ambiguos ─────────────
                if query_intent == "DOCS":
                    history_preview = ""
                    for turn in conversation_history[-6:]:
                        role = "U" if turn["role"] == "user" else "A"
                        history_preview += f"[{role}]: {turn['content'][:150]}\n"

                    router_system = (
                        "Clasifica la intención de la query en UNA de estas tres categorías:\n"
                        "MEMORY  → el usuario pregunta sobre el historial de esta conversación "
                        "(ej: qué preguntas hice, de qué hablamos, qué respondiste, resumen de la charla)\n"
                        "BOTH    → referencia al historial Y necesita información de documentos "
                        "(ej: lo que dijiste sobre X, antes mencionaste Y, amplía lo que explicaste)\n"
                        "DOCS    → consulta documentos sin referencia al historial\n\n"
                        "Responde ÚNICAMENTE con una de estas palabras: MEMORY, BOTH, DOCS"
                    )
                    router_user = (
                        f"HISTORIAL RECIENTE:\n{history_preview}\n\n"
                        f"QUERY: {query_for_retrieval}\n\n"
                        f"Clasificación:"
                    )

                    try:
                        router_client = openai.AsyncOpenAI(api_key=config.Constants.KEY_OPEN_AI)
                        router_response = await asyncio.wait_for(
                            router_client.chat.completions.create(
                                model=MODEL_ROUTER,
                                messages=[
                                    {"role": "system", "content": router_system},
                                    {"role": "user", "content": router_user},
                                ],
                                max_completion_tokens=2000,
                                reasoning_effort="low",
                            ),
                            timeout=8.0
                        )
                        raw_intent = router_response.choices[0].message.content.strip().upper()
                        # extraer primera palabra válida por si el modelo añade texto extra
                        for _word in raw_intent.split():
                            if _word in ("MEMORY", "BOTH", "DOCS"):
                                raw_intent = _word
                                break
                        query_intent = raw_intent if raw_intent in ("MEMORY", "BOTH", "DOCS") else "DOCS"
                        print(
                            f"[ROUTER] intent={query_intent} (capa 2 LLM) | {_time.time() - _router_t0:.2f}s | query='{query[:80]}'")
                    except Exception as router_err:
                        print(f"[ROUTER] ⚠️ Clasificador LLM falló, default DOCS: {router_err}")
                        query_intent = "DOCS"

                # ── Guarda anti-falso-positivo de MEMORY ──────────────────
                # MEMORY responde SIN tocar Weaviate (solo historial). Una pregunta
                # documental mid-conversation que comparte vocabulario con el tema
                # ("¿quién de los candidatos trabaja en Sevilla?") se parece por
                # embeddings a un prototipo de memoria ("¿qué me contaste sobre los
                # candidatos?") y caía en MEMORY → "no hay info en el historial".
                # Solo es MEMORY de verdad si refiere EXPLÍCITAMENTE al diálogo mismo.
                if query_intent == "MEMORY" and not _META_CONV_MARKERS.search(query):
                    print(f"[ROUTER] ↩️ MEMORY→DOCS: sin marcador de meta-conversación "
                          f"(pregunta documental) | query='{query[:80]}'")
                    query_intent = "DOCS"

                yield emit("status", {"message": f"🔀 Modo: {query_intent}", "phase": 0})

            # ═══════════════════════════════════════════════════════════
            # MODO MEMORY — responde desde historial sin tocar Weaviate
            # ═══════════════════════════════════════════════════════════

            if query_intent == "MEMORY":
                yield emit("status", {
                    "message": "📜 Consultando historial de conversación...",
                    "phase": 1
                })

                history_text = ""
                for i, turn in enumerate(conversation_history, 1):
                    role_label = "Usuario" if turn["role"] == "user" else "Assistant"
                    history_text += f"[{i}] {role_label}: {turn['content'][:500]}\n\n"

                memory_system = (
                    f"Eres {personality}. "
                    f"El usuario pregunta sobre el historial de esta conversación. "
                    f"Responde en HTML válido. Sé preciso y directo. "
                    f"NO inventes preguntas ni respuestas que no estén en el historial. "
                    f"Si no hay información suficiente, dilo claramente."
                )

                memory_user = (
                    f"HISTORIAL COMPLETO:\n\n{history_text}\n\n"
                    f"PREGUNTA: {query_original_user}\n\n"
                    f"Responde basándote ÚNICAMENTE en el historial mostrado arriba."
                )

                memory_response = None
                try:
                    mem_client = openai.AsyncOpenAI(api_key=config.Constants.KEY_OPEN_AI)
                    mem_resp = await asyncio.wait_for(
                        mem_client.chat.completions.create(
                            model=MODEL_COREFERENCE,
                            messages=[
                                {"role": "system", "content": memory_system},
                                {"role": "user", "content": memory_user},
                            ],
                            max_completion_tokens=1500,
                            reasoning_effort="low",
                        ),
                        timeout=20.0
                    )
                    memory_response = mem_resp.choices[0].message.content.strip()

                except Exception as mem_err:
                    print(f"[MEMORY] ⚠️ Handler falló: {mem_err}")
                    turns_user = [t for t in conversation_history if t["role"] == "user"]
                    items = "".join(f"<li>{t['content'][:300]}</li>" for t in turns_user)
                    memory_response = (
                        f"<div><h3>Preguntas registradas en esta sesión</h3>"
                        f"<ul>{items}</ul></div>"
                    )

                for i in range(0, len(memory_response), 100):
                    yield emit("chunk", {"message": memory_response[i:i + 100]})
                    await asyncio.sleep(0.01)

                await _save(query_original_user, memory_response)

                elapsed = time.time() - start_time_total
                yield emit("complete", {
                    "message": f"✅ Historial consultado en {elapsed:.2f}s",
                    "time": round(elapsed, 2)
                })
                return

            # BOTH: historial ya en conversation_history, coreference ya reescribió
            # query_for_retrieval. El sintetizador recibirá conversation_history
            # para poder anclar la respuesta en lo conversado previamente.
            if query_intent == "BOTH":
                yield emit("status", {"message": "🔗 Modo BOTH — historial + documentos", "phase": 0})

            # ═══════════════════════════════════════════════════════════
            # 🌐 PRIORIDAD 1: WEBSEARCH
            # ═══════════════════════════════════════════════════════════

            if websearch:
                yield emit("status", {"message": "🌐 Modo WebSearch activado...", "phase": 0})

                try:
                    result = await self._handle_web_search(
                        query, user, conversation_id, timestamp, config_chatbot
                    )

                    if not result:
                        result = "❌ No se pudo obtener respuesta del web search."
                        logger.error("_handle_web_search retornó None o vacío")

                    yield emit("chunk", {"text": result})

                    elapsed = time.time() - start_time_total
                    await _save(query_original_user, result)

                    yield emit("complete", {
                        "message": f"✅ WebSearch completado en {elapsed:.2f}s",
                        "time": round(elapsed, 2)
                    })
                    return

                except Exception as e:
                    raise RuntimeError(f"Error en web search: {str(e)}")

            # ═══════════════════════════════════════════════════════════
            # 🤖 PRIORIDAD 2: CLAUDE DIRECTO (LLM Assistant)
            # ═══════════════════════════════════════════════════════════

            if llm_direct:
                yield emit("status", {"message": "🤖 🐿️🐿️ Modo LLM Directo activado (sin RAG) 🐿️🐿️...", "phase": 0})

                max_query_length = 50000 if context else 9000
                if len(query) > max_query_length:
                    raise ValueError(f"La consulta es demasiado larga ({len(query)} caracteres)")

                yield emit("status", {"message": "🐿️🐿️ Cargando Assistant 🐿️🐿️...", "phase": 1})

                claude_api_key = os.getenv("ANTHROPIC_API_KEY")
                if not claude_api_key:
                    try:
                        config_manager = ConfigurationManagerSelected()
                        config_data = config_manager.load_configuration("manager_extractor")
                        claude_api_key = config_data.anthropic.get_api_key()
                    except Exception as config_err:
                        logger.warning(f"No se pudo cargar config de Anthropic: {config_err}")

                if not claude_api_key:
                    raise RuntimeError("Claude API key no configurada. Configura ANTHROPIC_API_KEY")

                claude_client = AsyncAnthropic(api_key=claude_api_key)

                claude_model = model if (model and model.startswith("claude")) else CLAUDE_SONNET

                yield emit("status", {"message": "✅ Usando assistant ..", "phase": 1})

                system_prompt = f"""Eres un {personality}.

    FORMATO DE RESPUESTA OBLIGATORIO - MUY IMPORTANTE:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    TODAS tus respuestas DEBEN estar en formato HTML puro, NO en markdown.

    ❌ NO USES NUNCA:
    - Markdown headers: ##, ###
    - Markdown bold: **texto**
    - Markdown italic: *texto*
    - Markdown lists: - item
    - Markdown code: ```code```

    ✅ USA SIEMPRE:
    - HTML headers: <h2>Título</h2>, <h3>Subtítulo</h3>
    - HTML bold: <strong>texto</strong>
    - HTML italic: <em>texto</em>
    - HTML lists: <ul><li>item</li></ul>
    - HTML code: <pre><code>código</code></pre>

    ESTRUCTURA REQUERIDA:
    <div>
      <h2>Título Principal</h2>
      <p>Párrafo explicativo con <strong>énfasis</strong> cuando sea necesario.</p>
      <h3>Subtítulo</h3>
      <ul>
        <li>Primer punto importante</li>
        <li>Segundo punto importante</li>
      </ul>
      <p>Más contenido en párrafos separados.</p>
    </div>

    PARA CÓDIGO:
    <pre><code>
    // Tu código aquí
    function ejemplo() {{
      return true;
    }}
    </code></pre>

    PARA TABLAS:
    <table border="1" cellpadding="5">
      <tr><th>Columna 1</th><th>Columna 2</th></tr>
      <tr><td>Dato 1</td><td>Dato 2</td></tr>
    </table>
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    INSTRUCCIONES GENERALES:
    - Mantén un tono {tone}
    - Tipo de documentos: {doc_type}
    - Usuario: {user}
    - Empresa: {company}
    - Configuración: {config_chatbot}
    - Responde de forma directa, profesional y precisa
    - Si necesitas más información, pídela claramente

    ⛔ GUARDRAIL ANTI-FABRICACIÓN (OBLIGATORIO):
    - Respondé SOLO con datos que provengan de la base de datos consultada, de los documentos recuperados o de una herramienta real de esta sesión. NUNCA inventes ni estimes precios, productos, cifras, fechas, porcentajes ni fuentes que no estén en los datos recuperados.
    - Si el usuario pide comparar con datos externos (mercado, competencia, precios de otros lugares/comercios, valores actuales) y no hay una fuente real disponible, DECILO explícitamente (por ejemplo: "no tengo datos de mercado ni de competencia en esta base") y ofrecé usar la herramienta de investigación web (Investigar) en lugar de inventar. NO armes tablas de "precios aproximados" ni comparativas sin fuente real.
    - Si sólo tenés un valor de la base, mostrá ese valor y aclará que no hay datos de comparación disponibles.

    RECUERDA: Tu respuesta debe ser HTML válido que se pueda renderizar directamente en un navegador.
    """

                if context:
                    yield emit("status", {"message": "📊 Agregando contexto SQL...", "phase": 1})
                    system_prompt += f"""

    DATOS SQL DISPONIBLES:
    ═══════════════════════════════════════════════════════════════
    {context}
    ═══════════════════════════════════════════════════════════════

    IMPORTANTE:
    - Los datos SQL mostrados arriba son la fuente PRINCIPAL para responder
    - Úsalos con precisión en números, nombres y datos específicos de la tabla
    - Presenta los datos en formato HTML usando tablas cuando sea apropiado
    - Para números importantes, usa <strong> para resaltarlos

    ⛔ GUARDRAIL ANTI-FABRICACIÓN (OBLIGATORIO):
    - Respondé SOLO con datos que provengan de estos datos SQL, de los documentos recuperados o de una herramienta real de esta sesión. NUNCA inventes ni estimes precios, productos, cifras, fechas, porcentajes ni fuentes que no estén en los datos recuperados.
    - Si el usuario pide comparar con datos externos (mercado, competencia, precios de otros lugares/comercios, valores actuales) y no hay una fuente real disponible, DECILO explícitamente (por ejemplo: "no tengo datos de mercado ni de competencia en esta base") y ofrecé usar la herramienta de investigación web (Investigar) en lugar de inventar. NO armes tablas de "precios aproximados" ni comparativas sin fuente real.
    - Si sólo tenés el valor de la base, mostrá ese valor y aclará que no hay datos de comparación disponibles.
    """

                messages = []
                for turn in conversation_history:
                    messages.append({"role": turn["role"], "content": turn["content"]})
                messages.append({"role": "user", "content": query})

                yield emit("status", {"message": "💬 Consultando a Assistant LLM...", "phase": 2})

                try:
                    full_response = ""
                    markdown_detected = False

                    async with claude_client.messages.stream(
                            model=claude_model,
                            max_tokens=4096,
                            temperature=0.7,
                            system=system_prompt,
                            messages=messages
                    ) as stream:
                        async for text in stream.text_stream:
                            full_response += text
                            if any(marker in text for marker in ["**", "##", "- ", "* ", "```"]):
                                markdown_detected = True
                            await asyncio.sleep(0.01)

                    if markdown_detected or any(
                            marker in full_response for marker in ["**", "##", "- ", "* ", "```", "__"]):
                        yield emit("status", {"message": "🔄 Convirtiendo markdown a HTML...", "phase": 2})
                        try:
                            import mistune
                            full_response = mistune.html(full_response)
                            full_response = self.compact_html(full_response)
                            yield emit("status", {"message": "✅ HTML compactado", "phase": 2})
                        except ImportError:
                            full_response = re.sub(r"^### (.+)$", r"<h3>\1</h3>", full_response, flags=re.MULTILINE)
                            full_response = re.sub(r"^## (.+)$", r"<h2>\1</h2>", full_response, flags=re.MULTILINE)
                            full_response = re.sub(r"^# (.+)$", r"<h1>\1</h1>", full_response, flags=re.MULTILINE)
                            full_response = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", full_response)
                            full_response = re.sub(r"__(.+?)__", r"<strong>\1</strong>", full_response)
                            full_response = re.sub(r"\*([^\*]+?)\*", r"<em>\1</em>", full_response)
                            full_response = re.sub(r"_([^_]+?)_", r"<em>\1</em>", full_response)
                            full_response = re.sub(r"```(.+?)```", r"<pre><code>\1</code></pre>", full_response,
                                                   flags=re.DOTALL)
                            full_response = re.sub(r"`(.+?)`", r"<code>\1</code>", full_response)

                            lines = full_response.split("\n")
                            in_list = False
                            result = []
                            for line in lines:
                                stripped = line.strip()
                                if stripped.startswith(("- ", "* ")):
                                    if not in_list:
                                        result.append("<ul>")
                                        in_list = True
                                    result.append(f"<li>{stripped[2:]}</li>")
                                else:
                                    if in_list:
                                        result.append("</ul>")
                                        in_list = False
                                    if stripped:
                                        result.append(line if stripped.startswith("<") else f"<p>{line}</p>")
                            if in_list:
                                result.append("</ul>")
                            full_response = "\n".join(result)

                    for i in range(0, len(full_response), 100):
                        yield emit("chunk", {"message": full_response[i:i + 100]})
                        await asyncio.sleep(0.01)

                    await _save(query_original_user, full_response)

                    elapsed = time.time() - start_time_total
                    yield emit("complete", {
                        "message": f"✅ Claude respondió directamente en {elapsed:.2f}s",
                        "time": round(elapsed, 2)
                    })
                    return

                except anthropic.APIConnectionError as e:
                    raise RuntimeError(f"Error de conexión con Claude: {str(e)}")
                except RuntimeError as e:
                    raise RuntimeError(f"Rate limit de Claude excedido: {str(e)}")
                except anthropic.APIStatusError as e:
                    raise RuntimeError(f"Error de Claude API: {e.message}")
                except Exception as e:
                    raise RuntimeError(f"Error inesperado con Claude: {str(e)}")

            # ═══════════════════════════════════════════════════════════
            # 📚 PRIORIDAD 3: PIPELINE RAG COMPLETO (DOCS / BOTH)
            # ═══════════════════════════════════════════════════════════

            yield emit("status", {"message": "📚 Modo RAG activado - búsqueda en documentos...", "phase": 0})

            max_query_length = 10000 if context else 5000
            if len(query) > max_query_length:
                raise ValueError(f"La consulta es demasiado larga ({len(query)} caracteres)")

            yield emit("status", {"message": "🔍 Verificando ambigüedad...", "phase": 0})
            ambiguity_check = self._detect_query_ambiguity(query)
            if ambiguity_check["is_ambiguous"]:
                yield emit("chunk", {"text": ambiguity_check["response"]})
                yield emit("complete", {"message": "Query ambigua - clarificación enviada", "time": 0})
                return

            yield emit("status", {"message": "✅ Query no ambigua", "phase": 0})

            timestamp = timestamp or datetime.datetime.utcnow()
            collection_name = re.sub(r"[^a-zA-Z0-9_]", "_", (company or "").upper().replace(" ", "_"))
            yield emit("status", {"message": f"📚 Colección: {collection_name}", "phase": 0})

            # ═══════════════════════════════════════════════════════════
            # FASE 1: CONFIGURACIÓN
            # ═══════════════════════════════════════════════════════════

            yield emit("status", {"message": "⚙️ Configurando sistema RAG...", "phase": 1})

            try:
                if not model:
                    model = MODEL_DECOMPOSER

                validation_task = await self.custom.exists_weaviate_class(collection_name)
                if asyncio.iscoroutine(validation_task) or asyncio.isfuture(validation_task):
                    class_exists = await validation_task
                else:
                    class_exists = validation_task

                if isinstance(class_exists, Exception):
                    raise RuntimeError(f"Error verificando colección: {str(class_exists)}")

                if not class_exists:
                    raise RuntimeError(f"No se encontró la colección '{collection_name}'")

                yield emit("status", {"message": "✅ Colección verificada", "phase": 1})

                simple_model_executor, vector_search_func = self._create_executors(
                    model, None, False, user, company, collection_name
                )
                model_executor = self._create_main_executor(model, None, False, tone, personality)

                yield emit("status", {"message": "✅ Sistema RAG configurado", "phase": 1})

            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"Error en configuración del sistema: {str(e)}")

            yield emit("status", {"message": "✅ Procesamiento directo", "phase": 2})

            # ═══════════════════════════════════════════════════════════
            # AGREGAR CONTEXTO SQL
            # ═══════════════════════════════════════════════════════════

            if context:
                yield emit("status", {"message": "📊 Agregando contexto SQL", "phase": 2})

                sql_context_header = f"""
    ═══════════════════════════════════════════════════════════════
    DATOS SQL DISPONIBLES
    ═══════════════════════════════════════════════════════════════

    {context}

    ═══════════════════════════════════════════════════════════════

    INSTRUCCIONES IMPORTANTES:
    - Los datos SQL mostrados arriba son la fuente PRINCIPAL para responder
    - Si la pregunta menciona leyes/normativas, COMBINA datos SQL + base documental
    - Sé PRECISO con números, nombres y datos específicos de la tabla
    - Mantén el tono: {tone}

    """
                query_for_retrieval = f"{sql_context_header}\nPREGUNTA DEL USUARIO:\n{query_for_retrieval}"

            # ── historial para el sintetizador en modo BOTH ───────────────────
            synth_conversation_history = conversation_history if query_intent == "BOTH" else []

            yield emit("status", {"message": "🔍 Búsqueda vectorial...", "phase": 3})

            try:
                requires_aggregation, initial_docs = await asyncio.gather(
                    self._check_aggregation_needed(query_for_retrieval, simple_model_executor),
                    self._do_multi_query_search(query_for_retrieval, vector_search_func,
                                                simple_model_executor, doc_type)
                )

                if isinstance(initial_docs, dict) and initial_docs.get("error"):
                    await _save(query_original_user, "")
                    yield emit("error", {
                        "code": "VECTOR_SEARCH_ERROR",
                        "message": f"Error en búsqueda vectorial: {initial_docs.get('mensaje', 'Error desconocido')}",
                        "details": {"collection": collection_name, "query_preview": query_for_retrieval[:100]}
                    })
                    return

                doc_count = len(initial_docs) if isinstance(initial_docs, list) else 0

                if doc_count == 0:
                    await _save(query_original_user, "")
                    yield emit("error", {
                        "code": "NO_DOCUMENTS_FOUND",
                        "message": f"No se encontraron documentos en la colección \"{collection_name}\"",
                        "details": {
                            "collection": collection_name,
                            "query_preview": query_for_retrieval[:100],
                            "usuario": user,
                            "company": company,
                        },
                        "suggestions": [
                            "Verifica que los documentos estén indexados en Weaviate",
                            f"Consulta: curl \"http://localhost:8087/v1/objects?class={collection_name}&limit=1\""
                        ]
                    })
                    return

                yield emit("status", {"message": f"✅ {doc_count} documentos encontrados", "phase": 3})

            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"Error en búsqueda inicial: {str(e)}")

            query_type = "UNKNOWN"
            num_partes = 0
            if isinstance(initial_docs, list) and initial_docs:
                first_doc = initial_docs[0]
                if isinstance(first_doc, dict):
                    query_type = first_doc.get("query_type", "UNKNOWN")
                    num_partes = first_doc.get("num_partes", 0)
                elif hasattr(first_doc, "metadata"):
                    meta = first_doc.metadata if isinstance(first_doc.metadata, dict) else {}
                    query_type = meta.get("query_type", "UNKNOWN")
                    num_partes = meta.get("num_partes", 0)

            yield emit("status", {"message": "🎯 Ordenando por relevancia...", "phase": 4})

            try:
                relevant_docs = await asyncio.wait_for(
                    self._rerank_and_diversify(query_for_retrieval, initial_docs), timeout=5.0
                )
                yield emit("status", {"message": f"✅ {len(relevant_docs)} docs seleccionados", "phase": 4})
            except TimeoutError:
                relevant_docs = initial_docs[:10]
                yield emit("status", {"message": "⏱️ Timeout rerank - usando top 10", "phase": 4})
            except Exception:
                relevant_docs = initial_docs[:10]
                yield emit("status", {"message": "⚠️ Error rerank - usando top 10", "phase": 4})

            yield emit("status", {"message": "📝 Procesando documentos...", "phase": 4})

            try:
                processed_chunks = self._process_documents(relevant_docs, query_for_retrieval, "")
                processed_chunks = self._consolidate_group_chunks(processed_chunks)
            except Exception:
                processed_chunks = relevant_docs

            CHUNKS_CONFIG = {"simple": 30, "multi": 15, "default": 30}
            is_simple_query = query_type in ("SIMPLE_EXACTA", "SIMPLE_CONCEPTUAL") and num_partes <= 2
            use_fast_model = is_simple_query or (query_type == "MULTI_PARTE" and num_partes <= 3)

            if is_simple_query:
                max_chunks = CHUNKS_CONFIG["simple"]
            elif query_type == "MULTI_PARTE":
                max_chunks = CHUNKS_CONFIG["multi"]
            else:
                max_chunks = min(len(processed_chunks), CHUNKS_CONFIG["default"])

            processed_chunks = processed_chunks[:max_chunks]
            chunk_executor = simple_model_executor if use_fast_model else model_executor

            yield emit("status", {"message": f"⚙️ Procesando {len(processed_chunks)} chunks...", "phase": 5})

            try:
                all_results = await self._process_chunks_in_batches(
                    processed_chunks, query_for_retrieval, chunk_executor
                )
                yield emit("status", {"message": f"✅ {len(all_results)} chunks procesados", "phase": 5})
            except Exception as e:
                raise RuntimeError(f"Error procesando chunks: {str(e)}")

            yield emit("status", {"message": "🤖 Generando respuesta final...", "phase": 6})

            response_text = None
            try:
                response_text = await asyncio.wait_for(
                    self.synthesize_answers(
                        all_results, query_original_user, model_executor, simple_model_executor,
                        query_type, requires_aggregation, user, company,
                        conversation_id, config_chatbot,
                        conversation_history=synth_conversation_history,  # FIX: contexto BOTH
                        fallback_chunks=processed_chunks,
                    ),
                    timeout=60.0
                )

                for i in range(0, len(response_text), 100):
                    yield emit("chunk", {"message": response_text[i:i + 100]})
                    await asyncio.sleep(0.01)

            except Exception as e:
                yield emit("status", {"message": f"⚠️ Error en síntesis: {str(e)[:100]}", "phase": 6})
                response_text = (
                    f"<div><h2>Resultados encontrados</h2>"
                    f"<p><strong>Consulta:</strong> {query_original_user}</p>"
                    f"<p><em>⚠️ Se encontraron {len(all_results)} fragmentos relevantes "
                    f"pero ocurrió un error en la síntesis final.</em></p></div>"
                )
                for i in range(0, len(response_text), 100):
                    yield emit("chunk", {"message": response_text[i:i + 100]})
                    await asyncio.sleep(0.01)

            await _save(query_original_user, response_text or "")

            elapsed_total = time.time() - start_time_total
            yield emit("complete", {"time": round(elapsed_total, 2)})

    # ═══════════════════════════════════════════════════════════════════════
    # 🔥 MANEJO DE ERRORES
    # ═══════════════════════════════════════════════════════════════════════

        except Exception as e:
            elapsed = time.time() - start_time_total


            error_str = str(e).lower()
            error_type_name = type(e).__name__

            if isinstance(e, TimeoutError) or "timeout" in error_str:
                error_code = "TIMEOUT"
                error_message = "La operación excedió el tiempo límite permitido"
            elif isinstance(e, ValueError) or "validation" in error_type_name.lower():
                error_code = "VALIDATION_ERROR"
                error_message = getattr(e, "message", str(e))
            elif any(x in error_str for x in ["openai", "anthropic", "claude", "llm", "model"]):
                error_code = "LLM_ERROR"
                error_message = f"Error en el modelo de lenguaje: {str(e)}"
            elif any(x in error_str for x in ["weaviate", "vector"]):
                error_code = "VECTOR_DB_ERROR"
                error_message = f"Error en base de datos vectorial: {str(e)}"
            elif any(x in error_str for x in ["database", "mongo", "postgres", "mysql"]):
                error_code = "DATABASE_ERROR"
                error_message = f"Error en base de datos: {str(e)}"
            elif any(x in error_str for x in ["connection", "network", "socket"]):
                error_code = "CONNECTION_ERROR"
                error_message = f"Error de conexión: {str(e)}"
            elif any(x in error_str for x in ["permission", "unauthorized", "forbidden"]):
                error_code = "PERMISSION_ERROR"
                error_message = f"Error de permisos: {str(e)}"
            else:
                error_code = "UNEXPECTED_ERROR"
                error_message = f"Error inesperado: {str(e)}"

            logger.error(
                f"[{error_code}] {error_message}",
                extra={
                    "request_id": request_id,
                    "usuario": user,
                    "company": company,
                    "query_length": len(query),
                    "elapsed_ms": round(elapsed * 1000, 2),
                    "error_type": error_type_name
                }
            )

            error_details = {
                "code": error_code,
                "message": error_message,
                "type": error_type_name,
                "timestamp": time.time(),
                "request_id": request_id,
                "elapsed_ms": round(elapsed * 1000, 2)
            }

            if os.getenv("ENVIRONMENT", "production") != "production":
                error_details["debug"] = {
                    "usuario": user,
                    "company": company,
                    "query_preview": query[:100],
                    "traceback": ""
                }

            yield emit("error", error_details)

    def compact_html(self, html_text: str) -> str:
        """
        Compacta HTML de forma balanceada:
        - Elimina saltos de línea excesivos
        - Elimina espacios entre tags HTML
        - Mantiene espacios entre palabras dentro del contenido

        Args:
            html_text: HTML con posibles espacios excesivos

        Returns:
            HTML compacto pero legible
        """

        # 1. Eliminar saltos de línea múltiples (dejar solo 1)
        html_text = re.sub(r'\n\s*\n+', '\n', html_text)

        # 2. Eliminar espacios/saltos ENTRE tags (pero NO dentro del contenido)
        # Esto transforma: ></strong>\n  <li> en: ></strong><li>
        html_text = re.sub(r'>\s+<', '><', html_text)

        # 3. Eliminar espacios al INICIO de tags block
        # Esto transforma: <p>  texto en: <p>texto
        html_text = re.sub(r'<(p|div|h[1-6]|li|td|th|tr|ul|ol|table)>\s+', r'<\1>', html_text)

        # 4. Eliminar espacios al FINAL de tags block
        # Esto transforma: texto  </p> en: texto</p>
        html_text = re.sub(r'\s+</(p|div|h[1-6]|li|td|th|tr|ul|ol|table)>', r'</\1>', html_text)

        # 5. Normalizar múltiples espacios DENTRO del texto (pero dejar 1)
        # Esto transforma: "palabra    palabra" en: "palabra palabra"
        html_text = re.sub(r'  +', ' ', html_text)

        # 6. Eliminar espacios al inicio/final de cada línea
        html_text = re.sub(r'^\s+|\s+$', '', html_text, flags=re.MULTILINE)

        # 7. Eliminar líneas completamente vacías
        lines = [line for line in html_text.split('\n') if line.strip()]

        # 8. Unir manteniendo estructura
        return '\n'.join(lines)

    def _create_claude_executor(
            self,
            claude_client: AsyncAnthropic,
            model: str,
            tone: str,
            personality: str
    ):
        """
        Crea un ejecutor que usa Claude API en lugar de OpenAI.

        Args:
            claude_client: Cliente AsyncAnthropic
            model: Modelo de Claude a usar
            tono: Tono de la respuesta
            personalidad: Personalidad del asistente

        Returns:
            Función async ejecutora
        """

        async def claude_executor(prompt: str, **kwargs) -> str:
            """
            Ejecutor que llama a Claude API.

            Args:
                prompt: Prompt a enviar a Claude
                **kwargs: Parámetros adicionales (system, max_tokens, temperature, etc.)

            Returns:
                str: Respuesta de Claude
            """
            try:
                # Extraer parámetros
                system_prompt = kwargs.get('system', f"Eres un {personality}. Usa un tono {tone}.")
                max_tokens = kwargs.get('max_tokens', 4096)
                temperature = kwargs.get('temperature', 0.7)

                # Construir mensajes
                messages = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

                # Llamar a Claude API
                response = await claude_client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=messages
                )

                # Extraer texto de la respuesta
                return response.content[0].text

            except anthropic.APIConnectionError as e:
                raise RuntimeError(f"Error de conexión con Claude API: {str(e)}")
            except RuntimeError as e:
                raise RuntimeError(f"Rate limit excedido en Claude API: {str(e)}")
            except anthropic.APIStatusError as e:
                raise RuntimeError(f"Error de Claude API: {e.message}")
            except Exception as e:
                raise RuntimeError(f"Error inesperado con Claude: {str(e)}")

        return claude_executor




