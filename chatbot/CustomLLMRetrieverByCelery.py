"""
"""

import concurrent.futures
import datetime
import io
import json
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Tuple

# from google import genai
import google.generativeai as genai
import numpy as np
import openai
import pypdfium2 as pdfium
import tiktoken
import weaviate
from PIL import Image
from celery.result import AsyncResult
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.llms.openai import OpenAI
from openai import AsyncOpenAI
from pytesseract import pytesseract
from redis import Redis
from tenacity import wait_exponential, stop_after_attempt, retry

from config.Constants import KEY_OPEN_AI, GPT_5_NANO, GEMINI_3_1_PRO, GPT_5, GPT_5_1, GPT_5_4, GPT_5_4_MINI
from framework.ConfigurationManager.ConfigurationManagerSelected import ConfigurationManagerSelected
from framework.Database.MongoDb.MongoDb import MongoDBManager
from framework.Embeddings.EmbeddingProcessor import EmbeddingProcessor

from framework.Rankers.ReRankers import FastOpenAIReranker
from framework.Redis.RedisConversationManager import RedisConversationManager
from framework.Splitters.ParallelDocumentSplitter import ParallelDocumentSplitter, SplitConfig
from framework.Splitters.SmartSplitAdvisor import SmartSplitAdvisor
from framework.Weaviate.ParallelWeaviateProcessor import ParallelWeaviateProcessor
from framework.Weaviate.VectorialSearchProcessor import VectorSearchFactory

os.environ['OPENAI_API_KEY'] = KEY_OPEN_AI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Enable DEBUG for this specific logger

# Also configure the handler
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


GHOST_ATTENTION_CONFIG = {
    "history_limit": 5,
    "context_max_chars": 8000,  # Increased from implicit
    "response_min_length": 50,  # Minimum to consider the answer valid
    "message_max_length": 4000,  # Por mensaje individual
    "injection_method": "sandwich",  # prefix, suffix, sandwich
    "enable_validation": True,
    "debug_mode": True
}

OCR_THREADS = int(os.getenv("OCR_THREADS", "4"))  # hilos por worker
MAX_PAGES_CONCURRENCY = int(os.getenv("OCR_PAGES_CONCURRENCY", "3"))  # pages concurrentes por PDF

    # Crea el pool una sola vez por proceso/worker (puede ir en __init__ de tu clase)
try:
  _ocr_pool  # noqa
except NameError:
   _ocr_pool = ThreadPoolExecutor(max_workers=OCR_THREADS)

# Inicializa el cliente
# anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)


import asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

from io import BytesIO




class CustomLLMRetrieverByCelery:
    def __init__(self, config, redis_client=Redis,max_tokens_context: int = 50000):
        #self.config = config  # Almacenar la configuration para OpenAI y Milvus
        self.memory = {}
        manager = ConfigurationManagerSelected()
        self.config = manager.load_configuration("manager_extractor")

        self.collection = MongoDBManager("chatbot_memory", config)
        self.weaviate_url = self.config['weaviate']['url']
        # self.client = weaviate.Client(self.weaviate_url)
        self.openai_api_key = self.config['openai_key']['key']
        self.asyncOpenAI = AsyncOpenAI(api_key=self.openai_api_key)
        # genai.configure(api_key="")
        self.tokenizer = tiktoken.get_encoding("o200k_base")
        self._call_count = 0
        self._last_reset = time.time()
        self._semaphore = asyncio.Semaphore(5)  # Max 5 llamadas concurrentes
        self.max_concurrent_requests = 3
        self.redis_client = redis_client
        genai.configure(api_key=self.config.google.get_gemini_api_key())
        self.clientG = genai
        self.client_async_openai = AsyncOpenAI(api_key=KEY_OPEN_AI)
        self.google_api_key = self.config.google.get_gemini_api_key()
        self.embedding_processor = EmbeddingProcessor(
            max_workers=4,
            batch_size=160
        )

        self.azure_endpoint = self.config.azure.get_document_intelligence_endpoint()
        self.azure_key = self.config.azure.get_document_intelligence_credential()

        self.azure_client = DocumentAnalysisClient(
            endpoint=self.config.azure.get_document_intelligence_endpoint(),
            credential=self.config.azure.get_document_intelligence_credential()
        )

        self.max_tokens_context = max_tokens_context
        self.encoding = tiktoken.get_encoding("cl100k_base")  # Para GPT-4
        self.batch_size = 32
        #self.embeddingRetVect  = RetVecBatchEmbedder()
        #self.client_open_ai = OpenAI()
        from openai import OpenAI as NativeOpenAI
        self.client_open_ai = NativeOpenAI(api_key=self.openai_api_key, max_retries=0)

        additional_headers = {
            "X-Openai-Api-Key": self.openai_api_key,
        }
        self.semaphore = asyncio.Semaphore(100)  # Limit to 60 concurrent requests
        self.lock = threading.Lock()
        # Crear cliente de Weaviate con los headers personalizados
        self.client = weaviate.Client(
            self.weaviate_url,
            startup_period=30,  # Tiempo de espera aumentado a 30 segundos
            additional_headers=additional_headers  # Pasar los headers con la API Key de OpenAI
        )
        # self.clientDocument = DocumentAnalysisClient(URL_AZURE, AzureKeyCredential(AZURE_KEY))
        openai.api_key = self.openai_api_key
        self.redis_manager = RedisConversationManager(
            host=config.redis.host,
            port=config.redis.port

        )
        self.parallel_processor = ParallelWeaviateProcessor(
            weaviate_client=self.client,
            embedding_processor=self.embedding_processor,
            max_concurrent_files=4,
            batch_insert_size=160
        )


        self.search_processor = VectorSearchFactory.create_with_embedding_processor(
            weaviate_client=self.client,
            embedding_processor=self.embedding_processor,  # ← AGREGAR
            max_workers=4
        )

    async def createCollectionRetrievalAugmentedGeneration(self, chunks: list, metadata: dict, embedding_fn_batch_local,
                                                           user: str, company: str):
        """
        Guarda los chunks en Weaviate con embedding y metadata - ESTRUCTURA UNIFICADA
        """
        import logging
        import json
        import numpy as np
        import uuid
        import hashlib
        import time
        from datetime import datetime

        logger.info(f"🔍 RAG - Chunks receiveds: {len(chunks)}")
        logger.info(f"🔍 RAG - Tipo chunks: {type(chunks)}")
        logger.info(f"🔍 RAG - Tipo primer chunk: {type(chunks[0]) if chunks else 'EMPTY'}")

        collection_name = re.sub(r"[^a-zA-Z0-9_]", "_", (company or "").upper().replace(" ", "_"))

        # Validation inicial
        if not chunks:
            logging.error("No hay chunks para procesar")
            return False

        if not callable(embedding_fn_batch_local):
            logging.error("embedding is not a valid function")
            return False

        # Paso 1: Crear/verificar clase con SCHEMA COMPLETO
        try:
            schema = self.client.schema.get()
            existing_classes = [c["class"] for c in schema.get("classes", [])]
            # ═══════════════════════════════════════════════════════════════
            # PREPARAR REFERENCIAS (siempre como array - tu schema usa text[])
            # ═══════════════════════════════════════════════════════════════

            refs_raw = metadata.get('referencias', [])

            # Detect the collection schema
            referencias_is_array = True  # Default
            try:
                class_schema = self.client.schema.get(collection_name)
                if class_schema and 'properties' in class_schema:
                    for prop in class_schema.get('properties', []):
                        if prop.get('name') == 'referencias':
                            referencias_is_array = 'text[]' in prop.get('dataType', [])
                            logging.info(f"[Schema] referencias es {'array' if referencias_is_array else 'string'}")
                            break
            except Exception as e:
                logging.warning(f"No se pudo detectar schema: {e}")

            # Preparar valor
            if referencias_is_array:
                if isinstance(refs_raw, list):
                    referencias_value = [str(r) for r in refs_raw if r]
                elif refs_raw:
                    referencias_value = [str(refs_raw)]
                else:
                    referencias_value = []
            else:
                if isinstance(refs_raw, list):
                    referencias_value = ' | '.join(str(r) for r in refs_raw) if refs_raw else ''
                else:
                    referencias_value = str(refs_raw) if refs_raw else ''

            referencias_text = ' | '.join(str(r) for r in refs_raw) if isinstance(refs_raw, list) else str(
                refs_raw or '')

            if collection_name not in existing_classes:
                class_obj = {
                    "class": collection_name,
                    "description": "Chunks de documentos con metadata completa",
                    "properties": [
                        # BASIC FIELDS
                        {"name": "name", "dataType": ["text"]},
                        {"name": "text", "dataType": ["text"]},
                        {"name": "user", "dataType": ["text"]},
                        {"name": "company", "dataType": ["text"]},

                        # METADATA DEL ARCHIVO
                        {"name": "file_id", "dataType": ["text"]},
                        {"name": "source", "dataType": ["text"]},
                        {"name": "filename", "dataType": ["text"]},
                        {"name": "doc_type", "dataType": ["text"]},

                        # METADATA DEL CHUNK
                        {"name": "chunk_length", "dataType": ["int"]},
                        {"name": "word_count", "dataType": ["int"]},
                        {"name": "content_type", "dataType": ["text"]},
                        {"name": "has_numbers", "dataType": ["boolean"]},
                        {"name": "digit_density", "dataType": ["number"]},

                        # REFERENCIAS
                        {"name": "referencias_text", "dataType": ["text"]},
                        {"name": "referencias", "dataType": ["text[]"]},

                        # TEMPORAL
                        {"name": "created_at", "dataType": ["number"]},
                        {"name": "created_date", "dataType": ["text"]},

                        # POSITION
                        {"name": "chunk_index", "dataType": ["int"]},
                        {"name": "total_chunks", "dataType": ["int"]},

                        # NAVIGATION
                        {"name": "previous_chunk_id", "dataType": ["text"]},
                        {"name": "next_chunk_id", "dataType": ["text"]},

                        # CONTENIDO
                        {"name": "language", "dataType": ["text"]},
                        {"name": "text_preview", "dataType": ["text"]},
                        {"name": "text_hash", "dataType": ["text"]},

                        # PROCESO
                        {"name": "schema_version", "dataType": ["text"]},
                        {"name": "processing_time", "dataType": ["number"]},
                        {"name": "embedding_model", "dataType": ["text"]},

                        # SOFT DELETE
                        {"name": "is_deleted", "dataType": ["boolean"]},
                        {"name": "deleted_at", "dataType": ["text"]},
                        {"name": "deleted_by", "dataType": ["text"]},
                        {"name": "deletion_reason", "dataType": ["text"]},

                        # AUDIT
                        {"name": "uploaded_by", "dataType": ["text"]},
                        {"name": "uploaded_by_name", "dataType": ["text"]},
                        {"name": "upload_ip", "dataType": ["text"]},

                        # BACKUP
                        {"name": "metadata_json", "dataType": ["text"]}
                    ]
                }
                self.client.schema.create_class(class_obj)
                logging.info(f"Clase {collection_name} creada en Weaviate.")
            else:
                logging.info(f"Clase {collection_name} ya existe.")
        except Exception as e:
            logging.error(f"Error al verificar/crear el esquema en Weaviate: {e}")
            return False

        # Paso 2: Pre-calcular embeddings en lote
        try:
            logging.info(f"Generando embeddings para {len(chunks)} chunks...")
            all_embeddings = await embedding_fn_batch_local(chunks)

            if not all_embeddings:
                logging.error("No se generaron embeddings")
                return False

            logging.info(f"Total embeddings: {len(all_embeddings)}")

        except Exception as e:
            logging.error(f"Error generando embeddings en lote: {e}")
            import traceback
            logging.error(f"Traceback completo: {traceback.format_exc()}")
            return False

        # Paso 3: Validar que tenemos embeddings para todos los chunks
        if len(all_embeddings) != len(chunks):
            logging.error(f"Mismatch: {len(chunks)} chunks pero {len(all_embeddings)} embeddings")
            return False

        # Paso 4: Insertar chunks con ESTRUCTURA UNIFICADA
        successful_insertions = 0
        failed_insertions = 0
        total_chunks = len(chunks)
        file_id = str(metadata.get('file_id', f'file_{uuid.uuid4().hex[:8]}'))
        filename = str(metadata.get('file_name', metadata.get('filename', 'unknown')))

        # ── EWS: cryptographic identity of the document ────────────────
        # Estas variables (doc_hash/doc_signature/indexed_at/doc_version) se usan
        # al construir cada data_object more abajo. En el path PDF las define
        # insert_document(); here (Word/Text path) they must be defined too, otherwise
        # cada insertion lanza NameError, la clase queda creada con 0 chunks y el .docx
        # it never gets indexed. The doc_hash is computed over the full text (concatenation
        # de chunks) con firma EWS opcional y fallback seguro.
        try:
            _full_text = "\n".join(
                c if isinstance(c, str) else str(c.get('text', c.get('content', c)))
                for c in chunks
            )
            from framework.compliance.doc_identity import prepare_doc_identity
            _ews = prepare_doc_identity(
                full_text=_full_text,
                filename=filename,
                tenant_id=company,
                weaviate_client=self.client,
                collection_name=collection_name,
            )
            doc_hash       = _ews["doc_hash"]
            doc_signature  = _ews["doc_signature"]
            indexed_at     = _ews["indexed_at"]
            indexed_at_iso = _ews["indexed_at_iso"]
            doc_version    = _ews["doc_version"]
        except Exception as _ews_err:
            logging.warning(f"EWS doc_identity failed (non-blocking): {_ews_err}")
            import time as _t, hashlib as _hlib
            from datetime import datetime as _dt, timezone as _tz
            _full_text = "\n".join(
                c if isinstance(c, str) else str(c.get('text', c.get('content', c)))
                for c in chunks
            )
            doc_hash       = _hlib.sha256(_full_text.encode('utf-8')).hexdigest()
            doc_signature  = ""
            indexed_at     = _t.time()
            indexed_at_iso = _dt.fromtimestamp(indexed_at, tz=_tz).isoformat()
            doc_version    = 1

        for i, chunk in enumerate(chunks):
            try:
                # Obtener y validar vector
                vector = all_embeddings[i]

                if isinstance(vector, np.ndarray):
                    vector = vector.tolist()
                elif not isinstance(vector, list):
                    vector = list(vector)

                vector = [float(x) for x in vector]

                if len(vector) == 0:
                    logging.error(f"[Chunk {i}] Vector is empty")
                    failed_insertions += 1
                    continue

                # Validar chunk de texto
                if isinstance(chunk, str):
                    text = chunk.strip()
                elif isinstance(chunk, dict):
                    text = str(chunk.get('text', chunk.get('content', ''))).strip()
                else:
                    text = str(chunk).strip()

                if not text:
                    logging.error(f"[Chunk {i}] Empty text chunk")
                    failed_insertions += 1
                    continue

                # ═══════════════════════════════════════════════════════
                # CONSTRUIR OBJETO CON ESTRUCTURA UNIFICADA (igual que PDF)
                # ═══════════════════════════════════════════════════════
                chunk_id = f"chunk-{file_id}_{uuid.uuid4().hex[:8]}_{i}"

                # Detectar idioma simple
                def detect_language(txt):
                    spanish_words = ['el', 'la', 'de', 'que', 'en', 'los', 'del', 'las', 'un', 'por', 'con', 'una',
                                     'su', 'para', 'es', 'al', 'lo', 'como', 'more', 'pero', 'sus', 'le', 'ya', 'o',
                                     'este', 'sí', 'porque', 'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre',
                                     'also', 'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos',
                                     'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante',
                                     'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro',
                                     'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada',
                                     'muchos', 'cual', 'poco', 'ella', 'estar']
                    words = txt.lower().split()[:100]
                    spanish_count = sum(1 for w in words if w in spanish_words)
                    return 'es' if spanish_count > len(words) * 0.1 else 'en'

                data_object = {
                    # BASIC FIELDS
                    "name": chunk_id,
                    "text": text,
                    "user": str(user),
                    "company": str(company),

                    # METADATA DEL ARCHIVO
                    "file_id": file_id,
                    "source": filename,
                    "filename": filename,
                    "doc_type": str(metadata.get('tipo_doc', metadata.get('doc_type', 'document'))),

                    # METADATA DEL CHUNK
                    "chunk_length": len(text),
                    "word_count": len(text.split()),
                    "content_type": "text",
                    "has_numbers": bool(any(c.isdigit() for c in text)),
                    "digit_density": sum(c.isdigit() for c in text) / len(text) if text else 0.0,

                    # REFERENCIAS
                    "referencias_text": referencias_text,
                    "referencias": referencias_value,  # ✅ Adaptado al schema

                    # TEMPORAL
                    "created_at": time.time(),
                    "created_date": datetime.now().isoformat(),

                    # EWS — cryptographic identity of the document (ETICS L1)
                    "doc_hash":      doc_hash,       # sha256 del texto completo
                    "doc_signature": doc_signature,  # firma BIP32 del tenant
                    "indexed_at":    indexed_at,     # Unix timestamp — activa BVG FreshnessVerifier
                    "indexed_at_iso": indexed_at_iso,
                    "doc_version":   doc_version,    # 1 = new, 2+ = update
                    "superseded":    False,           # True when a newer version exists

                    # POSITION
                    "chunk_index": i,
                    "total_chunks": total_chunks,

                    # NAVIGATION
                    "previous_chunk_id": f"chunk-{file_id}_{i - 1}" if i > 0 else "",
                    "next_chunk_id": f"chunk-{file_id}_{i + 1}" if i < total_chunks - 1 else "",

                    # CONTENIDO
                    "language": detect_language(text),
                    "text_preview": text[:200] + "..." if len(text) > 200 else text,
                    "text_hash": hashlib.md5(text.encode('utf-8')).hexdigest(),

                    # PROCESO
                    "schema_version": "1.1",
                    "processing_time": 0.0,
                    "embedding_model": "text-embedding-3-large",

                    # SOFT DELETE
                    "is_deleted": False,
                    "deleted_at": "",
                    "deleted_by": "",
                    "deletion_reason": "",

                    # AUDIT
                    "uploaded_by": str(user),
                    "uploaded_by_name": str(user),
                    "upload_ip": "",

                    # BACKUP
                    "metadata_json": json.dumps(metadata, ensure_ascii=False)
                }

                # Insertar en Weaviate
                self.client.data_object.create(
                    data_object=data_object,
                    class_name=collection_name,
                    vector=vector
                )

                successful_insertions += 1
                logging.info(f"[Chunk {i}] Guardado exitosamente")

            except Exception as e:
                logging.error(f"[Chunk {i}] Error: {e}")
                failed_insertions += 1
                continue

        # Resumen final
        success_rate = (successful_insertions / total_chunks) * 100 if total_chunks > 0 else 0

        logging.info(f"Resumen final:")
        logging.info(f"  Total chunks: {total_chunks}")
        logging.info(f"  Exitosos: {successful_insertions}")
        logging.info(f"  Fallidos: {failed_insertions}")
        logging.info(f"  Success rate: {success_rate:.1f}%")

        if successful_insertions == 0:
            logging.error("Could not save any chunk")
            return False

        return True

    def image_to_gemini_image_part(self, image: Image.Image) -> dict:
        """
        Convierte una imagen PIL a un objeto inline_data compatible con Gemini.

        Args:
            image (Image.Image): Imagen PIL (por ejemplo, una page de PDF convertida).

        Returns:
            dict: Objeto inline_data con MIME y bytes, para usar en Gemini.
        """
        buf = BytesIO()
        image.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        return {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": image_bytes
            }
        }


    def _process_page_with_gemini(self, img, page_number: int) -> Dict:
        """
        Process a single page with Gemini (synchronous operation).
        """
        try:
            import google.generativeai as genai

            # Convertir imagen a imagen Gemini compatible
            image_part = self.image_to_gemini_image_part(img)

            # Construir prompt OCR estructurado
            ocr_prompt = f"""
                You are an expert OCR reader. Extract ALL the text from this image with maximum fidelity.
            
                - Keep structure, headings, tables and columns.
                - If there is a table (products, items, etc.), render it as Markdown or a tabulated layout.
                - Do not fill in data that is not present.
                - Preserve the exact format of dates and amounts as they appear.
                - Language: Spanish.
                """

            model = genai.GenerativeModel("gemini-2.5-pro")
            response = model.generate_content([{"text": ocr_prompt}, image_part])

            text = response.text.strip()

            if not text:
                logging.warning(f"⚠️ Page {page_number} returned empty")
                return {
                    'page_number': page_number,
                    'success': False,
                    'text': '',
                    'error': 'Empty response'
                }
            else:
                print(f"📄 Page text {page_number}:\n{text[:100]}...")
                return {
                    'page_number': page_number,
                    'success': True,
                    'text': text,
                    'error': None
                }

        except Exception as e:
            return {
                'page_number': page_number,
                'success': False,
                'text': '',
                'error': str(e)
            }

    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(3)
    )
    async def _process_large_pdf_azure_parallel(
            self,
            pdf_binary: bytes,
            max_pages_per_call: int = 50
    ) -> Dict:
        """
        Large PDF dividido en chunks processeds EN PARALELO

        This function replaces _process_large_pdf_azure_with_retry
        by processing multiple chunks concurrently.
        """
        try:
            # Abrir PDF y contar pages
            pdf_doc = pdfium.PdfDocument(pdf_binary)
            total_pages = len(pdf_doc)

            # If it is small, process it whole
            if total_pages <= max_pages_per_call:
                pdf_doc.close()
                return await self._process_full_pdf_azure_with_retry(pdf_binary)

            print(f"📄 Large PDF: {total_pages} pages")
            print(f"🔀 Splitting into chunks of {max_pages_per_call} pages...")

            # Crear chunks
            chunks_info = []
            for start_page in range(0, total_pages, max_pages_per_call):
                end_page = min(start_page + max_pages_per_call, total_pages)
                chunk_id = f"Chunk-{start_page + 1}-{end_page}"

                chunks_info.append({
                    "start_page": start_page,
                    "end_page": end_page,
                    "chunk_id": chunk_id
                })

            print(f"📦 Total chunks: {len(chunks_info)}")

            # Semaphore to limit concurrency
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)

            # Crear tareas para todos los chunks
            tasks = []
            for chunk_info in chunks_info:
                # Extraer pages del chunk
                chunk_pdf_bytes = self._extract_page_range(
                    pdf_binary,
                    chunk_info["start_page"],
                    chunk_info["end_page"]
                )

                # Crear tarea
                task = self._process_chunk_with_semaphore(
                    semaphore,
                    chunk_pdf_bytes,
                    chunk_info["chunk_id"],
                    chunk_info["start_page"]
                )
                tasks.append(task)

            # 🚀 EJECUTAR EN PARALELO
            print(f"🚀 Processing {len(tasks)} chunks in parallel (max {self.max_concurrent_requests} concurrent)...")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            pdf_doc.close()

            # Procesar resultados
            all_text_chunks = []
            all_failed_pages = []
            failed_chunks = []

            # Ordenar por start_page para mantener orden
            valid_results = [r for r in results if isinstance(r, dict) and r.get("success")]
            valid_results.sort(key=lambda x: x["start_page"])

            for result in valid_results:
                if result["text"]:
                    all_text_chunks.append(result["text"])
                all_failed_pages.extend(result["failed_pages"])

            # Detectar chunks fallidos
            for result in results:
                if isinstance(result, dict) and not result.get("success"):
                    failed_chunks.append(result["chunk_id"])
                elif isinstance(result, Exception):
                    failed_chunks.append(f"Error: {str(result)}")

            final_text = "\n".join(all_text_chunks)

            print(f"✅ OCR paralelo finished:")
            print(f"   - Caracteres: {len(final_text)}")
            print(f"   - Pages: {total_pages - len(all_failed_pages)}/{total_pages}")
            print(f"   - Chunks OK: {len(valid_results)}/{len(chunks_info)}")

            if failed_chunks:
                print(f"   ⚠️ Chunks fallidos: {failed_chunks}")

            return {
                "text": final_text,
                "failed_pages": all_failed_pages,
                "failed_chunks": failed_chunks,
                "total_pages": total_pages,
                "processed_pages": total_pages - len(all_failed_pages)
            }

        except Exception as e:
            logging.error(f"Error procesando PDF: {e}")
            return {
                "text": "",
                "failed_pages": ["LARGE_PDF_ERROR"],
                "error": str(e)
            }

    async def extract_azure_with_handler_errors(self, pdf_binary: bytes, max_pages_per_call=500):
        """
        Optimized: sends the full PDF to Azure Document Intelligence with rate limiting
        """
        try:
            # Rate limiting integrado
            async with self._semaphore:
                await self._apply_rate_limiting()

                # Option 1: full PDF to Azure (MORE EFFICIENT)
                if len(pdf_binary) < 50_000_000:  # Azure limit ~50MB
                    return await self._process_full_pdf_azure_with_retry(pdf_binary)

                # Option 2: if the PDF is very large, split by pages
                return await self._process_large_pdf_azure_parallel(pdf_binary, max_pages_per_call)

        except Exception as e:
            logging.error(f"Error procesando PDF: {e}")
            return {"text": "", "failed_pages": ["GENERAL_ERROR"]}

    async def _apply_rate_limiting(self):
        """Aplica rate limiting inteligente"""
        current_time = time.time()

        # Reset contador cada minuto
        if current_time - self._last_reset > 60:
            self._call_count = 0
            self._last_reset = current_time

        self._call_count += 1

        # Estrategia de rate limiting por llamadas
        if self._call_count <= 20:
            await asyncio.sleep(0.1)  # Fast start
        elif self._call_count <= 50:
            await asyncio.sleep(0.5)  # Moderado
        elif self._call_count <= 80:
            await asyncio.sleep(1.0)  # Slower
        else:
            # Long pause near the limit
            print(f"Pausa preventiva: llamada #{self._call_count}")
            await asyncio.sleep(5.0)

        # Pausa extra cada 90 llamadas
        if self._call_count % 90 == 0:
            print("Pausa de cooldown de rate limiting...")
            await asyncio.sleep(15.0)

    async def _process_full_pdf_azure_with_retry(self, pdf_binary: bytes, chunk_id: str = ""):
        """Full PDF with automatic retry."""
        try:
            self._call_count += 1
            client = DocumentAnalysisClient(
                endpoint=self.azure_endpoint,
                credential=AzureKeyCredential(self.azure_key.key)
            )

            chunk_info = f" [{chunk_id}]" if chunk_id else ""
            print(f"[{self._call_count}] Enviando PDF{chunk_info} a Azure...")

            poller = client.begin_analyze_document(
                model_id="prebuilt-read",
                document=pdf_binary
            )
            result = poller.result(timeout=300)

            # Procesar resultado
            text_chunks = []
            failed_pages = []
            total_pages = len(result.pages) if result.pages else 0

            for page_num, page in enumerate(result.pages, 1):
                try:
                    page_text = ""
                    if hasattr(page, 'lines') and page.lines:
                        page_text = "\n".join([line.content for line in page.lines])
                    if page_text.strip():
                        text_chunks.append(page_text)
                    else:
                        failed_pages.append(page_num)
                except Exception as e:
                    logging.warning(f"Error on page {page_num}: {e}")
                    failed_pages.append(page_num)

            final_text = "\n".join(text_chunks)
            print(f"[{chunk_id or 'PDF'}] OCR finished: {len(final_text)} characters de {total_pages} pages")

            return {
                "text": final_text,
                "failed_pages": failed_pages,
                "chunk_id": chunk_id,
                "total_pages": total_pages
            }

        except Exception as e:
            error_msg = str(e).lower()
            if any(term in error_msg for term in ['rate', 'throttle', 'quota', 'limit']):
                print(f"[{chunk_id}] Rate limit detected: {e}")
                await asyncio.sleep(10.0)  # Pausa antes del retry
                raise  # Trigger retry
            else:
                logging.error(f"[{chunk_id}] Error no recuperable: {e}")
                raise

    async def _process_chunk_with_semaphore(
            self,
            semaphore: asyncio.Semaphore,
            chunk_pdf_bytes: bytes,
            chunk_id: str,
            start_page: int
    ) -> Dict:
        """
        Process a PDF chunk with concurrency control using a semaphore
        """
        async with semaphore:  # Limita requests concurrentes
            try:
                result = await self._process_full_pdf_azure_with_retry(
                    chunk_pdf_bytes,
                    chunk_id=chunk_id
                )

                # Adjust page numbers relative to the full document
                adjusted_failed = [fp + start_page for fp in result["failed_pages"]
                                   if isinstance(fp, int)]

                return {
                    "text": result["text"],
                    "failed_pages": adjusted_failed,
                    "chunk_id": chunk_id,
                    "start_page": start_page,
                    "success": True
                }

            except Exception as e:
                logging.error(f"Error procesando chunk {chunk_id}: {e}")
                return {
                    "text": "",
                    "failed_pages": [],
                    "chunk_id": chunk_id,
                    "start_page": start_page,
                    "success": False,
                    "error": str(e)
                }

    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(3)
    )
    async def _process_large_pdf_azure_parallel(
            self,
            pdf_binary: bytes,
            max_pages_per_call: int = 50
    ) -> Dict:
        """
        Large PDF dividido en chunks processeds EN PARALELO

        Args:
            pdf_binary: PDF en bytes
            max_pages_per_call: Max pages per chunk (default 50)

        Returns:
            Dict with extracted text and failed pages
        """
        try:
            # Abrir PDF para contar pages
            pdf_doc = pdfium.PdfDocument(io.BytesIO(pdf_binary))
            total_pages = len(pdf_doc)

            # If it is small, process it whole
            if total_pages <= max_pages_per_call:
                pdf_doc.close()
                return await self._process_full_pdf_azure_with_retry(pdf_binary)

            print(f"📄 Large PDF detected: {total_pages} pages")
            print(f"🔀 Splitting into chunks of {max_pages_per_call} pages...")

            # Crear lista de chunks a procesar
            chunks_info = []
            for start_page in range(0, total_pages, max_pages_per_call):
                end_page = min(start_page + max_pages_per_call, total_pages)
                chunk_id = f"Chunk-{start_page + 1}-{end_page}"

                chunks_info.append({
                    "start_page": start_page,
                    "end_page": end_page,
                    "chunk_id": chunk_id
                })

            print(f"📦 Total de chunks a procesar: {len(chunks_info)}")

            # Create a semaphore to limit concurrency
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)

            # Crear todas las tareas de procesamiento
            tasks = []
            for chunk_info in chunks_info:
                # Extraer pages del chunk
                chunk_pdf_bytes = self._extract_page_range(
                    pdf_binary,
                    chunk_info["start_page"],
                    chunk_info["end_page"]
                )

                # Crear tarea asynchronous
                task = self._process_chunk_with_semaphore(
                    semaphore,
                    chunk_pdf_bytes,
                    chunk_info["chunk_id"],
                    chunk_info["start_page"]
                )
                tasks.append(task)

            # Ejecutar todos los chunks EN PARALELO
            print(f"🚀 Processing {len(tasks)} chunks in parallel (max {self.max_concurrent_requests} concurrent)...")

            results = await asyncio.gather(*tasks, return_exceptions=True)

            pdf_doc.close()

            # Procesar resultados
            all_text_chunks = []
            all_failed_pages = []
            failed_chunks = []

            # Ordenar resultados por start_page para mantener orden correcto
            valid_results = [r for r in results if isinstance(r, dict) and r.get("success")]
            valid_results.sort(key=lambda x: x["start_page"])

            for result in valid_results:
                if result["text"]:
                    all_text_chunks.append(result["text"])
                all_failed_pages.extend(result["failed_pages"])

            # Detectar chunks fallidos
            for result in results:
                if isinstance(result, dict) and not result.get("success"):
                    failed_chunks.append(result["chunk_id"])
                elif isinstance(result, Exception):
                    failed_chunks.append(f"Error: {str(result)}")

            final_text = "\n".join(all_text_chunks)

            print(f"✅ OCR paralelo finished:")
            print(f"   - Total characters: {len(final_text)}")
            print(f"   - Pages processeds: {total_pages - len(all_failed_pages)}/{total_pages}")
            print(f"   - Chunks exitosos: {len(valid_results)}/{len(chunks_info)}")

            if failed_chunks:
                print(f"   ⚠️ Chunks fallidos: {failed_chunks}")

            return {
                "text": final_text,
                "failed_pages": all_failed_pages,
                "failed_chunks": failed_chunks,
                "total_pages": total_pages,
                "processed_pages": total_pages - len(all_failed_pages)
            }

        except Exception as e:
            logging.error(f"Error procesando Large PDF in parallel: {e}")
            return {
                "text": "",
                "failed_pages": ["LARGE_PDF_ERROR"],
                "error": str(e)
            }

    def _extract_page_range(self, pdf_binary: bytes, start_page: int, end_page: int) -> bytes:
        """
        Extrae un rango de pages de un PDF y retorna un nuevo PDF

        Args:
            pdf_binary: PDF original en bytes
            start_page: Page inicial (0-indexed)
            end_page: Page final (exclusivo, 0-indexed)

        Returns:
            bytes: Nuevo PDF con solo las pages seleccionadas
        """
        try:
            # Abrir PDF original
            src_pdf = pdfium.PdfDocument(io.BytesIO(pdf_binary))

            # Crear nuevo PDF
            dst_pdf = pdfium.PdfDocument.new()

            # Importar pages del rango
            dst_pdf.import_pages(src_pdf, list(range(start_page, end_page)))

            # Guardar a bytes
            output_buffer = io.BytesIO()
            dst_pdf.save(output_buffer)
            output_bytes = output_buffer.getvalue()

            # Cerrar documentos
            src_pdf.close()
            dst_pdf.close()
            output_buffer.close()

            return output_bytes

        except Exception as e:
            logging.error(f"Error extracting pages {start_page}-{end_page}: {e}")
            raise

    async def process_pdf(self, pdf_binary: bytes, max_pages_per_call: int = 50) -> Dict:
        """
        Main method to process a PDF (automatically parallel if large)

        Args:
            pdf_binary: PDF en bytes
            max_pages_per_call: Pages por chunk para PDFs grandes

        Returns:
            Dict with extracted text
        """
        try:
            # Contar pages
            pdf_doc = pdfium.PdfDocument(io.BytesIO(pdf_binary))
            total_pages = len(pdf_doc)
            pdf_doc.close()

            print(f"📄 PDF received: {total_pages} pages")

            # Decidir estrategia
            if total_pages <= max_pages_per_call:
                print("📝 Processing PDF completo...")
                return await self._process_full_pdf_azure_with_retry(pdf_binary)
            else:
                print("🔀 Processing Large PDF in parallel...")
                return await self._process_large_pdf_azure_parallel(
                    pdf_binary,
                    max_pages_per_call
                )

        except Exception as e:
            logging.error(f"Error en process_pdf: {e}")
            return {
                "text": "",
                "failed_pages": ["PDF_PROCESSING_ERROR"],
                "error": str(e)
            }

    def _extract_page_range_(self, pdf_binary: bytes, start_page: int, end_page: int) -> bytes:
        """Extrae rango de pages y devuelve PDF en bytes"""
        import pypdfium2 as pdfium

        # Abrir PDF original
        source_pdf = pdfium.PdfDocument(io.BytesIO(pdf_binary))

        # Crear nuevo PDF con pages seleccionadas
        new_pdf = pdfium.PdfDocument.new()

        for page_idx in range(start_page, end_page):
            if page_idx < len(source_pdf):
                page = source_pdf.get_page(page_idx)
                new_pdf.import_pages(source_pdf, [page_idx])

        # Convertir a bytes
        output_buffer = io.BytesIO()
        new_pdf.save(output_buffer)
        pdf_bytes = output_buffer.getvalue()

        # Limpiar
        source_pdf.close()
        new_pdf.close()
        output_buffer.close()

        return pdf_bytes

    async def _process_full_pdf_azure(self, pdf_binary: bytes):
        """Procesa PDF completo en una sola llamada a Azure"""
        try:

            # Inicializar cliente (mejor hacerlo una vez en __init__)
            client = DocumentAnalysisClient(
                endpoint=self.azure_endpoint,
                credential=AzureKeyCredential(
                    self.azure_key)
            )

            print(f"🚀 Enviando PDF completo a Azure Document Intelligence...")

            # Una sola llamada para todo el PDF
            # Sintaxis correcta para DocumentAnalysisClient
            poller = client.begin_analyze_document(
                model_id="prebuilt-layout",  # o "prebuilt-document"
                document=pdf_binary  # The parameter is called 'document', not 'analyze_request'
            )

            result = poller.result()

            # Extraer texto de todas las pages
            text_chunks = []
            failed_pages = []
            total_pages = len(result.pages) if result.pages else 0

            print(f"📄 Processing {total_pages} pages...")

            for page_num, page in enumerate(result.pages, 1):
                try:
                    page_text = ""
                    if hasattr(page, 'lines') and page.lines:
                        page_text = "\n".join([line.content for line in page.lines])

                    if page_text.strip():
                        text_chunks.append(page_text)
                        print(f"✅ Page {page_num}: {len(page_text)} characters")
                    else:
                        failed_pages.append(page_num)

                except Exception as e:
                    logging.warning(f"⚠️ Error on page {page_num}: {e}")
                    failed_pages.append(page_num)

            final_text = "\n".join(text_chunks)
            print(f"✅ OCR finished: {len(final_text)} characters de {total_pages} pages")

            return {
                "text": final_text,
                "failed_pages": failed_pages
            }

        except Exception as e:
            logging.error(f"❌ Error Azure Document Intelligence: {e}")
            return {"text": "", "failed_pages": ["AZURE_ERROR"]}

    async def _process_large_pdf_azure(self, pdf_binary: bytes, max_pages_per_call=500):
        """Para PDFs muy grandes: dividir en chunks de pages"""
        try:
            from PyPDF2 import PdfReader, PdfWriter
            import io

            # Leer PDF y dividir en chunks
            reader = PdfReader(io.BytesIO(pdf_binary))
            total_pages = len(reader.pages)

            print(f"📄 Large PDF detected: {total_pages} pages, dividiendo en chunks...")

            text_chunks = []
            failed_pages = []

            # Procesar en chunks de max_pages_per_call pages
            for start_page in range(0, total_pages, max_pages_per_call):
                end_page = min(start_page + max_pages_per_call, total_pages)

                print(f"🚀 Processing pages {start_page + 1} a {end_page}...")

                # Crear PDF chunk
                writer = PdfWriter()
                for page_num in range(start_page, end_page):
                    writer.add_page(reader.pages[page_num])

                # Convertir chunk a bytes
                chunk_buffer = io.BytesIO()
                writer.write(chunk_buffer)
                chunk_buffer.seek(0)
                chunk_bytes = chunk_buffer.getvalue()

                # Procesar chunk con Azure
                chunk_result = await self._process_full_pdf_azure(chunk_bytes)

                if chunk_result['text']:
                    text_chunks.append(chunk_result['text'])

                # Adjust failed page numbers
                for failed_page in chunk_result['failed_pages']:
                    if isinstance(failed_page, int):
                        failed_pages.append(failed_page + start_page)
                    else:
                        failed_pages.append(failed_page)

            final_text = "\n".join(text_chunks)
            return {
                "text": final_text,
                "failed_pages": failed_pages
            }

        except Exception as e:
            logging.error(f"❌ Error procesando Large PDF: {e}")
            return {"text": "", "failed_pages": ["LARGE_PDF_ERROR"]}

    async def extract_azure(self, pdf_binary: bytes, dpi=150, batch_size=10):
        """
        Convierte un PDF binario a texto utilizando OCR asynchronous con Azure.
        """
        try:
            from pdf2image import convert_from_bytes
            import io

            # Step 1: Convert PDF to images
            images = convert_from_bytes(pdf_binary, dpi=dpi)
            total_pages = len(images)
            print(f"📄 Total pages detected: {total_pages}")

            text_chunks = []

            # Paso 2: Procesar en lotes
            for i in range(0, total_pages, batch_size):
                batch_images = images[i:i + batch_size]
                print(f"🚀 Processing pages {i + 1} a {i + len(batch_images)}...")

                image_bytes_batch = []
                for idx, img in enumerate(batch_images):
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    image_bytes_batch.append(buf.getvalue())

                # Paso 3: Ejecutar OCR asynchronous
                tasks = [self.process_image_azure(img_bytes) for img_bytes in image_bytes_batch]
                ocr_results = await asyncio.gather(*tasks)

                for idx, res in enumerate(ocr_results):
                    print(f"📄 Page text {i + idx + 1}:\n{res[:100]}...")  # Shows the first characters
                    if res.strip():
                        text_chunks.append(res)

            final_text = "\n".join(text_chunks)
            print(f"✅ OCR finished. Total characters extracted: {len(final_text)}")
            return final_text

        except Exception as e:
            logging.error(f"❌ Error procesando el PDF: {e}")
            return ""

    def detect_tabla_heuristica(self,text: str) -> bool:
        if not text:
            return False
        lines = text.splitlines()
        count_verticals = sum(line.count('|') for line in lines)
        count_plus = text.count('+')
        count_dashes = text.count('-')

        return count_verticals >= 3 or count_plus >= 3 or count_dashes > 10
    async def detect_tables_azure(self, pdf_binary: bytes, dpi=150, batch_size=10):
        """
        Detecta tablas en un PDF utilizando OCR asynchronous con Azure.
        Return a list with per-page info: number, text, and whether it contains a table.
        """
        try:
            from pdf2image import convert_from_bytes
            import io

            # Step 1: Convert the binary PDF to images
            images = convert_from_bytes(pdf_binary, dpi=dpi)
            total_pages = len(images)
            print(f"📄 Total pages detected: {total_pages}")

            results = []

            # Paso 2: Procesar en lotes
            for i in range(0, total_pages, batch_size):
                batch_images = images[i:i + batch_size]
                print(f"🚀 Processing pages {i + 1} a {i + len(batch_images)}...")

                image_bytes_batch = []
                for img in batch_images:
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    image_bytes_batch.append(buf.getvalue())

                # Paso 3: Ejecutar OCR asynchronous
                tasks = [self.process_image_azure(img_bytes) for img_bytes in image_bytes_batch]
                ocr_results = await asyncio.gather(*tasks)

                # Step 4: Heuristic to detect tables
                for idx, text in enumerate(ocr_results):
                    num_pagina = i + idx + 1
                    tiene_tabla = self.detect_tabla_heuristica(text)
                    results.append({
                        "pagina": num_pagina,
                        "tiene_tabla": tiene_tabla,
                        "texto": text.strip()
                    })
                    print(f"📄 Page {num_pagina}: {'✅ Table detected' if tiene_tabla else '❌ No table'}")

            return results

        except Exception as e:
            logging.error(f"❌ Error procesando el PDF: {e}")
            return []

    async def process_image_azure(self, image_bytes: bytes) -> str:
        """
        Procesa una imagen (en formato de bytes) usando Azure AI Document Intelligence.

        Args:
            image_bytes (bytes): La imagen en formato PNG ya convertida a bytes.

        Returns:
            str: The text extracted from the image.
        """
        try:
            # Convertir los bytes en un stream para Azure
            img_stream = io.BytesIO(image_bytes)

            # Llamar a Azure OCR
            poller = self.azure_client.begin_analyze_document(
                model_id="prebuilt-layout",
                document=img_stream
            )
            result = poller.result()

            # Extraer el texto
            extracted_text = "\n".join(
                line.content for page in result.pages for line in page.lines
            )

            return extracted_text

        except Exception as e:
            logging.error(f"❌ Error procesando la imagen con Azure OCR: {e}")
            return ""

    async def remember_interaction(self, user:str = None,
                                   user_message:str = None,
                                   bot_response:str = None,
                                   conversation_id: str = None,
                                   timestamp=datetime.datetime.utcnow(),
                                   config_chatbot ="documental",
                                   files_ids=None
  ):
        """
        Store interactions in MongoDB associated with a user and a conversation identifier.

        Args:
            user (str): El identificador del user.
            user_message (str): Mensaje del user.
            bot_response (str): Respuesta generada por el bot.
            conversation_id (str): Unique conversation identifier. If not provided, a new one is generated.
        """
        interaction = {
            "user": user,
            "conversation_id": conversation_id,  # Associate the interaction with a unique conversation identifier
            "user_message": user_message,
            "bot_response": bot_response,
            "timestamp": timestamp,  # Llamar directamente
            "tipo_chatbot":config_chatbot,
            "files_ids": files_ids,
        }

        # Governance Ola 4: redactar PII AR antes de persistir si la policy del
        # tenant lo pide. Fail-open: si no se puede leer la policy o redactar, se
        # guarda el original (no perder datos ni romper el guardado existente).
        try:
            from framework.Audit import policy as _audit_policy
            _pol = _audit_policy.get_policy(getattr(self, "company", None),
                                            getattr(self, "config", None))
            if _pol.get("redactPII_on_store"):
                from framework.Anonimizator import pii_ar
                interaction = pii_ar.redact_dict(interaction)
        except Exception as _e:  # noqa: BLE001
            logging.debug(f"redactPII_on_store skip (fail-open): {_e}")

        # Make sure `self.collection` is a synchronous MongoDB instance.
        await asyncio.to_thread(self.collection.insert_one, interaction)

        return conversation_id  # Retornar el conversation_id para su uso posterior

    async def get_conversation_history(
            self,
            conversation_id: str,
            limit: int = 10
    ) -> list:
        if not conversation_id:
            return []

        def _query():
            # RECENT memory window: sort DESC + limit brings the last N
            # turns (it was ASC+limit before => returned the N OLDEST). It is reversed
            # afterwards to chronological order to build the history.
            return self.collection.find_all_chatbot(
                filtro={"conversation_id": conversation_id},
                proyeccion={"_id": 0, "user_message": 1, "bot_response": 1, "timestamp": 1},
                sort=[("timestamp", -1)],
                limit=limit
            )

        docs = await asyncio.to_thread(_query)
        docs = list(reversed(docs or []))

        history = []
        for doc in docs:
            if doc.get("user_message"):
                history.append({"role": "user", "content": doc["user_message"]})
            if doc.get("bot_response"):
                history.append({"role": "assistant", "content": doc["bot_response"]})

        return history

    def get_collections_by_user_and_company(self, user: str, company: str):
        """
        Return all collections (classes) in Weaviate that contain documents associated with a specific user and company.
        """
        try:
            # Obtener el esquema actual de Weaviate
            schema = self.client.schema.get()

            # Filtrar las clases que contienen los documentos del user/company.
            # FIX: the real property is 'user' y los campos son de tipo
            # 'text' → debe filtrarse con 'valueText' (no 'valueString'). Antes esto
            # NUNCA matcheaba, por lo que el borrado no limpiaba Weaviate.
            matching_classes = []
            for cls in schema['classes']:
                class_properties = cls.get('properties', [])
                has_user_property = any(prop.get('name') == 'user' for prop in class_properties)
                has_company_property = any(prop.get('name') == 'company' for prop in class_properties)
                if has_user_property and has_company_property:
                    response = self.client.query.get(cls['class'], ["user", "company"]) \
                        .with_where({
                        "operator": "And",
                        "operands": [
                            {"path": ["user"], "operator": "Equal", "valueText": user},
                            {"path": ["company"], "operator": "Equal", "valueText": company},
                        ]
                    }).do()

                    if response.get('data', {}).get('Get', {}).get(cls['class']):
                        matching_classes.append(cls['class'])

            # Sin coincidencias → lista empty (el caller lo maneja como "nada que borrar").
            return matching_classes

        except Exception as e:
            raise Exception(f"Error al obtener las colecciones: {str(e)}")



    def delete_all_occurrences_from_weaviate(self, user: str, company: str):
        """
        Delete all occurrences in Weaviate of the collections associated with a specific user and company
        y elimina las clases (colecciones) si quedan emptys.
        """
        try:
            # Obtener las colecciones (clases) en Weaviate asociadas al user y company
            collections = self.get_collections_by_user_and_company(user, company)

            if not collections:
                return {
                    "message": f"No hay datos que eliminar para el user '{user}', company '{company}'."}

            deleted_count = 0

            # Delete documents in each found collection/class.
            # FIX: real property 'user', type 'text' -> must use 'valueText'.
            for collection in collections:
                delete_response = self.client.batch.delete_objects(
                    class_name=collection,
                    where={
                        "operator": "And",
                        "operands": [
                            {"path": ["user"], "operator": "Equal", "valueText": user},
                            {"path": ["company"], "operator": "Equal", "valueText": company},
                        ]
                    }
                )
                deleted_count += delete_response.get("results", {}).get("successful", 0)

                # Check whether the class is empty and delete it if necessary.
                remaining_records = self.client.query.get(collection, ["user"]).with_limit(1).do()
                if not remaining_records.get('data', {}).get('Get', {}).get(collection, []):
                    self.client.schema.delete_class(collection)

            return {"message": f"Se eliminaron {deleted_count} documentos y {len(collections)} clase(s) en Weaviate."}

        except Exception as e:
            return {"message": f"Error al eliminar documentos y clases en Weaviate: {str(e)}"}

    # si
    def delete_collection(self, user: str, company: str) -> str:
        """
        Delete a class (collection) in Weaviate based on the generated class name
        for the company. It also validates that it contains related documents
        con el user especificado.
        """
        try:
            # Crear el nombre de la clase basado en company
            class_name = re.sub(r"[^a-zA-Z0-9_]", "_", (company or "").upper().replace(" ", "_"))

            # Obtener el esquema actual de Weaviate
            schema = self.client.schema.get()

            # Verificar si la clase existe antes de intentar eliminar
            class_exists = any(cls['class'] == class_name for cls in schema['classes'])

            if not class_exists:
                return f"La clase '{class_name}' no existe en Weaviate."

            # Realizar una consulta en la clase para verificar si existen objetos con los criterios proporcionados
            response = self.client.query.get(class_name, ["user", "company"]) \
                .with_where({
                "operator": "And",
                "operands": [
                    {
                        "path": ["user"],
                        "operator": "Equal",
                        "valueString": user
                    },
                    {
                        "path": ["company"],
                        "operator": "Equal",
                        "valueString": company
                    }
                ]
            }).do()

            # Si se encuentran objetos que coinciden con los criterios, eliminar la clase
            if response['data']['Get'][class_name]:
                self.client.schema.delete_class(class_name)
                return f"Deleted collection '{class_name}' related to user '{user}' y la company '{company}'."
            else:
                return f"No se encontraron documentos en la clase '{class_name}' que coincidan con los criterios proporcionados."

        except Exception as e:
            return f"Error deleting the collection: {str(e)}"



    def detect_document_type_sync(self, text: str) -> str:
        def run_async():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(self.detect_document_type(text))
            except Exception as e:
                logger.error(f"Error en detect_document_type (async): {e}")
                raise RuntimeError("Failed to detect the document type asynchronously.") from e
            finally:
                new_loop.close()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_async)
                return future.result()
        except Exception as e:
            logger.error(f"Error en detect_document_type_sync: {e}")
            raise RuntimeError("Failed in synchronous document type detection.") from e

    async def detect_document_type(self, text: str) -> str:
        text_lower = text.lower()

        # PRIORITY 1: Specific documents that get confused (detect first)
        # Boletos/Pasajes/Tickets
        if any(w in text_lower for w in [
            "boleto", "pasaje", "ticket", "boarding pass", "vuelo", "aerolínea",
            "aeropuerto", "embarque", "itinerario", "reserva de vuelo", "code de reserva",
            "gate", "terminal", "check-in", "equipaje de mano"
        ]):
            return "boleto"

        # Facturas/Comprobantes fiscales
        if any(w in text_lower for w in [
            "factura", "cuit", "cae", "afip", "iva", "monotributo",
            "responsable inscripto", "comprobante fiscal", "punto de venta"
        ]):
            return "factura"

        # Recibos/Comprobantes
        if any(w in text_lower for w in [
            "recibo", "comprobante de pago", "recibí", "pago realizado",
            "transferencia", "depósito", "voucher"
        ]):
            return "recibo"

        # Presupuestos/Cotizaciones
        if any(w in text_lower for w in [
            "presupuesto", "cotización", "estimación", "propuesta comercial",
            "lista de precios", "tarifa", "costo estimado"
        ]):
            return "presupuesto"

        # PRIORIDAD 2: Documentos legales/administrativos (after de descartar comerciales)
        # Contracts (more specific)
        if any(w in text_lower for w in [
            "contrato de", "las partes acuerdan", "cláusula", "incumplimiento",
            "rescisión", "vigencia del contrato", "objeto del contrato"
        ]) and not any(w in text_lower for w in [
            "boleto", "pasaje", "factura", "recibo", "ticket"
        ]):
            return "contrato"

        # Actas
        if any(w in text_lower for w in [
            "acta de", "reunión", "firmantes", "siendo las", "se da por",
            "orden del día", "asistentes"
        ]):
            return "acta"

        # Resoluciones
        if any(w in text_lower for w in [
            "resolución", "resuelve", "considerando", "por tanto",
            "artículo", "decreto", "disposición"
        ]):
            return "resolución"

        # Informes
        if any(w in text_lower for w in [
            "informe de", "evaluación", "conclusión", "análisis",
            "resultados", "diagnóstico", "recomendaciones"
        ]):
            return "informe"

        # Notas/Comunicaciones
        if any(w in text_lower for w in [
            "nota", "comunicamos", "solicitamos", "me dirijo",
            "por la presente", "saludamos"
        ]):
            return "nota"

        # Oficios
        if any(w in text_lower for w in [
            "oficio", "expediente", "juzgado", "tribunal",
            "causa", "proceso judicial"
        ]):
            return "oficio"

        # Cartas
        if any(w in text_lower for w in [
            "carta", "estimado", "querido", "cordialmente",
            "me es grato", "aprovecho la oportunidad"
        ]):
            return "carta"

        # Propuestas
        if any(w in text_lower for w in [
            "propuesta", "ofrecemos", "presentamos", "plan de trabajo",
            "metodología", "cronograma"
        ]):
            return "propuesta"

        # Presentaciones
        if any(w in text_lower for w in [
            "presentación", "diapositiva", "slide", "agenda",
            "introducción", "conclusiones"
        ]):
            return "presentación"

        # PRIORIDAD 3: LLM fallback con prompt mejorado
        try:
            completion = await self.asyncOpenAI.chat.completions.create(
                model=GPT_5_NANO,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un clasificador de documentos. Analiza el contenido PRINCIPAL del documento, no solo palabras sueltas.\n"
                            "TIPOS DISPONIBLES: boleto, factura, recibo, presupuesto, contrato, acta, resolución, informe, oficio, nota, carta, propuesta, presentación, documento_legal, desconocido\n\n"
                            "REGLAS IMPORTANTES:\n"
                            "- Si habla de vuelos/aerolíneas/pasajes = boleto\n"
                            "- Si tiene CUIT/CAE/AFIP = factura\n"
                            "- Si es pago/transferencia = recibo\n"
                            "- Si es cotización/precios = presupuesto\n"
                            "- SOLO usa 'contrato' si es realmente un acuerdo legal entre partes\n"
                            "- Responde con UNA SOLA PALABRA de la lista"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Clasifica este documento por su contenido principal:\n\n{text[:2000]}"
                    }
                ],
                #temperature=0,
                #max_tokens=10
                max_completion_tokens=10,
                reasoning_effort="low"
            )

            tipo = completion.choices[0].message.content.lower().strip()

            # Validar respuesta
            tipos_validos = [
                "boleto", "factura", "recibo", "presupuesto", "contrato",
                "acta", "resolución", "informe", "oficio", "nota", "carta",
                "propuesta", "presentación", "documento_legal"
            ]

            return tipo if tipo in tipos_validos else "desconocido"

        except Exception as e:
            print(f"[⚠️ GPT Fallback Error] {e}")
            return "desconocido"

    # Utilidad: cortar texto en chunks de tokens
    def dividir_por_tokens(self,text, model="gpt-5.6", max_tokens=3000, overlap=300):
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            # tiktoken no mapea gpt-5.x -> usar el tokenizer base (gpt-4o/gpt-5).
            enc = tiktoken.get_encoding("o200k_base")
        tokens = enc.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)
            start += max_tokens - overlap  # avanzar con solapamiento

        return chunks

    def _extract_section_title(self, text: str) -> str | None:
        """
        Extract a possible section or heading title from the chunk.
        """
        lines = text.strip().split("\n")
        for line in lines:
            if len(line.strip()) > 5 and line.isupper():
                return line.strip()
        return None


    from typing import Any, Dict

    async def embedding_fn_batch(self, texts: List[str]) -> List[List[float]]:
        """Fixed version that handles the OpenAI response correctly."""
        try:
            if not texts:
                return []

            valid_texts = [text.strip() for text in texts if text and text.strip()]
            if not valid_texts:
                return []

            print(f"📊 Generando {len(valid_texts)} embeddings en lote...")

            batch_size = 100
            all_embeddings = []

            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                print(f"  🚀 Processing lote {i // batch_size + 1}: {len(batch_texts)} textos")

                # Use the SYNCHRONOUS API WITHOUT AWAIT
                response = openai.embeddings.create(
                    input=batch_texts,
                    model="text-embedding-3-large"
                )

                # ✅ DEBUG: Verificar la respuesta
                print(f"  📊 Respuesta recibida: {type(response)}")
                print(f"  📊 Data type: {type(response.data)}")
                if response.data:
                    print(f"  📊 Primer item type: {type(response.data[0])}")
                    print(f"  📊 Primer embedding type: {type(response.data[0].embedding)}")

                # ✅ PROCESAR RESPUESTA DE FORMA SEGURA
                batch_embeddings = []
                for idx, item in enumerate(response.data):
                    try:
                        # Verificar que no sea ObjectRef
                        embedding = item.embedding
                        if hasattr(embedding, '__class__') and 'ObjectRef' in str(type(embedding)):
                            raise ValueError(f"❌ Embedding {idx} es ObjectRef: {type(embedding)}")

                        # Convertir a lista de Python
                        embedding_list = list(embedding) if hasattr(embedding, '__iter__') else [float(x) for x in
                                                                                                 embedding]
                        batch_embeddings.append(embedding_list)

                    except Exception as e:
                        logger.error(f"❌ Error procesando embedding {idx}: {e}")
                        batch_embeddings.append([0.0] * 1536)  # Embedding dummy

                all_embeddings.extend(batch_embeddings)

                # Rate limiting suave
                if i + batch_size < len(valid_texts):
                    await asyncio.sleep(0.1)

            print(f"✅ {len(all_embeddings)} embeddings processeds correctamente")
            return all_embeddings

        except Exception as e:
            logger.error(f"❌ Error en embedding_fn_batch: {e}")
            import traceback
            logger.error(f"❌ Stack trace: {traceback.format_exc()}")

            # Fallback de emergencia
            dummy_embeddings = [[0.0] * 1536 for _ in texts]
            logger.warning(f"⚠️ Usando embeddings dummy para {len(texts)} textos")
            return dummy_embeddings


    # Keep the original function for compatibility
    def embedding_fn(self, text: str) -> list[float]:
        """Original function kept for backward compatibility."""
        try:
            response = openai.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"❌ Error en embedding_fn individual: {e}")
            # Devolver embedding dummy en caso de error
            return [0.0] * 1536

    def insert_document_sync(
            self,
            user: str,
            company: str,
            content: Dict[str, Any],
            batch_size: int = 100,
            doc_type: str = "contrato",
            file_name: str = None

    ) -> str:
        def run_async():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    self.insert_document(user, company, content, batch_size, doc_type,file_name)
                )
            except Exception as e:
                logger.error(f"Error en insert_document (async) para archivo {file_name}: {e}")
                raise RuntimeError(f"Failed to insert the document '{file_name}' asynchronously.") from e
            finally:
                new_loop.close()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_async)
                return future.result()
        except Exception as e:
            logger.error(f"Error en insert_document_sync para archivo {file_name}: {e}")
            raise RuntimeError(f"Failed in synchronous insertion of the document '{file_name}'.") from e

    async def insert_document(
            self,
            user: str,
            company: str,
            content: Dict[str, Any],
            batch_size: int = 100,
            doc_type: str = "contrato",
            file_name: str = None
    ) -> str:
        """
        Inserta un documento en MongoDB y lo carga en Weaviate usando procesamiento paralelo optimizado.
        Robust version with exhaustive error handling and detailed logs.
        """

        # Variables para tracking
        collection_name = None
        mongo_id = None
        config = None
        chunks = None
        splited = None

        try:
            # ═══════════════════════════════════════════════════════════════
            # LOG INICIAL DESTACADO
            # ═══════════════════════════════════════════════════════════════
            logger.info("=" * 80)
            logger.info("INICIANDO INSERCION DE DOCUMENTO")
            logger.info(f"   Archivo: {file_name or 'unnamed'}")
            logger.info(f"   Usuario: {user}")
            logger.info(f"   Empresa: {company}")
            logger.info(f"   Batch size: {batch_size}")
            logger.info("=" * 80)

            # ═══════════════════════════════════════════════════════════════
            # VALIDACION BASICA
            # ═══════════════════════════════════════════════════════════════
            if not content:
                raise ValueError("El contenido no puede estar vacio")

            if 'text' not in content:
                raise ValueError("El contenido debe incluir el campo 'text'")

            if not content['text'] or not isinstance(content['text'], str):
                raise ValueError("El campo 'text' debe ser una cadena no vacia")

            # LOG: Informacion del contenido
            text_length = len(content['text'])
            logger.info(f"Contenido received: {text_length:,} characters")

            if text_length == 0:
                raise ValueError("El texto del documento esta vacio")

            # ═══════════════════════════════════════════════════════════════
            # NORMALIZACION DE NOMBRE DE COLECCION
            # ═══════════════════════════════════════════════════════════════
            try:
                collection_name = re.sub(r"[^a-zA-Z0-9_]", "_", (company or "").upper().replace(" ", "_"))
                logger.info(f"Coleccion destino: {collection_name}")
            except Exception as e:
                logger.error(f"Error normalizando nombre de coleccion: {e}")
                raise ValueError(f"Nombre de coleccion invalido: {e}")

            # Verifica si ya existe la clase en Weaviate
            try:
                exists = await self.exists_weaviate_class(collection_name)
                logger.info(f"Coleccion existe: {'Si' if exists else 'No (se creara)'}")
            except Exception as e:
                logger.error(f"Error verificando coleccion en Weaviate: {e}")
                raise Exception(f"Error de conexion con Weaviate: {e}")

            # ── EWS: cryptographic identity of the document ────────────────
            # Compute doc_hash, BIP32 signature, check existing version
            try:
                from framework.compliance.doc_identity import prepare_doc_identity
                _ews = prepare_doc_identity(
                    full_text=content['text'],
                    filename=file_name or 'unnamed',
                    tenant_id=company,
                    weaviate_client=self.client,
                    collection_name=collection_name,
                )
                if _ews["is_duplicate"]:
                    logger.info(
                        "✅ Documento duplicado (mismo contenido) — omitiendo indexado: %s",
                        file_name
                    )
                    return f"DUPLICATE:{_ews['doc_hash'][:16]}"
                doc_hash      = _ews["doc_hash"]
                doc_signature = _ews["doc_signature"]
                indexed_at    = _ews["indexed_at"]
                indexed_at_iso = _ews["indexed_at_iso"]
                doc_version   = _ews["doc_version"]
                if _ews["chunks_superseded"] > 0:
                    logger.info(
                        "📝 Previous version marked as superseded: %d chunks | v%d → v%d",
                        _ews["chunks_superseded"], doc_version - 1, doc_version
                    )
            except Exception as _ews_err:
                logger.warning("EWS doc_identity failed (non-blocking): %s", _ews_err)
                import time as _t, hashlib as _hlib
                doc_hash      = _hlib.sha256(content['text'].encode()).hexdigest()
                doc_signature = ""
                indexed_at    = _t.time()
                from datetime import datetime, timezone
                indexed_at_iso = datetime.fromtimestamp(indexed_at, tz=timezone.utc).isoformat()
                doc_version   = 1

            # Extrae metadatos
            metadata = {
                'file_name': file_name or 'unnamed',
                'user': user,
                'company': company,
                'tipo_doc': doc_type
            }

            # ═══════════════════════════════════════════════════════════════
            # PASO 1: ANALIZAR Y OBTENER CONFIG OPTIMA
            # ═══════════════════════════════════════════════════════════════
            logger.info("Paso 1/5: Analizando documento para configuracion optima...")
            try:
                advisor = SmartSplitAdvisor(provider="openai", model=GPT_5_NANO)
                config = advisor.advise_config(content['text'], metadata)

                # Manejar tanto dict como objeto SplitConfig
                logger.info("   Configuracion obtenida:")
                logger.info(f"      - Tipo: {type(config).__name__}")

                try:
                    if isinstance(config, dict):
                        chunk_size = config.get('chunk_size', 'N/A')
                        chunk_overlap = config.get('chunk_overlap', 'N/A')
                    else:
                        # Es un objeto, intentar varios nombres de atributos
                        chunk_size = getattr(config, 'chunk_size',
                                             getattr(config, 'size',
                                                     getattr(config, 'max_chunk_size', 'N/A')))
                        chunk_overlap = getattr(config, 'chunk_overlap',
                                                getattr(config, 'overlap',
                                                        getattr(config, 'chunk_overlap_size', 'N/A')))

                    logger.info(f"      - Chunk size: {chunk_size}")
                    logger.info(f"      - Overlap: {chunk_overlap}")
                except Exception as log_error:
                    logger.warning(f"      No se pudo leer detalles de configuracion: {log_error}")

            except Exception as e:
                logger.error(f"Error obteniendo configuracion de split: {e}")
                # Fallback a configuracion por defecto
                logger.warning("Usando configuracion por defecto")
                config = {'chunk_size': 1000, 'chunk_overlap': 200}

            # ═══════════════════════════════════════════════════════════════
            # PASO 2: SPLITEAR CON ESA CONFIG
            # ═══════════════════════════════════════════════════════════════
            logger.info("Paso 2/5: Splitting documento en chunks...")
            try:
                splitter = ParallelDocumentSplitter(config)
                chunks = splitter.split_document(content['text'], metadata)
                logger.info(f"📊 AFTER SPLIT: {len(chunks)} chunks")

                if not chunks or len(chunks) == 0:
                    raise ValueError("El documento no genero ningun chunk")

                logger.info(f"   Documento dividido en {len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Error dividiendo documento: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise Exception(f"Error en split de documento: {e}")

            # ═══════════════════════════════════════════════════════════════
            # PASO 3: NORMALIZAR PARA WEAVIATE
            # ═══════════════════════════════════════════════════════════════
            logger.info("Paso 3/5: Normalizando chunks para Weaviate...")
            try:
                splited = self._normalize_chunks_for_weaviate(chunks, metadata)
                logger.info(f"📊 AFTER NORMALIZE: {len(splited)} chunks")

                if not splited or len(splited) == 0:
                    raise ValueError("No se generaron chunks normalizados")

                logger.info(f"   {len(splited)} chunks normalizados")

                # VALIDACION: Verificar que los chunks tengan contenido
                empty_chunks = 0
                valid_chunks = 0
                for i, chunk in enumerate(splited):
                    if not chunk.get('text', '').strip():
                        empty_chunks += 1
                        logger.warning(f"   Chunk {i} esta vacio")
                    else:
                        valid_chunks += 1

                logger.info(f"   Chunks validos: {valid_chunks}, vacios: {empty_chunks}")

                if valid_chunks == 0:
                    raise ValueError("Ningun chunk tiene contenido valido")

                if empty_chunks > 0:
                    logger.warning(f"   Se encontraron {empty_chunks} chunks vacios")
                    # Filtrar chunks vacios
                    splited = [chunk for chunk in splited if chunk.get('text', '').strip()]
                    logger.info(f"   Chunks despues de filtrado: {len(splited)}")

            except Exception as e:
                logger.error(f"Error normalizando chunks: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise Exception(f"Error en normalizacion: {e}")

            # ═══════════════════════════════════════════════════════════════
            # PASO 4: INSERCION EN MONGODB
            # ═══════════════════════════════════════════════════════════════
            logger.info("Paso 4/5: Guardando metadata en MongoDB...")
            try:
                mongo = MongoDBManager(company, self.config)
                result = mongo.insert_one(metadata)
                mongo_id = result
                metadata.pop('_id', None)
                metadata['file_id'] = result
                logger.info(f"   Metadata guardada con ID: {result}")
            except Exception as e:
                logger.error(f"Error guardando en MongoDB: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise Exception(f"Error en MongoDB: {e}")

            # ═══════════════════════════════════════════════════════════════
            # PASO 5: INTEGRACION CON PROCESADOR PARALELO
            # ═══════════════════════════════════════════════════════════════
            logger.info("Paso 5/5: Iniciando procesamiento paralelo en Weaviate...")

            try:
                # Crear processedr de embeddings
                embedding_processor = EmbeddingProcessor(
                    max_workers=4,
                    batch_size=batch_size
                )
                logger.info(f"   Procesador de embeddings creado (workers: 4, batch: {batch_size})")

                # Crear processedr paralelo de Weaviate
                parallel_processor = ParallelWeaviateProcessor(
                    weaviate_client=self.client,
                    embedding_processor=embedding_processor,
                    max_concurrent_files=4,
                    batch_insert_size=batch_size
                )
                logger.info(f"   Procesador paralelo creado (batch_insert: {batch_size})")

                # Preparar datos en formato esperado
                files_data = [{
                    'chunks': splited,
                    'metadata': metadata
                }]
                logger.info(f"   Preparados {len(splited)} chunks para procesamiento")

                # LOG ANTES DE LLAMAR AL METODO CRITICO
                logger.info("")
                logger.info("LLAMANDO A process_multiple_files_parallel...")
                logger.info(f"   - Files data: {len(files_data)} archivo(s)")
                logger.info(f"   - Total chunks: {len(splited)}")
                logger.info(f"   - Usuario: {user}")
                logger.info(f"   - Empresa: {company}")
                logger.info("")

                # Procesar usando el sistema paralelo
                result_stats = await parallel_processor.process_multiple_files_parallel(
                    files_data=files_data,
                    user=user,
                    company=company,
                    uploaded_by_name=user,
                    upload_ip="192.168.1.33"
                )

                # LOG DESPUES DE RECIBIR RESPUESTA
                logger.info("")
                logger.info("RESPUESTA RECIBIDA de process_multiple_files_parallel")
                logger.info(f"   Success: {result_stats.get('success', False)}")
                logger.info(f"   Chunks processeds: {result_stats.get('chunks_processed', 0)}")
                logger.info(f"   Inserciones exitosas: {result_stats.get('successful_inserts', 0)}")
                logger.info(f"   Inserciones fallidas: {result_stats.get('failed_inserts', 0)}")
                logger.info("")

            except Exception as e:
                logger.error(f"Error creando processedres: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise Exception(f"Error en processedres: {e}")

            # ═══════════════════════════════════════════════════════════════
            # ✅ FIX: VALIDACION FLEXIBLE BASADA EN INSERCIONES REALES
            # ═══════════════════════════════════════════════════════════════

            successful_inserts = result_stats.get('successful_inserts', 0)
            failed_inserts = result_stats.get('failed_inserts', 0)
            is_success = result_stats.get('success', False)

            # ─────────────────────────────────────────────────────────────
            # RULE 1: If there are NO successful inserts → CRITICAL ERROR
            # ─────────────────────────────────────────────────────────────
            if successful_inserts == 0:
                error_msg = result_stats.get('error', 'Could not insert any chunk into Weaviate')
                logger.error(f"❌ CRITICAL: {error_msg}")
                logger.error(f"   Chunks processeds: {result_stats.get('chunks_processed', 0)}")
                logger.error(f"   Chunks fallidos: {failed_inserts}")
                raise Exception(f"Error en Weaviate: {error_msg}")

            # ─────────────────────────────────────────────────────────────
            # REGLA 2: Si hay inserciones PERO success=False → ADVERTIR
            # ─────────────────────────────────────────────────────────────
            if not is_success and successful_inserts > 0:
                logger.warning("=" * 80)
                logger.warning("⚠️ WARNING: strict validation failed but there are inserted chunks")
                logger.warning("=" * 80)
                logger.warning(f"   Inserciones exitosas: {successful_inserts}")
                logger.warning(f"   Inserciones fallidas: {failed_inserts}")

                # Analizar validation
                validation = result_stats.get('validation', {})
                stored_chunks = validation.get('stored_chunks', 0)
                expected_chunks = validation.get('expected_chunks', 0)

                logger.warning(f"   Validation esperaba: {expected_chunks} chunks")
                logger.warning(f"   Validation found: {stored_chunks} chunks")

                if stored_chunks < expected_chunks:
                    diff = expected_chunks - stored_chunks
                    logger.warning(f"   Diferencia: {diff} chunks no encontrados")
                    logger.warning(f"   Posibles causas:")
                    logger.warning(f"      1. Timing: Weaviate has not indexed everything yet")
                    logger.warning(f"      2. Filtros: La validation busca con filtros incorrectos")
                    logger.warning(f"      3. Metadata: Los chunks tienen metadata diferente")

                # Compute success rate
                total_attempts = successful_inserts + failed_inserts
                if total_attempts > 0:
                    success_rate = (successful_inserts / total_attempts) * 100
                    logger.warning(f"   Success rate de insertion: {success_rate:.1f}%")

                    # ─────────────────────────────────────────────────────
                    # DECISION: continue or fail?
                    # ─────────────────────────────────────────────────────
                    if success_rate >= 90:
                        logger.warning(f"   ✅ Tasa >= 90% - CONTINUANDO (validation puede estar mal)")
                        logger.warning(f"   Se insertaron {successful_inserts} chunks correctamente")
                        logger.warning("=" * 80)
                        # Do NOT raise an exception, continue
                    else:
                        logger.error(f"   ❌ Tasa < 90% - DEMASIADO BAJA")
                        logger.error(f"   Se perdieron demasiados chunks ({failed_inserts})")
                        logger.error("=" * 80)
                        raise Exception(f"Error en Weaviate: Success rate {success_rate:.1f}% es insuficiente")
                else:
                    # Should not happen (successful_inserts > 0)
                    logger.warning("   ⚠️ Could not compute success rate")
                    logger.warning("   CONTINUANDO de todos modos")
                    logger.warning("=" * 80)

            # ─────────────────────────────────────────────────────────────
            # REGLA 3: Si success=True → TODO OK
            # ─────────────────────────────────────────────────────────────
            else:
                logger.info(f"✅ Validation exitosa: {successful_inserts} chunks insertados correctamente")

            # ═══════════════════════════════════════════════════════════════
            # LOG DE ESTADISTICAS DETALLADAS
            # ═══════════════════════════════════════════════════════════════
            logger.info("")
            logger.info("=" * 80)
            logger.info("PROCESAMIENTO COMPLETADO EXITOSAMENTE")
            logger.info("=" * 80)
            logger.info("Estadisticas finales:")
            logger.info(f"   Chunks processeds: {result_stats.get('chunks_processed', 0)}")
            logger.info(f"   Inserciones exitosas: {successful_inserts}")
            logger.info(f"   Inserciones fallidas: {failed_inserts}")

            # VALIDACION DE COMPLETITUD
            validation = result_stats.get('validation', {})
            if validation:
                stored_chunks = validation.get('stored_chunks', 'N/A')
                expected_chunks = validation.get('expected_chunks', 'N/A')
                is_complete = validation.get('is_complete', False)

                logger.info(f"   Almacenados en Weaviate: {stored_chunks}")

                if is_complete:
                    logger.info("   ✅ DOCUMENTO COMPLETO: Todos los chunks almacenados")
                else:
                    logger.warning("   ⚠️ DOCUMENTO INCOMPLETO (per validation):")
                    logger.warning(f"      - Esperados: {expected_chunks}")
                    logger.warning(f"      - Encontrados: {stored_chunks}")
                    if isinstance(expected_chunks, int) and isinstance(stored_chunks, int):
                        logger.warning(f"      - Diferencia: {expected_chunks - stored_chunks}")
                    logger.warning(f"      NOTA: Esto puede ser un falso positivo de la validation")

            # Tiempos
            logger.info(f"   Tiempo embeddings: {result_stats.get('embedding_time', 0):.2f}s")
            logger.info(f"   Tiempo insercion: {result_stats.get('insert_time', 0):.2f}s")
            logger.info(f"   Tiempo total: {result_stats.get('total_time', 0):.2f}s")
            logger.info(f"   Velocidad: {result_stats.get('chunks_per_second', 0):.1f} chunks/s")

            # MUESTRA DE CONTENIDO
            content_sample = result_stats.get('content_sample', {})
            if content_sample and content_sample.get('chunks_preview'):
                logger.info("   Muestra de chunks almacenados:")
                for preview in content_sample['chunks_preview'][:3]:
                    try:
                        chunk_num = preview.get('chunk_number', '?')
                        text_len = preview.get('text_length', 0)
                        is_clean = preview.get('is_clean', False)
                        logger.info(f"      - Chunk {chunk_num}: {text_len} chars, limpio={is_clean}")
                    except Exception as preview_error:
                        logger.warning(f"      Error mostrando preview: {preview_error}")

            logger.info("=" * 80)
            logger.info("")

            # ═══════════════════════════════════════════════════════════════
            # ADVERTENCIAS ADICIONALES
            # ═══════════════════════════════════════════════════════════════
            if failed_inserts > 0:
                logger.warning(f"⚠️ ADVERTENCIA: Se perdieron {failed_inserts} chunks durante la insertion")
                total_inserts = successful_inserts + failed_inserts
                if total_inserts > 0:
                    success_rate = (successful_inserts / total_inserts * 100)
                    logger.warning(f"   Success rate: {success_rate:.1f}%")

            logger.info(
                f"✅ Documento '{file_name or 'unnamed'}' processed successfully in collection '{collection_name}'")

            # GDPR Art. 15/16/17: indexar las personas presentes en el documento para
            # so that the data subject's rights (access/erasure/rectification) work
            # sobre datos reales. Best-effort en hilo (Presidio es pesado y synchronous);
            # nunca rompe la ingesta.
            try:
                import hashlib as _hl
                from framework.compliance.personal_data_index import PersonalDataIndex
                _text = content.get("text", "") if isinstance(content, dict) else ""
                if _text:
                    _doc_hash = _hl.sha256(_text.encode()).hexdigest()
                    await asyncio.to_thread(
                        PersonalDataIndex(self.config).registrar_desde_pii_detection,
                        _text, _doc_hash, file_name or "unnamed", company,
                    )
            except Exception as _pii_e:
                logger.debug("PersonalDataIndex no poblado para %s: %s", file_name, _pii_e)

            return collection_name

        except ValueError as ve:
            logger.error(f"[Validacion] Error: {ve}")
            logger.error(f"   Archivo: {file_name}")
            logger.error(f"   Collection: {collection_name}")
            raise

        except Exception as e:
            logger.error(f"[Error] Problema durante la insercion del documento: {e}")
            logger.error(f"   Archivo: {file_name}")
            logger.error(f"   Usuario: {user}")
            logger.error(f"   Empresa: {company}")
            logger.error(f"   Collection: {collection_name}")
            logger.error(f"   MongoDB ID: {mongo_id}")
            logger.error(f"   Chunks generados: {len(chunks) if chunks else 0}")
            logger.error(f"   Chunks normalizados: {len(splited) if splited else 0}")

            import traceback
            logger.error("Traceback completo:")
            logger.error(traceback.format_exc())

            raise

    def get_llm(self, model_key: str, ollama_base_url: str = "http://localhost:11434"):

        if model_key.startswith("gemini-") and not hasattr(self, 'google_api_key'):
            raise ValueError("No se ha configurado google_api_key")
        if model_key.startswith("gemini-") and not self.google_api_key:
            raise ValueError("google_api_key is empty")

        if model_key.startswith("gemini-"):
            try:
                genai.configure(api_key=self.google_api_key)
                print(f"[Info] Cliente Gemini inicializado: {model_key}")
            except Exception as e:
                raise ValueError(f"Error inicializando Gemini: {str(e)}")

        def create_gemini_lambda():
            def gemini_wrapper(prompt):
                try:
                    print(f"[Gemini] Llamando {GEMINI_3_1_PRO} ({len(prompt)} chars)")
                    model = self.clientG.GenerativeModel(model_name=GEMINI_3_1_PRO)
                    response = model.generate_content(contents=prompt)
                    print(f"[Gemini] Respuesta recibida")
                    return response
                except Exception as e:
                    print(f"[Error Gemini] {str(e)}")

                    class ErrorResponse:
                        def __init__(self, error):
                            self.error = error
                            self.text = f"ERROR: {error}"

                    return ErrorResponse(str(e))

            return gemini_wrapper

        modelos = {
            # ── GPT-5 family ──────────────────────────────────────────
            GPT_5: lambda: ChatOpenAI(
                model_name=GPT_5,
                temperature=1,
                model_kwargs={"max_completion_tokens": 16000, "reasoning_effort": "medium"}
            ),

            GPT_5_1: lambda: ChatOpenAI(
                model_name=GPT_5_1,
                temperature=1,
                model_kwargs={"max_completion_tokens": 16000, "reasoning_effort": "medium"}
            ),

            GPT_5_NANO: lambda: ChatOpenAI(
                model_name=GPT_5_NANO,
                temperature=1,
                model_kwargs={"max_completion_tokens": 4500, "reasoning_effort": "low"}
            ),

            GPT_5_4_MINI: lambda: ChatOpenAI(
                model_name=GPT_5_4_MINI,
                temperature=1,
                model_kwargs={"max_completion_tokens": 8000, "reasoning_effort": "medium"}
            ),

            # ── Gemini ────────────────────────────────────────────────
            GEMINI_3_1_PRO: lambda: create_gemini_lambda(),

            # ── Local (Ollama) ────────────────────────────────────────
            "gpt-oss-20b": lambda: ChatOllama(
                model="gpt-oss:20b",
                base_url=ollama_base_url,
                temperature=1,
                num_predict=4096,
                timeout=120,
                keep_alive=-1,
                num_ctx=16384,
                repeat_penalty=1.1,
                top_k=40,
                top_p=0.9
            ),
        }

        if model_key not in modelos:
            raise ValueError(
                f"Modelo '{model_key}' no soportado. Disponibles: {list(modelos.keys())}")

        return modelos[model_key]()

    async def run_model(self, model: str, llm, rag_input: str, fallback_depth: int = 0) -> str:
        model = model.lower()
        start_time = time.time()

        timeouts = {
            GPT_5: 60.0,
            GPT_5_1: 60.0,
            GPT_5_NANO: 45.0,
            GPT_5_4_MINI: 45.0,
            GEMINI_3_1_PRO: 240.0,
            "gpt-oss-20b": 120.0,
        }

        timeout = timeouts.get(model, 60.0)
        print(f"[Info] Iniciando llamada al modelo {model} (timeout: {timeout}s)")

        try:
            # ── OpenAI ────────────────────────────────────────────────────
            if model in {GPT_5, GPT_5_1, GPT_5_NANO, GPT_5_4_MINI}:
                response = await asyncio.wait_for(
                    asyncio.to_thread(llm.invoke, rag_input),
                    timeout=timeout
                )
                result = getattr(response, "content", None)

            # ── Ollama ────────────────────────────────────────────────────
            elif model == "gpt-oss-20b":
                print(f"[Info] Llamando Ollama: {model}")
                try:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(llm.invoke, rag_input),
                        timeout=timeout
                    )
                    result = getattr(response, "content", None)
                    print(f"[Info] Ollama {model}: {len(result) if result else 0} chars")
                except Exception as e:
                    print(f"[Error Ollama] {model}: {str(e)}")
                    return await self._fallback_model(model, rag_input, str(e), fallback_depth, GPT_5_NANO)

            # ── Gemini ────────────────────────────────────────────────────
            elif model == GEMINI_3_1_PRO:
                show_progress = timeout > 120.0
                if show_progress:
                    progress_task = asyncio.create_task(self._show_progress(model, timeout))

                try:
                    model_instance = self.clientG.GenerativeModel(model_name=GEMINI_3_1_PRO)
                    response = await asyncio.wait_for(
                        asyncio.to_thread(lambda: model_instance.generate_content(rag_input)),
                        timeout=timeout
                    )

                    if show_progress and 'progress_task' in locals():
                        progress_task.cancel()

                    try:
                        result = response.text
                    except Exception:
                        result = str(response)

                except asyncio.TimeoutError:
                    if show_progress and 'progress_task' in locals():
                        progress_task.cancel()
                    return await self._fallback_model(
                        model, rag_input, f"Timeout after de {timeout}s", fallback_depth, GPT_5_NANO)

                except Exception as e:
                    if show_progress and 'progress_task' in locals():
                        progress_task.cancel()
                    return await self._fallback_model(model, rag_input, str(e), fallback_depth, GPT_5_NANO)

            else:
                raise ValueError(f"Modelo no soportado: {model}")

            # ── Validation resultado ───────────────────────────────────────
            if result is None or not result.strip():
                raise ValueError(f"Respuesta empty del modelo {model}")

            elapsed = time.time() - start_time
            print(f"[Performance] Modelo {model}: {elapsed:.2f}s")
            return result

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"[Timeout] {model} exceeded {timeout}s. Elapsed: {elapsed:.2f}s")
            return await self._fallback_model(
                model, rag_input, f"Timeout after de {elapsed:.2f}s", fallback_depth, GPT_5_NANO)

        except Exception as e:
            print(f"[Error] {model}: {str(e)}")
            return await self._fallback_model(model, rag_input, str(e), fallback_depth, GPT_5_NANO)

    async def _show_progress(self, model: str, total_timeout: float):
        """Muestra mensajes de progreso durante operaciones largas"""
        steps = min(int(total_timeout / 15), 10)  # Max 10 messages
        interval = total_timeout / steps

        for i in range(steps):
            try:
                await asyncio.sleep(interval)
                progress = int((i + 1) / steps * 100)
                print(
                    f"[Progress] {model} running... {progress}% (aprox. {(i + 1) * interval:.0f}s de {total_timeout:.0f}s)")
            except asyncio.CancelledError:
                # Task cancelled (normal when the main operation finishes)
                break

    async def _fallback_model(self, failed_model: str, rag_input: str, error_message: str,
                              depth: int = 0, force_model: str = None) -> str:
        if depth >= 2:
            return f"Sorry, multiple models failed. Original error: {error_message}"

        try:
            if force_model:
                backup_model = force_model
            else:
                fallback_map = {
                    GEMINI_3_1_PRO: GPT_5_1,
                    GPT_5_1: GPT_5_4_MINI,
                    GPT_5: GPT_5_4_MINI,
                    GPT_5_NANO: GPT_5_4_MINI,
                    GPT_5_4_MINI: GPT_5_1,
                    "gpt-oss-20b": GPT_5_4_MINI,
                }
                backup_model = fallback_map.get(failed_model, GPT_5_NANO)

            print(f"[Recovery] Fallback: {failed_model} → {backup_model}")

            backup_llm = self.get_llm(backup_model)
            final_prompt = self._prepare_recovery_prompt(rag_input, failed_model, error_message)
            return await self.run_model(backup_model, backup_llm, final_prompt, depth + 1)

        except Exception as e:
            return f"Error en modelo de respaldo: {str(e)}"

    def _prepare_recovery_prompt(self, rag_input: str, failed_model: str, error_message: str) -> str:
        """Prepara el prompt optimizado para el modelo de respaldo"""
        max_size = 12000 if "Timeout" in error_message else 16000

        if len(rag_input) <= max_size:
            return f"[FALLBACK] Model {failed_model} failed. Answer this request:\n\n{rag_input}"

        # Reducir el prompt preservando la estructura
        query_marker = "❓ Pregunta del user:"
        if query_marker in rag_input:
            context, query = rag_input.split(query_marker, 1)
            query_part = query_marker + query

            if len(context) > (max_size - len(query_part)):
                half_context = (max_size - len(query_part)) // 2
                reduced_context = (context[:half_context] +
                                   "\n...[Contexto reducido]...\n" +
                                   context[-half_context:])
                return reduced_context + query_part
            return rag_input

        # Standard reduction when there is no specific structure
        half_size = max_size // 2
        return (f"[FALLBACK] Answer this query (reduced content):\n\n"
                f"{rag_input[:half_size]}\n...[Content truncated]...\n{rag_input[-half_size:]}")

    def optimize_prompt_for_gemini(self, rag_input: str, model: str) -> str:
        """
        Optimize the prompt for Gemini models, especially preview versions
        that may struggle with very long or complex prompts.
        """
        prompt_length = len(rag_input)

        # Only optimize long prompts or preview models
        if prompt_length <= 20000 and "preview" not in model:
            return rag_input

        print(f"[Gemini Optimize] Optimizing prompt of {prompt_length} characters for {model}")

        # 1. Reduce complex instructions
        simplified = rag_input.replace(
            "Additional instructions:\n- If the context lacks enough information, say so clearly\n- Prioritize recent information over old data\n- Provide specific references when possible\n- Offer concrete, actionable recommendations\n- Structure your answer logically and easy to follow",
            "Instruction: Answer accurately based on the context."
        )

        # If the prompt is still very long, reduce its size
        if len(simplified) > 25000:
            # Identify key sections
            query_marker = "❓ Pregunta del user:"
            context_marker = "📌 Contexto relevante:"
            response_marker = "🤖 Respuesta:"

            # Preserve the question and the main instructions
            query_part = ""
            if query_marker in simplified:
                parts = simplified.split(query_marker)
                pre_query = parts[0]
                query_part = parts[1].split(response_marker)[0] if response_marker in parts[1] else parts[1]

            # Mainly reduce the context part
            if context_marker in simplified:
                context_parts = simplified.split(context_marker)
                pre_context = context_parts[0]
                context_content = context_parts[1].split(query_marker)[0] if query_marker in context_parts[1] else \
                    context_parts[1]

                # Preserve structure but reduce content
                max_context_size = 15000
                if len(context_content) > max_context_size:
                    reduced_context = context_content[:5000] + "\n...[Contexto resumido]...\n" + context_content[-5000:]

                    # Rebuild the prompt
                    optimized = pre_context + context_marker + reduced_context
                    if query_part:
                        optimized += query_marker + query_part
                    if response_marker in simplified:
                        optimized += response_marker + simplified.split(response_marker)[1]

                    simplified = optimized
            else:
                # If the context marker is not found, reduce uniformly
                if len(simplified) > 30000:
                    simplified = simplified[:15000] + "\n...[Contenido reducido]...\n" + simplified[-10000:]

        print(f"[Gemini Optimize] Prompt optimized to {len(simplified)} characters")
        return simplified

    async def create_empty_class(self, class_name: str):
        """
        Create an empty class in Weaviate with the full document-chunk structure.

        Args:
            class_name (str): Name of the class to create.
        """
        try:
            class_obj = {
                "class": class_name,
                "description": f"Document-chunk collection for {class_name}",
                "vectorizer": "text2vec-openai",  # Adjust per your configuration
                "moduleConfig": {
                    "text2vec-openai": {
                        "model": "ada",
                        "modelVersion": "002",
                        "type": "text"
                    }
                },
                "properties": [
                    # Identifiers
                    {"name": "name", "dataType": ["text"], "description": "Unique chunk ID"},
                    {"name": "file_id", "dataType": ["text"], "description": "Original file ID"},
                    {"name": "filename", "dataType": ["text"], "description": "File name"},
                    {"name": "text_hash", "dataType": ["text"], "description": "Content hash"},

                    # Content
                    {"name": "text", "dataType": ["text"], "description": "Full chunk content"},
                    {"name": "text_preview", "dataType": ["text"], "description": "Text preview"},
                    {"name": "content_type", "dataType": ["text"], "description": "Content type"},

                    # Chunk metadata
                    {"name": "chunk_index", "dataType": ["int"], "description": "Chunk index"},
                    {"name": "chunk_length", "dataType": ["int"], "description": "Chunk length"},
                    {"name": "total_chunks", "dataType": ["int"], "description": "Total chunks in the document"},
                    {"name": "previous_chunk_id", "dataType": ["text"], "description": "Previous chunk ID"},
                    {"name": "next_chunk_id", "dataType": ["text"], "description": "Next chunk ID"},

                    # Content analysis
                    {"name": "word_count", "dataType": ["int"], "description": "Word count"},
                    {"name": "has_numbers", "dataType": ["boolean"], "description": "Contains numbers"},
                    {"name": "digit_density", "dataType": ["number"], "description": "Digit density"},
                    {"name": "language", "dataType": ["text"], "description": "Detected language"},

                    # User and company
                    {"name": "user", "dataType": ["text"], "description": "User who uploaded the file"},
                    {"name": "uploaded_by", "dataType": ["text"], "description": "User email"},
                    {"name": "uploaded_by_name", "dataType": ["text"], "description": "User name"},
                    {"name": "company", "dataType": ["text"], "description": "User company"},
                    {"name": "upload_ip", "dataType": ["text"], "description": "Upload IP"},

                    # Classification
                    {"name": "doc_type", "dataType": ["text"], "description": "Document type"},
                    {"name": "source", "dataType": ["text"], "description": "Document source"},

                    # Dates and timestamps
                    {"name": "created_at", "dataType": ["number"], "description": "Creation timestamp"},
                    {"name": "created_date", "dataType": ["text"], "description": "ISO creation date"},
                    {"name": "deleted_at", "dataType": ["text"], "description": "Deletion date"},

                    # State and deletion
                    {"name": "is_deleted", "dataType": ["boolean"], "description": "Marked as deleted"},
                    {"name": "deleted_by", "dataType": ["text"], "description": "User who deleted"},
                    {"name": "deletion_reason", "dataType": ["text"], "description": "Deletion reason"},

                    # Embeddings and processing
                    {"name": "embedding_model", "dataType": ["text"], "description": "Embedding model used"},
                    {"name": "processing_time", "dataType": ["number"], "description": "Processing time"},

                    # References and metadata
                    {"name": "referencias", "dataType": ["text"], "description": "Chunk references"},
                    {"name": "referencias_text", "dataType": ["text"], "description": "References text"},
                    {"name": "metadata_json", "dataType": ["text"], "description": "Full metadata as JSON"},

                    # Version
                    {"name": "schema_version", "dataType": ["text"], "description": "Schema version"}
                ]
            }

            await asyncio.to_thread(self.client.schema.create_class, class_obj)
            print(f"✅ Clase '{class_name}' creada exitosamente con {len(class_obj['properties'])} propiedades.")

        except Exception as e:
            print(f"❌ Error al crear la clase '{class_name}': {e}")
            raise

    async def exists_weaviate_class(self, class_name: str) -> bool:
        """
        Verifica si una clase ya existe en Weaviate.
        Si no existe, la crea empty con la estructura completa.

        Args:
            class_name (str): Nombre exacto de la clase a verificar.

        Returns:
            bool: True si la clase existe o fue creada exitosamente, False en caso de error.
        """
        try:
            # Verificar si la clase existe
            schema = await asyncio.to_thread(self.client.schema.get)
            class_names = [c['class'] for c in schema.get('classes', [])]

            if class_name in class_names:
                print(f"✅ La clase '{class_name}' ya existe en Weaviate.")
                return True

            # Si no existe, crearla empty
            print(f"⚠️ La clase '{class_name}' no existe. Creando...")
            await self.create_empty_class(class_name)
            print(f"✅ Clase '{class_name}' creada y lista para usar.")
            return True

        except Exception as e:
            print(f"❌ Error al verificar/crear la clase '{class_name}': {e}")
            return False

    def get_default_model(self):
        """
        Retorna el modelo por defecto del sistema.

        Returns:
            str: Nombre del modelo por defecto (gpt-oss-20b)
        """
        default_model = GPT_5_NANO
        print(f"[Config] Usando modelo por defecto: {default_model}")
        return default_model

    def get_available_models(self):
        """
        Retorna la lista de todos los modelos disponibles en el sistema.

        Returns:
            dict: Dictionary with available categories and models
        """
        available_models = {
            "openai": [
                GPT_5,
                GPT_5_1,
                GPT_5_4_MINI,
                GPT_5_NANO,
            ],
            "google": [
                GEMINI_3_1_PRO,
            ],
            "ollama": [
                "gpt-oss-20b",
            ],
        }

        all_models = []
        for category in available_models.values():
            all_models.extend(category)

        print(f"[Config] Modelos disponibles: {len(all_models)} total")
        return {
            "by_category": available_models,
            "all_models": all_models,
            "default": self.get_default_model(),
            "fallback": self.get_fallback_model()
        }

    def set_default_model(self, model_name: str):
        """
        Allows changing the default model dynamically.

        Args:
            model_name (str): Nombre del nuevo modelo por defecto

        Returns:
            bool: True si el cambio fue exitoso, False si el modelo no existe
        """
        available = self.get_available_models()["all_models"]

        if model_name not in available:
            print(f"[Error] Modelo '{model_name}' is not available.")
            print(f"[Info] Modelos disponibles: {available}")
            return False

        # Store in an instance variable to persist during the session
        self._custom_default_model = model_name
        print(f"[Config] Modelo por defecto cambiado a: {model_name}")
        return True

    def get_default_model_dynamic(self):
        """
        Dynamic version that checks whether a custom model has been configured.

        Returns:
            str: Modelo por defecto actual (personalizado o sistema)
        """
        # Verificar si hay un modelo personalizado configurado
        if hasattr(self, '_custom_default_model') and self._custom_default_model:
            print(f"[Config] Usando modelo personalizado: {self._custom_default_model}")
            return self._custom_default_model

        # Si no hay personalizado, usar el por defecto del sistema
        return self.get_default_model()

    def set_fallback_model(self, model_name: str):
        """
        Allows changing the fallback model dynamically.

        Args:
            model_name (str): Nombre del nuevo modelo de fallback

        Returns:
            bool: True si el cambio fue exitoso, False si el modelo no existe
        """
        available = self.get_available_models()["all_models"]

        if model_name not in available:
            print(f"[Error] Modelo de fallback '{model_name}' is not available.")
            print(f"[Info] Modelos disponibles: {available}")
            return False

        # Guardar en una variable de instancia
        self._custom_fallback_model = model_name
        print(f"[Config] Modelo de fallback cambiado a: {model_name}")
        return True

    def get_fallback_model(self):
        """
        Retorna el modelo de fallback cuando el modelo principal falla.

        Returns:
            str: Nombre del modelo de fallback (gpt-4.1-nano)
        """
        fallback_model = GPT_5_1
        print(f"[Config] Modelo de fallback configurado: {fallback_model}")
        return fallback_model
    def get_fallback_model_dynamic(self):
        """
        Dynamic version that checks whether a custom fallback has been configured.

        Returns:
            str: Modelo de fallback actual (personalizado o sistema)
        """
        # Verificar si hay un fallback personalizado configurado
        if hasattr(self, '_custom_fallback_model') and self._custom_fallback_model:
            print(f"[Config] Usando fallback personalizado: {self._custom_fallback_model}")
            return self._custom_fallback_model

        # Si no hay personalizado, usar el por defecto del sistema
        return self.get_fallback_model()

    async def run_model_with_default(self, rag_input: str, model: str = None, fallback_depth: int = 0) -> str:
        """
        Ejecuta un modelo usando gpt-oss-20b por defecto si no se especifica.
        Includes automatic fallback handling.

        Args:
            rag_input (str): Texto de entrada para el modelo
            model (str, optional): Specific model to use. If None, uses the default
            fallback_depth (int): Profundidad del fallback para evitar loops infinitos

        Returns:
            str: Respuesta del modelo
        """
        # Usar modelo por defecto si no se especifica
        if model is None:
            model = self.get_default_model_dynamic()
            print(f"[Info] Usando modelo por defecto: {model}")

        try:
            # Obtener el LLM para el modelo especificado
            llm = self.get_llm(model)

            # Ejecutar el modelo
            return await self.run_model(model, llm, rag_input, fallback_depth)

        except Exception as e:
            print(f"[Error] Error con modelo {model}: {str(e)}")

            # Si ya estamos en fallback profundo, evitar loop infinito
            if fallback_depth >= 2:
                return f"❌ Critical error: all models failed. Last error: {str(e)}"

            # Intentar con modelo de fallback
            fallback_model = self.get_fallback_model_dynamic()

            # Avoid falling back to itself
            if fallback_model == model:
                return f"❌ Error: model {model} failed and no alternative fallback is available. Error: {str(e)}"

            print(f"[Fallback] Cambiando de {model} a {fallback_model}")

            try:
                return await self.run_model_with_default(rag_input, model=fallback_model,
                                                              fallback_depth=fallback_depth + 1)
            except Exception as fallback_error:
                return f"❌ Error: Tanto el modelo principal ({model}) como el fallback ({fallback_model}) fallaron. Error principal: {str(e)}, Error fallback: {str(fallback_error)}"

    def _clean_user_query(self, query: str) -> str:
        """
        Clean the user's query of automatic formatting suffixes.
        """
        if not query:
            return query

        format_patterns = [
            r"\s+que\s+la\s+salida\s+sea\s+html\s+y\s+en\s+un\s+formato\s+profesional\s*$",
            r"\s+que\s+la\s+salida\s+sea\s+html.*$",
            r"\s+en\s+un\s+formato\s+profesional.*$",
            r"\s+formato\s+html\s+profesional.*$",
            r"\s+salida\s+html.*$",
        ]

        cleaned = query
        for pattern in format_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        return cleaned.strip()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calcula similitud coseno entre dos vectores"""
        import numpy as np

        if len(a.shape) == 1:
            a = a.reshape(1, -1)
        if len(b.shape) == 1:
            b = b.reshape(1, -1)

        return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

    async def _save_interaction(
            self, query, prompt_clean, response_text, user,
            conversation_id, timestamp, config_chatbot
    ):
        """Store the interaction in the DB and Redis."""
        query_limpia = self._clean_user_query(prompt_clean or query)

        # Guardar en BD tradicional
        await self.remember_interaction(
            user=user,
            user_message=query_limpia,
            bot_response=response_text,
            conversation_id=conversation_id,
            timestamp=timestamp,
            config_chatbot=config_chatbot
        )

    async def generate_suggested_questions(
            self,
            context: str,
            num_questions: int = 30,
    ) -> List[str]:
        """
        Generate suggested questions based on a provided context.
        CONSTRAINT: only questions answerable from the corpus.
        """

        prompt_template = f"""You are an expert in content analysis and relevant question generation.

    PROVIDED CONTEXT:
    {context}

    TASK:
    Analyze the context above and generate exactly {num_questions} questions a user could ask about this content.

    CRITICAL CONSTRAINT — READ BEFORE GENERATING:
    Each question MUST be answerable ONLY with information explicitly present
    in the context provided above.
    Before including each question, check: 'Is the answer in the text?'
    - YES it is → include
    - NO it is not, or it requires external knowledge → discard and generate another

    ALLOWED QUESTION TYPES (only if the answer is in the context):
    - Conceptual: What is X? What does Y mean? (where X and Y appear in the text)
    - Procedural: How is X done? What are the steps for Y?
    - Temporal: When does X happen? Within what deadline Y?
    - Requirements: What is needed for X? What are the documents for Y?
    - Comparative: What is the difference between X and Y? (ONLY if both X and Y appear in the text)
    - Causal: Why X? What is Y for?

    ADDITIONAL CRITERIA:
    1. Topical diversity: cover different aspects of the context
    2. Varied levels: basic, intermediate and advanced
    3. Clarity: direct and understandable without extra context
    4. No repetitions


    OUTPUT FORMAT — respond ONLY with this JSON, no extra text, no markdown:
    {{
        "questions": [
            "Question 1?",
            "Question 2?",
            "Question 3?"
        ]
    }}

    Generate exactly {num_questions} questions. Each one must end with '?'.
    """

        try:
            estimated_tokens = (num_questions * 25) + 1000
            max_tokens_to_use = max(estimated_tokens, 2500)

            response = await self.asyncOpenAI.chat.completions.create(
                model=GPT_5_4,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at generating questions answerable from a corpus. "
                            "Respond ONLY with valid JSON, no markdown, no explanations."
                        )
                    },
                    {"role": "user", "content": prompt_template}
                ],
                max_completion_tokens=max_tokens_to_use

            )

            choice = response.choices[0]
            logger.info(f"[DEBUG] finish_reason: {choice.finish_reason}")
            logger.info(f"[DEBUG] refusal: {choice.message.refusal}")
            logger.info(f"[DEBUG] content raw: {repr(choice.message.content)}")

            response_text = (choice.message.content or "").strip()
            logger.info(f"Respuesta del LLM recibida, longitud: {len(response_text)} characters")

            # Strip markdown fences si el modelo las incluye de todas formas
            response_text = re.sub(r'^```(?:json)?\s*|\s*```$', '', response_text.strip())

            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not match:
                logger.error(f"No JSON found in the response: '{response_text[:200]}'")
                raise ValueError("No JSON found in the model response")
            response_text = match.group()

            try:
                data = json.loads(response_text)

                if isinstance(data, dict) and "questions" in data:
                    questions = data["questions"]
                elif isinstance(data, list):
                    questions = data
                else:
                    logger.warning(f"Estructura inesperada en JSON: {type(data)}")
                    questions = []

                if not isinstance(questions, list):
                    logger.warning("The questions are not in list format")
                    questions = []

                if len(questions) < num_questions:
                    logger.warning(f"Se generaron {len(questions)} preguntas de {num_questions} solicitadas")
                elif len(questions) > num_questions:
                    questions = questions[:num_questions]

                clean_questions = []
                for question in questions:
                    if isinstance(question, str) and question.strip():
                        p = question.strip()
                        if not p.endswith('?'):
                            p += '?'
                        clean_questions.append(p)
                        logger.info(f"  {len(clean_questions):02d}. {p}")

                if not clean_questions:
                    raise ValueError("Could not extract valid questions from the JSON")

                logger.info(f"Se generaron {len(clean_questions)} preguntas sugeridas exitosamente")

                if len(clean_questions) < num_questions:
                    logger.info(f"Completando con fallback: {num_questions - len(clean_questions)} adicionales")
                    clean_questions.extend(
                        self._generate_fallback_questions(
                            context,
                            num_questions - len(clean_questions),
                        )
                    )

                return clean_questions

            except json.JSONDecodeError as e:
                logger.error(f"Error al parsear JSON: {e}")
                logger.error(f"Respuesta (primeros 1000 chars): {response_text[:1000]}")

                fallback_questions = []
                for linea in response_text.split('\n'):
                    linea = linea.strip()
                    if linea and ('?' in linea or linea.startswith('¿')):
                        linea = linea.lstrip('0123456789.-) ').strip()
                        linea = linea.strip('",')
                        if linea.startswith('"') and linea.endswith('"'):
                            linea = linea[1:-1]
                        if not linea.endswith('?'):
                            linea += '?'
                        if len(linea) > 5:
                            fallback_questions.append(linea)

                if fallback_questions:
                    logger.info(f"Fallback parsing manual: {len(fallback_questions)} preguntas")
                    if len(fallback_questions) < num_questions:
                        fallback_questions.extend(
                            self._generate_fallback_questions(
                                context,
                                num_questions - len(fallback_questions),
                            )
                        )
                    return fallback_questions[:num_questions]
                else:
                    raise ValueError("No se pudieron extraer preguntas ni del JSON ni del parsing manual")

        except Exception as e:
            logger.error(f"Error al generar preguntas sugeridas: {str(e)}")
            logger.exception("Traceback completo:")
            logger.warning("Using full basic fallback questions")
            return self._generate_fallback_questions(context, num_questions)


    def _generate_fallback_questions(self, context: str, num_questions: int) -> List[str]:
        """
        Generate generic fallback questions based on the context.

        Args:
            contexto: Contexto original
            num_questions: Number of questions to generate

        Returns:
            Lista de preguntas de fallback
        """
        base_questions = [
            f"What key information does this document contain?",
            "What are the key concepts mentioned?",
            "How can I apply this information?",
            "What requirements are mentioned?",
            "What are the steps to follow?",
            "What documentation is needed?",
            "Within what deadline must the procedures be carried out?",
            "Where can I find more information?",
            "Who is responsible for this process?",
            "When does this information take effect?",
            "What rights are covered?",
            "What obligations are established?",
            "What exceptions exist?",
            "How is follow-up carried out?",
            "What forms must I fill out?",
            "Whom should I contact for more details?",
            "What happens if I do not meet the requirements?",
            "What are the applicable penalties?",
            "How can I request a review?",
            "What special cases are considered?",
            "What is the appeal procedure?",
            "What costs are associated?",
            "How can I obtain a copy of the document?",
            "What recent changes have been made?",
            "What are the recommended best practices?",
            "What common mistakes should I avoid?",
            "How does this relate to other regulations?",
            "What additional resources are available?",
            "When must I renew or update this information?",
            "What support is available for further questions?"
        ]

        # Retornar las primeras N preguntas necesarias
        return base_questions[:num_questions]
    #### query augmentatation jaime ####

    async def generate_query_augmentation(
            self,
            original_query: str,
            doc_type: str = "General",
            num_questions: int = 5,
            context: str = None
    ) -> List[str]:
        """
        Genera preguntas sugeridas ESTRICTAMENTE basadas en el contexto documental.
        Soporta:
        - Meta questions: "¿Qué puedo preguntar?"
        - Thematic questions: "¿Cómo pregunto sobre X?"
        - Preguntas generales: genera sugerencias variadas
        """

        print(f"\n{'=' * 60}")
        print(f"🧠 GENERADOR DE PREGUNTAS SUGERIDAS")
        print(f"{'=' * 60}")
        print(f"💬 Query: {original_query[:80]}...")
        print(f"📄 Tipo: {doc_type}")
        print(f"📋 Contexto: {len(context) if context else 0} chars")

        try:
            # ═══════════════════════════════════════════════════════════
            # VALIDAR CONTEXTO
            # ═══════════════════════════════════════════════════════════

            if not context or len(context.strip()) < 30:
                return ["❌ No hay contexto documental para generar preguntas."]

            # ═══════════════════════════════════════════════════════════
            # DETECTAR TIPO DE CONSULTA
            # ═══════════════════════════════════════════════════════════

            query_type, tema_interes = await self._detect_query_type(original_query)

            print(f"🎯 Tipo: {query_type} | Tema: {tema_interes or 'N/A'}")

            # ═══════════════════════════════════════════════════════════
            # BUILD PROMPT BY TYPE
            # ═══════════════════════════════════════════════════════════

            base_context = f"""
    CONTEXTO DOCUMENTAL (ÚNICA FUENTE DE VERDAD):
    ══════════════════════════════════════════════════════════════
    {context}
    ══════════════════════════════════════════════════════════════
    TIPO: {doc_type}
    """

            if query_type == "meta":
                # ════════════════════════════════════════════════════════
                # META QUESTION: "What can I ask?"
                # ════════════════════════════════════════════════════════
                prompt = f"""{base_context}

    TAREA: El user quiere saber QUÉ PUEDE PREGUNTAR sobre este contexto.

    Analiza el contexto y genera {num_questions} ejemplos de preguntas que se pueden responder.

    PROCESO:
    1. Identifica los TEMAS principales del contexto (personas, cargos, fechas, acciones, etc.)
    2. Genera UNA pregunta ejemplo por cada tema identificado
    3. Las preguntas deben mostrar al user qué tipo de información puede consultar

    🚨 REGLAS ANTI-ALUCINACIÓN:
    - SOLO genera preguntas sobre información que EXISTE en el contexto
    - USA datos REALES: nombres exactos, cargos exactos, fechas exactas
    - NO inventes información

    FORMATO (solo preguntas numeradas que muestren qué se puede consultar):
    1.
    2.
    3.
    4.
    5.
    """

            elif query_type == "tematica":
                # ════════════════════════════════════════════════════════
                # THEMATIC QUESTION: "How do I ask about X?"
                # ════════════════════════════════════════════════════════
                prompt = f"""{base_context}

    TAREA: El user quiere saber cómo consultar sobre "{tema_interes}".

    Genera {num_questions} preguntas sobre "{tema_interes}" que se pueden responder CON EL CONTEXTO.

    🚨 REGLAS ANTI-ALUCINACIÓN:
    1. SOLO genera preguntas si "{tema_interes}" aparece o se relaciona con el contexto
    2. USA datos REALES del contexto
    3. Si "{tema_interes}" NO está en el contexto, responde EXACTAMENTE:
       "El tema '{tema_interes}' no se encuentra en el contexto disponible."
       Y luego lista {num_questions} preguntas sobre temas que SÍ están.

    FORMATO (solo preguntas numeradas):
    1.
    2.
    3.
    4.
    5.
    """

            else:
                # ════════════════════════════════════════════════════════
                # PREGUNTA GENERAL: Sugerencias variadas
                # ════════════════════════════════════════════════════════
                prompt = f"""{base_context}

    TAREA: Genera {num_questions} preguntas sugeridas para explorar este contexto.

    🚨 REGLAS ANTI-ALUCINACIÓN:
    - CADA pregunta debe ser RESPONDIBLE con el contexto
    - USA datos EXACTOS que aparecen (nombres, cargos, fechas, acciones)
    - NO inventes información que no esté presente
    - Variedad: quién, qué, por qué, cuándo, cómo

    FORMATO (solo preguntas numeradas):
    1.
    2.
    3.
    4.
    5.
    """

            # ═══════════════════════════════════════════════════════════
            # EJECUTAR Y PARSEAR
            # ═══════════════════════════════════════════════════════════

            response = await self.run_model_with_default(prompt, model=GPT_5_1)

            questions = self._parse_generated_questions(response, num_questions)

            # Manejar respuesta de "tema no encontrado"
            if not questions and ("no se encuentra" in response.lower() or "no está" in response.lower()):
                return [response.strip()]

            if not questions:
                questions = ["What is the summary of the available information?"]

            print(f"\n✅ Preguntas generadas:")
            for i, p in enumerate(questions, 1):
                print(f"   {i}. {p}")

            return questions[:num_questions]

        except Exception as e:
            print(f"❌ ERROR: {e}")
            return [f"Error: {str(e)}"]

    async def _detect_query_type(self, query: str) -> tuple:
        """
        Detecta el tipo de consulta:
        - "meta": ¿Qué puedo preguntar? ¿Qué información hay?
        - "tematica": ¿Cómo pregunto sobre X?
        - "general": Cualquier otra cosa

        Returns: (tipo, tema_interes)
        """

        prompt = f"""
    Analiza esta consulta: "{query}"

    Clasifica en:
    1. "meta" → Pregunta QUÉ puede preguntar o QUÉ información hay (sin tema específico)
    2. "tematica" → Pregunta CÓMO PREGUNTAR sobre un tema específico
    3. "general" → Otra consulta

    Ejemplos:
    - "¿Qué puedo preguntar?" → meta, null
    - "¿Qué información hay disponible?" → meta, null
    - "¿Sobre qué puedo consultar?" → meta, null
    - "¿Qué datos tiene el documento?" → meta, null
    - "¿Cómo pregunto sobre salarios?" → tematica, salarios
    - "Dame ejemplos de preguntas sobre riesgos" → tematica, riesgos
    - "¿Qué puedo preguntar sobre certificaciones?" → tematica, certificaciones
    - "Muéstrame los datos" → general, null
    - "Resume el documento" → general, null

    JSON (sin explicaciones):
    {{"tipo": "meta|tematica|general", "tema": "tema o null"}}
    """

        try:
            response = await self.run_model_with_default(prompt, model=GPT_5_1)

            # Limpiar JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            data = json.loads(response)
            tipo = data.get("tipo", "general")
            tema = data.get("tema")

            if tema and tema.lower() in ['null', 'none', '']:
                tema = None

            return (tipo, tema)

        except Exception as e:
            print(f"   ⚠️ Error detectando tipo: {e}")
            return ("general", None)



    def _parse_generated_questions(self, response: str, num_expected: int) -> List[str]:
        """Parsea las preguntas generadas del LLM"""
        import re

        questions = []

        # Intentar diferentes patrones
        patterns = [
            r'^\d+\.\s*(.+\?)$',  # 1. Question?
            r'^\d+\)\s*(.+\?)$',  # 1) Question?
            r'^[-•]\s*(.+\?)$',  # - Question?
            r'^[✅✓]\s*"(.+\?)"$',  # ✅ "Question?"
            r'^[✅✓]\s*(.+\?)$',  # ✅ Question?
        ]

        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue

            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    question = match.group(1).strip()
                    question = question.strip('"').strip("'")
                    if question and question not in questions:
                        questions.append(question)
                    break

        # If not enough were found, look for lines with "?"
        if len(questions) < num_expected:
            for line in response.split('\n'):
                line = line.strip()
                if '?' in line and line not in questions:
                    line = re.sub(r'^[\d\.\)\-•✅✓\s]+', '', line)
                    line = line.strip('"').strip("'").strip()
                    if line and line not in questions:
                        questions.append(line)

        return questions[:num_expected]

    def _generate_fallback_question_from_context(self, context: str, index: int) -> str:
        """
        Genera una pregunta de fallback basada en el contexto.

        Args:
            contexto: El contexto original
            index: Question index (for variety)

        Returns:
            Pregunta de fallback
        """
        templates = [
            f"What key information does this document contain?",
            "What are the key concepts mentioned?",
            "How can I apply this information?",
            "What requirements are mentioned?",
            "What are the steps to follow?",
            "What documentation do I need?",
            "Within what deadline must I carry out the procedures?",
            "Where can I find more details?",
            "Who is responsible for this process?",
            "When does it take effect?",
            "What rights do I have under this document?",
            "What obligations are established?",
            "What costs are associated?",
            "How can I request more information?",
            "What exceptions exist?",
            "What is the detailed procedure?",
            "What forms must I fill out?",
            "Whom should I contact?",
            "What happens if I do not comply?",
            "How can I follow up?",
            "What recent changes are there?",
            "What are the special conditions?",
            "What practical cases exist?",
            "How does this apply to my situation?",
            "What alternatives do I have available?",
            "What is this document's validity period?",
            "What penalties apply?",
            "How can I appeal a decision?",
            "What security measures exist?",
            "Where do I submit my application?"
        ]

        return templates[index % len(templates)]

    def _fallback_questions_generation(self, context: str, num_questions: int) -> List[str]:
        """
        Genera preguntas de fallback cuando el proceso principal falla.

        Args:
            contexto: Contexto original
            num_questions: Number of questions to generate

        Returns:
            Lista de preguntas de fallback
        """
        print(f"[FALLBACK] Generating {num_questions} fallback questions")

        questions = []
        for i in range(num_questions):
            question = self._generate_fallback_question_from_context(context, i)
            questions.append(question)

        return questions

    def _normalize_chunks_for_weaviate(self, splited, metadata):
        """
        Normaliza chunks para Weaviate incluyendo filename en el texto
        """
        if not splited:
            return []

        # Convierte metadata a JSON string una sola vez
        md_json = json.dumps(metadata or {}, ensure_ascii=False)

        # Extraer filename del metadata
        filename = metadata.get('file_name', 'archivo_desconocido.pdf') if metadata else 'archivo_desconocido.pdf'

        normalized_chunks = []

        for chunk in splited:
            if isinstance(chunk, str):
                # Incluir filename al inicio del chunk
                enriched_text = f"ARCHIVO: {filename}\n{chunk}"

                normalized_chunks.append({
                    "text": enriched_text,  # Texto con filename incluido
                    "metadata_json": md_json
                })
            elif isinstance(chunk, dict):
                # Si ya es dict, extraer texto y enriquecerlo
                original_text = chunk.get("text", "")
                enriched_text = f"ARCHIVO: {filename}\n{original_text}"

                normalized_chunks.append({
                    "text": enriched_text,  # Texto enriquecido con filename
                    "metadata_json": chunk.get("metadata_json", md_json)
                })

        return normalized_chunks

    class AsyncBatchClient:
        async def __aenter__(self):
            print("Inicializando cliente de base de datos...")
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            print("Cerrando cliente de base de datos...")

        async def add_data_object(self, data_object, vector):
            print(f"Insertando: {data_object} con vector {vector[:5]}...")
            await asyncio.sleep(0.1)  # Simular latencia de base de datos


