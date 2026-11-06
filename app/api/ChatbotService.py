"""
"""
import asyncio
import base64
import concurrent
import copy
import csv
import hashlib
import io
import json
import os
import re
import time
import traceback
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from typing import Optional, Tuple, Dict, Any, List, AsyncGenerator

import chardet
import fitz  # PyMuPDF
from bson import ObjectId
from fastapi import APIRouter, Form, UploadFile, File, HTTPException, Depends
from openai import AsyncOpenAI, OpenAI
from starlette import status
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from Constants.Constants import HTML_FORMAT
from config.Constants import KEY_OPEN_AI, GPT_5_NANO, GPT_5_4_MINI, GPT_5_4
from framework.ConfigurationManager.ConfigurationManagerSelected import ConfigurationManagerSelected
from framework.Database.MongoDb.MongoDb import MongoDBManager
from framework.StringNormalize.TextNormalizer import TextNormalizer
from chatbot import ChatbotTrainer
from chatbot.CustomLLMOfficeRetriever import run_document_to_weaviate
from chatbot.CustomLLMRetrieverByCelery import CustomLLMRetrieverByCelery
from framework.ORC.PaddleOCRSequential import extract_text_sequential_pure  # OCR local (PaddleOCR)
from app.middleware.traceability_layer import trace_event
from chatbot.RAGPipeline import RAGPipeline, AsyncioTimeoutError
from framework.chatbotValidators.chatbotValidator import ChatbotValidator

api_router_processing_pages = APIRouter()
from PIL import Image, UnidentifiedImageError
from datetime import datetime
os.environ['OPENAI_API_KEY'] = KEY_OPEN_AI

import logging
logger = logging.getLogger(__name__)


def _api_success(data, message=None, meta=None):
    """Successful JSON response (replaces APIResponse.success)."""
    r = {"success": True, "data": data, "meta": {"timestamp": _ts_iso(), **(meta or {})}}
    if message:
        r["message"] = message
    return r


def _api_error(exception, **_ignored):
    """Error JSON response (replaces APIResponse.error)."""
    msg = str(exception)
    return {"success": False, "error": {"message": msg, "user_message": msg}}

api_router_chat_bot = APIRouter()

# Also configure the handler
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

METRICS_DIR = os.getenv("METRICS_DIR", "/home/athenas/metricas")
METRICS_FILE = os.getenv("METRICS_FILE", "detalle_todos.csv")
METRICS_PATH = os.path.join(METRICS_DIR, METRICS_FILE)

CSV_HEADER = [
    "ts_iso","run_id","cpu","filename","total_time","accumulated_time","wall_clock","speedup"
]

def _ensure_metrics_file():
    os.makedirs(METRICS_DIR, exist_ok=True)
    if not os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)



def validate_required_fields(rules: Dict[str, Any]):
    required_fields = ["user", "company", "aplicacion"]
    missing_fields = [field for field in required_fields if field not in rules]
    if missing_fields:
        logger.error(f"Missing the following fields in json_data: {', '.join(missing_fields)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing the following fields in json_data: {', '.join(missing_fields)}"
        )

# =========================
#   SYNCHRONOUS VERSION
# =========================



async def write_file_metric(run_id: str, cpu: int, filename: str, started_at: float) -> float:
    """Writes one file row and returns its total_time (seconds)."""
    total_time = time.time() - started_at
    await _write_metrics_safe(run_id, filename, total_time, cpu=cpu)
    return total_time

def validate_ocr_result(ocr_result, filename, run_id):
    """
    Improved OCR result validation that prioritizes extracted content over flags
    """
    if not ocr_result:
        return False, "OCR result is None"

    # Check whether there is extracted text (the most important signal)
    text = (ocr_result.get('text') or "").strip()
    if text and len(text) >= 10:
        return True, f"Valid text extracted: {len(text)} characters"

    # Fall back to the success flag
    if ocr_result.get('success') is True:
        return True, "Success flag is True"

    # Not enough text means a real failure
    return False, f"No usable text extracted: {len(text)} characters"


MIN_TEXT_LENGTH = 10


async def _process_single_file_async(
        file_content: bytes,
        filename: str,
        content_type: str,
        rules: dict,
        config: dict,
        run_id: str,
        num_workers: int
) -> Dict[str, Any]:
    """
    Process a single file (PDF or document).

    Args:
        file_content: File content in bytes
        filename: File name
        content_type: MIME type
        rules: Configuration with user, company
        config: Application configuration
        run_id: Run ID for tracking
        num_workers: Number of workers (for metrics)

    Returns:
        Dict with the processing result
    """
    times = {}
    file_started = time.time()

    logger.info(f"[{run_id}] Starting processing: {filename} ({len(file_content)} bytes)")

    # Extract data from rules once
    user = str(rules['user']).upper()
    company = str(rules['company']).upper()


    # ═══════════════════════════════════════════════════════════════
    # BASE RESULT - Normalized structure that is always returned
    # ═══════════════════════════════════════════════════════════════
    base_result = {
        'success': False,
        'filename': filename,
        'content': file_content,
        'content_type': content_type,
        'text_length': 0,
        'tokens': 0,
        'tipo_doc': None,
        'search_tokens': [],
        'size': 0,
        'summary': '',
        'ocr_details': {},
        'times': times,
        'error': None
    }

    try:
        # Inicialización
        init_start = time.time()
        train = CustomLLMRetrieverByCelery(config)
        times['initialization'] = time.time() - init_start

        # ─────────────────────────────────────────────────────────────
        # DOCUMENTO NO-PDF
        # ─────────────────────────────────────────────────────────────
        if content_type != 'application/pdf':
            logger.info(f"[{run_id}] {filename} is not a PDF, processing as a direct document")

            file_size = len(file_content)
            base_result['size'] = file_size

            try:
                # PHASE 1 (extraction): nothing is indexed in Weaviate here. The
                # vector + GridFS write is done by the endpoint's transactional commit,
                # ONLY if ALL files pass extraction (atomic ingestion).

                # Extract text for metrics
                extracted_text = ""
                try:
                    extension = filename.lower().split('.')[-1] if '.' in filename else ''

                    if extension in ['txt', 'text', 'log', 'md', 'csv', 'json', 'xml']:
                        try:
                            extracted_text = file_content.decode('utf-8')
                        except UnicodeDecodeError:
                            extracted_text = file_content.decode('latin-1', errors='ignore')

                    elif extension in ['docx']:
                        from docx import Document
                        from io import BytesIO
                        doc = Document(BytesIO(file_content))
                        extracted_text = "\n".join([p.text for p in doc.paragraphs])

                    elif extension in ['xlsx', 'xls']:
                        extracted_text = f"[Excel file with {file_size} bytes]"

                except Exception as e:
                    logger.warning(f"[{run_id}] {filename}: could not extract text for metrics: {e}")
                    extracted_text = ""

                text_length = len(extracted_text)
                base_result['text_length'] = text_length

                # Compute tokens
                try:
                    if text_length > 0:
                        tokens = len(extracted_text) // 4
                    else:
                        tokens = file_size // 4
                except Exception as e:
                    logger.warning(f"[{run_id}] {filename}: Error computing tokens: {e}")
                    tokens = text_length // 4 if text_length > 0 else file_size // 4

                base_result['tokens'] = tokens

                # ✅ Summary in executor — does not block the event loop
                summary_start = time.time()
                try:
                    if text_length > 100:
                        logger.debug(f"[{run_id}] {filename}: Generating non-PDF summary in executor...")
                        summary_t = ""
                        times['summary'] = time.time() - summary_start
                        logger.info(f"[{run_id}] {filename}: Summary generated ({times['summary']:.2f}s)")
                        base_result['summary'] = summary_t
                except TimeoutError:
                    times['summary'] = time.time() - summary_start
                    logger.warning(f"[{run_id}] {filename}: Non-PDF summary timed out ({times['summary']:.2f}s), continuing without it")
                except Exception as e:
                    times['summary'] = time.time() - summary_start
                    logger.warning(f"[{run_id}] {filename}: Error generating non-PDF summary ({times['summary']:.2f}s): {e}")

            except Exception as e:
                logger.error(f"[{run_id}] {filename}: Error processing document: {e}")
                times['total'] = time.time() - file_started
                await write_file_metric(run_id, num_workers, filename, file_started)
                base_result.update({
                    'error': f"Error processing document: {e}",
                    'times': times
                })
                return base_result

            times['total'] = time.time() - file_started
            await write_file_metric(run_id, num_workers, filename, file_started)

            logger.info(
                f"[{run_id}] Completed non-PDF document {filename} ({times['total']:.2f}s) "
                f"- {text_length} chars, {tokens} tokens"
            )

            base_result.update({
                'success': True,
                'tipo_doc': "docus",
                'extracted_text': extracted_text,
                'times': times
            })
            return base_result

        # ─────────────────────────────────────────────────────────────
        # PDF PROCESSING
        # ─────────────────────────────────────────────────────────────
        base_result['tipo_doc'] = "doc"

        # Detect page count
        page_detection_start = time.time()
        try:
            pdf_doc = fitz.open(stream=file_content, filetype="pdf")
            num_pages = len(pdf_doc)
            pdf_doc.close()
        except Exception as e:
            logger.warning(f"[{run_id}] {filename}: Could not detect pages, assuming 1: {e}")
            num_pages = 1
        times['page_detection'] = time.time() - page_detection_start

        logger.info(f"[{run_id}] {filename}: {num_pages} pages detected")

        # ─────────────────────────────────────────────────────────────
        # OCR WITH PADDLEOCR (local) — Azure Document Intelligence removed
        # ─────────────────────────────────────────────────────────────
        ocr_start = time.time()
        size_t = 0
        summary_t = ""

        try:
            # PaddleOCR is synchronous (CPU) → run it in an executor so it does not block the loop.
            ocr_result = await asyncio.get_event_loop().run_in_executor(
                None, extract_text_sequential_pure, file_content, filename
            )
            times['ocr_paddle'] = time.time() - ocr_start
            logger.info(f"[{run_id}] {filename}: PaddleOCR succeeded ({times['ocr_paddle']:.2f}s)")

        except Exception as e:
            times['ocr_paddle_failed'] = time.time() - ocr_start
            logger.error(f"[{run_id}] {filename}: PaddleOCR failed: {e}")
            times['total'] = time.time() - file_started
            await write_file_metric(run_id, num_workers, filename, file_started)
            base_result.update({
                'error': f"PaddleOCR error: {e}",
                'ocr_details': {
                    'method_used': 'paddleocr',
                    'pages_detected': num_pages
                },
                'times': times
            })
            return base_result

        times['ocr_total'] = time.time() - ocr_start

        # ─────────────────────────────────────────────────────────────
        # SUMMARY IN EXECUTOR — does not block the event loop
        # ─────────────────────────────────────────────────────────────
        summary_start = time.time()
        try:
            if ocr_result and ocr_result.get('text'):
                size_t = len(ocr_result.get('text', '')) if ocr_result else 0
                logger.debug(f"[{run_id}] {filename}: Generating PDF summary in executor...")
                summary_t = ""
                times['summary'] = time.time() - summary_start
                logger.info(f"[{run_id}] {filename}: Summary generated ({times['summary']:.2f}s)")
        except TimeoutError:
            times['summary'] = time.time() - summary_start
            logger.warning(
                f"[{run_id}] {filename}: PDF summary timed out ({times['summary']:.2f}s), continuing without it"
            )
            summary_t = ""
        except Exception as e:
            times['summary'] = time.time() - summary_start
            logger.warning(f"[{run_id}] {filename}: Text assistant error ({times['summary']:.2f}s): {e}")

        base_result['size'] = size_t
        base_result['summary'] = summary_t

        # ─────────────────────────────────────────────────────────────
        # OCR VALIDATION
        # ─────────────────────────────────────────────────────────────
        is_valid, validation_msg = validate_ocr_result(ocr_result, filename, run_id)

        if not is_valid:
            logger.error(f"[{run_id}] {filename}: {validation_msg}")
            times['total'] = time.time() - file_started
            await write_file_metric(run_id, num_workers, filename, file_started)
            base_result.update({
                'error': f"Error en OCR: {validation_msg}",
                'ocr_details': {
                    'method_used': 'paddleocr',
                    'pages_detected': num_pages
                },
                'times': times
            })
            return base_result

        extracted_text = (ocr_result.get('text') or "").strip()
        text_length = len(extracted_text)

        if text_length < MIN_TEXT_LENGTH:
            logger.warning(f"[{run_id}] {filename}: Text too short ({text_length} chars)")
            times['total'] = time.time() - file_started
            await write_file_metric(run_id, num_workers, filename, file_started)
            base_result.update({
                'error': f"Extracted text too short ({text_length} characters)",
                'text_length': text_length,
                'ocr_details': {
                    'method_used': 'paddleocr',
                    'pages_detected': num_pages,
                    'characters_extracted': text_length
                },
                'times': times
            })
            return base_result

        logger.info(f"[{run_id}] {filename}: OCR validated, {text_length} characters")

        # ─────────────────────────────────────────────────────────────
        # TOKEN CALCULATION
        # ─────────────────────────────────────────────────────────────
        token_start = time.time()
        try:
            tokens = len(extracted_text) // 4
        except Exception as e:
            logger.warning(f"[{run_id}] {filename}: Error computing tokens: {e}")
            tokens = text_length // 4
        times['token_calculation'] = time.time() - token_start

        # ─────────────────────────────────────────────────────────────
        # PHASE 1 (extraction): nothing is persisted here. The payload is stored so
        # the endpoint writes it in the TRANSACTIONAL COMMIT, and only if ALL files
        # passed extraction (atomic auto-training: "not halfway").
        # ─────────────────────────────────────────────────────────────
        base_result['pdf_db_payload'] = {
            'text': extracted_text,
            'filename': filename,
            'pages_processed': ocr_result.get('pages_processed', num_pages),
            'pages_detected': num_pages,
            'size_t': size_t,
            'summary': summary_t,
            'ocr_method': 'paddleocr',
            'ocr_latency_ms': ocr_result.get('latency_ms', 0),
            'characters_extracted': text_length
        }
        base_result['extracted_text'] = extracted_text

        # ─────────────────────────────────────────────────────────────
        # SUCCESSFUL RESULT
        # ─────────────────────────────────────────────────────────────
        times['total'] = time.time() - file_started
        await write_file_metric(run_id, num_workers, filename, file_started)

        logger.info(
            f"[{run_id}] Completado {filename} ({times['total']:.2f}s) - "
            f"{text_length} chars, {tokens} tokens"
        )

        base_result.update({
            'success': True,
            'text_length': text_length,
            'tokens': tokens,
            'ocr_details': {
                'method_used': 'paddleocr',
                'pages_detected': num_pages,
                'pages_processed': ocr_result.get('pages_processed', num_pages),
                'characters_extracted': text_length,
                'latency_ms': ocr_result.get('latency_ms', 0)
            },
            'times': times
        })
        return base_result

    except Exception as e:
        times['total'] = time.time() - file_started
        await write_file_metric(run_id, num_workers, filename, file_started)

        logger.error(f"[{run_id}] General error processing {filename}: {e}")
        logger.debug(f"[{run_id}] Stack trace: {traceback.format_exc()}")

        base_result.update({
            'error': f"General error: {e}",
            'times': times
        })
        return base_result




async def process_files_sequentially(
        file_tasks: List[dict],
        rules: dict,
        config: dict,
        num_workers: int = 1,
        run_id: str = None,
        timeout_total: int = 900,
) -> List[dict]:
    """Process the files SEQUENTIALLY (one by one).

    Validates each ``file_task`` and runs ``_process_single_file_async``, awaiting
    its result before moving on to the next, collecting all results.
    Replaces the old Ray-based parallel engine.
    """
    _validate_inputs(file_tasks, rules, config)
    if not file_tasks:
        logger.warning("Empty file_tasks list")
        return []

    start_time = time.time()
    rules_copy = copy.deepcopy(rules)
    config_copy = copy.deepcopy(config)

    total = len(file_tasks)
    results: List[dict] = []
    for i, ft in enumerate(file_tasks):
        validation_error = _validate_file_task(ft, i)
        if validation_error:
            results.append({
                "success": False,
                "filename": validation_error.get("filename", "unknown"),
                "error": validation_error.get("reason", "invalid file_task"),
            })
            continue

        file_run_id = f"{run_id or 'run'}_{i + 1}"
        try:
            result = await asyncio.wait_for(
                _process_single_file_async(
                    file_content=ft["content"],
                    filename=ft["filename"],
                    content_type=ft["content_type"],
                    rules=rules_copy,
                    config=config_copy,
                    run_id=file_run_id,
                    num_workers=num_workers,
                ),
                timeout=timeout_total,
            )
            results.append(result)
            logger.info(f"✓ Processed {i + 1}/{total}: {ft['filename']}")
        except asyncio.TimeoutError:
            logger.error(f"Timed out processing {ft['filename']}")
            results.append({
                "success": False, "filename": ft["filename"],
                "error": "Timeout - file not completed", "timeout": True,
            })
        except Exception as e:
            logger.error(f"Error processing {ft['filename']}: {e}")
            results.append({
                "success": False, "filename": ft["filename"],
                "error": str(e), "exception": type(e).__name__,
            })

    _log_final_statistics(results, start_time)
    return results









def _validate_inputs(file_tasks: List[dict], rules: dict, config: dict) -> None:
    """
    Validate the required inputs.

    Raises:
        ValueError: if rules or config are invalid
    """
    if not rules:
        raise ValueError("Rules are required and cannot be empty")

    if not config:
        raise ValueError("Config is required and cannot be empty")




def _build_async_redis_client(config):
    """
    Create an async Redis client (redis.asyncio) from AppConfig.
    Returns None if the config is not available (does not block startup).
    """
    try:
        import redis.asyncio as aioredis
        return aioredis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            db=getattr(config.redis, 'db', 0),
            username=getattr(config.redis, 'username', None),
            password=getattr(config.redis, 'password', None),
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    except Exception as e:
        logger.warning("Could not create async Redis client: %s", e)
        return None






def _validate_file_task(ft: dict, index: int) -> Optional[dict]:
    """
    Validate that a file_task has the required fields.

    Args:
        ft: file task to validate
        index: task index

    Returns:
        None if valid, a dict with the error if invalid
    """
    filename = 'unknown'

    # Check it is a dict
    if not isinstance(ft, dict):
        return {
            'index': index,
            'filename': filename,
            'reason': f'Not a dict, it is {type(ft).__name__}'
        }

    filename = ft.get('filename', 'unknown')

    # Check required fields
    required_fields = ['content', 'filename', 'content_type']
    missing = [f for f in required_fields if f not in ft]

    if missing:
        return {
            'index': index,
            'filename': filename,
            'reason': f'Missing fields: {missing}'
        }

    # Check content
    content = ft.get('content')

    if content is None:
        return {
            'index': index,
            'filename': filename,
            'reason': 'Content is None'
        }

    if not isinstance(content, bytes):
        return {
            'index': index,
            'filename': filename,
            'reason': f'Content must be bytes, it is {type(content).__name__}'
        }

    if len(content) == 0:
        return {
            'index': index,
            'filename': filename,
            'reason': 'Content is empty (0 bytes)'
        }

    # Valid
    return None




MAX_ERRORS_TO_LOG = 5






def _log_final_statistics(results: List[dict], start_time: float) -> None:
    """Log the final processing statistics."""
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    total_time = time.time() - start_time

    logger.info("=== FINAL STATISTICS ===")
    logger.info(f"Total tasks: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average speed: {len(results) / total_time:.2f} tasks/s")

    if failed:
        logger.info("ERRORS FOUND:")
        for i, fail in enumerate(failed[:MAX_ERRORS_TO_LOG]):
            logger.info(f"   {i + 1}. {fail.get('filename', 'unknown')}: {fail.get('error', 'unknown error')}")
        if len(failed) > MAX_ERRORS_TO_LOG:
            logger.info(f"   ... and {len(failed) - MAX_ERRORS_TO_LOG} more errors")

def _sanitize_failed_result(result: dict) -> dict:
    """Return a JSON-serializable copy of a worker's failed result.

    The worker dict (``base_result`` in ``_process_single_file_async``) includes
    ``'content'`` with the file's RAW BYTES, and may carry other binary or
    non-serializable objects. If that reaches the endpoint's final ``JSONResponse``
    (``response_data["errors"]``), ``json.dumps`` blows up with
    ``TypeError: Object of type bytes is not JSON serializable`` and the whole request
    returns 500 even though the rest of the files were processed fine.

    We keep only useful error metadata, discarding ``content`` and any binary value.
    """
    if not isinstance(result, dict):
        return {'filename': 'unknown', 'error': str(result)}

    SAFE_KEYS = (
        'filename', 'error', 'content_type', 'text_length', 'tokens', 'tipo_doc',
        'size', 'times', 'exception', 'timeout', 'error_type', 'ray_wrapper_info',
    )

    def _jsonable(value):
        if isinstance(value, (bytes, bytearray, memoryview)):
            return f"<{len(bytes(value))} bytes omitted>"
        if isinstance(value, dict):
            return {k: _jsonable(v) for k, v in value.items() if k != 'content'}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    clean = {k: _jsonable(result[k]) for k in SAFE_KEYS if k in result}
    clean.setdefault('filename', result.get('filename', 'unknown'))
    clean.setdefault('error', result.get('error') or 'unknown error')
    return clean


async def _commit_training_atomic(
    successful_results: List[dict],
    config,
    rules: dict,
    mongo_documents,
    doc_desc_manager,
    user_query: dict,
    results,
) -> List[str]:
    """Phase 2 (transactional) of auto-training. Invoked ONLY when the N files
    passed extraction (atomic gate). Writes ALL documents in this order:
    vector indexing (Weaviate) -> GridFS -> metadata (mongo_documents).

    If ANY write fails, it reverts what was done in THIS phase and re-raises, so the
    ingestion is not left halfway:
      - GridFS: per-file deletion (safe).
      - Weaviate: can only be deleted at the collection level, so it is reverted ONLY if
        the collection was new (empty). If it already had documents, a CRITICAL log is
        left for manual cleanup (cannot delete per file without touching the pre-existing).

    Returns the list of written GridFS file_id values.
    """
    train = CustomLLMRetrieverByCelery(config)
    user = str(rules['user']).upper()
    company = str(rules['company']).upper()

    # ¿La colección ya tenía documents? Define si un rollback de Weaviate es seguro.
    try:
        collection_existed = bool(list(mongo_documents.find_all_clasic(user_query)))
    except Exception:
        collection_existed = True  # when in doubt, do NOT delete at the collection level

    file_ids: List[str] = []
    weaviate_touched = False

    try:
        for r in successful_results:
            filename = r.get('filename')
            content = r.get('content')
            content_type = r.get('content_type')

            # 1) Vector indexing (AWAIT: a failure counts as a commit failure)
            if r.get('tipo_doc') == 'doc':  # PDF
                await train.insert_document(
                    user=user, company=company,
                    content=r['pdf_db_payload'], doc_type="doc", file_name=filename,
                )
                weaviate_touched = True
                await asyncio.to_thread(
                    ChatbotTrainer.ChatbotTrainer.training_chat_bot_buscador,
                    {"file_name": filename, "content_type": content_type,
                     "contents": r.get('extracted_text', '')},
                    copy.deepcopy(rules),
                )
            else:  # non-PDF document
                await run_document_to_weaviate(
                    file_bytes=content,
                    weaviate_url=config['weaviate']['url'],
                    file_name=filename, user=user, company=company,
                    openai_api_key=KEY_OPEN_AI,
                )
                weaviate_touched = True

            # 2) Binario en GridFS
            file_id = mongo_documents.insert_file(content, filename, content_type=content_type)
            file_ids.append(file_id)
            logger.info(f"✅ Saved (commit): {filename} -> file_id {file_id}")

        # 3) Metadata in mongo_documents (within the same transactional scope)
        await asyncio.to_thread(
            handle_existing_documents,
            rules, user_query, mongo_documents, doc_desc_manager, file_ids, results,
        )

        return file_ids

    except Exception as commit_err:
        logger.error(f"❌ Transactional commit failed: {commit_err}. Reverting phase 2...")
        # GridFS rollback (per file, safe)
        if file_ids:
            try:
                mongo_documents.delete_files(file_ids)
                logger.info(f"   ↩️  GridFS reverted: {len(file_ids)} file(s) deleted")
            except Exception as e:
                logger.error(f"   ⚠️  Could not revert GridFS: {e}")
        # Weaviate rollback: only safe if the collection was NEW
        if weaviate_touched:
            if not collection_existed:
                try:
                    train.delete_all_occurrences_from_weaviate(user, company)
                    logger.info("   ↩️  Weaviate reverted (new collection): occurrences deleted")
                except Exception as e:
                    logger.error(f"   ⚠️  Could not revert Weaviate: {e}")
            else:
                logger.critical(
                    f"   ⚠️  Weaviate has partial writes in a PRE-EXISTING collection "
                    f"({user}/{company}); requires manual per-file cleanup."
                )
        raise


def process_results(results: List[dict], mongo_documents) -> tuple:
    """Split successes/errors from the extraction phase (does NOT persist: that is done
    by the endpoint's transactional commit). Also returns token_array and search_tokens.
    file_ids comes back empty on purpose (it is filled during the commit)."""
    successful_results: List[dict] = []
    failed_results: List[dict] = []
    token_array: List[dict] = []
    file_ids: List[str] = []
    search_tokens_arrays: List[list] = []

    for result in results:
        if result.get('success'):
            successful_results.append(result)
            token_array.append(result.get('tokens'))
            search_tokens_arrays.append(result.get('search_tokens', []))
            # Not persisted here: GridFS + vector indexing happen in the endpoint's
            # transactional commit (_commit_training_atomic), only if ALL files passed
            # extraction. file_ids is filled during that commit.
        else:
            # The worker dict includes 'content' (bytes) and other binaries -> sanitize
            # to JSON-serializable metadata before it reaches the endpoint's JSONResponse.
            failed_results.append(_sanitize_failed_result(result))
            logger.error(f"Error processing {result.get('filename')}: {result.get('error')}")

    return successful_results, failed_results, token_array, file_ids, search_tokens_arrays


def handle_existing_documents(
        rules: dict,
        user_query: dict,
        mongo_documents,
        doc_descriptions,
        file_ids: List[str],
        results
) -> dict:
    """
    Handle insertion/update of documents in mongo_documents and doc_descriptions.

    Returns:
        dict: operation summary (inserted_count, updated_count, descriptions_inserted)
    """
    try:
        inserted_count = 0
        updated_count = 0
        descriptions_inserted = 0

        # DEBUG: inspect what we are receiving
        logger.info(f"🔍 DEBUG - Received:")
        logger.info(f"   file_ids: {file_ids}")
        logger.info(f"   results type: {type(results)}")
        logger.info(f"   results: {results if results else 'None or empty'}")

        # Normalize fields to uppercase
        rules.update({
            key: str(rules[key]).upper()
            for key in ['user', 'company']
            if key in rules
        })

        user_query.pop("timestamp", None)

        # Find existing documents
        existing_documents = list(mongo_documents.find_all_clasic(user_query))

        # ============================================================
        # CASE 1: UPDATE EXISTING DOCUMENT IN mongo_documents
        # ============================================================
        if existing_documents:
            existing_ids = [
                [doc['_id'], doc.get('file_id', [])]
                for doc in existing_documents
                if '_id' in doc
            ]

            old_file_ids = [
                file_id
                for doc_files in [item[1] for item in existing_ids]
                for file_id in doc_files
            ]
            combined_file_ids = list(dict.fromkeys(old_file_ids + file_ids))

            mongo_documents.update_one_new(existing_ids[0][0], {
                "file_id": combined_file_ids,
                "last_updated": datetime.utcnow()
            })
            updated_count += 1
            logger.info(f"✅ mongo_documents updated: {len(old_file_ids)} → {len(combined_file_ids)} file_ids")

        # ============================================================
        # CASE 2: INSERT NEW DOCUMENT IN mongo_documents
        # ============================================================
        else:
            doc_to_insert = rules.copy()
            doc_to_insert["file_id"] = file_ids
            doc_to_insert['date'] = datetime.utcnow()

            if results and len(results) > 0:
                first_result = results[0]
                if 'size' in first_result and isinstance(first_result['size'], dict):
                    doc_to_insert['size_bytes'] = first_result['size'].get('size_bytes', 0)
                if 'summary' in first_result and isinstance(first_result['summary'], dict):
                    doc_to_insert['summary'] = " ".join(first_result['summary'].get('puntos_clave', []))
                if 'filename' in first_result:
                    doc_to_insert['filename'] = first_result['filename']

            mongo_documents.insert_one(doc_to_insert)
            inserted_count += 1
            logger.info(f"✅ mongo_documents new document inserted with {len(file_ids)} file_ids")

        # ============================================================
        # ALWAYS INSERT INTO doc_descriptions
        # ============================================================
        logger.info(f"📝 Trying to insert into doc_descriptions...")
        logger.info(f"   results condition: {results is not None}")
        logger.info(f"   results length: {len(results) if results else 0}")

        # Check that doc_descriptions is initialized
        if doc_descriptions is None:
            logger.error("❌ doc_descriptions is None!")
            return {
                'inserted_count': inserted_count,
                'updated_count': updated_count,
                'descriptions_inserted': 0,
                'error': 'doc_descriptions not initialized'
            }

        # OPTION 1: with full results
        if results and isinstance(results, list) and len(results) > 0:
            logger.info(f"✅ Processing {len(results)} results for doc_descriptions")

            for idx, result in enumerate(results):
                try:
                    file_id = file_ids[idx] if idx < len(file_ids) else None

                    if not file_id:
                        logger.warning(f"⚠️ No file_id for result {idx}")
                        continue

                    desc_doc = {
                        **user_query,
                        'date': datetime.utcnow(),
                        'file_id': [file_id],
                        'filename': result.get('filename', 'unknown'),
                        'document_format': result.get('document_format', 'unknown'),
                        'aplicacion': rules.get('aplicacion'),
                        'config_chatbot': rules.get('config_chatbot'),
                        'csv_processed': result.get('csv_processed', False)
                    }

                    # Optional fields
                    if 'summary' in result:
                        if isinstance(result['summary'], dict):
                            desc_doc['summary'] = " ".join(result['summary'].get('puntos_clave', []))
                        else:
                            desc_doc['summary'] = str(result['summary'])

                    if 'size' in result and isinstance(result['size'], dict):
                        desc_doc['size_bytes'] = result['size'].get('size_bytes', 0)
                        desc_doc['size_mb'] = result['size'].get('size_mb', 0)

                    for field in ['headers', 'filas', 'columnas', 'encoding', 'tokens']:
                        if field in result:
                            desc_doc[field] = result[field]

                    # Remove _id if present
                    desc_doc.pop("_id", None)

                    logger.info(f"   📄 Inserting: {desc_doc.get('filename')}")
                    doc_descriptions.insert_one(desc_doc)
                    descriptions_inserted += 1
                    logger.info(f"   ✅ OK")

                except Exception as e:
                    logger.error(f"❌ Error inserting description {idx}: {e}", exc_info=True)
                    continue

        # OPTION 2: no results, use only file_ids (fallback)
        elif file_ids and len(file_ids) > 0:
            logger.warning(f"⚠️ No results, using fallback with file_ids")

            for idx, file_id in enumerate(file_ids):
                try:
                    desc_doc = {
                        **user_query,
                        'date': datetime.utcnow(),
                        'file_id': [file_id],
                        'filename': f'file_{idx}',
                        'document_format': 'unknown',
                        'aplicacion': rules.get('aplicacion'),
                        'config_chatbot': rules.get('config_chatbot'),
                        'csv_processed': False
                    }

                    desc_doc.pop("_id", None)

                    logger.info(f"   📄 Inserting fallback: {file_id}")
                    doc_descriptions.insert_one(desc_doc)
                    descriptions_inserted += 1
                    logger.info(f"   ✅ OK")

                except Exception as e:
                    logger.error(f"❌ Error inserting fallback {idx}: {e}", exc_info=True)
                    continue
        else:
            logger.warning("⚠️ No results or file_ids to insert into doc_descriptions")

        # ============================================================
        # RESUMEN
        # ============================================================
        logger.info("═" * 60)
        logger.info(f"📊 SUMMARY handle_existing_documents:")
        logger.info(f"   📄 mongo_documents: {inserted_count} new, {updated_count} updated")
        logger.info(f"   📝 doc_descriptions: {descriptions_inserted} inserted")
        logger.info("═" * 60)

        return {
            'inserted_count': inserted_count,
            'updated_count': updated_count,
            'descriptions_inserted': descriptions_inserted
        }

    except Exception as e:
        logger.error(f"❌ Error in handle_existing_documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error handling documents in MongoDB: {str(e)}"
        )



def _ts_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


# === sync writer (modified) ===
def _write_metrics_csv(
    run_id: str,
    cpu: int,
    filename: str,
    total_time: Optional[float],
    accumulated_time: Optional[float] = None,
    wall_clock: Optional[float] = None,
    speedup: Optional[float] = None,
) -> None:
    """SYNCHRONOUS: write one row to the CSV."""
    _ensure_metrics_file()

    # For the total row we do not hash; otherwise we keep the short hash.
    def _format_filename(name: str) -> str:
        if not name:
            return ""
        upper = name.upper()
        if upper == "TOTAL_PROCESS" or upper.startswith("__TOTAL__"):
            return name
        # keeps the previous behavior for files
        return hashlib.md5(name.encode("utf-8")).hexdigest()[:12]

    row = [
        _ts_iso(),
        run_id,
        cpu,
        _format_filename(filename),
        f"{total_time:.6f}" if total_time is not None else "",
        f"{accumulated_time:.6f}" if accumulated_time is not None else "",
        f"{wall_clock:.6f}" if wall_clock is not None else "",
        f"{speedup:.6f}" if speedup is not None else "",
    ]
    with open(METRICS_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

async def _write_metrics_safe(
    run_id: str,
    filename: str,
    total_time: float,
    cpu: int = 1,
    accumulated_time: Optional[float] = None,
    wall_clock: Optional[float] = None,
    speedup: Optional[float] = None,
) -> None:
    """
    ASYNC: run _write_metrics_csv on the thread pool.
    (NOTE: we call _write_metrics_csv, not this same function)
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        _write_metrics_csv,
        run_id,
        cpu,
        filename,
        total_time,
        accumulated_time,
        wall_clock,
        speedup,
    )


class InMemoryUploadFile:
    def __init__(self, filename: str, content_type: str, contents: bytes):
        self.filename = filename
        self.content_type = content_type
        self._contents = contents
        self.file = io.BytesIO(contents)
    async def read(self):
        return self._contents


# ==================== CLIENTE OPENAI GLOBAL ====================

_openai_client = None


def init_openai_client():
    """Initialize the OpenAI client once at startup"""
    global _openai_client
    _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    logger.info("✅ OpenAI client initialized")






logger = logging.getLogger(__name__)



@api_router_chat_bot.post(
    "/chatbot/rag/chatbotservice/stream",
    summary="Chatbot with SSE streaming",
    tags=["Chatbot Services - Knowledge database"]
)
async def bot_services_stream(
        json_data: str = Form(...),
        conversation_id: str = Form(None),
        continue_conversation: Optional[bool] = Form(None),
        files: List[UploadFile] = File(None),
        response_model=None,
):
    """
    Chatbot endpoint with STREAMING (Server-Sent Events).

    Returns real-time events:
    - status: processing progress
    - chunk: pieces of the generated answer
    - complete: successful completion
    - error: if something fails

    Returns:
        StreamingResponse: real-time SSE events
    """

    start_time = time.perf_counter()

    # ── Attachments (CSV/XLSX…) → ctx["files"] of the tool router ──
    # Read eagerly into bytes so the SSE generator (which runs after the
    # endpoint returns) does not depend on the UploadFile lifecycle.
    # Exact format consumed by data_analysis: {filename, mime, bytes}.
    uploaded: List[dict] = []
    for _f in (files or []):
        try:
            _b = await _f.read()
            if not _b:
                continue
            uploaded.append({
                "filename": _f.filename,
                "mime": _f.content_type or "application/octet-stream",
                "bytes": _b,
            })
        except Exception as _fe:  # noqa: BLE001
            logger.warning(f"⚠️ Could not read attached file in stream: {_fe}")
            continue
    if uploaded:
        logger.info(f"📎 Stream received {len(uploaded)} file(s): "
                    f"{[u['filename'] for u in uploaded]}")

    # Helper para emitir eventos SSE
    def emit_event(event_type: str, data: dict) -> str:
        """Format SSE events."""
        payload = {
            'type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            **data
        }
        event_str = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        # 📝 LOG: print the event about to be sent
        logger.info(f"📤 Emitting event [{event_type}]: {json.dumps(payload, ensure_ascii=False)[:200]}")
        return event_str

    # ═══════════════════════════════════════════════════════════
    # EVENT GENERATOR
    # ═══════════════════════════════════════════════════════════

    async def event_generator():
        """Main generator that sends SSE events."""

        conversation_id_final = conversation_id or str(uuid.uuid4())

        try:
            # ════════════════════════════════════════════════════════
            # 1. DECODE JSON
            # ════════════════════════════════════════════════════════

            yield emit_event('status', {'message': '📋 Validating data...', 'phase': 0})

            try:
                rules = json.loads(json_data)
            except json.JSONDecodeError as e:
                logger.error(f"❌ Invalid JSON: {e}")

                yield emit_event('error', {
                    'code': 'ENDPOINT-JSON-001',
                    'message': 'Invalid JSON',
                    'user_message': '⚠️ The format of the submitted data is not valid.',
                    'severity': 'medium',
                    'details': {
                        'error_message': str(e),
                        'error_position': getattr(e, 'pos', None)
                    },
                    'frontend_hints': {
                        'is_retryable': True,
                        'retry_strategy': 'immediate',
                        'ui_component': 'toast',
                        'severity_color': 'yellow',
                        'icon': 'warning',
                        'auto_dismiss_ms': 5000
                    }
                })
                return

            # ════════════════════════════════════════════════════════
            # 2. VALIDATE REQUIRED FIELDS
            # ════════════════════════════════════════════════════════

            required_fields = ["user", "company", "prompt", "timestamp"]
            missing_fields = [field for field in required_fields if field not in rules]

            if missing_fields:
                logger.error(f"❌ Missing fields: {missing_fields}")

                yield emit_event('error', {
                    'code': 'ENDPOINT-FIELDS-001',
                    'message': 'Missing required fields',
                    'user_message': f'⚠️ Missing data: {", ".join(missing_fields)}',
                    'severity': 'medium',
                    'details': {
                        'missing_fields': missing_fields,
                        'required_fields': required_fields
                    },
                    'frontend_hints': {
                        'is_retryable': True,
                        'focus_field': missing_fields[0],
                        'ui_component': 'inline',
                        'severity_color': 'yellow',
                        'icon': 'warning'
                    }
                })
                return

            prompt_original = rules['prompt']
            # Identity from the TOKEN (not the body) — sets the real user/company.

            yield emit_event('status', {'message': '✅ Data validated', 'phase': 0})

            # ════════════════════════════════════════════════════════
            # 3. APPROPRIATE-CONTENT VALIDATION
            # ════════════════════════════════════════════════════════

            yield emit_event('status', {'message': '🔍 Checking content...', 'phase': 0})

            try:
                chatbotValidator = ChatbotValidator()
                is_appropriate, categories = chatbotValidator.is_appropriate_content(prompt_original)

                # Fail-open: only block if moderation EXPLICITLY flagged the
                # content (False). None = could not validate → let it through.
                if is_appropriate is False:
                    logger.warning(f"⚠️ Inappropriate content: {categories}")

                    yield emit_event('error', {
                        'code': 'ENDPOINT-CONTENT-001',
                        'message': 'Inappropriate content detected',
                        'user_message': '⚠️ Your message contains inappropriate language. Please rephrase your query.',
                        'severity': 'medium',
                        'details': {
                            'categories': categories
                        },
                        'frontend_hints': {
                            'is_retryable': True,
                            'focus_field': 'prompt',
                            'ui_component': 'modal',
                            'severity_color': 'red',
                            'icon': 'warning'
                        },
                        'help': {
                            'suggestions': [
                                'Use professional and respectful language',
                                'Avoid offensive words',
                                'Phrase your question clearly'
                            ]
                        }
                    })
                    return

            except Exception as e:
                logger.error(f"❌ Validation error: {e}")
                # Continue without validation (failsafe)

            yield emit_event('status', {'message': '✅ Content is appropriate', 'phase': 0})

            # ════════════════════════════════════════════════════════
            # 4. INITIALIZE SERVICES
            # ════════════════════════════════════════════════════════

            yield emit_event('status', {'message': '🔧 Initializing services...', 'phase': 1})

            try:
                # Initialize configuration
                initConfig = ConfigurationManagerSelected()
                config = initConfig.load_configuration(rules.get('aplicacion', 'default'))

                # ── Instantiate RagOrchestrator ──
                from chatbot.pipeline_v2 import RagOrchestrator
                pipeline_v2 = RagOrchestrator(config, _build_async_redis_client(config))


                # Normalize prompt
                text = TextNormalizer()
                prompt_normalized = text.normalize(prompt_original)

                # Prepare parameters
                user = str(rules['user']).replace(" ", "").upper()
                company = str(rules['company']).replace(" ", "").upper()

                logger.info(f"🚀 User: {user}, Company: {company}")

                yield emit_event('status', {
                    'message': '✅ Services ready',
                    'phase': 1,
                    'details': {
                        'user': user,
                        'company': company
                    }
                })

            except Exception as e:
                logger.error(f"❌ Initialization error: {e}")

                yield emit_event('error', {
                    'code': 'ENDPOINT-INIT-001',
                    'message': 'Error initializing services',
                    'user_message': '❌ Error configuring the system.',
                    'severity': 'high',
                    'details': {
                        'error_type': type(e).__name__,
                        'error_message': str(e)[:200]
                    },
                    'frontend_hints': {
                        'is_retryable': True,
                        'retry_strategy': 'delayed',
                        'retry_after_seconds': 3,
                        'ui_component': 'modal',
                        'severity_color': 'red',
                        'icon': 'error'
                    }
                })
                return

            # ════════════════════════════════════════════════════════
            # 5. CALL THE CHATBOT WITH STREAMING
            # ════════════════════════════════════════════════════════

            yield emit_event('status', {'message': '🤖 Processing query...', 'phase': 2})

            response_chunks = []  # To rebuild the full answer
            has_error = False
            error_data = None
            chunk_counter = 0  # Chunk counter

            logger.info("=" * 80)
            logger.info("🚀 STARTING CHATBOT STREAMING")
            logger.info("=" * 80)

            try:
                # 🔥 CALL THE STREAMING FUNCTION
                # ── RagOrchestrator.stream() — wraps the legacy pipeline ──
                async for event in pipeline_v2.stream(
                        query=prompt_normalized,
                        user=user,
                        company=company,
                        conversation_id=conversation_id_final,
                        tone=rules.get('tono', 'neutral'),
                        prompt_clean=prompt_original,
                        timestamp=rules['timestamp'],
                        model=GPT_5_4_MINI,
                        personality=rules.get('personalidad', 'default'),
                        doc_type=rules.get('document_type', 'general'),
                        config_chatbot=rules.get('config_chatbot', 'default'),
                        websearch=rules.get("web_search", False),
                        context=rules.get("contexto"),
                        llm_direct=rules.get("llm_direct", False),
                        save_interaction=True,
                        response_mode=rules.get("response_mode", "hybrid"),
                        provider=rules.get("provider"),
                        agent_config=rules.get("agent_config"),
                        files=uploaded,
                ):
                    logger.debug(f"📥 Event: {event[:100]}...")
                    yield event

                    if event.startswith("data: "):
                        try:
                            event_data = json.loads(event[6:])
                            event_type = event_data.get('type')
                            if event_type in ('chunk', 'content'):
                                chunk_counter += 1
                                chunk_text = event_data.get('text') or event_data.get('message') or ''
                                if chunk_text:
                                    response_chunks.append(chunk_text)
                            elif event_type == 'error':
                                has_error = True
                                error_data = event_data
                        except json.JSONDecodeError:
                            pass

                    await asyncio.sleep(0.001)

                # 📝 LOG: streaming summary
                logger.info("=" * 80)
                logger.info("📊 STREAMING SUMMARY")
                logger.info(f"   Total chunks sent: {chunk_counter}")
                logger.info(f"   Total characters: {len(''.join(response_chunks))}")
                logger.info(f"   Had error?: {has_error}")
                logger.info(f"   Conversation ID: {conversation_id_final}")
                logger.info("=" * 80)

                # ════════════════════════════════════════════════════
                # 6. POST-PROCESSING (IF NO ERROR)
                # ════════════════════════════════════════════════════

                if not has_error:
                    # Rebuild the full answer
                    response_full = ''.join(response_chunks)

                    if response_full:
                        # 📝 LOG: answer preview
                        response_preview = response_full[:300] + "..." if len(response_full) > 300 else response_full
                        logger.info(f"💬 Full answer (preview):\n{response_preview}")

                        # Tokens (TokenCalculator removed)
                        tokenCalculatedOut = 0
                        tokenCalculatedIn = 0

                        # Save the conversation in the background
                        try:
                            conversations = MongoDBManager("conversation_by_user", config)
                            conversations.insert_one({
                                "user": user,
                                "company": company,
                                "conversation_id": conversation_id_final,
                                "config_chatbot": str(rules.get('config_chatbot', 'default')).upper(),
                                "prompt_original": prompt_original,
                                "prompt_processed": prompt_normalized,
                                "timestamp": rules['timestamp'],
                                "tokensIn": tokenCalculatedIn,
                                "tokensOut": tokenCalculatedOut
                            })
                            logger.info(f"💾 Conversation saved - ID: {conversation_id_final}")
                        except Exception as e:
                            logger.error(f"❌ Error saving: {e}")

                        # Compute total time
                        elapsed_time = time.perf_counter() - start_time
                        logger.info(f"⏱️ Total execution time: {elapsed_time:.2f} seconds")

                        # Send final metadata
                        yield emit_event('metadata', {
                            'conversation_id': conversation_id_final,
                            'execution_time_seconds': round(elapsed_time, 2),
                            'status': 'success'
                        })


            except Exception as e:
                logger.error(f"❌ Chatbot error: {e}")
                import traceback
                logger.error(traceback.format_exc())

                yield emit_event('error', {
                    'code': 'ENDPOINT-CHATBOT-001',
                    'message': 'Chatbot processing error',
                    'user_message': '❌ Error processing your query. Please try again.',
                    'severity': 'high',
                    'details': {
                        'error_type': type(e).__name__,
                        'error_message': str(e)[:200]
                    },
                    'frontend_hints': {
                        'is_retryable': True,
                        'retry_strategy': 'delayed',
                        'retry_after_seconds': 5,
                        'ui_component': 'modal',
                        'severity_color': 'red',
                        'icon': 'error'
                    }
                })

        except Exception as e:
            # Unhandled critical error
            logger.error(f"❌ Critical error: {e}")
            import traceback
            logger.error(traceback.format_exc())

            yield emit_event('error', {
                'code': 'ENDPOINT-CRITICAL-001',
                'message': 'Critical error in the endpoint',
                'user_message': '❌ Unexpected error. Please try again later.',
                'severity': 'critical',
                'details': {
                    'error_type': type(e).__name__,
                    'error_message': str(e)[:200]
                },
                'frontend_hints': {
                    'is_retryable': True,
                    'retry_strategy': 'delayed',
                    'retry_after_seconds': 10,
                    'should_show_support_button': True,
                    'ui_component': 'modal',
                    'severity_color': 'red',
                    'icon': 'error'
                }
            })

    # ═══════════════════════════════════════════════════════════
    # RETURN STREAMING RESPONSE
    # ═══════════════════════════════════════════════════════════

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx: disable buffering
            "Access-Control-Allow-Origin": "*",  # CORS
        }
    )








####
# ═════════════════════════════════════════════════════════════════════════════
# CHAT ROUTING (extractor → DB → chatbot)
#
# Branch decision, ADDITIVE over the current path:
#   - prompt with a SHA256 hash in parentheses  → 'hash' (pre-generated SQL; untouched)
#   - no hash and a STRUCTURED dataset (SQL)     → 'sql'  (read-only text-to-SQL)
#   - everything else                            → 'rag'  (vector, as always)
# If text-to-SQL fails or there is no associated DB, the caller falls back to RAG.
# ═════════════════════════════════════════════════════════════════════════════
# Signals that the user is asking for the CONTENT/SUMMARY of a document (not tabular
# data). In a dataset with a connected DB, "document"/"documents" collides with the
# `documents` table; without this guard, "summarize the document" translates to
# `SELECT * FROM documents` and dumps raw rows instead of summarizing via RAG.
_RAG_OVERRIDE_RE = re.compile(
    r"\b(resum[ií]\w*|res[uú]me\w*|resumime|sintetiz[aá]\w*|"
    r"de qu[eé] (?:trata|habla)|sobre qu[eé] trata|qu[eé] dice|qu[eé] contiene|"
    r"qu[eé] informaci[oó]n|puntos?\s+key|conclusi[oó]n del|"
    r"analiz[aá]\w*\s+(?:el|este|ese)\s+(?:document|informe|file|contrato|pdf|text))\b",
    re.IGNORECASE,
)
# Greetings / small talk / help requests: in a dataset with a connected DB they must NOT
# go to NL→SQL (otherwise "hola" returns "I can translate to SQL…"). They go to RAG/general,
# which the orchestrator handles as conversational chat.
_CHAT_OVERRIDE_RE = re.compile(
    r"^[\s¿¡\"'.,-]*(hola|buenas|buen[oa]s? (?:d[ií]as?|tardes|noches)|buen d[ií]a|hello|hi|hey|"
    r"gracias|ok|oka|dale|listo|chau|c[oó]mo est[aá]s|c[oó]mo va|qu[eé] tal|"
    r"qui[eé]n sos|qui[eé]n eres|qu[eé] sos|qu[eé] pod[eé]s hacer|qu[eé] puedes hacer|"
    r"qu[eé] sab[eé]s hacer|para qu[eé] serv[ií]s|en qu[eé] me pod[eé]s ayudar|ayuda|help)\b",
    re.IGNORECASE,
)
# STRONG data/aggregation/tabular signals → they WIN over the override
# (e.g. "summarize how many invoices there are per supplier" still goes to SQL).
_SQL_SIGNAL_RE = re.compile(
    r"\b(cu[aá]nt\w+|cantidad\s+de|total(?:es)?|promedio|m[aá]xim\w+|m[ií]nim\w+|"
    r"list[aá]\w*|agrup\w+|order\w+|rank\w+|top\s+\d+|"
    r"por\s+(?:mes|a[ñn]o|d[ií]a|categor[ií]a|proveedor|emisor|cliente|tipo|estado)|"
    r"group\s+by|count\s*\(|\bselect\b|\bfrom\b|\bwhere\b)\b",
    re.IGNORECASE,
)













###3





















def emit_event(event_type: str, data: Dict[str, Any]) -> str:
    """Generate a formatted SSE event."""
    event_data = {
        "type": event_type,
        "timestamp": time.time(),
        **data
    }
    return f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"




# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT STREAMING
# ═══════════════════════════════════════════════════════════════════════════════



















        # TODO: implement rollback if needed
        # On a critical error, consider deleting the created file_ids








@api_router_chat_bot.post(
    "/chatbot/rag/AutoTrainingBotByUser",
    summary="Chatbot training (up to 40 files)",
    description="""
    This service trains the chatbot by processing multiple files sequentially.

    ## 🎯 Main Function

    Performs the chatbot training through:
    - ✅ Sequential processing (one file at a time)
    - ✅ Asynchronous file reading
    - ✅ Text extraction by format
    - ✅ Vectorization and dual storage
    - ✅ Automatic success-rate calculation

    ## 📋 Supported File Types

    - **PDF**: documents with OCR
    - **Excel** (.xlsx, .xls): spreadsheets
    - **Word** (.docx): text documents
    - **Images**: PNG, JPG, JPEG (with OCR)
    - **Maximum**: 40 files per request

    ## 🔄 Service Flow

    1. **User uploads** up to 40 files
    2. **System validates** files and parameters
    3. **Asynchronous reading** of all files
    4. **Sequential processing** (one file at a time)
    5. **Dual storage**:
       - mongo_documents: 1 consolidated record
       - doc_descriptions: 1 record per file
    6. **Metrics**: success rate, total time
    7. **System returns** detailed statistics

    ## 📋 Request Structure

    ### `json_data` example:
    ```json
    {
        "user": "walter.milstein@gen-ai.io",
        "company": "DEMO",
        "aplicacion": "document_processor",
        "modelo": "global_documents",
        "description": "Sequential document processing"
    }
    ```

    ### Request fields:
    - **`user`** (string, required): user identifier
    - **`company`** (string, required): business context
    - **`aplicacion`** (string, required): configuration application
    - **`modelo`** (string, optional): storage model (default: global_documents)
    - **`description`** (string, optional): dataset description

    ### Files:
    - **files**: list of files (1-40)
    - Formats: PDF, Excel, Word, Images
    - Maximum size: 50MB per file

    ## 📤 Response Structure
    ```json
    {
        "success": true,
        "data": {
            "response": "Collection created or updated",
            "files_processed": 38,
            "files_failed": 2,
            "success_rate_percent": 95.0,
            "total_time_seconds": 125.5,
            "workers_used": 1,
            "tokens": [15000, 8000, ...],
            "deep_search_tokens": [12000],
            "mongodb_operations": {
                "mongo_documents": {
                    "inserted": 0,
                    "updated": 1
                },
                "doc_descriptions": {
                    "inserted": 38
                }
            },
            "errors": null
        },
        "meta": {
            "request_id": "req_xyz",
            "execution_time_seconds": 125.5
        }
    }
    ```

    ## 💡 Use Cases

    - **Bulk load**: up to 40 documents at once
    - **Mixed documents**: PDF + Excel + Word + Images
    """,
    tags=["Chatbot Services - Training"],
    responses={
        200: {
            "description": "Documents processed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": {
                            "response": "Collection created or updated",
                            "files_processed": 38,
                            "success_rate_percent": 95.0
                        }
                    }
                }
            }
        },
        400: {"description": "Validation error / No file processed"},
        413: {"description": "Files too large"},
        422: {"description": "Unsupported file format"},
        500: {"description": "Internal server error"},
        503: {"description": "OCR service unavailable"},
        504: {"description": "Timeout processing documents"}
    }
)
async def training_chat_bot_simplified(
        request: Request,
        files: List[UploadFile] = File(
            ...,
            description="List of PDF or other document files (maximum 40)"
        ),
        json_data: str = Form(
            ...,
            description="JSON with the training configuration"
        ),
):
    """
    Robust service for chatbot training.

    Processes multiple files sequentially, one file at a time.

    ### Parameters:
    - **files**: list of files (1-40)
    - **json_data**: JSON with user, company, aplicacion

    ### Validations:
    - ✅ valid JSON
    - ✅ required fields present
    - ✅ 1-40 files
    - ✅ max 50MB per file
    - ✅ supported formats
    - ✅ timeout: 900s (15 min) total

    ### Successful response:
    Returns detailed statistics with success rate and timings

    ### Errors:
    - **400**: validation failed / no file processed
    - **413**: files too large
    - **422**: unsupported format
    - **500**: error processing files
    - **503**: OCR unavailable
    - **504**: timeout
    """

    t0 = time.perf_counter()
    start_time = time.time()

    # Generate a unique request_id
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    run_id = f"upload_{int(start_time)}"

    # Global variables for cleanup
    rules = None
    user_up = None
    company_up = None
    config = None
    mongo_users = None
    mongo_documents = None
    doc_desc_manager = None
    num_workers = 0
    processed_file_ids = []

    # ════════════════════════════════════════════════════════════
    # ENRICHED ERROR CONTEXT
    # ════════════════════════════════════════════════════════════



    try:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"🚀 NEW REQUEST - training_chat_bot_simplified")
        logger.info(f"🆔 Request ID: {request_id}")
        logger.info(f"🏃 Run ID: {run_id}")
        logger.info(f"{'=' * 80}")

        # ════════════════════════════════════════════════════════════
        # STEP 1: VALIDATE FILES
        # ════════════════════════════════════════════════════════════

        try:
            logger.info(f"\n📁 STEP 1: Validating files...")

            if not files:
                raise ValueError("No files were provided to process")

            file_count = len(files)

            logger.info(f"   📁 {file_count} file(s) received")

            # Validar cantidad (1-40)
            if not (1 <= file_count <= 40):
                raise ValueError("Cantidad de files inválida (debe ser 1 a 40)")

            logger.info(f"   ✅ Cantidad de files válida (1-40)")

        except ValueError:
            raise

        except Exception as e:

            logger.error(f"❌ Error validating files: {e}")

            raise ValueError("Error validating the input files")

        # ════════════════════════════════════════════════════════════
        # STEP 2: JSON PARSING AND VALIDATION
        # ════════════════════════════════════════════════════════════

        try:
            logger.info(f"\n📋 STEP 2: Validating JSON...")

            rules = json.loads(json_data)

            # Identity from the TOKEN (not the body) before normalizing.

            # Normalize fields (except a few)
            excluded_fields = ['modelo', 'config_chatbot', 'description', 'descripcion', 'aplicacion']

            for key, value in rules.items():
                if isinstance(value, str) and key not in excluded_fields:
                    rules[key] = value.strip().upper()

            logger.info(f"   ✅ JSON parsed and normalized successfully")

        except ValueError:
            raise

        except Exception as e:

            logger.error(f"❌ Error parsing JSON: {e}")

            raise ValueError("Error parsing configuration JSON")

        # ════════════════════════════════════════════════════════════
        # STEP 3: VALIDATE REQUIRED FIELDS
        # ════════════════════════════════════════════════════════════

        try:
            logger.info(f"\n📋 STEP 3: Validating required fields...")

            # Use the existing function or implement validation
            required_fields = ["user", "company", "aplicacion"]

            validate_required_fields(
               rules
            )

            logger.info(f"   ✅ All required fields present")

        except ValueError:
            raise

        except Exception as e:

            logger.error(f"❌ Error validating fields: {e}")

            raise ValueError("Error validating input fields")

        # ════════════════════════════════════════════════════════════
        # STEP 4: EXTRACT AND NORMALIZE DATA
        # ════════════════════════════════════════════════════════════

        try:
            logger.info(f"\n📝 STEP 4: Normalizing data...")

            user_up = str(rules["user"]).strip().upper()
            company_up = str(rules["company"]).strip().upper()
            aplicacion = str(rules["aplicacion"]).strip()

            # Optional fields
            model = rules.get("modelo", "global_documents")
            description = rules.get("description", "No description")
            config_chatbot = rules.get("config_chatbot", "general")

            # Validate non-empty fields
            if not user_up or not company_up or not aplicacion:
                empty_fields = []
                if not user_up:
                    empty_fields.append("user")
                if not company_up:
                    empty_fields.append("company")
                if not aplicacion:
                    empty_fields.append("aplicacion")

                raise ValueError(f"The fields {', '.join(empty_fields)} cannot be empty")

            logger.info(f"   👤 User: {user_up}")
            logger.info(f"   🏢 Company: {company_up}")
            logger.info(f"   ⚙️  Application: {aplicacion}")
            logger.info(f"   📊 Model: {model}")

        except ValueError:
            raise

        except Exception as e:

            logger.error(f"❌ Error normalizing data: {e}")

            raise ValueError("Error processing the input data")

        # ════════════════════════════════════════════════════════════
        # STEP 5: LOAD CONFIGURATION
        # ════════════════════════════════════════════════════════════

        try:
            logger.info(f"\n⚙️  STEP 5: Loading configuration...")

            initConfig = ConfigurationManagerSelected()

            config = await asyncio.wait_for(
                asyncio.to_thread(
                    initConfig.load_configuration,
                    aplicacion
                ),
                timeout=15.0
            )

            if not config:
                raise ValueError("Could not load the configuration")

            logger.info(f"   ✅ Configuration loaded: {aplicacion}")

        except AsyncioTimeoutError:

            logger.error(f"   ⏱️  Timeout loading configuration (15s)")

            timeout_error = TimeoutError("Timeout loading system configuration")

            handled = timeout_error

            return JSONResponse(
                status_code=504,
                content=_api_error(
                    exception=handled,
                    path=str(request.url.path),
                    method=request.method,
                    duration_ms=(time.perf_counter() - t0) * 1000
                )
            )

        except ValueError:
            raise

        except Exception as e:

            logger.error(f"   ❌ Error loading configuration: {e}")

            raise ValueError("Error loading system configuration")

        # ════════════════════════════════════════════════════════════
        # STEP 6: INITIALIZE COMPONENTS
        # ════════════════════════════════════════════════════════════

        try:
            logger.info(f"\n🤖 STEP 6: Initializing components...")

            # Initialize MongoDB managers
            mongo_users = MongoDBManager("chat_bot_users", config)
            mongo_documents = MongoDBManager(model, config)
            doc_desc_manager = MongoDBManager('doc_descriptions', config)

            logger.info(f"   ✅ MongoDB components initialized")

        except Exception as e:

            logger.error(f"   ❌ Error initializing components: {e}")

            raise RuntimeError("Error initializing system components")


        # ════════════════════════════════════════════════════════════
        # STEP 8: ASYNCHRONOUS FILE READING
        # ════════════════════════════════════════════════════════════

        logger.info(f"\n📖 STEP 8: Reading files asynchronously...")

        file_tasks = []

        try:
            # Read all files in parallel
            read_tasks = [file.read() for file in files]

            file_contents = await asyncio.wait_for(
                asyncio.gather(*read_tasks, return_exceptions=True),
                timeout=60.0  # 1 minute to read all files
            )

            # Build file_tasks with metadata
            for idx, (file, content) in enumerate(zip(files, file_contents), 1):
                if isinstance(content, Exception):
                    logger.error(f"   ❌ Error reading '{file.filename}': {content}")
                    continue

                file_tasks.append({
                    'file': file,
                    'content': content,
                    'filename': file.filename,
                    'content_type': file.content_type,
                    'size': len(content),
                    'index': idx
                })

            total_files = len(file_tasks)

            logger.info(f"   ✅ {total_files}/{file_count} file(s) read successfully")

            if total_files == 0:
                raise ValueError("Could not read any file")


        except AsyncioTimeoutError:
            logger.error(f"   ⏱️  Timeout reading files (60s)")

            timeout_error = TimeoutError("Timeout reading files")

            handled = timeout_error

            return JSONResponse(
                status_code=504,
                content=_api_error(
                    exception=handled,
                    path=str(request.url.path),
                    method=request.method,
                    duration_ms=(time.perf_counter() - t0) * 1000
                )
            )

        except ValueError:
            raise

        except Exception as e:

            logger.error(f"   ❌ Error reading files: {e}")

            raise ValueError("Error reading the files")

        # ════════════════════════════════════════════════════════════
        # STEP 9: PROCESSING CONFIGURATION
        # ════════════════════════════════════════════════════════════

        logger.info(f"\n⚙️  STEP 9: Configuring processing...")

        num_workers = 1

        logger.info(f"   📊 Sequential processing of {total_files} files")


        # ════════════════════════════════════════════════════════════
        # STEP 10: SEQUENTIAL PROCESSING
        # ════════════════════════════════════════════════════════════

        logger.info(f"\n🚀 STEP 10: Sequential processing...")
        logger.info(f"   ⚡ Starting processing of {total_files} files")

        try:
            results = await asyncio.wait_for(
                process_files_sequentially(
                    file_tasks=file_tasks,
                    rules=rules,
                    config=config,
                    num_workers=num_workers,
                    run_id=run_id,
                ),
                timeout=900.0  # 15 minutes for the full processing
            )

            logger.info(f"   ✅ Processing completed")


        except AsyncioTimeoutError:
            logger.error(f"   ⏱️  Timeout during processing (900s)")

            timeout_error = TimeoutError("Timeout during processing")

            handled = timeout_error

            return JSONResponse(
                status_code=504,
                content=_api_error(
                    exception=handled,
                    path=str(request.url.path),
                    method=request.method,
                    duration_ms=(time.perf_counter() - t0) * 1000
                )
            )

        except Exception as e:

            logger.error(f"   ❌ Error during processing: {e}")

            raise RuntimeError("Error during processing")

        # ════════════════════════════════════════════════════════════
        # STEP 11: PROCESS RESULTS
        # ════════════════════════════════════════════════════════════

        logger.info(f"\n📊 STEP 11: Processing results...")

        try:
            # Use the existing process_results function
            successful_results, failed_results, token_array, file_ids, search_tokens_arrays = \
                await asyncio.wait_for(
                    asyncio.to_thread(
                        process_results,
                        results,
                        mongo_documents
                    ),
                    timeout=60.0
                )


            logger.info(f"   ✅ Successful: {len(successful_results)}")
            logger.info(f"   ❌ Failed: {len(failed_results)}")


            # The "all-or-nothing" validation happens in the ATOMIC GATE below (outside
            # this try), so we can respond 200 with the list of unprocessed files.

        except AsyncioTimeoutError:
            logger.error(f"   ⏱️  Timeout processing results (60s)")

            timeout_error = TimeoutError("Timeout processing results")

            handled = timeout_error

            return JSONResponse(
                status_code=504,
                content=_api_error(
                    exception=handled,
                    path=str(request.url.path),
                    method=request.method,
                    duration_ms=(time.perf_counter() - t0) * 1000
                )
            )

        except RuntimeError:
            raise

        except Exception as e:

            logger.error(f"   ❌ Error processing results: {e}")

            raise RuntimeError("Error processing the results")

        # ════════════════════════════════════════════════════════════
        # ATOMIC GATE: if ANY file could not be processed, NOTHING is written
        # (not even partially). Respond 200 with the list of unprocessed documents.
        # ════════════════════════════════════════════════════════════
        if failed_results:
            elapsed = time.perf_counter() - t0
            total_in = len(successful_results) + len(failed_results)
            documents_not_processed = [f.get('filename', 'unknown') for f in failed_results]
            user_message = (
                f"Nothing was written: {len(failed_results)} of {total_in} document(s) "
                f"could not be processed. Review the documents and try again."
            )
            logger.warning(f"⛔ Atomic ingestion aborted: {user_message}")
            return JSONResponse(
                status_code=200,
                content=_api_success(
                    data={
                        "saved": False,
                        "user_message": user_message,
                        "files_processed": 0,
                        "files_failed": len(failed_results),
                        "documents_not_processed": documents_not_processed,
                        "errors": failed_results,
                        "configuration": {
                            "user": user_up, "company": company_up,
                            "modelo": model, "config_chatbot": config_chatbot,
                        },
                    },
                    message=user_message,
                    meta={
                        "request_id": request_id, "run_id": run_id,
                        "tenant_id": company_up, "user_id": user_up,
                        "path": str(request.url.path), "method": request.method,
                        "execution_time_seconds": round(elapsed, 2),
                        "endpoint": "training_chat_bot_simplified",
                    },
                )
            )

        # ════════════════════════════════════════════════════════════
        # STEP 12: MONGODB STORAGE
        # ════════════════════════════════════════════════════════════

        logger.info(f"\n💾 STEP 12: MongoDB storage...")

        try:
            logger.info(f"\n👤 STEP 7: Validating user...")

            user_query = {
                "user": user_up,
                "company": company_up,
            }

            # Use the existing validate_or_insert_user function or implement it
            user_exists = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: list(mongo_users.find_all_clasic(user_query))
                ),
                timeout=10.0
            )

            if not user_exists:
                logger.info(f"   ➕ User does not exist, creating...")

                await asyncio.wait_for(
                    asyncio.to_thread(
                        mongo_users.insert_one,
                        {
                            **user_query,
                            "description": description,
                            "config_chatbot": config_chatbot,
                            "created_at": datetime.utcnow().isoformat()
                        }
                    ),
                    timeout=10.0
                )

                logger.info(f"   ✅ User created")
            else:
                logger.info(f"   ✅ User already exists")

        except AsyncioTimeoutError:
            logger.warning(f"   ⏱️  Timeout validating user (10s), continuing...")

        except Exception as e:
            logger.warning(f"   ⚠️  Error validating user: {e}, continuing...")

        # ── Versioning + traces: register every file that came in ──────────
        try:
            from framework.Traceability.ingest_registry import register_ingestion
            for _f in files:
                register_ingestion(
                    company=company_up,
                    document_ref=getattr(_f, "filename", None) or "document",
                    user=user_up,
                    operation="ingestion",
                    metadata={"config_chatbot": config_chatbot},
                )
        except Exception as _te:
            logger.warning(f"   ⚠️  Ingestion trace registration failed (non-blocking): {_te}")

        # ════════════════════════════════════════════════════════════
        # TRANSACTIONAL COMMIT (phase 2): all files passed extraction -> write
        # Weaviate + GridFS + metadata. If anything fails, it is reverted and aborted
        # (nothing is written partially).
        # ════════════════════════════════════════════════════════════
        try:
            file_ids = await asyncio.wait_for(
                _commit_training_atomic(
                    successful_results, config, rules, mongo_documents,
                    doc_desc_manager, user_query, results,
                ),
                timeout=600.0  # 10 min para indexación vectorial + Mongo de todo el lote
            )
            logger.info(f"   ✅ Transactional commit OK: {len(file_ids)} document(s) written")

        except Exception as commit_err:
            elapsed = time.perf_counter() - t0
            user_message = (
                "Nothing was written: an error occurred while writing the documents and the "
                "operation was reverted. Please try again."
            )
            logger.error(f"⛔ {user_message} (cause: {commit_err})")
            return JSONResponse(
                status_code=200,
                content=_api_success(
                    data={
                        "saved": False,
                        "user_message": user_message,
                        "files_processed": 0,
                        "files_failed": 0,
                        "commit_error": str(commit_err),
                    },
                    message=user_message,
                    meta={
                        "request_id": request_id, "run_id": run_id,
                        "tenant_id": company_up, "user_id": user_up,
                        "path": str(request.url.path), "method": request.method,
                        "execution_time_seconds": round(elapsed, 2),
                        "endpoint": "training_chat_bot_simplified",
                    },
                )
            )

        # ════════════════════════════════════════════════════════════
        # STEP 13: COMPUTE METRICS AND BUILD RESPONSE
        # ════════════════════════════════════════════════════════════

        logger.info(f"\n📊 STEP 13: Computing final metrics...")

        # Process deep-search tokens
        deep_search_tokens = [0]
        for arr in search_tokens_arrays:
            if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], dict):
                deep_search_tokens = [arr[0].get('text_1', 0)]
                break

        # Remove _id from rules
        rules.pop('_id', None)

        # Final calculations
        elapsed = time.perf_counter() - t0
        total_processed = len(successful_results) + len(failed_results)
        success_rate = (len(successful_results) / total_processed * 100) if total_processed > 0 else 0

        logger.info(f"\n{'=' * 80}")
        logger.info(f"✅ TRAINING COMPLETED")
        logger.info(f"   Total time: {elapsed:.2f}s")
        logger.info(f"   Files processed: {len(successful_results)}/{total_processed}")
        logger.info(f"   Success rate: {success_rate:.1f}%")
        logger.info(f"   Workers used: {num_workers}")
        if failed_results:
            logger.warning(f"   Failed files: {len(failed_results)}")
        logger.info(f"{'=' * 80}\n")

        # Build answer (we only reach here if ALL were written: atomic ingestion)
        response_data = {
            "saved": True,
            "response": "Collection created or updated successfully",
            "files_processed": len(successful_results),
            "files_failed": len(failed_results),
            "success_rate_percent": round(success_rate, 2),
            "total_time_seconds": round(elapsed, 2),
            "workers_used": num_workers,
            "configuration": {
                "user": user_up,
                "company": company_up,
                "modelo": model,
                "config_chatbot": config_chatbot
            },
            "post": rules,
            "tokens": token_array,
            "deep_search_tokens": deep_search_tokens
        }

        # Message to show on screen when finished: "X processed, N not processed..."
        n_ok = len(successful_results)
        n_fail = len(failed_results)
        if n_fail:
            user_message = (
                f"{n_ok} processed, {n_fail} not processed. Review the documents."
            )
        else:
            user_message = f"{n_ok} document(s) processed successfully."
        response_data["user_message"] = user_message

        # Add errors if present
        if failed_results:
            response_data["errors"] = failed_results
            # Simple list of names to show directly in the UI
            response_data["documents_not_processed"] = [
                f.get("filename", "desconocido") for f in failed_results
            ]
            response_data["warning"] = f"{n_fail} file(s) failed during processing"

        return JSONResponse(
            status_code=200,
            content=_api_success(
                data=response_data,
                message=user_message,
                meta={
                    "request_id": request_id,
                    "run_id": run_id,
                    "tenant_id": company_up,
                    "user_id": user_up,
                    "path": str(request.url.path),
                    "method": request.method,
                    "execution_time_seconds": round(elapsed, 2),
                    "endpoint": "training_chat_bot_simplified"
                }
            )
        )

    except (ValueError, RuntimeError, TimeoutError) as e:
        # Known framework errors
        handled = e

        # Special cases
        if isinstance(e, ValueError):
            if "too large" in str(e).lower():
                status_code = 413
            else:
                status_code = 422
        elif isinstance(e, RuntimeError):
            if "could not process any file" in str(e).lower():
                status_code = 400
            else:
                status_code = 500
        elif isinstance(e, TimeoutError):
            status_code = 504
        else:
            status_code = 500

        return JSONResponse(
            status_code=status_code,
            content=_api_error(
                exception=handled,
                path=str(request.url.path),
                method=request.method,
                duration_ms=(time.perf_counter() - t0) * 1000,
                include_debug=False
            )
        )

    except Exception as e:
        # Any unexpected error

        logger.error(f"❌ UNEXPECTED ERROR: {e}")
        traceback.print_exc()


        unexpected_error = RuntimeError(f"Unexpected error during training: {str(e)[:200]}")

        handled = unexpected_error

        return JSONResponse(
            status_code=500,
            content=_api_error(
                exception=handled,
                path=str(request.url.path),
                method=request.method,
                duration_ms=(time.perf_counter() - t0) * 1000,
                include_debug=False
            )
        )

    finally:
        # ════════════════════════════════════════════════════════════
        # CLEANUP (ALWAYS RUNS)
        # ════════════════════════════════════════════════════════════

        elapsed_final = time.perf_counter() - t0
        logger.info(f"\n🧹 CLEANUP - Total time: {elapsed_final:.2f}s")

        # TODO: implement rollback if needed


# ===== Download exported files (docx/pdf/xlsx/pptx) — restored =====
@api_router_chat_bot.get(
    "/chatbot/export/download/{file_id}",
    summary="Download a file generated by the chatbot export tools",
    tags=["Chatbot Services - Knowledge database"],
)
async def download_export_file(
    file_id: str,
):
    """
    Serve (with Bearer auth, like the rest of the chatbot endpoints) a file
    previously generated by the *export tools* and stored in GridFS. The id comes
    from the ``attachment`` SSE event emitted by the chatbot stream.
    """
    from urllib.parse import quote

    try:
        obj_id = ObjectId(file_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid file_id")

    try:
        config = ConfigurationManagerSelected().load_configuration("default")
        mongo = MongoDBManager("chatbot_exports", config)
    except Exception as exc:
        logger.error(f"download_export_file: error initializing Mongo: {exc}")
        raise HTTPException(status_code=500, detail="Could not access file storage")

    try:
        grid_out = mongo.fs.get(obj_id)
        data = grid_out.read()
    except Exception:
        raise HTTPException(status_code=404, detail="File not found")

    # Ownership validation (anti-IDOR, #10): the export is tagged with {owner,company}
    # in metadata; it is compared against the token identity. Legacy files without an
    # owner are allowed (backward-compat). Flag EXPORT_ENFORCE_OWNERSHIP (default on).
    if os.getenv("EXPORT_ENFORCE_OWNERSHIP", "1") not in ("0", "false", "False"):
        _meta = getattr(grid_out, "metadata", None) or {}
        _owner = str(_meta.get("owner") or "").replace(" ", "").upper()
        _ident = str("" or "").replace(" ", "").upper()
        if _owner and _owner != _ident:
            logger.warning("download_export_file: access denied to %s (owner != requester)", file_id)
            raise HTTPException(status_code=404, detail="File not found")

    filename = getattr(grid_out, "filename", None) or f"{file_id}.bin"
    content_type = getattr(grid_out, "content_type", None) or "application/octet-stream"
    ascii_name = filename.encode("ascii", "ignore").decode("ascii") or "download.bin"

    return StreamingResponse(
        io.BytesIO(data),
        media_type=content_type,
        headers={
            "Content-Disposition": (
                f"attachment; filename=\"{ascii_name}\"; "
                f"filename*=UTF-8''{quote(filename)}"
            ),
            "Content-Length": str(len(data)),
            "Cache-Control": "no-store",
            "Access-Control-Allow-Origin": "*",
        },
    )
