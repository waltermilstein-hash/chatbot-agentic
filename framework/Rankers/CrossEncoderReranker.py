# =============================================================================
# =============================================================================
"""
CrossEncoderReranker — local reranker without API calls.

Replaces filter_chunks_reflexive (50 LLM calls × ~0.5s = ~25s)
with a local cross-encoder that scores all chunks in ~200ms.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~50MB, runs on CPU
  - Downloaded from HuggingFace the first time
  - Singleton: loaded once per process

Automatic fallback if the model is unavailable:
  orders by the chunk's certainty/score field (0ms, no model).
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker_instance = None
_load_failed = False   # avoid retrying if it already failed


def _get_reranker():
    """Lazy singleton — loads the model only once."""
    global _reranker_instance, _load_failed
    if _reranker_instance is not None:
        return _reranker_instance
    if _load_failed:
        return None
    try:
        from sentence_transformers import CrossEncoder
        _reranker_instance = CrossEncoder(_MODEL_NAME, max_length=512)
        logger.info("CrossEncoderReranker: model '%s' loaded", _MODEL_NAME)
        return _reranker_instance
    except Exception as e:
        _load_failed = True
        logger.warning("CrossEncoderReranker: could not load the model (%s) — using score fallback", e)
        return None


def _extract_text(chunk: Any) -> str:
    """Extract the chunk's text regardless of its format."""
    if isinstance(chunk, str):
        return chunk[:800]
    if isinstance(chunk, dict):
        text = (
            chunk.get("content") or chunk.get("contenido") or
            chunk.get("text") or chunk.get("page_content") or ""
        )
        return str(text)[:800]
    return str(chunk)[:800]


def _extract_score(chunk: Any) -> float:
    """Extract the chunk's existing similarity score."""
    if isinstance(chunk, dict):
        return float(
            chunk.get("certainty") or chunk.get("score") or
            chunk.get("similitud") or chunk.get("distance") or 0.0
        )
    return 0.0


def rerank(query: str, chunks: List[Any], top_k: int = 20) -> List[Any]:
    """
    Score and order chunks by relevance to the query.

    Synchronous — call from asyncio with asyncio.to_thread().

    Args:
        query:   The user's query
        chunks:  List of chunks (dict, str, or Document)
        top_k:   Maximum number of chunks to return

    Returns:
        A list of up to top_k chunks ordered by descending relevance.
    """
    if not chunks or not query:
        return chunks or []

    model = _get_reranker()

    if model is not None:
        # ── Cross-encoder: one forward pass for all pairs ────────────
        try:
            pairs = [(query, _extract_text(c)) for c in chunks]
            scores: List[float] = model.predict(pairs).tolist()
            scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
            result = [c for _, c in scored[:top_k]]
            logger.info("CrossEncoderReranker: %d→%d chunks | top score=%.3f",
                        len(chunks), len(result), scored[0][0] if scored else 0)
            return result
        except Exception as e:
            logger.warning("CrossEncoderReranker: predict error (%s) — score fallback", e)

    # ── Fallback: order by the existing vector score ─────────────────
    logger.debug("CrossEncoderReranker: falling back to score-based ordering")
    sorted_chunks = sorted(chunks, key=_extract_score, reverse=True)
    return sorted_chunks[:top_k]


async def rerank_async(query: str, chunks: List[Any], top_k: int = 20) -> List[Any]:
    """Async version — delegates to a thread to avoid blocking the event loop."""
    return await asyncio.to_thread(rerank, query, chunks, top_k)
