# =============================================================================
# =============================================================================
"""
ReflexiveChunkFilter — Reflexive chunk filter extracted from the RagEngine.

Changes from the original:
  - Limit of chunks to evaluate: 50 (not 500)
  - Batch evaluation so the API isn't overwhelmed
  - Logging with a logger instead of print()
  - Configurable per-chunk timeout
"""

import asyncio
import logging
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

PROMPT_RELEVANCE = """Decide whether this document fragment is relevant to answer the following query.
Answer ONLY with "YES" or "NO".

Query: {query}
Fragment: {fragment}"""

MAX_CHUNKS_TO_EVALUATE = 50
TIMEOUT_PER_CHUNK = 4.0
BATCH_SIZE = 10


class ReflexiveChunkFilter:
    """
    Filters the retrieved chunks by evaluating their relevance with an LLM.
    Processes in batches to control API usage.
    """

    def __init__(
        self,
        model_executor: Callable,
        max_chunks: int = MAX_CHUNKS_TO_EVALUATE,
        timeout_per_chunk: float = TIMEOUT_PER_CHUNK,
        batch_size: int = BATCH_SIZE,
    ):
        self.executor = model_executor
        self.max_chunks = max_chunks
        self.timeout = timeout_per_chunk
        self.batch_size = batch_size

    async def filter(self, chunks: List, query: str) -> List:
        """
        Filter chunks by relevance.

        - If there are ≤ 5 chunks, return them all without filtering.
        - Limit the evaluation to max_chunks to avoid a cost explosion.
        - If the filter leaves 0 chunks, return the top 3 by original score.
        """
        if not chunks or not query:
            return chunks or []

        if len(chunks) <= 5:
            return chunks

        candidates = chunks[:self.max_chunks]
        logger.info("ReflexiveFilter | evaluating %d/%d chunks", len(candidates), len(chunks))

        relevant = []
        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i: i + self.batch_size]
            results = await self._evaluar_batch(batch, query)
            for chunk, is_relevant in zip(batch, results):
                if is_relevant:
                    relevant.append(chunk)

        if not relevant:
            logger.warning("ReflexiveFilter | 0 relevant, using top-3 by score")
            relevant = self._top_by_score(chunks, n=3)

        logger.info("ReflexiveFilter | %d/%d relevant chunks", len(relevant), len(candidates))
        return relevant

    async def _evaluar_batch(self, batch: List, query: str) -> List[bool]:
        tasks = [
            asyncio.wait_for(self._evaluar_chunk(chunk, query, i), timeout=self.timeout)
            for i, chunk in enumerate(batch)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            r if isinstance(r, bool) else True  # on error, include the chunk
            for r in results
        ]

    async def _evaluar_chunk(self, chunk: Any, query: str, idx: int) -> bool:
        try:
            fragment = str(
                chunk.get("content") or chunk.get("contenido") or chunk.get("text") or chunk
            )[:600]
            prompt = PROMPT_RELEVANCE.format(query=query[:300], fragment=fragment)
            answer = await self.executor(prompt)
            return "YES" in str(answer).upper()
        except Exception as e:
            logger.debug("Error evaluating chunk %d: %s", idx, e)
            return True

    @staticmethod
    def _top_by_score(chunks: List, n: int) -> List:
        def get_score(c):
            return float(c.get("certainty") or c.get("score") or c.get("similitud") or 0.0)

        return sorted(chunks, key=get_score, reverse=True)[:n]
