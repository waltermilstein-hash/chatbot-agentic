# =============================================================================
# =============================================================================
"""
QueryCache — Redis cache for chatbot answers.

Avoids re-running the full pipeline for identical or very similar
queries within a time window.

Default TTL: 5 minutes (300s).
Only caches DOCS-type queries (not MEMORY or BOTH).
Does not cache when web_search=True (results change constantly).

Key: qcache:{company}:{sha256(normalized_query)[:16]}
"""

import hashlib
import logging
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

_TTL_SECONDS = 300   # 5 minutes


def _normalize_cache_key(text: str) -> str:
    """Lowercase, no accents, collapsed whitespace."""
    t = text.lower().strip()
    t = "".join(
        c for c in unicodedata.normalize("NFD", t)
        if unicodedata.category(c) != "Mn"
    )
    return " ".join(t.split())


def _cache_key(company: str, query: str) -> str:
    query_norm = _normalize_cache_key(query)
    digest = hashlib.sha256(query_norm.encode()).hexdigest()[:16]
    return f"qcache:{company.upper()}:{digest}"


class QueryCache:
    """
    Redis-backed cache of chatbot answers.

    Usage:
        cache = QueryCache(redis_client)
        answer = await cache.get(company, query)
        if answer:
            return answer   # cache hit, skip pipeline
        ...
        await cache.set(company, query, generated_answer)
    """

    def __init__(self, redis_client, ttl: int = _TTL_SECONDS):
        self.redis = redis_client
        self.ttl = ttl
        self._enabled = redis_client is not None

    async def get(self, company: str, query: str) -> Optional[str]:
        """Return the cached answer or None if it does not exist."""
        if not self._enabled:
            return None
        try:
            key = _cache_key(company, query)
            value = await self.redis.get(key)
            if value:
                logger.info("QueryCache HIT | %s | query=%.60s", company, query)
                return value.decode() if isinstance(value, bytes) else value
        except Exception as e:
            logger.debug("QueryCache.get error: %s", e)
        return None

    async def set(self, company: str, query: str, answer: str) -> None:
        """Store the answer in the cache with TTL."""
        if not self._enabled or not answer:
            return
        try:
            key = _cache_key(company, query)
            await self.redis.setex(key, self.ttl, answer)
            logger.info("QueryCache SET | %s | ttl=%ds | query=%.60s",
                        company, self.ttl, query)
        except Exception as e:
            logger.debug("QueryCache.set error: %s", e)

    async def invalidate(self, company: str) -> int:
        """Invalidate all entries for a company (use after re-ingestion)."""
        if not self._enabled:
            return 0
        try:
            pattern = f"qcache:{company.upper()}:*"
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
                logger.info("QueryCache INVALIDATE | %s | %d keys", company, len(keys))
            return len(keys)
        except Exception as e:
            logger.debug("QueryCache.invalidate error: %s", e)
            return 0
