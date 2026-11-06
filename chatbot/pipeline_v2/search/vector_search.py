# =============================================================================
# =============================================================================
"""
VectorSearchEngine — Vector search engine extracted from the RagEngine.

Centralizes the search configuration in one place and exposes
clear parameters instead of scattered constants.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional

from framework.Embeddings.EmbeddingProcessor import EmbeddingProcessor
from framework.Weaviate.VectorialSearchProcessor import SearchConfig, VectorSearchFactory

logger = logging.getLogger(__name__)


@dataclass
class SearchParams:
    """
    Vector search parameters.

    Safe values for the legal domain:
      - reduced limit (50 vs original 500): avoids noise in the context
      - high certainty (0.65 vs original 0.25): only truly relevant chunks
    """
    limit: int = 50
    certainty: float = 0.65        # 25% was too permissive → noise in the answer
    remove_duplicates: bool = True
    max_chunks_per_file: int = 20
    enable_parallel: bool = True


class VectorSearchEngine:
    """
    Wraps VectorSearchFactory with a conservative configuration
    and structured logging.
    """

    def __init__(
        self,
        weaviate_client,
        openai_api_key: str,
        params: Optional[SearchParams] = None,
    ):
        self.params = params or SearchParams()
        embedding_processor = EmbeddingProcessor(
            max_workers=4,
            batch_size=160,
        )
        config = SearchConfig(
            limit=self.params.limit,
            certainty=self.params.certainty,
            remove_duplicates=self.params.remove_duplicates,
            max_chunks_per_file=self.params.max_chunks_per_file,
            enable_parallel=self.params.enable_parallel,
        )
        self.processor = VectorSearchFactory.create_with_embedding_processor(
            weaviate_client=weaviate_client,
            embedding_processor=embedding_processor,
            max_workers=4,
        )
        self._config = config

    async def search(
        self,
        query: str,
        collection_name: str,
        user: str,
        company: str,
        timeout: float = 15.0,
    ) -> List:
        """
        Vector search with timeout and logging.

        Returns:
            List of chunks (empty on failure or timeout).
        """
        logger.info("VectorSearch | query=%.60s | collection=%s | limit=%d | certainty=%.2f",
                    query, collection_name, self.params.limit, self.params.certainty)
        try:
            # search_similar_docs is the main method of the semantic search processor.
            # Receives: user, company, class_name, query, config
            # class_name == collection_name in Weaviate.
            docs = await asyncio.wait_for(
                self.processor.search_similar_docs(
                    user=user,
                    company=company,
                    class_name=collection_name,
                    query=query,
                    config=self._config,   # ← certainty=0.65 and limit=50 applied here
                ),
                timeout=timeout,
            )
            logger.info("VectorSearch | found=%d docs", len(docs) if docs else 0)
            return docs or []
        except asyncio.TimeoutError:
            logger.warning("VectorSearch TIMEOUT (%.0fs) | query=%.60s", timeout, query)
            return []
        except Exception as e:
            logger.error("VectorSearch ERROR: %s", e)
            return []
