"""
"""
import asyncio
import time
import logging
import concurrent.futures
from typing import List, Dict, Any, Union, Optional
import openai

logger = logging.getLogger(__name__)

# Global constants - CHANGED TO LARGE
DEFAULT_MODEL = "text-embedding-3-large"
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536
}


class EmbeddingProcessor:
    """Optimized embedding processor with ThreadPool"""

    def __init__(
            self,
            max_workers: int = 4,
            batch_size: int = 100,
            model: str = DEFAULT_MODEL
    ):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.model = model
        self.embedding_dimensions = MODEL_DIMENSIONS.get(model, 3072)  # Default to 3072

        logger.info(f"EmbeddingProcessor initialized:")
        logger.info(f"  - Model: {self.model}")
        logger.info(f"  - Dimensions: {self.embedding_dimensions}")
        logger.info(f"  - Workers: {max_workers}")
        logger.info(f"  - Batch size: {batch_size}")

        # Statistics
        self.total_embeddings_generated = 0
        self.total_processing_time = 0.0

    async def embedding_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a query (async method for VectorialSearchProcessor)
        """
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning(f"Empty or invalid query for embedding")
            return [0.0] * self.embedding_dimensions

        try:
            logger.debug(f"Generating embedding for query: '{text[:100]}...'")

            # Use the existing async method
            return await self.embedding_fn(text.strip())

        except Exception as e:
            logger.error(f"Error generating embedding for query: {e}")
            return [0.0] * self.embedding_dimensions

    def _validate_and_clean_texts(
            self,
            texts: List[Union[str, Any]]
    ) -> tuple[List[str], List[int]]:
        """
        Validate and clean texts, return valid texts and their original indices
        """
        valid_texts = []
        valid_indices = []

        for i, text in enumerate(texts):
            try:
                if text is None:
                    continue

                if not isinstance(text, str):
                    text = str(text)

                text = text.strip()

                if len(text) < 3:
                    continue

                valid_texts.append(text)
                valid_indices.append(i)

            except Exception as e:
                logger.warning(f"Error validating text {i}: {e}")
                continue

        filtered_count = len(texts) - len(valid_texts)
        if filtered_count > 0:
            logger.info(
                f"Validation: {len(valid_texts)}/{len(texts)} valid texts "
                f"({filtered_count} filtered out)"
            )

        return valid_texts, valid_indices

    async def embedding_fn_batch_parallel(
            self,
            texts: List[Union[str, Any]]
    ) -> List[List[float]]:
        """
        Generate embeddings in parallel with automatic dimension detection
        """
        start_time = time.time()

        try:
            if not texts:
                logger.warning("Empty text list")
                return []

            original_count = len(texts)
            logger.info(f"Processing {original_count} texts for embeddings with {self.model}...")

            # Validate and clean
            valid_texts, valid_indices = self._validate_and_clean_texts(texts)

            if not valid_texts:
                logger.error("No valid texts")
                return [
                    [0.0] * self.embedding_dimensions
                    for _ in range(original_count)  # ← FIXED: underscore was missing
                ]

            # Process with ThreadPool
            results = await self._process_with_threads(valid_texts)

            # Combine results
            valid_embeddings = self._combine_results(results, len(valid_texts))

            if not valid_embeddings:
                raise ValueError("No embeddings were generated")

            # Detect actual dimensions
            if valid_embeddings:
                detected_dims = len(valid_embeddings[0])
                if detected_dims != self.embedding_dimensions:
                    logger.warning(
                        f"Detected dimensions ({detected_dims}) != "
                        f"expected ({self.embedding_dimensions}). Adjusting..."
                    )
                    self.embedding_dimensions = detected_dims

            # Rebuild the full list with dummies for invalid texts
            final_embeddings = self._reconstruct_embeddings(
                valid_embeddings,
                valid_indices,
                original_count
            )

            # Final validation
            if len(final_embeddings) != original_count:
                final_embeddings = self._fix_embedding_count(
                    final_embeddings,
                    original_count
                )

            # Statistics
            processing_time = time.time() - start_time
            self.total_embeddings_generated += len(valid_texts)
            self.total_processing_time += processing_time

            logger.info(
                f"Embeddings completed: {len(final_embeddings)} total, "
                f"{len(valid_texts)} valid, "
                f"{original_count - len(valid_texts)} dummies, "
                f"{self.embedding_dimensions} dims, "
                f"{processing_time:.2f}s, "
                f"{len(valid_texts) / processing_time:.1f} emb/s"
            )

            return final_embeddings

        except Exception as e:
            logger.error(f"Critical error in embedding_fn_batch_parallel: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Fallback
            return [
                [0.0] * self.embedding_dimensions
                for _ in range(len(texts))
            ]

    def _reconstruct_embeddings(
            self,
            valid_embeddings: List[List[float]],
            valid_indices: List[int],
            total_count: int
    ) -> List[List[float]]:
        """Rebuild the full list with valid embeddings and dummies"""
        result = []
        valid_idx = 0

        for i in range(total_count):
            if valid_idx < len(valid_indices) and i == valid_indices[valid_idx]:
                result.append(valid_embeddings[valid_idx])
                valid_idx += 1
            else:
                result.append([0.0] * self.embedding_dimensions)

        return result

    def _fix_embedding_count(
            self,
            embeddings: List[List[float]],
            expected: int
    ) -> List[List[float]]:
        """Adjust the number of embeddings to the expected count"""
        current = len(embeddings)

        if current < expected:
            missing = expected - current
            logger.warning(f"Adding {missing} dummy embeddings ({self.embedding_dimensions} dims)")
            embeddings.extend([
                [0.0] * self.embedding_dimensions
                for _ in range(missing)
            ])
        elif current > expected:
            logger.warning(f"Truncating {current - expected} extra embeddings")
            embeddings = embeddings[:expected]

        return embeddings

    async def _process_with_threads(
            self,
            texts: List[str]
    ) -> List[Dict[str, Any]]:
        """Process with ThreadPoolExecutor"""
        batches = self._create_batches(texts)
        logger.info(f"Creating {len(batches)} ThreadPool batches with {self.model}...")

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
        ) as executor:
            loop = asyncio.get_event_loop()

            tasks = [
                loop.run_in_executor(
                    executor,
                    self._process_batch_sync,
                    batch_texts,
                    batch_id
                )
                for batch_texts, batch_id in batches
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            return self._handle_thread_results(results, batches)

    def _process_batch_sync(
            self,
            batch_texts: List[str],
            batch_id: int
    ) -> Dict[str, Any]:
        """Process a batch synchronously"""
        start_time = time.time()

        try:
            response = openai.embeddings.create(
                input=batch_texts,
                model=self.model
            )

            embeddings = [
                list(item.embedding)
                for item in response.data
            ]

            return {
                'batch_id': batch_id,
                'embeddings': embeddings,
                'success': True,
                'count': len(embeddings),
                'processing_time': time.time() - start_time
            }

        except Exception as e:
            logger.error(f"Batch {batch_id} error: {e}")
            return {
                'batch_id': batch_id,
                'embeddings': [
                    [0.0] * self.embedding_dimensions
                    for _ in batch_texts
                ],
                'success': False,
                'error': str(e),
                'count': len(batch_texts),
                'processing_time': time.time() - start_time
            }

    def _handle_thread_results(
            self,
            results: List,
            batches: List[tuple]
    ) -> List[Dict[str, Any]]:
        """Handle thread results, including exceptions"""
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Thread batch {i} failed: {result}")
                batch_size = len(batches[i][0])
                processed.append({
                    'batch_id': i,
                    'embeddings': [
                                      [0.0] * self.embedding_dimensions
                                  ] * batch_size,
                    'success': False,
                    'error': str(result)
                })
            else:
                processed.append(result)
        return processed

    def _create_batches(
            self,
            texts: List[str]
    ) -> List[tuple[List[str], int]]:
        """Create text batches"""
        return [
            (texts[i:i + self.batch_size], i // self.batch_size)
            for i in range(0, len(texts), self.batch_size)
        ]

    def _combine_results(
            self,
            results: List[Dict[str, Any]],
            expected_total: int
    ) -> List[List[float]]:
        """Combine and sort results"""
        results.sort(key=lambda x: x.get('batch_id', 0))

        all_embeddings = []
        successful = sum(1 for r in results if r.get('success', False))

        for result in results:
            all_embeddings.extend(result.get('embeddings', []))

        logger.info(
            f"Successful batches: {successful}/{len(results)}, "
            f"Embeddings: {len(all_embeddings)}/{expected_total}"
        )

        return all_embeddings

    # Compatibility methods
    async def embedding_fn_batch(
            self,
            texts: List[Union[str, Any]]
    ) -> List[List[float]]:
        """Alias for compatibility"""
        return await self.embedding_fn_batch_parallel(texts)

    async def embedding_fn(self, text: str) -> List[float]:
        """Generate a single embedding (async)"""
        if not text or not isinstance(text, str) or not text.strip():
            return [0.0] * self.embedding_dimensions

        try:
            response = openai.embeddings.create(
                input=[text.strip()],
                model=self.model
            )
            return list(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error in embedding_fn: {e}")
            return [0.0] * self.embedding_dimensions

    def embedding_fn_sync(self, text: str) -> List[float]:
        """Generate a single embedding (sync)"""
        if not text or not isinstance(text, str) or not text.strip():
            return [0.0] * self.embedding_dimensions

        try:
            response = openai.embeddings.create(
                input=[text.strip()],
                model=self.model
            )
            return list(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error in embedding_fn_sync: {e}")
            return [0.0] * self.embedding_dimensions

    def get_stats(self) -> Dict[str, Any]:
        """Return processor statistics"""
        return {
            'total_embeddings': self.total_embeddings_generated,
            'total_time': self.total_processing_time,
            'avg_speed': (
                self.total_embeddings_generated / self.total_processing_time
                if self.total_processing_time > 0 else 0
            ),
            'model': self.model,
            'dimensions': self.embedding_dimensions,
        }


# Default global instance
_default_processor: Optional[EmbeddingProcessor] = None


def get_embedding_processor(
        max_workers: int = 4,
        batch_size: int = 100,
        model: str = DEFAULT_MODEL
) -> EmbeddingProcessor:
    """Get or create the global processor instance"""
    global _default_processor

    if _default_processor is None:
        _default_processor = EmbeddingProcessor(
            max_workers=max_workers,
            batch_size=batch_size,
            model=model
        )

    return _default_processor


# Convenience functions
async def embedding_fn_batch(
        texts: List[Union[str, Any]]
) -> List[List[float]]:
    """Global function to generate embeddings in batch"""
    processor = get_embedding_processor()
    return await processor.embedding_fn_batch_parallel(texts)
