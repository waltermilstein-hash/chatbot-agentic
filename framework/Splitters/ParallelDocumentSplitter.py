"""
"""

# parallel_document_splitter.py
import os
import time
import logging
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass
import math
import gc

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for the parallel splitter"""
    chunk_size: int = 800
    overlap: int = 100
    batch_size: int = 1000
    max_workers: int = None
    use_processes: bool = False
    preserve_paragraphs: bool = True
    min_chunk_size: int = 50


class ParallelDocumentSplitter:
    """
    Document Splitter with parallelization, smart batching and enriched metadata

    IMPROVEMENTS:
    - Automatic reference extraction (articles, sections, tables, etc.)
    - Per-chunk specific metadata
    - Content-type detection
    - Compatible with hybrid search
    """

    def __init__(self, config: SplitConfig = None):
        self.config = config or SplitConfig()
        self.max_workers = self.config.max_workers or min(8, os.cpu_count() or 4)

        # Thresholds to decide the strategy
        self.parallel_threshold = 5000
        self.batch_threshold = 50000

        # Statistics
        self.total_chunks_processed = 0
        self.total_time = 0.0
        self.batches_processed = 0

        logger.info(f"ParallelDocumentSplitter initialized - Workers: {self.max_workers}")

    # ═══════════════════════════════════════════════════════════════
    #  NEW FUNCTION: PER-CHUNK METADATA EXTRACTION
    # ═══════════════════════════════════════════════════════════════

    def _extract_chunk_metadata(
            self,
            chunk_text: str,
            chunk_index: int,
            doc_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        🔍 Extract per-chunk specific metadata.

        CRITICAL TO SOLVE THE "ARTÍCULO 27B" PROBLEM:
        - Detects legal references (artículo X, sección Y)
        - Detects technical references (Tabla X, Figura Y, ISO-X)
        - Identifies content type
        - Adds chunk metrics

        Args:
            chunk_text: The chunk's text
            chunk_index: The chunk's index in the document
            doc_metadata: The document's general metadata

        Returns:
            Dict with enriched metadata
        """
        metadata = doc_metadata.copy() if doc_metadata else {}

        # Chunk identifier
        metadata['chunk_id'] = chunk_index
        metadata['chunk_length'] = len(chunk_text)

        # ═══ REFERENCE EXTRACTION ═══
        # NOTE: the reference patterns and their labels match Spanish document
        # content and are stored/indexed for Spanish-language search — keep them.
        referencias = []

        # Patterns for LEGAL documents
        legal_patterns = [
            (r'\bartículo\s+(\d+[a-z]?)\b', 'artículo'),
            (r'\bart\.\s*(\d+[a-z]?)\b', 'artículo'),
            (r'\bsección\s+(\d+[a-z]?)\b', 'sección'),
            (r'\bsec\.\s*(\d+[a-z]?)\b', 'sección'),
            (r'\bcapítulo\s+([IVX]+|\d+)\b', 'capítulo'),
            (r'\bcap\.\s*([IVX]+|\d+)\b', 'capítulo'),
            (r'\bdisposición\s+(\w+)\b', 'disposición'),
            (r'\bdisp\.\s*(\w+)\b', 'disposición'),
        ]

        # Patterns for TECHNICAL documents
        tech_patterns = [
            (r'\btabla\s+(\d+)\b', 'tabla'),
            (r'\bfigura\s+(\d+)\b', 'figura'),
            (r'\bfig\.\s*(\d+)\b', 'figura'),
            (r'\banexo\s+([IVX]+|\d+)\b', 'anexo'),
            (r'\bapéndice\s+([A-Z]|\d+)\b', 'apéndice'),
            (r'\bISO[-\s]?(\d+)\b', 'ISO'),
            (r'\bCNAE[-\s]?(\d+)\b', 'CNAE'),
            (r'\bUNE[-\s]?(\d+)\b', 'UNE'),
        ]

        chunk_lower = chunk_text.lower()

        # Extract legal references
        for pattern, tipo in legal_patterns:
            matches = re.findall(pattern, chunk_lower, re.IGNORECASE)
            for match in matches:
                ref = f"{tipo} {match}"
                if ref not in referencias:
                    referencias.append(ref)

        # Extract technical references
        for pattern, tipo in tech_patterns:
            matches = re.findall(pattern, chunk_lower, re.IGNORECASE)
            for match in matches:
                ref = f"{tipo} {match}"
                if ref not in referencias:
                    referencias.append(ref)

        # Store references
        if referencias:
            if isinstance(referencias, list):
                metadata['referencias'] = ', '.join(str(ref) for ref in referencias if ref) if referencias else ''
            else:
                metadata['referencias'] = str(referencias) if referencias else ''

            # Searchable field for fast filtering in the vector DB
            metadata['referencias_text'] = ' | '.join(referencias).lower()

        # ═══ CONTENT-TYPE DETECTION ═══
        # Tables (markdown or ASCII art)
        if re.search(r'\|.*\|.*\|', chunk_text) or re.search(r'\+[-+]+\+', chunk_text):
            metadata['content_type'] = 'table'

        # Long numbered lists
        elif len(re.findall(r'^\s*\d+[\.)]\s+', chunk_text, re.MULTILINE)) > 3:
            metadata['content_type'] = 'list'

        # Code (programming symbols)
        elif chunk_text.count('{') > 5 and chunk_text.count('}') > 5:
            metadata['content_type'] = 'code'

        # Definitions/glossary (many ":" followed by text)
        elif len(re.findall(r'^[^:\n]{1,50}:\s+\w', chunk_text, re.MULTILINE)) > 3:
            metadata['content_type'] = 'definition'

        # Narrative text
        else:
            metadata['content_type'] = 'text'

        # ═══ CHUNK METRICS ═══
        words = chunk_text.split()
        metadata['word_count'] = len(words)
        metadata['has_numbers'] = bool(re.search(r'\d+', chunk_text))

        # Digit density (useful to detect tables/data)
        digit_count = len(re.findall(r'\d', chunk_text))
        metadata['digit_density'] = round(digit_count / len(chunk_text), 3) if chunk_text else 0

        # Detect whether it has URLs
        metadata['has_urls'] = bool(re.search(r'https?://', chunk_text))

        # Detect whether it has emails
        metadata['has_emails'] = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', chunk_text))

        return metadata

    # ═══════════════════════════════════════════════════════════════
    #  MODIFIED FUNCTION: _add_metadata_to_chunk
    # ═══════════════════════════════════════════════════════════════

    def _add_metadata_to_chunk(self, chunk_data: Tuple[str, Dict[str, Any], int]) -> Dict[str, Any]:
        """
        Add enriched metadata to a chunk.

        IMPORTANT CHANGE: it now returns a Dict instead of a str

        Args:
            chunk_data: Tuple of (chunk_text, doc_metadata, chunk_index)

        Returns:
            Dict with the structure:
            {
                'text': str,
                'metadata': Dict,
                'chunk_index': int
            }
        """
        chunk_text, doc_metadata, chunk_index = chunk_data

        try:
            # Extract the chunk's specific metadata
            chunk_metadata = self._extract_chunk_metadata(
                chunk_text=chunk_text,
                chunk_index=chunk_index,
                doc_metadata=doc_metadata or {}
            )

            return {
                'text': chunk_text,
                'metadata': chunk_metadata,
                'chunk_index': chunk_index
            }

        except Exception as e:
            logger.error(f"Error extracting metadata from chunk {chunk_index}: {e}")
            # Fallback: minimal metadata
            return {
                'text': chunk_text,
                'metadata': {
                    **(doc_metadata or {}),
                    'chunk_id': chunk_index,
                    'chunk_length': len(chunk_text),
                    'error': str(e)
                },
                'chunk_index': chunk_index
            }

    # [... rest of the functions unchanged ...]

    def _calculate_optimal_strategy(self, text_length: int) -> Dict[str, Any]:
        """Calculate the optimal strategy based on text size"""
        estimated_chunks = math.ceil(text_length / (self.config.chunk_size - self.config.overlap))

        if text_length < self.parallel_threshold:
            strategy = "sequential"
            use_batching = False
            optimal_batch_size = estimated_chunks
        elif text_length < self.batch_threshold:
            strategy = "parallel"
            use_batching = False
            optimal_batch_size = estimated_chunks
        else:
            strategy = "parallel_batched"
            use_batching = True
            target_batches = max(2, min(self.max_workers * 2, estimated_chunks // 100))
            optimal_batch_size = max(self.config.batch_size, estimated_chunks // target_batches)

        return {
            "strategy": strategy,
            "use_batching": use_batching,
            "estimated_chunks": estimated_chunks,
            "optimal_batch_size": optimal_batch_size,
            "estimated_batches": math.ceil(estimated_chunks / optimal_batch_size) if use_batching else 1
        }

    def _smart_text_split(self, text: str) -> List[str]:
        """Smart split that respects paragraphs and sentences"""
        if len(text) <= self.config.chunk_size:
            return [text]

        chunks = []
        current_chunk = ""

        if self.config.preserve_paragraphs:
            paragraphs = text.split('\n\n')
        else:
            paragraphs = [text]

        for paragraph in paragraphs:
            if len(paragraph) > self.config.chunk_size:
                sentences = self._split_into_sentences(paragraph)

                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 <= self.config.chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
            else:
                if len(current_chunk) + len(paragraph) + 2 <= self.config.chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        chunks = [chunk for chunk in chunks if len(chunk) >= self.config.min_chunk_size]

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _process_text_chunk(self, text_segment: str) -> List[str]:
        """Process a text segment (function for parallelization)"""
        try:
            return self._smart_text_split(text_segment)
        except Exception as e:
            logger.error(f"Error processing text segment: {e}")
            from textwrap import wrap
            return wrap(text_segment, self.config.chunk_size)

    def _split_text_sequential(self, text: str) -> List[str]:
        """Split text sequentially"""
        logger.info("🔄 Using SEQUENTIAL strategy")
        return self._smart_text_split(text)

    def _split_text_parallel(self, text: str) -> List[str]:
        """Split text in parallel (no batching)"""
        logger.info(f"⚡ Using PARALLEL strategy with {self.max_workers} workers")

        text_length = len(text)
        segment_size = max(5000, text_length // self.max_workers)

        segments = []
        for i in range(0, text_length, segment_size):
            segment = text[i:i + segment_size]
            if i + segment_size < text_length:
                last_space = segment.rfind(' ')
                if last_space > segment_size * 0.8:
                    segment = segment[:last_space]
            segments.append(segment)

        all_chunks = []

        if self.config.use_processes:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._process_text_chunk, segment) for segment in segments]

                for future in as_completed(futures):
                    try:
                        chunks = future.result(timeout=60)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        logger.error(f"Error in parallel processing: {e}")
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._process_text_chunk, segment) for segment in segments]

                for future in as_completed(futures):
                    try:
                        chunks = future.result(timeout=60)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        logger.error(f"Error in parallel processing: {e}")

        return all_chunks

    def _split_text_parallel_batched(self, text: str, strategy: Dict[str, Any]) -> List[str]:
        """Split text in parallel with smart batching"""
        logger.info(f"⚡📦 Using PARALLEL + BATCHING strategy")
        logger.info(f"   Estimated batches: {strategy['estimated_batches']}")
        logger.info(f"   Batch size: {strategy['optimal_batch_size']}")

        basic_chunks = self._smart_text_split(text)

        if len(basic_chunks) <= strategy['optimal_batch_size']:
            return basic_chunks

        chunk_batches = []
        for i in range(0, len(basic_chunks), strategy['optimal_batch_size']):
            batch = basic_chunks[i:i + strategy['optimal_batch_size']]
            chunk_batches.append(batch)

        logger.info(f"📦 Created {len(chunk_batches)} chunk batches")

        all_processed_chunks = []

        for batch_num, batch in enumerate(chunk_batches, 1):
            batch_start = time.time()

            if len(batch) > 1 and self.max_workers > 1:
                workers_for_batch = min(self.max_workers, len(batch))

                with ThreadPoolExecutor(max_workers=workers_for_batch) as executor:
                    futures = [executor.submit(self._refine_chunk, chunk) for chunk in batch]

                    batch_results = []
                    for future in as_completed(futures):
                        try:
                            refined_chunks = future.result(timeout=30)
                            batch_results.extend(refined_chunks)
                        except Exception as e:
                            logger.error(f"Error refining chunk in batch {batch_num}: {e}")

                all_processed_chunks.extend(batch_results)
            else:
                for chunk in batch:
                    refined = self._refine_chunk(chunk)
                    all_processed_chunks.extend(refined)

            batch_time = time.time() - batch_start
            logger.info(f"✅ Batch {batch_num}/{len(chunk_batches)} completed in {batch_time:.1f}s")
            gc.collect()

        self.batches_processed = len(chunk_batches)
        return all_processed_chunks

    def _refine_chunk(self, chunk: str) -> List[str]:
        """Refine a chunk (split it further if necessary)"""
        if len(chunk) <= self.config.chunk_size:
            return [chunk]
        return self._smart_text_split(chunk)

    # ═══════════════════════════════════════════════════════════════
    #  MAIN FUNCTION MODIFIED: split_document
    # ═══════════════════════════════════════════════════════════════

    def split_document(
            self,
            text: str,
            metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        🚀 MAIN FUNCTION: Split a document with enriched metadata

        IMPORTANT CHANGE: it now returns List[Dict] instead of List[str]

        Each element has:
        {
            'text': str,           # The chunk's content
            'metadata': Dict,      # Specific metadata (with references!)
            'chunk_index': int     # Position in the document
        }

        Args:
            text: Text to split
            metadata: The document's general metadata

        Returns:
            List of chunks enriched with metadata
        """
        start_time = time.time()

        if not text or not text.strip():
            return []

        text_length = len(text)
        logger.info(f"📄 Starting document split: {text_length:,} characters")

        # ═══ STEP 1: DETERMINE STRATEGY ═══
        strategy = self._calculate_optimal_strategy(text_length)
        logger.info(f"🎯 Selected strategy: {strategy['strategy'].upper()}")

        # ═══ STEP 2: RUN SPLIT ACCORDING TO STRATEGY ═══
        if strategy['strategy'] == 'sequential':
            chunks = self._split_text_sequential(text)
        elif strategy['strategy'] == 'parallel':
            chunks = self._split_text_parallel(text)
        else:
            chunks = self._split_text_parallel_batched(text, strategy)

        split_time = time.time() - start_time

        # ═══ STEP 3: ADD ENRICHED METADATA IN PARALLEL ═══
        metadata_start = time.time()
        enriched_chunks = self._add_metadata_parallel(chunks, metadata or {})
        metadata_time = time.time() - metadata_start

        total_time = time.time() - start_time

        # ═══ FINAL STATISTICS ═══
        self.total_chunks_processed = len(enriched_chunks)
        self.total_time = total_time

        # Count chunks with references (useful for debugging)
        chunks_with_refs = sum(
            1 for c in enriched_chunks
            if c['metadata'].get('referencias')
        )

        # Content types found
        content_types = {}
        for c in enriched_chunks:
            ctype = c['metadata'].get('content_type', 'unknown')
            content_types[ctype] = content_types.get(ctype, 0) + 1

        logger.info(f"✅ Split complete:")
        logger.info(f"   📊 Chunks generated: {len(enriched_chunks):,}")
        logger.info(f"   🔗 Chunks with references: {chunks_with_refs}")
        logger.info(f"   📝 Content types: {content_types}")
        logger.info(f"   ⏱️  Split time: {split_time:.1f}s")
        logger.info(f"   ⏱️  Metadata time: {metadata_time:.1f}s")
        logger.info(f"   ⏱️  Total time: {total_time:.1f}s")
        logger.info(f"   🚀 Throughput: {len(enriched_chunks) / total_time:.1f} chunks/sec")

        if strategy['use_batching']:
            logger.info(f"   📦 Batches processed: {self.batches_processed}")

        return enriched_chunks

    # ═══════════════════════════════════════════════════════════════
    #  MODIFIED FUNCTION: _add_metadata_parallel
    # ═══════════════════════════════════════════════════════════════

    def _add_metadata_parallel(
            self,
            chunks: List[str],
            metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Add enriched metadata to chunks in parallel.

        CHANGE: returns List[Dict] instead of List[str]
        """
        if len(chunks) <= 10:
            # Few chunks - sequential processing
            enriched_chunks = []
            for i, chunk in enumerate(chunks):
                enriched = self._add_metadata_to_chunk((chunk, metadata, i))
                enriched_chunks.append(enriched)
            return enriched_chunks

        # Many chunks - parallel processing
        chunk_data = [(chunk, metadata, i) for i, chunk in enumerate(chunks)]
        enriched_chunks = [None] * len(chunks)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self._add_metadata_to_chunk, data): i
                for i, data in enumerate(chunk_data)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    enriched_chunk = future.result(timeout=10)
                    enriched_chunks[index] = enriched_chunk
                except Exception as e:
                    logger.error(f"Error adding metadata to chunk {index}: {e}")
                    # Fallback
                    enriched_chunks[index] = {
                        'text': chunks[index],
                        'metadata': {**metadata, 'chunk_id': index, 'error': str(e)},
                        'chunk_index': index
                    }

        return enriched_chunks

    def get_stats(self) -> Dict[str, Any]:
        """Get the splitter's statistics"""
        return {
            "total_chunks_processed": self.total_chunks_processed,
            "total_time": self.total_time,
            "chunks_per_second": self.total_chunks_processed / max(0.001, self.total_time),
            "batches_processed": self.batches_processed,
            "max_workers": self.max_workers,
            "config": self.config.__dict__
        }
