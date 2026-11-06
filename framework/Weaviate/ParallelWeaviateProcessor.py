"""
"""

import asyncio
import concurrent.futures
import hashlib
import logging
import json
import uuid
import time
import re
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class ParallelWeaviateProcessor:
    """
    Parallel Weaviate processor with support for enriched chunks.

    V2 improvements:
    - Support for dict chunks with structured metadata
    - Automatic reference detection
    - Enriched schema for hybrid search
    - 'filename' field for search by file name
    - Backward compatible with string chunks
    """

    def __init__(self, weaviate_client, embedding_processor, max_concurrent_files: int = 4,
                 batch_insert_size: int = 100):
        self.client = weaviate_client
        self.embedding_processor = embedding_processor
        self.max_concurrent_files = max_concurrent_files
        self.batch_insert_size = batch_insert_size

    # ═══════════════════════════════════════════════════════════════
    #  MÉTODO NUEVO: _extract_filename
    # ═══════════════════════════════════════════════════════════════

    def _extract_filename(self, source: str) -> str:
        """
        Extract a clean file name from the source path.

        Examples:
        - '/path/to/convenio.pdf' → 'convenio.pdf'
        - 'https://example.com/docs/manual.pdf' → 'manual.pdf'
        - 'C:\\Users\\docs\\documento.pdf' → 'documento.pdf'
        - 'convenio.pdf' → 'convenio.pdf'
        - '' → ''

        Args:
            source: Full path or URL of the file

        Returns:
            Just the file name
        """
        if not source:
            return ''

        try:
            # Extract only the file name (without path)
            # Works with Unix, Windows and URL paths
            filename = os.path.basename(source)

            # If it is a URL, it may have query params; strip them
            if '?' in filename:
                filename = filename.split('?')[0]

            return filename.strip()

        except Exception as e:
            logger.warning(f"Error extracting filename from '{source}': {e}")
            return source  # Fallback: devolver source original

    # ═══════════════════════════════════════════════════════════════
    #  METHOD: _extract_text_from_chunk (unchanged)
    # ═══════════════════════════════════════════════════════════════

    def _extract_text_from_chunk(self, chunk) -> str:
        """
        Extract clean text from the chunk, handling different formats.

        SUPPORTS:
        - str: "direct text"
        - old dict: {"text": "...", "metadata_json": "..."}
        - new dict: {"text": "...", "metadata": {...}, "chunk_index": 0}
        """
        try:
            if isinstance(chunk, str):
                return chunk.strip()

            elif isinstance(chunk, dict):
                if 'text' in chunk:
                    text = chunk['text']

                    if isinstance(text, str) and text.strip().startswith('{'):
                        try:
                            import ast
                            parsed = ast.literal_eval(text)
                            if isinstance(parsed, dict) and 'text' in parsed:
                                return str(parsed['text']).strip()
                        except:
                            pass

                    return str(text).strip()
                else:
                    return str(chunk).strip()
            else:
                return str(chunk).strip()

        except Exception as e:
            logger.warning(f"Error extracting chunk text: {e}")
            return str(chunk).strip()

    # ═══════════════════════════════════════════════════════════════
    #  METHOD: _extract_metadata_from_chunk (unchanged)
    # ═══════════════════════════════════════════════════════════════

    def _extract_metadata_from_chunk(
            self,
            chunk,
            file_metadata: Dict,
            chunk_index: int  # ✅ Agregar este parámetro
    ) -> Dict:
        """
        Extract chunk metadata: combines file + chunk + automatic analysis.

        Args:
            chunk: May be a str or a dict
            file_metadata: Metadata of the full file
            chunk_index: Index of the chunk within the document

        Returns:
            Dict with fully enriched metadata
        """
        # 1. Extract text
        chunk_text = self._extract_text_from_chunk(chunk)

        # 2. Base file metadata
        combined_metadata = file_metadata.copy()

        # 3. If the chunk already has metadata, merge it
        if isinstance(chunk, dict) and 'metadata' in chunk:
            chunk_metadata = chunk['metadata']
            if isinstance(chunk_metadata, dict):
                combined_metadata.update(chunk_metadata)

        # ✅ 4. EXTRAER METADATA ENRIQUECIDA (referencias, métricas, etc.)
        enriched_metadata = self._extract_chunk_metadata(
            chunk_text=chunk_text,
            chunk_index=chunk_index,
            doc_metadata=combined_metadata
        )

        return enriched_metadata

    def _extract_chunk_metadata(
            self,
            chunk_text: str,
            chunk_index: int,
            doc_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        🔍 Extract metadata specific to each chunk.

        CRITICAL FOR SOLVING THE "ARTÍCULO 27B" PROBLEM:
        - Detects legal references (article X, section Y)
        - Detects technical references (Table X, Figure Y, ISO-X)
        - Identifies content type
        - Adds chunk metrics

        Args:
            chunk_text: Chunk text
            chunk_index: Index of the chunk within the document
            doc_metadata: General document metadata

        Returns:
            Dict with enriched metadata
        """
        import re

        metadata = doc_metadata.copy() if doc_metadata else {}

        # Chunk identifier
        metadata['chunk_id'] = chunk_index
        metadata['chunk_length'] = len(chunk_text)

        # ═══ REFERENCE EXTRACTION ═══
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

        # Extrae referencias legales
        for pattern, tipo in legal_patterns:
            matches = re.findall(pattern, chunk_lower, re.IGNORECASE)
            for match in matches:
                ref = f"{tipo} {match}"
                if ref not in referencias:
                    referencias.append(ref)

        # Extrae referencias técnicas
        for pattern, tipo in tech_patterns:
            matches = re.findall(pattern, chunk_lower, re.IGNORECASE)
            for match in matches:
                ref = f"{tipo} {match}"
                if ref not in referencias:
                    referencias.append(ref)

        # Store references
        if referencias:
            #metadata['referencias'] = referencias
            if isinstance(referencias, list):
                metadata['referencias'] = ', '.join(str(ref) for ref in referencias if ref) if referencias else ''
            else:
                metadata['referencias'] = str(referencias) if referencias else ''
            # Searchable field for fast filtering in the vector DB
            metadata['referencias_text'] = ' | '.join(referencias).lower()

        # ═══ CONTENT TYPE DETECTION ═══
        # Tablas (markdown o ASCII art)
        if re.search(r'\|.*\|.*\|', chunk_text) or re.search(r'\+[-+]+\+', chunk_text):
            metadata['content_type'] = 'table'

        # Listas numeradas extensas
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

        # Digit density (useful for detecting tables/data)
        digit_count = len(re.findall(r'\d', chunk_text))
        metadata['digit_density'] = round(digit_count / len(chunk_text), 3) if chunk_text else 0

        # Detect whether it has URLs
        metadata['has_urls'] = bool(re.search(r'https?://', chunk_text))

        # Detect whether it has emails
        metadata['has_emails'] = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', chunk_text))

        return metadata
    # ═══════════════════════════════════════════════════════════════
    #  METHOD: process_multiple_files_parallel (unchanged)
    # ═══════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════
    #  MÉTODOS NUEVOS: SOFT DELETE
    # ═══════════════════════════════════════════════════════════════

    def soft_delete_document(
            self,
            collection_name: str,
            file_id: str,
            user: str,
            company: str,
            deleted_by: str,
            deletion_reason: str = "User deleted the document"
    ) -> Dict[str, Any]:
        """
        Mark a whole document as deleted (soft delete).

        Args:
            collection_name: Collection name
            file_id: ID of the file to delete
            user: Owning user
            company: Company
            deleted_by: Email of the user performing the deletion
            deletion_reason: Reason for deletion

        Returns:
            Dict with the operation result
        """
        try:
            # Get all chunks of the document
            where_filter = {
                "operator": "And",
                "operands": [
                    {"path": ["file_id"], "operator": "Equal", "valueText": str(file_id)},
                    {"path": ["user"], "operator": "Equal", "valueText": str(user)},
                    {"path": ["company"], "operator": "Equal", "valueText": str(company)},
                    {"path": ["is_deleted"], "operator": "Equal", "valueBoolean": False}
                ]
            }

            result = (
                self.client.query
                .get(collection_name, ["name"])
                .with_where(where_filter)
                .with_limit(10000)
                .do()
            )

            chunks = result.get('data', {}).get('Get', {}).get(collection_name, [])

            if not chunks:
                return {
                    'success': False,
                    'error': 'Document not found or already deleted',
                    'chunks_deleted': 0
                }

            # Mark each chunk as deleted
            deleted_count = 0
            deletion_time = datetime.now().isoformat()

            for chunk in chunks:
                chunk_id = chunk.get('name')

                try:
                    # Update chunk
                    self.client.data_object.update(
                        uuid=chunk_id,
                        class_name=collection_name,
                        data_object={
                            "is_deleted": True,
                            "deleted_at": deletion_time,
                            "deleted_by": str(deleted_by),
                            "deletion_reason": str(deletion_reason)
                        }
                    )
                    deleted_count += 1

                except Exception as e:
                    logger.error(f"Error marking chunk {chunk_id} as deleted: {e}")

            logger.info(f"🗑️  Document {file_id} marked as deleted ({deleted_count} chunks)")

            return {
                'success': True,
                'file_id': file_id,
                'chunks_deleted': deleted_count,
                'deleted_by': deleted_by,
                'deleted_at': deletion_time,
                'deletion_reason': deletion_reason
            }

        except Exception as e:
            logger.error(f"❌ Error in soft delete: {e}")
            return {
                'success': False,
                'error': str(e),
                'chunks_deleted': 0
            }

    def restore_document(
            self,
            collection_name: str,
            file_id: str,
            user: str,
            company: str,
    ) -> Dict[str, Any]:
        """
        Restore a deleted document (undo soft delete).

        Args:
            collection_name: Collection name
            file_id: ID of the file to restore
            user: Owning user
            company: Company

        Returns:
            Dict with the operation result
        """
        try:
            # Get deleted chunks
            where_filter = {
                "operator": "And",
                "operands": [
                    {"path": ["file_id"], "operator": "Equal", "valueText": str(file_id)},
                    {"path": ["user"], "operator": "Equal", "valueText": str(user)},
                    {"path": ["company"], "operator": "Equal", "valueText": str(company)},
                    {"path": ["is_deleted"], "operator": "Equal", "valueBoolean": True}
                ]
            }

            result = (
                self.client.query
                .get(collection_name, ["name"])
                .with_where(where_filter)
                .with_limit(10000)
                .do()
            )

            chunks = result.get('data', {}).get('Get', {}).get(collection_name, [])

            if not chunks:
                return {
                    'success': False,
                    'error': 'Document not found in trash',
                    'chunks_restored': 0
                }

            # Restore each chunk
            restored_count = 0

            for chunk in chunks:
                chunk_id = chunk.get('name')

                try:
                    # Restore chunk
                    self.client.data_object.update(
                        uuid=chunk_id,
                        class_name=collection_name,
                        data_object={
                            "is_deleted": False,
                            "deleted_at": "",
                            "deleted_by": "",
                            "deletion_reason": ""
                        }
                    )
                    restored_count += 1

                except Exception as e:
                    logger.error(f"Error restoring chunk {chunk_id}: {e}")

            logger.info(f"♻️  Document {file_id} restored ({restored_count} chunks)")

            return {
                'success': True,
                'file_id': file_id,
                'chunks_restored': restored_count
            }

        except Exception as e:
            logger.error(f"❌ Error restoring document: {e}")
            return {
                'success': False,
                'error': str(e),
                'chunks_restored': 0
            }

    def get_deleted_documents(
            self,
            collection_name: str,
            user: str,
            company: str,
            limit: int = 50
    ) -> List[Dict]:
        """
        Get the list of deleted documents (trash).

        Returns:
            List of documents with deletion info
        """
        try:
            where_filter = {
                "operator": "And",
                "operands": [
                    {"path": ["user"], "operator": "Equal", "valueText": str(user)},
                    {"path": ["company"], "operator": "Equal", "valueText": str(company)},
                    {"path": ["is_deleted"], "operator": "Equal", "valueBoolean": True}
                ]
            }

            result = (
                self.client.query
                .get(collection_name, [
                    "file_id", "filename", "deleted_at", "deleted_by",
                    "deletion_reason", "chunk_index", "total_chunks"
                ])
                .with_where(where_filter)
                .with_limit(limit)
                .do()
            )

            chunks = result.get('data', {}).get('Get', {}).get(collection_name, [])

            # Group by file_id
            documents = {}
            for chunk in chunks:
                file_id = chunk.get('file_id')
                if file_id not in documents:
                    documents[file_id] = {
                        'file_id': file_id,
                        'filename': chunk.get('filename', ''),
                        'deleted_at': chunk.get('deleted_at', ''),
                        'deleted_by': chunk.get('deleted_by', ''),
                        'deletion_reason': chunk.get('deletion_reason', ''),
                        'total_chunks': chunk.get('total_chunks', 0)
                    }

            deleted_docs = list(documents.values())

            logger.info(f"🗑️  Encontrados {len(deleted_docs)} documentos eliminados")

            return deleted_docs

        except Exception as e:
            logger.error(f"❌ Error fetching deleted documents: {e}")
            return []

    def permanent_delete_document(
            self,
            collection_name: str,
            file_id: str,
            user: str,
            company: str,
    ) -> Dict[str, Any]:
        """
        Permanently delete a document (hard delete).
        ⚠️ IRREVERSIBLE - use with caution

        Args:
            collection_name: Collection name
            file_id: ID of the file to delete permanently
            user: Owning user
            company: Company

        Returns:
            Dict with the operation result
        """
        try:
            # Only remove documents already marked as deleted
            where_filter = {
                "operator": "And",
                "operands": [
                    {"path": ["file_id"], "operator": "Equal", "valueText": str(file_id)},
                    {"path": ["user"], "operator": "Equal", "valueText": str(user)},
                    {"path": ["company"], "operator": "Equal", "valueText": str(company)},
                    {"path": ["is_deleted"], "operator": "Equal", "valueBoolean": True}
                ]
            }

            result = (
                self.client.query
                .get(collection_name, ["name"])
                .with_where(where_filter)
                .with_limit(10000)
                .do()
            )

            chunks = result.get('data', {}).get('Get', {}).get(collection_name, [])

            if not chunks:
                return {
                    'success': False,
                    'error': 'Document not found in trash or not marked as deleted',
                    'chunks_deleted': 0
                }

            # Permanently delete each chunk
            deleted_count = 0

            for chunk in chunks:
                chunk_id = chunk.get('name')

                try:
                    self.client.data_object.delete(
                        uuid=chunk_id,
                        class_name=collection_name
                    )
                    deleted_count += 1

                except Exception as e:
                    logger.error(f"Error permanently deleting chunk {chunk_id}: {e}")

            logger.warning(f"⚠️  Document {file_id} PERMANENTLY DELETED ({deleted_count} chunks)")

            return {
                'success': True,
                'file_id': file_id,
                'chunks_deleted_permanently': deleted_count,
                'warning': 'Eliminación permanente - IRREVERSIBLE'
            }

        except Exception as e:
            logger.error(f"❌ Error in permanent deletion: {e}")
            return {
                'success': False,
                'error': str(e),
                'chunks_deleted': 0
            }

    async def process_multiple_files_parallel(
            self,
            files_data: List[Dict[str, Any]],
            user: str,
            company: str,
            uploaded_by_name: str = None,  # ✅ NUEVO parámetro
            upload_ip: str = None  # ✅ NUEVO parámetro
    ) -> Dict[str, Any]:

        start_time = time.time()
        total_files = len(files_data)

        logger.info(f"🚀 Processing {total_files} files in parallel")

        # Step 1: create/verify collection
        collection_name = await self._ensure_collection_exists(company, user)
        if not collection_name:
            return {'success': False, 'error': 'Could not create the collection'}

        # Step 2: collect all chunks
        all_chunks_data = []

        for file_idx, file_data in enumerate(files_data):
            chunks = file_data.get('chunks', [])
            file_metadata = file_data.get('metadata', {})

            for chunk_idx, chunk in enumerate(chunks):
                chunk_text = self._extract_text_from_chunk(chunk)

                # ✅ CAMBIO: Pasar chunk_idx como tercer parámetro
                chunk_metadata = self._extract_metadata_from_chunk(chunk, file_metadata, chunk_idx)

                file_id = file_metadata.get('file_id', f'file_{file_idx}')
                chunk_id = f"chunk-{file_id}_{uuid.uuid4().hex[:8]}_{chunk_idx}"

                all_chunks_data.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata,
                    'file_idx': file_idx,
                    'chunk_idx': chunk_idx,
                    'chunk_id': chunk_id
                })

        total_chunks = len(all_chunks_data)
        logger.info(f"📊 Total chunks to process: {total_chunks} from {total_files} files")

        if total_chunks == 0:
            return {'success': True, 'files_processed': 0, 'chunks_processed': 0}

        # Metadata statistics
        chunks_with_refs = sum(
            1 for c in all_chunks_data
            if c['metadata'].get('referencias')
        )

        content_types = {}
        for c in all_chunks_data:
            ctype = c['metadata'].get('content_type', 'unknown')
            content_types[ctype] = content_types.get(ctype, 0) + 1

        if chunks_with_refs > 0:
            logger.info(f"🔗 Chunks with references: {chunks_with_refs}/{total_chunks}")

            sample_refs = []
            for c in all_chunks_data[:10]:
                refs = c['metadata'].get('referencias', [])
                if refs:
                    sample_refs.extend(refs[:2])

            if sample_refs:
                logger.info(f"📌 Reference examples: {', '.join(sample_refs[:15])}")

        if content_types:
            logger.info(f"📝 Content types: {content_types}")

        # Paso 3: Generar embeddings
        try:
            logger.info("🧠 Generando embeddings en paralelo...")
            all_texts = [chunk_data['text'] for chunk_data in all_chunks_data]

            embedding_start = time.time()
            all_embeddings = await self.embedding_processor.embedding_fn_batch_parallel(all_texts)
            embedding_time = time.time() - embedding_start

            logger.info(f"✅ {len(all_embeddings)} embeddings en {embedding_time:.2f}s")

            if len(all_embeddings) != len(all_chunks_data):
                logger.error(f"❌ Mismatch: {len(all_chunks_data)} chunks, {len(all_embeddings)} embeddings")
                return {'success': False, 'error': 'Mismatch en embeddings'}

        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {e}")
            return {'success': False, 'error': f'Error in embeddings: {str(e)}'}

        # Paso 4: Inserción
        try:
            logger.info("💾 Inserting into Weaviate...")

            chunks_with_embeddings = []
            for i, chunk_data in enumerate(all_chunks_data):
                chunks_with_embeddings.append({
                    **chunk_data,
                    'embedding': all_embeddings[i]
                })
            ### @Jaime
            insert_results = await self._batch_insert_sequential(
                chunks_with_embeddings,
                collection_name,
                user,
                company,
                uploaded_by_name=uploaded_by_name,  # ✅ NUEVO
                upload_ip=upload_ip  # ✅ NUEVO
            )

            result = insert_results[0] if insert_results else {}
            successful = result.get('successful_count', 0)
            failed = result.get('failed_count', 0)

            # Validación
            logger.info("🔍 Validando almacenamiento...")

            validation = self.validate_document_storage(
                collection_name=collection_name,
                user=user,
                company=company,
                expected_chunks=total_chunks,
                file_id=files_data[0].get('metadata', {}).get('file_id') if files_data else None
            )

            content_validation = self.validate_document_content(
                collection_name=collection_name,
                user=user,
                company=company,
                sample_size=5
            )

            total_time = time.time() - start_time

            stats = {
                'success': successful > 0 and validation.get('is_complete', True),
                'total_time': total_time,
                'files_processed': total_files,
                'chunks_processed': total_chunks,
                'successful_inserts': successful,
                'failed_inserts': failed,
                'embedding_time': embedding_time,
                'insert_time': result.get('processing_time', 0),
                'chunks_per_second': total_chunks / total_time if total_time > 0 else 0,
                'validation': validation,
                'content_sample': content_validation,
                'chunks_with_references': chunks_with_refs,
                'content_types': content_types
            }

            # Logs finales
            logger.info(f"📊 ESTADÍSTICAS FINALES:")
            logger.info(f"   ✅ Chunks exitosos: {successful}/{total_chunks}")
            logger.info(f"   ❌ Chunks fallidos: {failed}")
            logger.info(f"   🔗 Chunks with references: {chunks_with_refs}")
            logger.info(f"   📝 Content types: {content_types}")
            logger.info(f"   💾 Almacenados: {validation.get('stored_chunks', 0)}")
            logger.info(f"   ⏱️ Tiempo total: {total_time:.2f}s")
            logger.info(f"   🧠 Tiempo embeddings: {embedding_time:.2f}s")
            logger.info(f"   💾 Tiempo inserción: {stats['insert_time']:.2f}s")
            logger.info(f"   🚀 Velocidad: {stats['chunks_per_second']:.1f} chunks/s")

            if validation.get('is_complete'):
                logger.info(f"   ✅ COMPLETE DOCUMENT")
            else:
                logger.warning(f"   ⚠️ INCOMPLETE DOCUMENT")

            return stats

        except Exception as e:
            logger.error(f"❌ Error in insertion: {e}", exc_info=True)
            return {'success': False, 'error': f'Error in insertion: {str(e)}'}


    # ═══════════════════════════════════════════════════════════════
    #  MÉTODO MODIFICADO: _batch_insert_sequential
    # ═══════════════════════════════════════════════════════════════

    def get_collection_stats(
            self,
            collection_name: str,
            user: str,
            company: str,
    ) -> Dict:
        """
        Detailed collection statistics.
        """
        where_filter = {
            "operator": "And",
            "operands": [
                {"path": ["user"], "operator": "Equal", "valueText": user},
                {"path": ["company"], "operator": "Equal", "valueText": company},
            ]
        }

        # Get data
        result = (
            self.client.query
            .get(collection_name, [
                "filename", "content_type", "language",
                "chunk_length", "word_count", "created_date"
            ])
            .with_where(where_filter)
            .with_limit(10000)
            .do()
        )

        chunks = result['data']['Get'][collection_name]

        # Calcular estadísticas
        stats = {
            'total_chunks': len(chunks),
            'unique_files': len(set(c['filename'] for c in chunks)),
            'languages': {},
            'content_types': {},
            'avg_chunk_length': 0,
            'avg_word_count': 0,
            'date_range': {
                'oldest': None,
                'newest': None
            }
        }

        # Group by language
        for chunk in chunks:
            lang = chunk.get('language', 'unknown')
            stats['languages'][lang] = stats['languages'].get(lang, 0) + 1

            ctype = chunk.get('content_type', 'unknown')
            stats['content_types'][ctype] = stats['content_types'].get(ctype, 0) + 1

        # Promedios
        if chunks:
            stats['avg_chunk_length'] = sum(c.get('chunk_length', 0) for c in chunks) / len(chunks)
            stats['avg_word_count'] = sum(c.get('word_count', 0) for c in chunks) / len(chunks)

        return stats

    async def _batch_insert_sequential(
            self,
            chunks_with_embeddings: List[Dict],
            collection_name: str,
            user: str,
            company: str,
            uploaded_by_name: str = None,  # ✅ NUEVO parámetro
            upload_ip: str = None  # ✅ NUEVO parámetro
    ) -> List[Dict]:
        """
        Insert chunks with soft delete + auditing.
        """
        logger.info(f"💾 Inserting {len(chunks_with_embeddings)} chunks in a batch...")
        start_time = time.time()

        successful = 0
        failed = 0
        errors = []

        try:
            with self.client.batch as batch:
                batch.batch_size = 100

                for i, chunk_data in enumerate(chunks_with_embeddings, 1):
                    try:
                        # ═══ PREPARE VECTOR ═══
                        vector = chunk_data['embedding']
                        if isinstance(vector, np.ndarray):
                            vector = vector.tolist()
                        elif not isinstance(vector, list):
                            vector = list(vector)
                        vector = [float(x) for x in vector]

                        # ═══ CLEAN TEXT ═══
                        text = chunk_data['text']
                        if not isinstance(text, str):
                            text = str(text)

                        if text.startswith('{"text":') or text.startswith("{'text':"):
                            logger.warning(f"Nested JSON in chunk {i}, cleaning up...")
                            text = self._extract_text_from_chunk(text)

                        # ═══ PREPARAR METADATA ═══
                        metadata = chunk_data['metadata']

                        # Extraer filename
                        filename = ''
                        if 'file_name' in metadata and metadata['file_name']:
                            filename = str(metadata['file_name']).strip()
                        if not filename and 'filename' in metadata and metadata['filename']:
                            filename = str(metadata['filename']).strip()
                        if not filename and 'source' in metadata and metadata['source']:
                            source_val = str(metadata['source']).strip()
                            if source_val:
                                filename = self._extract_filename(source_val)
                        if not filename:
                            file_id = metadata.get('file_id', '')
                            if file_id:
                                filename = f"file_{file_id}"

                        source = filename

                        # ═══ CHUNKS VECINOS ═══
                        chunk_idx = chunk_data.get('chunk_idx', 0)
                        file_id = metadata.get('file_id', '')

                        previous_chunk_id = None
                        if chunk_idx > 0:
                            previous_chunk_id = f"chunk-{file_id}_{chunk_idx - 1}"

                        next_chunk_id = None
                        if chunk_idx < len(chunks_with_embeddings) - 1:
                            next_chunk_id = f"chunk-{file_id}_{chunk_idx + 1}"

                        # ═══════════════════════════════════════════════════════
                        # BUILD OBJECT FOR WEAVIATE
                        # ═══════════════════════════════════════════════════════
                        data_object = {
                            # CAMPOS BÁSICOS
                            "name": chunk_data['chunk_id'],
                            "text": text.strip(),
                            "user": str(user),
                            "company": str(company),

                            # FILE METADATA
                            "file_id": str(file_id),
                            "source": source,
                            "filename": filename,
                            "doc_type": str(metadata.get('doc_type', '')),

                            # CHUNK METADATA
                            "chunk_length": int(metadata.get('chunk_length', 0)),
                            "word_count": int(metadata.get('word_count', 0)),
                            "content_type": str(metadata.get('content_type', 'text')),
                            "has_numbers": bool(metadata.get('has_numbers', False)),
                            "digit_density": float(metadata.get('digit_density', 0.0)),

                            # REFERENCIAS
                            "referencias_text": str(metadata.get('referencias_text', '')),

                            # TEMPORAL
                            "created_at": time.time(),
                            "created_date": datetime.now().isoformat(),

                            # POSICIÓN
                            "chunk_index": int(chunk_idx),
                            "total_chunks": len(chunks_with_embeddings),

                            # NAVEGACIÓN
                            "previous_chunk_id": previous_chunk_id if previous_chunk_id else "",
                            "next_chunk_id": next_chunk_id if next_chunk_id else "",

                            # CONTENT
                            "language": self._detect_language(text),
                            "text_preview": text[:200] + "..." if len(text) > 200 else text,
                            "text_hash": hashlib.md5(text.encode('utf-8')).hexdigest(),

                            # PROCESO
                            "schema_version": "1.1",  # ✅ Actualizada versión
                            "processing_time": chunk_data.get('processing_time', 0.0),
                            "embedding_model": "text-embedding-3-large",

                            # ═══════════════════════════════════════════════════
                            # 🔥 NUEVOS CAMPOS: SOFT DELETE
                            # ═══════════════════════════════════════════════════
                            "is_deleted": False,
                            "deleted_at": "",
                            "deleted_by": "",
                            "deletion_reason": "",

                            # ═══════════════════════════════════════════════════
                            # 🔥 NUEVOS CAMPOS: UPLOADED BY (AUDITORÍA)
                            # ═══════════════════════════════════════════════════
                            "uploaded_by": str(user),
                            "uploaded_by_name": str(uploaded_by_name) if uploaded_by_name else str(user),
                            "upload_ip": str(upload_ip) if upload_ip else "",

                            # BACKUP
                            "metadata_json": json.dumps(metadata, ensure_ascii=False)
                        }



                        # REFERENCIAS COMO ARRAY
                        # ═══════════════════════════════════════════════════════════════
                        # REFERENCES - adapt to the collection schema
                        # ═══════════════════════════════════════════════════════════════

                        refs_raw = metadata.get('referencias', [])

                        # Detect whether the collection uses text[] or text
                        referencias_is_array = True  # Default for new collections
                        try:
                            class_schema = self.client.schema.get(collection_name)
                            if class_schema and 'properties' in class_schema:
                                for prop in class_schema.get('properties', []):
                                    if prop.get('name') == 'referencias':
                                        referencias_is_array = 'text[]' in prop.get('dataType', [])
                                        break
                        except:
                            pass

                        # Prepare the value according to its type
                        if referencias_is_array:
                            # Collection expects an array
                            if isinstance(refs_raw, list):
                                data_object["referencias"] = [str(r) for r in refs_raw if r]
                            elif refs_raw:
                                data_object["referencias"] = [str(refs_raw)]
                            else:
                                data_object["referencias"] = []
                        else:
                            # Collection expects a string
                            if isinstance(refs_raw, list):
                                data_object["referencias"] = ' | '.join(str(r) for r in refs_raw) if refs_raw else ''
                            else:
                                data_object["referencias"] = str(refs_raw) if refs_raw else ''

                        # ADD TO BATCH
                        batch.add_data_object(
                            data_object=data_object,
                            class_name=collection_name,
                            vector=vector
                        )

                        successful += 1

                        if i % 100 == 0:
                            logger.info(f"📊 Progreso: {i}/{len(chunks_with_embeddings)}")

                    except Exception as e:
                        logger.error(f"❌ Error in chunk {i}: {e}")
                        failed += 1
                        errors.append(str(e))

            total_time = time.time() - start_time
            logger.info(f"✅ Inserción: {successful} exitosos, {failed} fallidos en {total_time:.2f}s")

            return [{
                'batch_id': 0,
                'success': successful > 0,
                'successful_count': successful,
                'failed_count': failed,
                'processing_time': total_time,
                'errors': errors[:10]
            }]

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ Critical error in batch insert: {e}")

            return [{
                'batch_id': 0,
                'success': False,
                'successful_count': successful,
                'failed_count': len(chunks_with_embeddings) - successful,
                'processing_time': total_time,
                'error': str(e)
            }]
    # ═══════════════════════════════════════════════════════════════
    #  HELPER METHOD: LANGUAGE DETECTION
    # ═══════════════════════════════════════════════════════════════

    def _detect_language(self, text: str) -> str:
        """
        Detect the text language.

        Uses a simple Spanish/English heuristic.
        For more accurate detection, install: pip install langdetect
        """
        try:
            # Option 1: langdetect (if installed)
            try:
                from langdetect import detect
                return detect(text[:500])  # Solo primeros 500 chars
            except ImportError:
                pass

            # Option 2: simple Spanish/English heuristic
            spanish_words = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'por', 'para', 'los', 'las']
            english_words = ['the', 'and', 'of', 'to', 'in', 'a', 'is', 'that', 'for', 'with']

            text_lower = text.lower()[:1000]  # Solo primeros 1000 chars

            spanish_count = sum(1 for w in spanish_words if f' {w} ' in text_lower)
            english_count = sum(1 for w in english_words if f' {w} ' in text_lower)

            if spanish_count > english_count:
                return 'es'
            elif english_count > spanish_count:
                return 'en'
            else:
                return 'unknown'

        except Exception as e:
            logger.warning(f"Error detecting language: {e}")
            return 'unknown'
    # async def _batch_insert_sequential(
    #         self,
    #         chunks_with_embeddings: List[Dict],
    #         collection_name: str,
    #         user: str,
    #         company: str,
    # ) -> List[Dict]:
    #     """
    #     Insert chunks into Weaviate with enriched metadata.
    #     """
    #     logger.info(f"💾 Inserting {len(chunks_with_embeddings)} chunks in a batch...")
    #     start_time = time.time()
    #
    #     successful = 0
    #     failed = 0
    #     errors = []
    #
    #     try:
    #         with self.client.batch as batch:
    #             batch.batch_size = 100
    #
    #             for i, chunk_data in enumerate(chunks_with_embeddings, 1):
    #                 try:
    #                     # Prepare vector
    #                     vector = chunk_data['embedding']
    #                     if isinstance(vector, np.ndarray):
    #                         vector = vector.tolist()
    #                     elif not isinstance(vector, list):
    #                         vector = list(vector)
    #                     vector = [float(x) for x in vector]
    #
    #                     # Clean text
    #                     text = chunk_data['text']
    #                     if not isinstance(text, str):
    #                         text = str(text)
    #
    #                     if text.startswith('{"text":') or text.startswith("{'text':"):
    #                         logger.warning(f"Nested JSON in chunk {i}, cleaning up...")
    #                         text = self._extract_text_from_chunk(text)
    #
    #                     # Preparar metadata
    #                     metadata = chunk_data['metadata']
    #
    #                     # ═══════════════════════════════════════════════════════
    #                     # Fix: try multiple fields for source/filename
    #                     # ═══════════════════════════════════════════════════════
    #
    #                     # Try to get source from several possible fields
    #                     source = (
    #                             str(metadata.get('source', '')) or
    #                             str(metadata.get('file_name', '')) or
    #                             str(metadata.get('filename', '')) or
    #                             ''
    #                     )
    #
    #                     # If source is still empty, try file_id as a fallback
    #                     if not source:
    #                         file_id = metadata.get('file_id', '')
    #                         if file_id:
    #                             source = f"file_{file_id}"
    #
    #                     # Extraer filename
    #                     filename = self._extract_filename(source)
    #
    #                     # If filename is still empty, use file_name directly if present
    #                     if not filename and metadata.get('file_name'):
    #                         filename = metadata.get('file_name')
    #
    #                     # Object for Weaviate
    #                     data_object = {
    #                         # Campos básicos
    #                         "name": chunk_data['chunk_id'],
    #                         "text": text.strip(),
    #                         "user": str(user),
    #                         "company": str(company),
    #
    #                         # FILE METADATA
    #                         "file_id": str(metadata.get('file_id', '')),
    #                         "source": source,  # Puede ser path, file_name, o file_id
    #                         "filename": filename,  # should now have a value
    #                         "doc_type": str(metadata.get('doc_type', '')),
    #
    #                         # CHUNK METADATA
    #                         "chunk_length": int(metadata.get('chunk_length', 0)),
    #                         "word_count": int(metadata.get('word_count', 0)),
    #                         "content_type": str(metadata.get('content_type', 'text')),
    #                         "has_numbers": bool(metadata.get('has_numbers', False)),
    #                         "digit_density": float(metadata.get('digit_density', 0.0)),
    #
    #                         # ✅ REFERENCIAS
    #                         "referencias_text": str(metadata.get('referencias_text', '')),
    #
    #                         # Metadata completa como JSON (backup)
    #                         "metadata_json": json.dumps(metadata, ensure_ascii=False)
    #                     }
    #
    #                     # Add references as an array if present
    #                     if 'referencias' in metadata and metadata['referencias']:
    #                         refs = metadata['referencias']
    #                         if isinstance(refs, list):
    #                             data_object["referencias"] = [str(r) for r in refs]
    #
    #                     # Add to the batch
    #                     batch.add_data_object(
    #                         data_object=data_object,
    #                         class_name=collection_name,
    #                         vector=vector
    #                     )
    #
    #                     successful += 1
    #
    #                     if i % 100 == 0:
    #                         logger.info(f"📊 Progreso: {i}/{len(chunks_with_embeddings)}")
    #
    #                 except Exception as e:
    #                     logger.error(f"❌ Error in chunk {i}: {e}")
    #                     failed += 1
    #                     errors.append(str(e))
    #
    #         total_time = time.time() - start_time
    #         logger.info(f"✅ Inserción: {successful} exitosos, {failed} fallidos en {total_time:.2f}s")
    #
    #         return [{
    #             'batch_id': 0,
    #             'success': successful > 0,
    #             'successful_count': successful,
    #             'failed_count': failed,
    #             'processing_time': total_time,
    #             'errors': errors[:10]
    #         }]
    #
    #     except Exception as e:
    #         total_time = time.time() - start_time
    #         logger.error(f"❌ Critical error: {e}")
    #
    #         return [{
    #             'batch_id': 0,
    #             'success': False,
    #             'successful_count': successful,
    #             'failed_count': len(chunks_with_embeddings) - successful,
    #             'processing_time': total_time,
    #             'error': str(e)
    #         }]
    # ═══════════════════════════════════════════════════════════════
    #  MÉTODO MODIFICADO: _ensure_collection_exists
    # ═══════════════════════════════════════════════════════════════

    async def _ensure_collection_exists(self, company: str, user: str) -> str:
        """
        Create a collection with an IMPROVED schema + Soft Delete + Uploaded By.
        """
        collection_name = re.sub(r"[^a-zA-Z0-9_]", "_", (company or "").upper().replace(" ", "_"))

        try:
            schema = self.client.schema.get()
            existing_classes = [c["class"] for c in schema.get("classes", [])]

            if collection_name not in existing_classes:
                class_obj = {
                    "class": collection_name,
                    "description": "Collection optimized with soft delete and auditing",
                    "properties": [
                        # ═══ CAMPOS BÁSICOS ═══
                        {"name": "name", "dataType": ["text"]},
                        {"name": "text", "dataType": ["text"]},
                        {"name": "user", "dataType": ["text"]},
                        {"name": "company", "dataType": ["text"]},

                        # ═══ FILE METADATA ═══
                        {"name": "file_id", "dataType": ["text"]},
                        {"name": "source", "dataType": ["text"]},
                        {"name": "filename", "dataType": ["text"]},
                        {"name": "doc_type", "dataType": ["text"]},

                        # ═══ CHUNK METADATA ═══
                        {"name": "chunk_length", "dataType": ["int"]},
                        {"name": "word_count", "dataType": ["int"]},
                        {"name": "content_type", "dataType": ["text"]},
                        {"name": "has_numbers", "dataType": ["boolean"]},
                        {"name": "digit_density", "dataType": ["number"]},

                        # ═══ REFERENCIAS ═══
                        {"name": "referencias", "dataType": ["text[]"]},
                        {"name": "referencias_text", "dataType": ["text"]},

                        # ═══ TEMPORAL ═══
                        {"name": "created_at", "dataType": ["number"]},
                        {"name": "created_date", "dataType": ["text"]},

                        # ═══ POSICIÓN Y CONTEXTO ═══
                        {"name": "chunk_index", "dataType": ["int"]},
                        {"name": "total_chunks", "dataType": ["int"]},
                        {"name": "previous_chunk_id", "dataType": ["text"]},
                        {"name": "next_chunk_id", "dataType": ["text"]},

                        # ═══ CONTENT ═══
                        {"name": "language", "dataType": ["text"]},
                        {"name": "text_preview", "dataType": ["text"]},
                        {"name": "text_hash", "dataType": ["text"]},

                        # ═══ PROCESS METADATA ═══
                        {"name": "schema_version", "dataType": ["text"]},
                        {"name": "processing_time", "dataType": ["number"]},
                        {"name": "embedding_model", "dataType": ["text"]},

                        # ═══════════════════════════════════════════════════
                        # 🔥 NUEVOS CAMPOS: SOFT DELETE + UPLOADED BY
                        # ═══════════════════════════════════════════════════

                        # Soft Delete
                        {
                            "name": "is_deleted",
                            "dataType": ["boolean"],
                            "description": "Marks whether the chunk is deleted (soft delete)"
                        },
                        {
                            "name": "deleted_at",
                            "dataType": ["text"],
                            "description": "ISO timestamp when it was deleted"
                        },
                        {
                            "name": "deleted_by",
                            "dataType": ["text"],
                            "description": "Email of the user who deleted it"
                        },
                        {
                            "name": "deletion_reason",
                            "dataType": ["text"],
                            "description": "Reason for deletion"
                        },

                        # Uploaded By (Auditoría)
                        {
                            "name": "uploaded_by",
                            "dataType": ["text"],
                            "description": "Email of the user who uploaded the document"
                        },
                        {
                            "name": "uploaded_by_name",
                            "dataType": ["text"],
                            "description": "Full name of the user who uploaded it"
                        },
                        {
                            "name": "upload_ip",
                            "dataType": ["text"],
                            "description": "IP the document was uploaded from (optional)"
                        },

                        # Backup completo
                        {"name": "metadata_json", "dataType": ["text"]}
                    ],

                    # ═══ ÍNDICES OPTIMIZADOS ═══
                    "invertedIndexConfig": {
                        "indexTimestamps": True,
                        "indexNullState": True,
                        "indexPropertyLength": True
                    }
                }
                self.client.schema.create_class(class_obj)
                logger.info(f"✅ Collection {collection_name} created with soft delete + auditing")

            return collection_name

        except Exception as e:
            logger.error(f"❌ Error creating collection: {e}")
            return None


    # async def _ensure_collection_exists(self, company: str, user: str) -> str:
    #     """
    #     Create a collection with an enriched schema.
    #
    #     SCHEMA WITH:
    #     - File metadata (file_id, source, filename, doc_type)
    #     - Chunk metadata (chunk_length, content_type, etc.)
    #     - References (for hybrid search)
    #     """
    #     collection_name = re.sub(r"[^a-zA-Z0-9_]", "_", (company or "").upper().replace(" ", "_"))
    #
    #     try:
    #         schema = self.client.schema.get()
    #         existing_classes = [c["class"] for c in schema.get("classes", [])]
    #
    #         if collection_name not in existing_classes:
    #             class_obj = {
    #                 "class": collection_name,
    #                 "description": "Collection with enriched chunks, structured metadata and filename",
    #                 "properties": [
    #                     # Campos básicos
    #                     {"name": "name", "dataType": ["text"]},
    #                     {"name": "text", "dataType": ["text"]},
    #                     {"name": "user", "dataType": ["text"]},
    #                     {"name": "company", "dataType": ["text"]},
    #
    #                     # FILE METADATA
    #                     {"name": "file_id", "dataType": ["text"]},
    #                     {"name": "source", "dataType": ["text"]},  # Path completo
    #                     {"name": "filename", "dataType": ["text"]},  # New: name only
    #                     {"name": "doc_type", "dataType": ["text"]},
    #
    #                     # CHUNK METADATA
    #                     {"name": "chunk_length", "dataType": ["int"]},
    #                     {"name": "word_count", "dataType": ["int"]},
    #                     {"name": "content_type", "dataType": ["text"]},
    #                     {"name": "has_numbers", "dataType": ["boolean"]},
    #                     {"name": "digit_density", "dataType": ["number"]},
    #
    #                     # ✅ REFERENCIAS
    #                     {"name": "referencias", "dataType": ["text[]"]},
    #                     {"name": "referencias_text", "dataType": ["text"]},
    #
    #                     # Backup completo
    #                     {"name": "metadata_json", "dataType": ["text"]}
    #                 ]
    #             }
    #             self.client.schema.create_class(class_obj)
    #             logger.info(f"✅ Collection {collection_name} created with filename support")
    #
    #         return collection_name
    #
    #     except Exception as e:
    #         logger.error(f"❌ Error creating collection: {e}")
    #         return None

    # ═══════════════════════════════════════════════════════════════
    #  MÉTODO NUEVO: search_by_filename
    # ═══════════════════════════════════════════════════════════════

    def search_by_filename(
            self,
            collection_name: str,
            user: str,
            company: str,
            filename_query: str,
            limit: int = 10
    ) -> List[Dict]:
        """
        Search chunks by file name.

        Args:
            collection_name: Collection name
            user: User
            company: Company
            filename_query: File name or part of it
            limit: Maximum number of results

        Returns:
            List of chunks belonging to files with that name

        Example:
            search_by_filename(..., filename_query="convenio")
            → Returns chunks from "convenio.pdf", "convenio_metal.pdf", etc.
        """
        try:
            where_filter = {
                "operator": "And",
                "operands": [
                    {"path": ["user"], "operator": "Equal", "valueText": str(user)},
                    {"path": ["company"], "operator": "Equal", "valueText": str(company)},
                    {"path": ["filename"], "operator": "Like", "valueText": f"*{filename_query}*"}
                ]
            }

            result = (
                self.client.query
                .get(collection_name, ["name", "text", "filename", "source", "referencias_text"])
                .with_where(where_filter)
                .with_limit(limit)
                .do()
            )

            chunks = result.get('data', {}).get('Get', {}).get(collection_name, [])

            logger.info(f"🔍 Found {len(chunks)} file chunks with '{filename_query}'")

            return chunks

        except Exception as e:
            logger.error(f"❌ Error searching by filename: {e}")
            return []

    # ═══════════════════════════════════════════════════════════════
    #  MÉTODO NUEVO: get_unique_filenames
    # ═══════════════════════════════════════════════════════════════

    def get_unique_filenames(
            self,
            collection_name: str,
            user: str,
            company: str,
    ) -> List[str]:
        """
        Get the list of unique file names in the collection.

        Returns:
            List of unique filenames

        Example:
            ['convenio.pdf', 'manual.pdf', 'reglamento.pdf']
        """
        try:
            where_filter = {
                "operator": "And",
                "operands": [
                    {"path": ["user"], "operator": "Equal", "valueText": str(user)},
                    {"path": ["company"], "operator": "Equal", "valueText": str(company)},
                ]
            }

            # Query to fetch all filenames
            result = (
                self.client.query
                .get(collection_name, ["filename"])
                .with_where(where_filter)
                .with_limit(10000)  # High limit to fetch them all
                .do()
            )

            chunks = result.get('data', {}).get('Get', {}).get(collection_name, [])

            # Extraer filenames únicos
            filenames = set()
            for chunk in chunks:
                fn = chunk.get('filename', '')
                if fn:
                    filenames.add(fn)

            unique_filenames = sorted(list(filenames))

            logger.info(f"📁 Unique files in collection: {len(unique_filenames)}")

            return unique_filenames

        except Exception as e:
            logger.error(f"❌ Error fetching unique filenames: {e}")
            return []

    # ═══ VALIDATION METHODS ═══

    def validate_document_storage(self, collection_name: str, user: str, company: str,
                                  expected_chunks: int = None,
                                  file_id: str = None) -> Dict:
        """Validate that the document is fully stored."""
        try:
            where_filter = {
                "operator": "And",
                "operands": [
                    {"path": ["user"], "operator": "Equal", "valueText": str(user)},
                    {"path": ["company"], "operator": "Equal", "valueText": str(company)},
                ]
            }

            if file_id:
                where_filter["operands"].append({
                    "path": ["file_id"],
                    "operator": "Equal",
                    "valueText": str(file_id)
                })

            result = (
                self.client.query
                .aggregate(collection_name)
                .with_where(where_filter)
                .with_meta_count()
                .do()
            )

            stored_count = result.get('data', {}).get('Aggregate', {}).get(collection_name, [{}])[0].get('meta',
                                                                                                         {}).get(
                'count', 0)
            is_complete = stored_count == expected_chunks if expected_chunks else None

            return {
                'success': True,
                'stored_chunks': stored_count,
                'expected_chunks': expected_chunks,
                'is_complete': is_complete,
                'collection_name': collection_name
            }

        except Exception as e:
            logger.error(f"❌ Error validating storage: {e}")
            return {
                'success': False,
                'error': str(e),
                'stored_chunks': 0,
                'expected_chunks': expected_chunks,
                'is_complete': False
            }

    def validate_document_content(self, collection_name: str, user: str, company: str,
                                  file_id: str = None,
                                  sample_size: int = 5) -> Dict:
        """Validate stored content with samples."""
        try:
            where_filter = {
                "operator": "And",
                "operands": [
                    {"path": ["user"], "operator": "Equal", "valueText": str(user)},
                    {"path": ["company"], "operator": "Equal", "valueText": str(company)},
                ]
            }

            result = (
                self.client.query
                .get(collection_name, ["name", "text", "filename", "referencias_text", "content_type"])
                .with_where(where_filter)
                .with_limit(sample_size)
                .do()
            )

            chunks = result.get('data', {}).get('Get', {}).get(collection_name, [])

            validation = {
                'success': True,
                'sample_size': len(chunks),
                'chunks_preview': []
            }

            for i, chunk in enumerate(chunks, 1):
                text = chunk.get('text', '')
                is_clean = not (text.startswith('{"text":') or text.startswith("{'text':"))

                chunk_info = {
                    'chunk_number': i,
                    'chunk_id': chunk.get('name'),
                    'filename': chunk.get('filename', ''),  # ✅ Incluye filename
                    'text_length': len(text),
                    'text_preview': text[:100] + '...' if len(text) > 100 else text,
                    'is_clean': is_clean,
                    'referencias': chunk.get('referencias_text', ''),
                    'content_type': chunk.get('content_type', '')
                }

                validation['chunks_preview'].append(chunk_info)

            return validation

        except Exception as e:
            logger.error(f"❌ Error validating content: {e}")
            return {
                'success': False,
                'error': str(e),
                'sample_size': 0,
                'chunks_preview': []
            }

    # ═══ PARALLEL METHOD (optional, unchanged) ═══

    async def _parallel_batch_insert(self, chunks_with_embeddings: List[Dict],
                                     collection_name: str, user: str, company: str) -> List[Dict]:
        """Insert chunks using parallel batches (same change as _batch_insert_sequential)."""
        # Implementación similar a _batch_insert_sequential pero en paralelo
        pass


# ═══════════════════════════════════════════════════════════════
#  FUNCIÓN CONVENIENTE ACTUALIZADA
# ═══════════════════════════════════════════════════════════════

async def process_files_parallel_weaviate(
        weaviate_client,
        embedding_processor,
        files_data: List[Dict],
        user: str,
        company: str,
        max_concurrent: int = 4
) -> Dict[str, Any]:
    """
    Convenience function to process files with enriched chunks.

    FEATURES:
    - Enriched chunks with references
    - Automatic filename extraction
    - Search by file name
    - Backward compatible

    files_data: [
        {
            'chunks': [...],  # List[str] o List[Dict]
            'metadata': {
                'file_id': '...',
                'source': '/path/to/archivo.pdf',  # ← filename se extrae de aquí
                ...
            }
        },
        ...
    ]
    """
    processor = ParallelWeaviateProcessor(
        weaviate_client=weaviate_client,
        embedding_processor=embedding_processor,
        max_concurrent_files=max_concurrent,
        batch_insert_size=100
    )

    return await processor.process_multiple_files_parallel(files_data, user, company)