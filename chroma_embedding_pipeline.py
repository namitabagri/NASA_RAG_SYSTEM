"""Core ChromaDB embedding pipeline for NASA space mission text data.

This module implements the pipeline class that orchestrates text file
processing, OpenAI embedding generation, and ChromaDB collection management.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from text_processing import (
    build_text_file_metadata,
    chunk_text,
    extract_mission_from_path,
    scan_text_files_only,
)

logger = logging.getLogger(__name__)


class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings - text files only."""

    def __init__(
        self,
        openai_api_key: str,
        chroma_persist_directory: str = './chroma_db',
        collection_name: str = 'nasa_space_missions_text',
        embedding_model: str = 'text-embedding-3-small',
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        if not openai_api_key:
            raise ValueError('OpenAI API key is required')

        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_directory)
        self.embedding_function = OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=embedding_model,
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        return chunk_text(text, metadata, self.chunk_size, self.chunk_overlap)

    def check_document_exists(self, doc_id: str) -> bool:
        try:
            result = self.collection.get(ids=[doc_id])
            return bool(result.get('ids'))
        except Exception:
            return False

    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        try:
            embedding = self.get_embedding(text)
            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding],
            )
            logger.debug(f'Updated document: {doc_id}')
            return True
        except Exception as e:
            logger.error(f'Error updating document {doc_id}: {e}')
            return False

    def delete_documents_by_source(self, source_pattern: str) -> int:
        try:
            all_docs = self.collection.get()
            ids_to_delete = []
            for i, metadata in enumerate(all_docs.get('metadatas', [])):
                if source_pattern in metadata.get('source', ''):
                    ids_to_delete.append(all_docs['ids'][i])

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f'Deleted {len(ids_to_delete)} documents matching source pattern: {source_pattern}')
                return len(ids_to_delete)

            logger.info(f'No documents found matching source pattern: {source_pattern}')
            return 0
        except Exception as e:
            logger.error(f'Error deleting documents by source: {e}')
            return 0

    def get_file_documents(self, file_path: Path) -> List[str]:
        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)
            all_docs = self.collection.get()
            file_doc_ids = []
            for i, metadata in enumerate(all_docs.get('metadatas', [])):
                if metadata.get('source') == source and metadata.get('mission') == mission:
                    file_doc_ids.append(all_docs['ids'][i])
            return file_doc_ids
        except Exception as e:
            logger.error(f'Error getting file documents: {e}')
            return []

    def get_embedding(self, text: str) -> List[float]:
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_model,
        )
        return response.data[0].embedding

    def generate_document_id(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        mission = metadata.get('mission', 'unknown')
        source = metadata.get('source', file_path.stem)
        chunk_index = metadata.get('chunk_index', 0)
        return f'{mission}_{source}_chunk_{chunk_index:04d}'

    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                return []

            metadata = build_text_file_metadata(file_path, content)
            return self.chunk_text(content, metadata)
        except Exception as e:
            logger.error(f'Error processing text file {file_path}: {e}')
            return []

    def scan_text_files_only(self, base_path: str) -> List[Path]:
        return scan_text_files_only(base_path, logger)

    def add_documents_to_collection(
        self,
        documents: List[Tuple[str, Dict[str, Any]]],
        file_path: Path,
        batch_size: int = 50,
        update_mode: str = 'skip',
    ) -> Dict[str, int]:
        stats = {'added': 0, 'updated': 0, 'skipped': 0}
        if update_mode == 'replace':
            self.delete_documents_by_source(file_path.stem)

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            for text, metadata in batch:
                doc_id = self.generate_document_id(file_path, metadata)
                exists = self.check_document_exists(doc_id)

                if exists and update_mode == 'skip':
                    stats['skipped'] += 1
                    continue
                if exists and update_mode == 'update':
                    if self.update_document(doc_id, text, metadata):
                        stats['updated'] += 1
                    continue

                embedding = self.get_embedding(text)
                self.collection.add(
                    ids=[doc_id],
                    documents=[text],
                    metadatas=[metadata],
                    embeddings=[embedding],
                )
                stats['added'] += 1

        return stats

    def process_all_text_data(self, base_path: str, update_mode: str = 'skip', batch_size: int = 50) -> Dict[str, Any]:
        stats = {
            'files_processed': 0,
            'documents_added': 0,
            'documents_updated': 0,
            'documents_skipped': 0,
            'errors': 0,
            'total_chunks': 0,
            'missions': {}
        }

        files = self.scan_text_files_only(base_path)

        for file_path in files:
            try:
                documents = self.process_text_file(file_path)
                file_stats = self.add_documents_to_collection(
                    documents,
                    file_path,
                    batch_size=batch_size,
                    update_mode=update_mode,
                )

                stats['files_processed'] += 1
                stats['documents_added'] += file_stats['added']
                stats['documents_updated'] += file_stats['updated']
                stats['documents_skipped'] += file_stats['skipped']
                stats['total_chunks'] += len(documents)

                mission = self.extract_mission_from_path(file_path)
                mission_stats = stats['missions'].setdefault(
                    mission,
                    {'files': 0, 'chunks': 0, 'added': 0, 'updated': 0, 'skipped': 0},
                )
                mission_stats['files'] += 1
                mission_stats['chunks'] += len(documents)
                mission_stats['added'] += file_stats['added']
                mission_stats['updated'] += file_stats['updated']
                mission_stats['skipped'] += file_stats['skipped']
            except Exception as e:
                logger.error(f'Error processing {file_path}: {e}')
                stats['errors'] += 1

        return stats

    def get_collection_info(self) -> Dict[str, Any]:
        return {
            'name': self.collection.name,
            'count': self.collection.count(),
            'metadata': getattr(self.collection, 'metadata', {})
        }

    def query_collection(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            all_docs = self.collection.get()
            metadatas = all_docs.get('metadatas', [])
            if not metadatas:
                return {'error': 'No documents in collection'}

            stats: Dict[str, Any] = {
                'total_documents': len(metadatas),
                'missions': {},
                'data_types': {},
                'document_categories': {},
                'file_types': {}
            }

            for metadata in metadatas:
                mission = metadata.get('mission', 'unknown')
                data_type = metadata.get('data_type', 'unknown')
                doc_category = metadata.get('document_category', 'unknown')
                file_type = metadata.get('file_type', 'unknown')

                stats['missions'][mission] = stats['missions'].get(mission, 0) + 1
                stats['data_types'][data_type] = stats['data_types'].get(data_type, 0) + 1
                stats['document_categories'][doc_category] = stats['document_categories'].get(doc_category, 0) + 1
                stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1

            return stats
        except Exception as e:
            logger.error(f'Error getting collection stats: {e}')
            return {'error': str(e)}

    def extract_mission_from_path(self, file_path: Path) -> str:
        return extract_mission_from_path(file_path)
