"""Command line entrypoint for the NASA ChromaDB embedding pipeline.

This module parses CLI arguments, initializes the embedding pipeline, and
executes ingest, deletion, stats, and query operations.
"""

import argparse
import logging
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from chroma_embedding_pipeline import ChromaEmbeddingPipelineTextOnly

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chroma_embedding_text_only.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Parse command-line arguments and run the embedding pipeline."""
    parser = argparse.ArgumentParser(description='ChromaDB Embedding Pipeline for NASA Data')
    parser.add_argument('--data-path', default='.', help='Path to data directories')
    parser.add_argument('--openai-key', help='OpenAI API key (or set OPENAI_API_KEY in environment or .env)')
    parser.add_argument('--chroma-dir', default='./chroma_db_openai', help='ChromaDB persist directory')
    parser.add_argument('--collection-name', default='nasa_space_missions_text', help='Collection name')
    parser.add_argument('--embedding-model', default='text-embedding-3-small', help='OpenAI embedding model')
    parser.add_argument('--chunk-size', type=int, default=500, help='Text chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=100, help='Chunk overlap size')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--update-mode', choices=['skip', 'update', 'replace'], default='skip',
                        help='How to handle existing documents: skip, update, or replace')
    parser.add_argument('--test-query', help='Test query after processing')
    parser.add_argument('--stats-only', action='store_true', help='Only show collection statistics')
    parser.add_argument('--delete-source', help='Delete all documents from a specific source pattern')

    args = parser.parse_args()

    # Support providing the OpenAI key either via CLI or environment (.env supported by load_dotenv)
    openai_key = args.openai_key or os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_KEY')
    if not openai_key:
        parser.error('OpenAI API key required. Provide --openai-key or set OPENAI_API_KEY in your environment or a .env file')

    logger.info('Initializing ChromaDB Embedding Pipeline...')
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if args.delete_source:
        deleted_count = pipeline.delete_documents_by_source(args.delete_source)
        logger.info(f'Deleted {deleted_count} documents matching source pattern: {args.delete_source}')
        return

    if args.stats_only:
        logger.info('Collection Statistics:')
        stats = pipeline.get_collection_stats()
        for key, value in stats.items():
            logger.info(f'{key}: {value}')
        return

    logger.info(f'Starting text data processing with update mode: {args.update_mode}')
    start_time = time.time()
    stats = pipeline.process_all_text_data(args.data_path, update_mode=args.update_mode, batch_size=args.batch_size)
    processing_time = time.time() - start_time

    logger.info('=' * 60)
    logger.info('PROCESSING COMPLETE')
    logger.info('=' * 60)
    logger.info(f'Files processed: {stats["files_processed"]}')
    logger.info(f'Total chunks created: {stats["total_chunks"]}')
    logger.info(f'Documents added to collection: {stats["documents_added"]}')
    logger.info(f'Documents updated in collection: {stats["documents_updated"]}')
    logger.info(f'Documents skipped (already exist): {stats["documents_skipped"]}')
    logger.info(f'Errors: {stats["errors"]}')
    logger.info(f'Processing time: {processing_time:.2f} seconds')

    logger.info('\nMission breakdown:')
    for mission, mission_stats in stats['missions'].items():
        logger.info(f'  {mission}: {mission_stats["files"]} files, {mission_stats["chunks"]} chunks')
        logger.info(f'    Added: {mission_stats["added"]}, Updated: {mission_stats["updated"]}, Skipped: {mission_stats["skipped"]}')

    collection_info = pipeline.get_collection_info()
    logger.info(f"\nCollection: {collection_info.get('name', 'N/A')}")
    logger.info(f"Total documents in collection: {collection_info.get('count', 'N/A')}")

    if args.test_query:
        logger.info(f"\nTesting query: '{args.test_query}'")
        results = pipeline.query_collection(args.test_query)
        documents = results.get('documents', [])
        if documents:
            logger.info(f'Found {len(documents[0])} results:')
            for i, doc in enumerate(documents[0][:3]):
                logger.info(f'Result {i + 1}: {doc[:200]}...')

    logger.info('Pipeline completed successfully!')


if __name__ == '__main__':
    main()
