"""Utility helpers for NASA text file ingestion and metadata processing.

This module provides reusable functions for scanning mission data folders,
extracting metadata, and splitting text into chunked documents for
ChromaDB embedding ingestion.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging
from datetime import datetime


def chunk_text(text: str, metadata: Dict[str, Any], chunk_size: int, chunk_overlap: int) -> List[Tuple[str, Dict[str, Any]]]:
    """Split text into chunks with metadata."""
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [(text, {**metadata, 'chunk_index': 0, 'total_chunks': 1})]

    chunks: List[Tuple[str, Dict[str, Any]]] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            brk_point = text.rfind('.', start, end)
            if brk_point > start:
                end = brk_point + 1

        chunk = text[start:end].strip()
        if chunk:
            chunk_metadata = {**metadata, 'chunk_index': chunk_index}
            chunks.append((chunk, chunk_metadata))
            chunk_index += 1

        if end >= len(text):
            break

        start = max(end - chunk_overlap, end)

    total_chunks = len(chunks)
    for _, meta in chunks:
        meta['total_chunks'] = total_chunks

    return chunks


def extract_mission_from_path(file_path: Path) -> str:
    """Extract mission name from file path."""
    path_str = str(file_path).lower()
    if 'apollo11' in path_str or 'apollo_11' in path_str:
        return 'apollo_11'
    if 'apollo13' in path_str or 'apollo_13' in path_str:
        return 'apollo_13'
    if 'challenger' in path_str:
        return 'challenger'
    return 'unknown'


def extract_data_type_from_path(file_path: Path) -> str:
    """Extract data type from file path."""
    path_str = str(file_path).lower()
    if 'transcript' in path_str:
        return 'transcript'
    if 'textract' in path_str:
        return 'textract_extracted'
    if 'audio' in path_str:
        return 'audio_transcript'
    if 'flight_plan' in path_str:
        return 'flight_plan'
    return 'document'


def extract_document_category_from_filename(filename: str) -> str:
    """Extract document category from filename for better organization."""
    filename_lower = filename.lower()
    if 'pao' in filename_lower:
        return 'public_affairs_officer'
    if 'cm' in filename_lower:
        return 'command_module'
    if 'tec' in filename_lower:
        return 'technical'
    if 'flight_plan' in filename_lower:
        return 'flight_plan'
    if 'mission_audio' in filename_lower:
        return 'mission_audio'
    if 'ntrs' in filename_lower:
        return 'nasa_archive'
    if '19900066485' in filename_lower:
        return 'technical_report'
    if '19710015566' in filename_lower:
        return 'mission_report'
    if 'full_text' in filename_lower:
        return 'complete_document'
    return 'general_document'


def build_text_file_metadata(file_path: Path, content: str) -> Dict[str, Any]:
    """Build metadata for a text file document."""
    return {
        'source': file_path.stem,
        'file_path': str(file_path),
        'file_type': 'text',
        'content_type': 'full_text',
        'mission': extract_mission_from_path(file_path),
        'data_type': extract_data_type_from_path(file_path),
        'document_category': extract_document_category_from_filename(file_path.name),
        'file_size': len(content),
        'processed_timestamp': datetime.now().isoformat()
    }


def scan_text_files_only(base_path: str, logger: logging.Logger) -> List[Path]:
    """Scan data directories for text files only (avoiding JSON duplicates)."""
    base_path_obj = Path(base_path)
    files_to_process: List[Path] = []
    data_dirs = ['apollo11', 'apollo13', 'challenger']

    for data_dir in data_dirs:
        dir_path = base_path_obj / data_dir
        if dir_path.exists():
            logger.info(f"Scanning directory: {dir_path}")
            text_files = list(dir_path.glob('**/*.txt'))
            files_to_process.extend(text_files)
            logger.info(f"Found {len(text_files)} text files in {data_dir}")

    filtered_files: List[Path] = []
    for file_path in files_to_process:
        if (file_path.name.startswith('.') or
                'summary' in file_path.name.lower() or
                file_path.suffix.lower() != '.txt'):
            continue
        filtered_files.append(file_path)

    logger.info(f"Total text files to process: {len(filtered_files)}")

    mission_counts: Dict[str, int] = {}
    for file_path in filtered_files:
        mission = extract_mission_from_path(file_path)
        mission_counts[mission] = mission_counts.get(mission, 0) + 1

    logger.info("Files by mission:")
    for mission, count in mission_counts.items():
        logger.info(f"  {mission}: {count} files")

    return filtered_files
