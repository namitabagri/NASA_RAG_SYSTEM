# NASA RAG System

A Retrieval-Augmented Generation (RAG) system for NASA space mission data, featuring ChromaDB embeddings with OpenAI integration and conversational LLM capabilities.

## Project Structure

The project is organized into focused, reusable modules:

### Core Modules

#### `text_processing.py`
**Purpose:** Utility functions for text ingestion and metadata extraction

- `chunk_text()` — Split text into manageable chunks with overlap preservation
- `extract_mission_from_path()` — Identify mission type from file paths (Apollo 11, Apollo 13, Challenger)
- `extract_data_type_from_path()` — Classify document types (transcript, textract, audio, flight plan)
- `extract_document_category_from_filename()` — Categorize documents (PAO, command module, technical, etc.)
- `build_text_file_metadata()` — Generate rich metadata for documents
- `scan_text_files_only()` — Recursively discover text files in mission directories

#### `chroma_embedding_pipeline.py`
**Purpose:** Core pipeline orchestrating ChromaDB and OpenAI embedding operations

**Main Class:** `ChromaEmbeddingPipelineTextOnly`

Key methods:
- `__init__()` — Initialize ChromaDB client with OpenAI embedding function
- `process_text_file()` — Read and chunk individual text files
- `process_all_text_data()` — Batch process all mission data
- `add_documents_to_collection()` — Ingest documents with configurable update modes (skip/update/replace)
- `query_collection()` — Search embeddings for similar documents
- `get_collection_stats()` — Analyze collection breakdown by mission, data type, category
- `delete_documents_by_source()` — Remove documents matching source patterns

#### `pipeline_cli.py`
**Purpose:** Command-line interface for the embedding pipeline

Provides:
- Argument parsing for all pipeline operations
- Logging configuration
- Execution orchestration via `main()`

Supported operations:
- Full data ingestion with configurable parameters
- Collection statistics viewing
- Document deletion by source pattern
- Test queries after processing

#### `conversational_llm_client.py`
**Purpose:** OpenAI conversational client with history management

Features:
- Conversation history tracking
- Context-aware response generation
- Single client instance for efficiency
- NASA space exploration domain knowledge

### Entry Points

#### `pipeline_entrypoint.py`
**Purpose:** Backwards-compatible legacy entrypoint

Delegates to `pipeline_cli.py` for backwards compatibility while maintaining new module structure.

## Installation

```bash
pip install -r requirements.txt
```

Ensure `.env` file contains:
```
OPENAI_API_KEY=<your-key>
```

## Usage

### Run the Embedding Pipeline

```bash
python pipeline_entrypoint.py \
  --openai-key $OPENAI_API_KEY \
  --data-path /path/to/mission/data \
  --chroma-dir ./chroma_db \
  --chunk-size 500 \
  --chunk-overlap 100
```

### View Collection Statistics

```bash
python pipeline_entrypoint.py \
  --openai-key $OPENAI_API_KEY \
  --stats-only
```

### Delete Documents by Source

```bash
python pipeline_entrypoint.py \
  --openai-key $OPENAI_API_KEY \
  --delete-source apollo11
```

### Update Mode Options

- `skip` — Skip existing documents (default)
- `update` — Update existing documents with new embeddings
- `replace` — Delete all documents from file and re-ingest

## Module Dependencies

```
text_processing.py
    ↓
chroma_embedding_pipeline.py
    ↓
pipeline_cli.py
    ↓
pipeline_entrypoint.py
```

`conversational_llm_client.py` is standalone for RAG response generation.

## Supported Data Sources

- Apollo 11 extracted data (text files)
- Apollo 13 extracted data (text files)
- Apollo 11 Textract extracted data (text files)
- Challenger transcribed audio data (text files)

## Configuration

### Chunking Parameters

- `--chunk-size` (default: 500) — Maximum characters per chunk
- `--chunk-overlap` (default: 100) — Overlap between chunks
- `--batch-size` (default: 50) — Documents processed per batch

### Embedding Parameters

- `--embedding-model` (default: text-embedding-3-small) — OpenAI embedding model
- `--collection-name` (default: nasa_space_missions_text) — ChromaDB collection name

## Output

Processing logs are written to:
- Console (stderr)
- `chroma_embedding_text_only.log`

Statistics include:
- Files processed
- Total chunks created
- Documents added/updated/skipped
- Per-mission breakdown
- Processing time
