"""ChromaDB helper utilities.

Make the chromadb import optional so the rest of the app can still
import this module when chromadb isn't installed. When unavailable,
functions will return safe fallbacks and errors rather than raising
ImportError at module import time.
"""

from typing import Dict, List, Optional
from pathlib import Path

# Optional import of chromadb
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except Exception:
    chromadb = None
    Settings = None
    CHROMADB_AVAILABLE = False


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends: Dict[str, Dict[str, str]] = {}

    if not CHROMADB_AVAILABLE:
        # chromadb isn't installed; return empty set so the app can show
        # a helpful message instead of crashing.
        return backends

    current_dir = Path(".")

    # Look for ChromaDB directories that match specific criteria
    chroma_dirs = [d for d in current_dir.iterdir() if d.is_dir() and "chroma" in d.name.lower()]

    for chroma_dir in chroma_dirs:
        try:
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )

            collections = client.list_collections()

            for collection in collections:
                key = f"{chroma_dir.name}/{collection.name}"
                try:
                    count = collection.count()
                except Exception:
                    count = "N/A"

                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name": f"{chroma_dir.name} - {collection.name} ({count} docs)",
                    "count": str(count)
                }

        except Exception as e:
            error_preview = str(e)[:50]
            backends[str(chroma_dir)] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} (Error: {error_preview}...)",
                "count": "0"
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend.

    Returns a tuple (collection, success, error_message).
    """
    if not CHROMADB_AVAILABLE:
        return None, False, "chromadb package not installed"

    try:
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(collection_name)
        return collection, True, ""
    except Exception as e:
        return None, False, str(e)


def retrieve_documents(collection, query: str, n_results: int = 3,
                       mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    if collection is None:
        return None

    where_filter = None

    if mission_filter and mission_filter.lower() not in ("all", "none", ""):
        where_filter = {"mission": mission_filter}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )

    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    context_parts = ["=== RELEVANT CONTEXT ==="]

    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        mission = meta.get("mission", "unknown")
        mission = mission.replace("_", " ").capitalize()

        source = meta.get("source", "unknown")

        category = meta.get("category", "general")
        category = category.replace("_", " ").capitalize()

        header = f"\n[Source {i}] Mission: {mission} | File: {source} | Category: {category}"
        context_parts.append(header)

        max_length = 800
        content = doc[:max_length] + "..." if len(doc) > max_length else doc
        context_parts.append(content)

    return "\n".join(context_parts)