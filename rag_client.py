import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
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
                    "dir": str(chroma_dir),
                    "collection": collection.name,
                    "display_name": f"{chroma_dir.name} - {collection.name} ({count} docs)",
                    "count": str(count)
                }

        except Exception as e:
            error_preview = str(e)[:50]
            backends[str(chroma_dir)] = {
                "dir": str(chroma_dir),
                "collection": "",
                "display_name": f"{chroma_dir.name} (Error: {error_preview}...)",
                "count": "0"
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_collection(collection_name)


def retrieve_documents(collection, query: str, n_results: int = 3,
                       mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
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