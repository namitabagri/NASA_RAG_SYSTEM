import json
import os
from pathlib import Path
from typing import Any, Dict, List

from rag_client import initialize_rag_system, retrieve_documents, format_context


DATASET_PATH = Path(__file__).resolve().parent / "test_questions.json"
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nasa_space_missions_text")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")


def load_test_questions(path: Path = DATASET_PATH) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_evaluation() -> None:
    questions = load_test_questions()

    if not questions:
        print("No evaluation questions were loaded.")
        return

    print(f"Loaded {len(questions)} evaluation questions from {DATASET_PATH.name}")

    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY is not set. The evaluator will only show retrieved context and skip answer generation.")

    try:
        collection = initialize_rag_system(CHROMA_DIR, COLLECTION_NAME)
    except Exception as exc:
        print(f"Unable to initialize RAG collection: {exc}")
        print("Make sure the Chroma database exists and the collection name is correct.")
        collection = None

    for item in questions:
        question = item.get("question", "")
        category = item.get("category", "unknown")
        print(f"\n[{category}] {question}")

        if collection is None:
            print("Response: RAG collection unavailable")
            continue

        try:
            docs_result = retrieve_documents(collection, question, n_results=3)
            if docs_result and docs_result.get("documents"):
                context = format_context(docs_result["documents"][0], docs_result["metadatas"][0])
                if OPENAI_API_KEY:
                    from conversational_llm_client import generate_response

                    response = generate_response(
                        question,
                        [],
                        context,
                        model=MODEL_NAME,
                    )
                else:
                    response = "OpenAI API key not configured"
                print("Response:")
                print(response)
            else:
                print("Response: No relevant documents found")
        except Exception as exc:
            print(f"Response: Evaluation failed: {exc}")


if __name__ == "__main__":
    run_evaluation()
