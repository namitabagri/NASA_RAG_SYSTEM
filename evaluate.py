import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List
from dotenv import load_dotenv

from rag_client import initialize_rag_system, retrieve_documents, format_context


DATASET_PATH = Path(__file__).resolve().parent / "test_questions.json"
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nasa_space_missions_text")
# Load environment variables from a .env file (if present) so
# os.getenv can pick up values when scripts are run directly.
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "their", "this", "to",
    "what", "when", "where", "which", "who", "why", "with"
}


def load_test_questions(path: Path = DATASET_PATH) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", _normalize_text(text))


def compute_answer_similarity(actual_answer: str, expected_answer: str) -> float:
    if not actual_answer and not expected_answer:
        return 0.0

    actual_tokens = _tokenize(actual_answer)
    expected_tokens = _tokenize(expected_answer)
    if not actual_tokens or not expected_tokens:
        return float(SequenceMatcher(None, _normalize_text(actual_answer), _normalize_text(expected_answer)).ratio())

    overlap = set(actual_tokens) & set(expected_tokens)
    if not overlap:
        return float(SequenceMatcher(None, _normalize_text(actual_answer), _normalize_text(expected_answer)).ratio())

    precision = len(overlap) / len(actual_tokens)
    recall = len(overlap) / len(expected_tokens)
    if precision + recall == 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))


def compute_faithfulness(actual_answer: str, expected_topics: List[str]) -> float:
    if not expected_topics:
        return 0.0

    actual_lower = actual_answer.lower()
    matched = sum(1 for topic in expected_topics if topic.lower() in actual_lower)
    return float(matched / len(expected_topics))


def compute_relevancy(question: str, actual_answer: str) -> float:
    if len(actual_answer.strip()) <= 50:
        return 0.0

    keywords = [
        token for token in re.findall(r"\w+", question.lower())
        if len(token) > 2 and token not in STOPWORDS
    ]
    if not keywords:
        return 1.0

    matches = [token for token in keywords if token in actual_answer.lower()]
    return 1.0 if matches else 0.0


def summarize_metrics(rows: List[Dict[str, Any]]) -> None:
    print(f"Aggregate Metrics (n={len(rows)})")
    print("─────────────────────────────")

    if not rows:
        print("answer_similarity : mean=0.00  min=0.00  max=0.00")
        print("faithfulness      : mean=0.00  min=0.00  max=0.00")
        print("relevancy         : mean=0.00  min=0.00  max=0.00")
        return

    for metric_name in ("answer_similarity", "faithfulness", "relevancy"):
        values = [float(row[metric_name]) for row in rows if row[metric_name] is not None]
        if not values:
            values = [0.0]
        print(
            f"{metric_name:<17}: mean={mean(values):.2f}  min={min(values):.2f}  max={max(values):.2f}"
        )


def run_evaluation() -> None:
    questions = load_test_questions()

    if not questions:
        print("No evaluation questions were loaded.")
        return

    print(f"Loaded {len(questions)} evaluation questions from {DATASET_PATH.name}")

    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY is not set. The evaluator will use retrieved context when generation is unavailable.")

    try:
        rag_result = initialize_rag_system(CHROMA_DIR, COLLECTION_NAME)
        if isinstance(rag_result, tuple) and len(rag_result) >= 3:
            collection, success, error_message = rag_result
        else:
            collection, success, error_message = rag_result, True, ""
    except Exception as exc:
        print(f"Unable to initialize RAG collection: {exc}")
        print("Make sure the Chroma database exists and the collection name is correct.")
        collection = None
        success = False
        error_message = str(exc)

    if not success:
        print(f"RAG initialization failed: {error_message}")

    rows: List[Dict[str, Any]] = []
    for item in questions:
        question = item.get("question", "")
        category = item.get("category", "unknown")
        expected_answer = item.get("expected_answer", "")
        expected_topics = item.get("expected_topics", [])
        entry_id = item.get("id", "unknown")

        try:
            if collection is None:
                raise RuntimeError(error_message or "RAG collection unavailable")

            docs_result = retrieve_documents(collection, question, n_results=3)
            if docs_result and docs_result.get("documents"):
                context = format_context(docs_result["documents"][0], docs_result["metadatas"][0])
                if OPENAI_API_KEY:
                    from conversational_llm_client import generate_response

                    actual_answer = generate_response(
                        question,
                        [],
                        context,
                        model=MODEL_NAME,
                    )
                else:
                    actual_answer = context if context else "No relevant documents found"
            else:
                actual_answer = "No relevant documents found"
        except Exception:
            actual_answer = "Evaluation failed"
            rows.append(
                {
                    "id": entry_id,
                    "category": category,
                    "answer_similarity": 0.0,
                    "faithfulness": 0.0,
                    "relevancy": 0.0,
                }
            )
            continue

        answer_similarity = compute_answer_similarity(actual_answer, expected_answer)
        faithfulness = compute_faithfulness(actual_answer, expected_topics)
        relevancy = compute_relevancy(question, actual_answer)
        rows.append(
            {
                "id": entry_id,
                "category": category,
                "answer_similarity": answer_similarity,
                "faithfulness": faithfulness,
                "relevancy": relevancy,
            }
        )

    print("\nPer-question Summary")
    print("| id | category | answer_similarity | faithfulness | relevancy |")
    print("| --- | --- | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['id']} | {row['category']} | {row['answer_similarity']:.2f} | {row['faithfulness']:.2f} | {row['relevancy']:.2f} |"
        )

    summarize_metrics(rows)


if __name__ == "__main__":
    run_evaluation()
