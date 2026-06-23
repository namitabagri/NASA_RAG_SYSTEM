from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports (optional). Wrap all ragas imports in a try/except so
# importing this module does not raise if the `ragas` package is not
# installed. When unavailable, set RAGAS_AVAILABLE = False and provide
# safe None fallbacks for names used elsewhere.
try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas import SingleTurnSample
    from ragas.metrics import (
        BleuScore,
        NonLLMContextPrecisionWithReference,
        ResponseRelevancy,
        Faithfulness,
        RougeScore,
    )
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except Exception:
    # ragas not installed or import failed; provide safe fallbacks
    LangchainLLMWrapper = None
    LangchainEmbeddingsWrapper = None
    SingleTurnSample = None
    BleuScore = None
    NonLLMContextPrecisionWithReference = None
    ResponseRelevancy = None
    Faithfulness = None
    RougeScore = None
    evaluate = None
    RAGAS_AVAILABLE = False


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        NonLLMContextPrecisionWithReference(),
        BleuScore(),
        RougeScore(),
    ]

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=answer  
    )

    results = {}
    for metric in metrics:
        try:
            score = metric.single_turn_score(sample)
            results[metric.name] = round(score, 4)
        except Exception as e:
            results[metric.name] = f"Error: {str(e)[:50]}"

    return results