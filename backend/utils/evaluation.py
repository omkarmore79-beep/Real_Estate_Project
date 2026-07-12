"""
RAG Evaluation Framework.
Tracks quality metrics: Precision, Recall, MRR, NDCG, Hallucination Rate, and Citation Accuracy.
Stores metrics in MongoDB or a local JSON file.
"""

from __future__ import annotations

import logging
import os
import json
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

EVAL_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "storage", "evaluation_metrics.json")

def evaluate_rag_response(
    question: str,
    answer: str,
    retrieved_results: list[dict],
    citations: list[dict]
) -> dict[str, Any]:
    """
    Perform a programmatic evaluation on the generated RAG response.
    Metrics evaluated:
      - Citation Accuracy (percentage of citations that match retrieved sources)
      - Factual Grounding (faithfulness score based on semantic overlap)
      - Average Retrieval Score (NDCG/MRR proxy using cosine scores)
    """
    # 1. Compute Citation Accuracy
    # Check if citations match the actual retrieved document IDs
    retrieved_doc_ids = {r.get("document_id") for r in retrieved_results if r.get("document_id")}
    valid_citations = 0
    for cit in citations:
        doc_id = cit.get("document_id")
        if doc_id in retrieved_doc_ids:
            valid_citations += 1
            
    citation_accuracy = (valid_citations / len(citations)) if citations else 1.0

    # 2. Compute Factual Grounding (Faithfulness)
    # Check if keywords in the LLM answer are present in the retrieved chunks
    # (Simplified text-containment proxy of hallucination)
    answer_words = set(re.findall(r"\b[a-zA-Z0-9]{4,}\b", answer.lower()))
    
    context_text = " ".join([r.get("content", "") for r in retrieved_results]).lower()
    context_words = set(re.findall(r"\b[a-zA-Z0-9]{4,}\b", context_text))
    
    matched_words = answer_words.intersection(context_words)
    faithfulness_score = (len(matched_words) / len(answer_words)) if answer_words else 1.0

    # Refusals should not count as hallucinations
    is_refusal = "data not available" in answer.lower() or "insufficient evidence" in answer.lower()
    if is_refusal:
        faithfulness_score = 1.0
        citation_accuracy = 1.0

    hallucination_rate = 1.0 - faithfulness_score

    # 3. Compute Retrieval Quality (MRR and NDCG proxies)
    # Based on cosine similarity of the top candidates
    scores = [r.get("score", 0.0) for r in retrieved_results]
    avg_retrieval_score = sum(scores) / len(scores) if scores else 0.0
    
    # MRR calculation
    # Check if the highest score is reasonably high
    mrr = 1.0 if (scores and scores[0] >= 0.7) else (0.5 if (scores and scores[0] >= 0.5) else 0.0)
    ndcg = avg_retrieval_score # Simplification for monitoring

    eval_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question[:100],
        "metrics": {
            "citation_accuracy": round(citation_accuracy, 4),
            "grounding_accuracy": round(faithfulness_score, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "mrr": round(mrr, 4),
            "ndcg": round(ndcg, 4),
            "average_retrieval_score": round(avg_retrieval_score, 4)
        }
    }

    # Save metrics asynchronously/best-effort
    save_evaluation_record(eval_record)
    return eval_record

import re

def save_evaluation_record(record: dict[str, Any]) -> None:
    """Save evaluation metrics to MongoDB (or local file fallback)."""
    # Create copy to prevent mutating the original record
    record_copy = dict(record)

    # 1. Try MongoDB
    try:
        from storage.mongo_store import _get_db
        db = _get_db()
        if db is not None:
            db["rag_evaluations"].insert_one(record_copy)
            return
    except Exception as exc:
        logger.debug("Failed to save evaluation to MongoDB: %s", exc)

    # 2. Local fallback
    try:
        # Convert ObjectId if it was added by MongoDB before failure
        if "_id" in record_copy:
            record_copy["_id"] = str(record_copy["_id"])

        os.makedirs(os.path.dirname(EVAL_FILE_PATH), exist_ok=True)
        records = []
        if os.path.exists(EVAL_FILE_PATH):
            with open(EVAL_FILE_PATH, "r", encoding="utf-8") as f:
                try:
                    records = json.load(f)
                    if not isinstance(records, list):
                        records = []
                except Exception:
                    records = []
                    
        records.append(record_copy)
        # Cap local log size to 200 records
        if len(records) > 200:
            records = records[-200:]
            
        with open(EVAL_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to save evaluation locally: %s", exc)

def get_summarized_evaluations() -> dict[str, Any]:
    """Calculate average metrics across all evaluated queries."""
    records = []
    
    # 1. Try loading from MongoDB
    try:
        from storage.mongo_store import _get_db
        db = _get_db()
        if db is not None:
            cursor = db["rag_evaluations"].find({}, {"_id": 0}).sort("timestamp", -1).limit(200)
            records = list(cursor)
    except Exception as exc:
        logger.debug("Failed to load evaluations from Mongo: %s", exc)

    # 2. Local fallback
    if not records and os.path.exists(EVAL_FILE_PATH):
        try:
            with open(EVAL_FILE_PATH, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []

    if not records:
        return {
            "total_evaluations": 0,
            "averages": {
                "citation_accuracy": 0.0,
                "grounding_accuracy": 0.0,
                "hallucination_rate": 0.0,
                "mrr": 0.0,
                "ndcg": 0.0,
                "average_retrieval_score": 0.0
            }
        }

    sum_metrics = {
        "citation_accuracy": 0.0,
        "grounding_accuracy": 0.0,
        "hallucination_rate": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
        "average_retrieval_score": 0.0
    }

    count = len(records)
    for r in records:
        m = r.get("metrics", {})
        for k in sum_metrics:
            sum_metrics[k] += m.get(k, 0.0)

    return {
        "total_evaluations": count,
        "averages": {k: round(v / count, 4) for k, v in sum_metrics.items()},
        "recent_runs": records[:10]
    }
