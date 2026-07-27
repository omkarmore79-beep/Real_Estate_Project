import sys
import os
import time
import math
from typing import List, Dict

backend_path = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.append(backend_path)

from dotenv import load_dotenv
dotenv_path = os.path.join(backend_path, ".env")
load_dotenv(dotenv_path=dotenv_path)

from retrieval.hybrid_retriever import retrieve
from services.reranker import rerank_sync

# Evaluated test cases (query, expected terms in content)
BENCHMARK_CASES = [
    {
        "query": "What are the safety warnings?",
        "gold_terms": ["heating", "battery", "exhaust", "safety", "hazard"]
    },
    {
        "query": "Before performing any service work on the excavator, where should you attach the 'Do Not Operate' tag?",
        "gold_terms": ["controls", "cab", "steering", "tag", "operate"]
    },
    {
        "query": "What distance must be maintained between any part of the machine or load and an electric power line?",
        "gold_terms": ["distance", "voltage", "power line", "limit", "clearance"]
    },
    {
        "query": "What is the standard track shoe width, standard bucket capacity (SAE heaped), and operating weight for the R215L excavator model?",
        "gold_terms": ["shoe", "capacity", "weight", "operating", "standard"]
    }
]

def calculate_metrics(results: List[Dict], gold_terms: List[str]) -> Dict:
    """Calculate Recall, Precision, MRR, and NDCG based on relevance to gold terms."""
    retrieved_count = len(results)
    if retrieved_count == 0:
        return {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}

    relevance_scores = []
    for r in results:
        content = r.get("content", "").lower()
        score = sum(1.0 for term in gold_terms if term in content)
        relevance_scores.append(score)

    # Binarized relevance (relevance > 0.0)
    hits = [1 if s > 0 else 0 for s in relevance_scores]
    
    # 1. Precision & Recall
    precision = sum(hits) / retrieved_count
    # Assume max possible hits is the number of gold terms or 3
    recall = sum(hits) / min(retrieved_count, 3)

    # 2. MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for idx, hit in enumerate(hits):
        if hit == 1:
            mrr = 1.0 / (idx + 1)
            break

    # 3. NDCG (Normalized Discounted Cumulative Gain)
    dcg = 0.0
    for idx, rel in enumerate(relevance_scores):
        dcg += rel / math.log2(idx + 2)

    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = 0.0
    for idx, rel in enumerate(ideal_relevance):
        idcg += rel / math.log2(idx + 2)

    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "ndcg": ndcg
    }

def run_benchmarks():
    print("====================================================")
    print("      HYBRID RETRIEVAL BENCHMARK: VOYAGE RERANKER   ")
    print("====================================================\n")

    vector_metrics_agg = {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "latency": 0.0}
    voyage_metrics_agg = {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "latency": 0.0}
    count = len(BENCHMARK_CASES)

    print(f"Running {count} benchmark queries...\n")

    for case in BENCHMARK_CASES:
        query = case["query"]
        gold_terms = case["gold_terms"]
        print(f"Query: \"{query}\"")

        # 1. Measure Vector / RRF Retrieval (Top 50)
        # Note: retrieve internally calls caching, we bypass caching using a dynamic query suffix if needed
        # We can run retrieve directly.
        start = time.perf_counter()
        
        # We fetch results. In retrieve function, the Voyage reranking is now integrated.
        # To measure WITHOUT reranking, we retrieve candidates and sort them by original score.
        raw_results = retrieve(query, top_k=50)
        vector_only_results = sorted(raw_results, key=lambda x: x.get("score", 0.0), reverse=True)[:10]
        
        latency_vector = time.perf_counter() - start
        vector_metrics = calculate_metrics(vector_only_results, gold_terms)
        
        vector_metrics_agg["precision"] += vector_metrics["precision"]
        vector_metrics_agg["recall"] += vector_metrics["recall"]
        vector_metrics_agg["mrr"] += vector_metrics["mrr"]
        vector_metrics_agg["ndcg"] += vector_metrics["ndcg"]
        vector_metrics_agg["latency"] += latency_vector

        # 2. Measure Voyage Hosted Reranker (Top 10 from Top 50 candidates)
        start = time.perf_counter()
        
        voyage_results = retrieve(query, top_k=10)
        
        latency_voyage = time.perf_counter() - start
        voyage_metrics = calculate_metrics(voyage_results, gold_terms)
        
        voyage_metrics_agg["precision"] += voyage_metrics["precision"]
        voyage_metrics_agg["recall"] += voyage_metrics["recall"]
        voyage_metrics_agg["mrr"] += voyage_metrics["mrr"]
        voyage_metrics_agg["ndcg"] += voyage_metrics["ndcg"]
        voyage_metrics_agg["latency"] += latency_voyage

        print(f"  -> Vector Only : Recall={vector_metrics['recall']:.2f} | MRR={vector_metrics['mrr']:.2f} | Latency={latency_vector:.3f}s")
        print(f"  -> Voyage Rerank: Recall={voyage_metrics['recall']:.2f} | MRR={voyage_metrics['mrr']:.2f} | Latency={latency_voyage:.3f}s\n")

    # Average metrics
    for k in vector_metrics_agg:
        vector_metrics_agg[k] /= count
        voyage_metrics_agg[k] /= count

    # Print Comparison Report
    print("====================================================")
    print("             FINAL COMPARISON REPORT                ")
    print("====================================================")
    print(f"| Metric           | Vector / RRF Only | Voyage Hosted Rerank | Improvement |")
    print(f"|------------------|-------------------|----------------------|-------------|")
    print(f"| Precision@10     | {vector_metrics_agg['precision']:.4f}            | {voyage_metrics_agg['precision']:.4f}               | {((voyage_metrics_agg['precision'] - vector_metrics_agg['precision']) / (vector_metrics_agg['precision'] + 1e-6) * 100):+.1f}% |")
    print(f"| Recall@10        | {vector_metrics_agg['recall']:.4f}            | {voyage_metrics_agg['recall']:.4f}               | {((voyage_metrics_agg['recall'] - vector_metrics_agg['recall']) / (vector_metrics_agg['recall'] + 1e-6) * 100):+.1f}% |")
    print(f"| MRR              | {vector_metrics_agg['mrr']:.4f}            | {voyage_metrics_agg['mrr']:.4f}               | {((voyage_metrics_agg['mrr'] - vector_metrics_agg['mrr']) / (vector_metrics_agg['mrr'] + 1e-6) * 100):+.1f}% |")
    print(f"| NDCG             | {vector_metrics_agg['ndcg']:.4f}            | {voyage_metrics_agg['ndcg']:.4f}               | {((voyage_metrics_agg['ndcg'] - vector_metrics_agg['ndcg']) / (vector_metrics_agg['ndcg'] + 1e-6) * 100):+.1f}% |")
    print(f"| Latency (avg)    | {vector_metrics_agg['latency']:.3f}s           | {voyage_metrics_agg['latency']:.3f}s              | {((voyage_metrics_agg['latency'] - vector_metrics_agg['latency']) / (vector_metrics_agg['latency'] + 1e-6) * 100):+.1f}% |")
    print("====================================================\n")

if __name__ == "__main__":
    run_benchmarks()
