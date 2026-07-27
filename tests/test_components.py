"""
Unit tests for upgraded RAG system components.
Run with: python -m unittest tests/test_components.py
"""

import unittest
import os
import sys
import tempfile
import json
import shutil
import csv

# Add backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from processing.document_processor import validate_and_hash_file
from utils.cache import set_cache, get_cache, set_json_cache, get_json_cache, clear_cache, get_cache_metrics
from ingestion.multiformat_parser import parse_document
from ingestion.chunker import split_into_sentences, semantic_chunk_text
from retrieval.query_analyzer import classify_query_intent, expand_query
from utils.evaluation import evaluate_rag_response, get_summarized_evaluations


class TestRAGComponents(unittest.TestCase):

    def setUp(self):
        clear_cache()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    # ── 1. Hashing & Security Validation ──────────────────────────────────────
    def test_file_validation_and_hashing(self):
        # Test normal file hashing
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Hello World, Real Estate Hybrid RAG")
            
        file_hash = validate_and_hash_file(test_file)
        self.assertTrue(len(file_hash) > 0)
        
        # Test executable rejection
        exec_file = os.path.join(self.temp_dir, "malicious.exe")
        with open(exec_file, "wb") as f:
            f.write(b"MZ\x90\x00\x03\x00\x00\x00")
            
        with self.assertRaises(ValueError) as context:
            validate_and_hash_file(exec_file)
        self.assertIn("Executable files are not permitted", str(context.exception))

    # ── 2. Caching ────────────────────────────────────────────────────────────
    def test_caching_layer(self):
        # String cache
        set_cache("key1", "value1")
        self.assertEqual(get_cache("key1"), "value1")
        
        # JSON cache
        data = {"price": 1000000, "builder": "Hiranandani"}
        set_json_cache("key_json", data)
        cached_data = get_json_cache("key_json")
        self.assertEqual(cached_data["price"], 1000000)
        self.assertEqual(cached_data["builder"], "Hiranandani")
        
        # Metrics
        metrics = get_cache_metrics()
        self.assertTrue("hit_rate" in metrics)
        self.assertEqual(metrics["hits"], 2)

    # ── 3. Multiformat Parser ──────────────────────────────────────────────────
    def test_multiformat_parser_txt(self):
        test_file = os.path.join(self.temp_dir, "test.txt")
        content = "Golden Willows by Hiranandani is located at Panvel."
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(content)
            
        res = parse_document(test_file, "doc_1", "test.txt")
        self.assertEqual(res["total_pages"], 1)
        self.assertIn("Golden Willows", res["full_text"])

    def test_multiformat_parser_csv(self):
        test_file = os.path.join(self.temp_dir, "test.csv")
        with open(test_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Project", "Builder", "Price"])
            writer.writerow(["Golden Willows", "Hiranandani", "1.2 Cr"])
            
        res = parse_document(test_file, "doc_2", "test.csv")
        self.assertEqual(res["total_pages"], 1)
        self.assertIn("| Project | Builder | Price |", res["full_text"])
        self.assertIn("Golden Willows", res["full_text"])

    # ── 4. Semantic Chunking ──────────────────────────────────────────────────
    def test_sentence_splitting(self):
        text = "Golden Willows Panvel is nice. The possession starts in 2026! What is the price?"
        sentences = split_into_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "Golden Willows Panvel is nice.")
        self.assertEqual(sentences[1], "The possession starts in 2026!")

    # ── 5. Query Understanding ────────────────────────────────────────────────
    def test_query_intent_classification(self):
        q1 = "what is the price of 3 bhk"
        intents = classify_query_intent(q1)
        self.assertIn("Pricing", intents)
        self.assertIn("Floor Plan", intents)
        
        q2 = "show the clubhouse and swimming pool"
        intents2 = classify_query_intent(q2)
        self.assertIn("Amenities", intents2)

    # ── 6. Evaluation Framework ───────────────────────────────────────────────
    def test_rag_evaluation(self):
        question = "Where is Golden Willows located?"
        answer = "Golden Willows is located in Panvel."
        retrieved_results = [
            {"document_id": "doc_123", "content": "Golden Willows is located in Panvel, Navi Mumbai."}
        ]
        citations = [
            {"document_id": "doc_123", "page_number": 2, "source_file": "brochure.pdf"}
        ]
        
        eval_res = evaluate_rag_response(question, answer, retrieved_results, citations)
        self.assertIn("metrics", eval_res)
        metrics = eval_res["metrics"]
        self.assertEqual(metrics["citation_accuracy"], 1.0)
        self.assertTrue(metrics["grounding_accuracy"] > 0.5)
        
        summary = get_summarized_evaluations()
        self.assertTrue(summary["total_evaluations"] > 0)


if __name__ == "__main__":
    unittest.main()
