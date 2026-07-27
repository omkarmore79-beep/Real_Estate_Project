"""
Unit tests for Excavator RAG Scaffolding.
Run with: python -m unittest tests/test_excavator.py
"""

import unittest
import os
import sys

# Add backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from chatbot.query_router import classify_im_query
from fastapi.testclient import TestClient
from app import app


class TestExcavatorRAG(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    # ── 1. Query Router Intent & Extraction Tests ──────────────────────────────
    def test_query_router_diagnostic(self):
        # Query with fault code and starter keyword
        query = "Excavator cranks but wont start, showing fault code E-042 on cluster"
        result = classify_im_query(query)
        
        # Checking classification categories and extractions
        self.assertEqual(result["category"], "diagnostic")
        self.assertIn("E-042", result["dtc_codes"])
        self.assertIn("starter_motor", result["components"])

    def test_query_router_procedural(self):
        # Query asking for how-to instructions
        query = "How do I replace the swing motor oil seal?"
        result = classify_im_query(query)
        
        self.assertEqual(result["category"], "procedural")
        self.assertIn("swing_motor", result["components"])

    def test_query_router_informational(self):
        # General spec lookup
        query = "what is the hydraulic pump flow rate specification?"
        result = classify_im_query(query)
        
        self.assertEqual(result["category"], "informational")
        self.assertIn("hydraulic_pump", result["components"])

    # ── 2. FastAPI Endpoint Scaffolding Route Tests ────────────────────────────
    def test_excavator_chat_endpoint(self):
        payload = {
            "domain": "excavator",
            "message": "Engine is smoking under heavy load, fault code E102",
        }
        response = self.client.post("/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("routing", data)
        self.assertIn("category", data["routing"])
        self.assertEqual(data["routing"]["category"], "diagnostic")
        self.assertIn("[SCANNED ROUTE: DIAGNOSTIC]", data["answer"])
        self.assertIn("E102", data["routing"]["dtc_codes"])
        self.assertIn("engine", data["routing"]["components"])


if __name__ == "__main__":
    unittest.main()
