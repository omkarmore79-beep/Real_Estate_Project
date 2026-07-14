"""
Unit and Integration Tests for Voyage AI Multimodal 3.5 Upgrade & Conversational Memory.
Run with: python -m unittest tests/test_voyage_multimodal.py
"""

import unittest
import os
import sys
from unittest.mock import MagicMock, patch

# Add backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from retrieval.embeddings import TextEmbedder, ImageEmbedder
from utils.memory import memory_manager, SessionMemory


class TestVoyageAndMemory(unittest.TestCase):

    # ── 1. Conversational Memory Tests ───────────────────────────────────────
    def test_session_memory_history(self):
        session = SessionMemory(max_history_pairs=2)
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")
        
        history = session.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["content"], "Hello")
        self.assertEqual(history[1]["content"], "Hi there!")

    def test_session_memory_eviction(self):
        session = SessionMemory(max_history_pairs=2)
        # Add 3 pairs (6 messages)
        for i in range(3):
            session.add_message("user", f"Question {i}")
            session.add_message("assistant", f"Answer {i}")
            
        history = session.get_history()
        # Max pairs is 2, so it should retain the last 4 messages (last 2 pairs)
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]["content"], "Question 1")
        self.assertEqual(history[-1]["content"], "Answer 2")

    def test_memory_manager_singleton(self):
        s1 = memory_manager.get_session("session_abc")
        s2 = memory_manager.get_session("session_abc")
        self.assertIs(s1, s2)
        
        s3 = memory_manager.get_session("session_xyz")
        self.assertIsNot(s1, s3)

    # ── 2. Voyage AI Embedding Call Mocks ──────────────────────────────────────
    @patch("retrieval.embeddings.get_voyage_client")
    def test_text_embedder_calls_voyage(self, mock_get_client):
        # Setup mock client
        mock_client = MagicMock()
        mock_res = MagicMock()
        mock_res.embeddings = [[0.1] * 1024]
        mock_client.multimodal_embed.return_value = mock_res
        mock_get_client.return_value = mock_client

        embedder = TextEmbedder()
        # Clear cache first to force live call
        with patch("utils.cache.get_json_cache", return_value=None):
            vec = embedder.embed("Test query", input_type="query")

        self.assertEqual(len(vec), 1024)
        mock_client.multimodal_embed.assert_called_once_with(
            inputs=[["Test query"]],
            model="voyage-multimodal-3.5",
            input_type="query"
        )

    @patch("retrieval.embeddings.get_voyage_client")
    def test_image_embedder_interleaved_calls_voyage(self, mock_get_client):
        mock_client = MagicMock()
        mock_res = MagicMock()
        mock_res.embeddings = [[0.2] * 1024]
        mock_client.multimodal_embed.return_value = mock_res
        mock_get_client.return_value = mock_client

        # Create a temp dummy image file
        import tempfile
        from PIL import Image
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = Image.new("RGB", (100, 100), color="red")
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            embedder = ImageEmbedder()
            with patch("utils.cache.get_json_cache", return_value=None):
                vec = embedder.embed_interleaved("Caption text", tmp_path, input_type="document")

            self.assertEqual(len(vec), 1024)
            # Ensure multimodal_embed was called with interleaved string and PIL Image
            call_args = mock_client.multimodal_embed.call_args[1]
            self.assertEqual(call_args["model"], "voyage-multimodal-3.5")
            self.assertEqual(call_args["input_type"], "document")
            
            inputs = call_args["inputs"]
            self.assertEqual(len(inputs), 1)
            self.assertEqual(inputs[0][0], "Caption text")
            self.assertIsInstance(inputs[0][1], Image.Image)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
