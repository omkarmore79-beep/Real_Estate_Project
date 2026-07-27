import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

from services.reranker import rerank_sync, rerank_async, get_chunk_text_representation, get_rerank_metrics

class TestVoyageReranker(unittest.TestCase):

    def setUp(self):
        self.query = "What is the price of the 3 BHK?"
        self.chunks = [
            {
                "id": "c1",
                "source_type": "text",
                "content": "A 3 BHK luxury apartment starting at 1.5 Cr.",
                "score": 0.8,
                "metadata": {"project": "Marina", "document_id": "doc1"}
            },
            {
                "id": "c2",
                "source_type": "image",
                "content": "Image description fallback",
                "caption": "3 BHK Floor Plan",
                "score": 0.6,
                "metadata": {
                    "project": "Marina",
                    "document_id": "doc1",
                    "nearby_text": "Close to the clubhouse",
                    "ocr_context": "RERA: PRM/KA/RERA/1234"
                }
            }
        ]

    def test_chunk_text_representation(self):
        # Text chunk representation
        repr_text = get_chunk_text_representation(self.chunks[0])
        self.assertEqual(repr_text, "A 3 BHK luxury apartment starting at 1.5 Cr.")

        # Image chunk representation
        repr_image = get_chunk_text_representation(self.chunks[1])
        self.assertIn("Image Caption: 3 BHK Floor Plan", repr_image)
        self.assertIn("Nearby Paragraphs: Close to the clubhouse", repr_image)
        self.assertIn("OCR Labels inside diagram: RERA: PRM/KA/RERA/1234", repr_image)

    @patch("voyageai.Client")
    def test_rerank_sync_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        # Mock Voyage API response
        mock_result1 = MagicMock(index=1, relevance_score=0.95)
        mock_result2 = MagicMock(index=0, relevance_score=0.72)
        mock_response = MagicMock(results=[mock_result1, mock_result2])
        mock_client.rerank.return_value = mock_response

        # Execute
        res = rerank_sync(self.query, self.chunks, top_k=2)

        # Assertions
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["id"], "c2")  # Index 1 was ranked top
        self.assertEqual(res[0]["rerank_score"], 0.95)
        self.assertEqual(res[1]["id"], "c1")
        self.assertEqual(res[1]["rerank_score"], 0.72)

    @patch("voyageai.Client")
    def test_rerank_sync_api_failure_fallback(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.rerank.side_effect = Exception("API Key Invalid")

        # Should fall back to vector scores and not crash
        res = rerank_sync(self.query, self.chunks, top_k=2)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["id"], "c1")  # Kept original order
        self.assertEqual(res[0]["rerank_score"], 0.8)

    @patch("voyageai.Client")
    def test_rerank_sync_empty_response(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock(results=[])
        mock_client.rerank.return_value = mock_response

        res = rerank_sync(self.query, self.chunks, top_k=2)
        self.assertEqual(len(res), 2)  # Returns all inputs with fallback scores
        self.assertEqual(res[0]["rerank_score"], 0.0)

    @patch("voyageai.AsyncClient")
    def test_rerank_async_success(self, mock_async_client_cls):
        mock_client = MagicMock()
        mock_async_client_cls.return_value = mock_client
        
        mock_result1 = MagicMock(index=0, relevance_score=0.9)
        mock_result2 = MagicMock(index=1, relevance_score=0.8)
        mock_response = MagicMock(results=[mock_result1, mock_result2])
        
        # AsyncMock for async method rerank
        mock_client.rerank = AsyncMock(return_value=mock_response)

        # Run async function
        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(rerank_async(self.query, self.chunks, top_k=2))

        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["id"], "c1")
        self.assertEqual(res[0]["rerank_score"], 0.9)

    @patch("voyageai.AsyncClient")
    def test_rerank_async_timeout(self, mock_async_client_cls):
        mock_client = MagicMock()
        mock_async_client_cls.return_value = mock_client
        
        async def slow_rerank(*args, **kwargs):
            await asyncio.sleep(5)
            return MagicMock()
            
        mock_client.rerank = AsyncMock(side_effect=slow_rerank)

        # Patch timeout to be small
        with patch("services.reranker.RERANK_TIMEOUT", 0.1):
            loop = asyncio.get_event_loop()
            res = loop.run_until_complete(rerank_async(self.query, self.chunks, top_k=2))
            
            # Should fallback gracefully
            self.assertEqual(len(res), 2)
            self.assertEqual(res[0]["rerank_score"], 0.8)

    @patch("voyageai.Client")
    def test_rerank_sync_retry_logic(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        # Fail twice, succeed on third attempt
        mock_result = MagicMock(index=0, relevance_score=0.99)
        mock_response = MagicMock(results=[mock_result])
        
        mock_client.rerank.side_effect = [
            Exception("Rate Limit Exceeded (429)"),
            Exception("Timeout"),
            mock_response
        ]

        with patch("time.sleep", return_value=None):
            res = rerank_sync(self.query, [self.chunks[0]], top_k=1)
            self.assertEqual(res[0]["rerank_score"], 0.99)
            self.assertEqual(mock_client.rerank.call_count, 3)

    def test_metrics_collection(self):
        metrics = get_rerank_metrics()
        self.assertIn("total_requests", metrics)
        self.assertIn("average_latency", metrics)
        self.assertIn("total_failures", metrics)
