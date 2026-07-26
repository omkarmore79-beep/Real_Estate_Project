"""
Qdrant Cloud client singletons.

Prefer gRPC for throughput-critical bulk writes and keep the synchronous client
available for existing search/read operations.
"""
import os
from qdrant_client import AsyncQdrantClient, QdrantClient

_api_key = os.getenv("QDRANT_API_KEY") or None
_qdrant_url = os.getenv("QDRANT_URL")

client = QdrantClient(
    url=_qdrant_url,
    api_key=_api_key,
    prefer_grpc=True,
    timeout=120,
)

async_client = AsyncQdrantClient(
    url=_qdrant_url,
    api_key=_api_key,
    prefer_grpc=True,
    timeout=120,
)
