import os
from qdrant_client import QdrantClient

# Centralized Qdrant Cloud Client instance
# Loaded from QDRANT_URL and QDRANT_API_KEY
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY") if os.getenv("QDRANT_API_KEY") else None,
    timeout=60
)
