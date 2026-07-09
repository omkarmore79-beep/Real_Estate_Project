import sys
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import os
from urllib.parse import urlparse

# Load environment
BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")

url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

print("Qdrant URL configured:", bool(url))
print("Qdrant API Key configured:", bool(api_key))

if not url:
    raise SystemExit("QDRANT_URL missing in backend/.env")

# Extract host safely without secrets
try:
    parsed = urlparse(url)
    host_only = parsed.netloc or parsed.path
except Exception:
    host_only = "unknown"

print("Qdrant Cloud host:", host_only)

try:
    print("Connecting to Qdrant Cloud...")
    client = QdrantClient(url=url, api_key=api_key if api_key else None, timeout=60)
    collections = client.get_collections().collections
    print("Qdrant connection: OK")
    print("Available collections:", [c.name for c in collections])
except Exception as e:
    print("Qdrant connection FAILED:")
    print(type(e).__name__)
    print(str(e))
    sys.exit(1)
