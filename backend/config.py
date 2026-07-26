"""
Backend configuration.

Loads `.env` from the backend directory (where the file actually lives).
Falls back to the project root for backward compatibility.

Qdrant validation rules:
  - QDRANT_URL is REQUIRED — no localhost fallback.
  - If the URL is a Qdrant Cloud URL (*.cloud.qdrant.io), QDRANT_API_KEY is also REQUIRED.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
_env_source = str(BASE_DIR / ".env")


# ── Groq / LLM ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview")

# ── MongoDB ────────────────────────────────────────────────────────────────────
MONGODB_URI = os.getenv("MONGODB_URI")
# Strip surrounding quotes that python-dotenv sometimes preserves
MONGODB_URI = (MONGODB_URI or "").strip().strip('"').strip("'")
MONGODB_DB = os.getenv("MONGODB_DB", "real_estate_chatbot")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "projects")
_MONGODB_URI = MONGODB_URI

ALLOW_UPLOAD_WITHOUT_MONGODB = os.getenv("ALLOW_UPLOAD_WITHOUT_MONGODB", "false").lower() == "true"

# ── Qdrant ─────────────────────────────────────────────────────────────────────
# No default — QDRANT_URL must be set explicitly.
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION_TEXT = os.getenv("QDRANT_COLLECTION_TEXT", "real_estate_text_chunks")
QDRANT_COLLECTION_IMAGES = os.getenv("QDRANT_COLLECTION_IMAGES", "real_estate_image_chunks")

# Extract Qdrant host safely
from urllib.parse import urlparse
try:
    _parsed_qdrant = urlparse(QDRANT_URL)
    _qdrant_host = _parsed_qdrant.netloc or _parsed_qdrant.path
except Exception:
    _qdrant_host = "unknown"

# ── Qdrant validation ─────────────────────────────────────────────────────────
if not QDRANT_URL:
    logger.error("QDRANT_URL is not set (dotenv source: %s).", _env_source)
    raise EnvironmentError(
        "QDRANT_URL is required. Set it in your .env file."
    )

_is_cloud = "cloud.qdrant.io" in QDRANT_URL
if _is_cloud and not QDRANT_API_KEY:
    logger.error("QDRANT_API_KEY is required for Qdrant Cloud (%s).", _qdrant_host)
    raise EnvironmentError(
        "QDRANT_API_KEY is required when connecting to Qdrant Cloud."
    )

# ── Voyage AI ─────────────────────────────────────────────────────────────────
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

# ── Embedding & Reranker Models ────────────────────────────────────────────────
TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "voyage-3")
IMAGE_EMBEDDING_MODEL = os.getenv("IMAGE_EMBEDDING_MODEL", "voyage-multimodal-3.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
VOYAGE_RERANK_MODEL = os.getenv("VOYAGE_RERANK_MODEL", "rerank-2")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "50"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "10"))
RERANK_TIMEOUT = int(os.getenv("RERANK_TIMEOUT", "30"))
VOYAGE_TPM_LIMIT = int(os.getenv("VOYAGE_TPM_LIMIT", "20000"))

# Vector dimensions
TEXT_VECTOR_DIM = int(os.getenv("TEXT_VECTOR_DIM", "1024"))   # Voyage dense dim
IMAGE_VECTOR_DIM = int(os.getenv("IMAGE_VECTOR_DIM", "1024"))  # Voyage multimodal dim

# ── OCR ────────────────────────────────────────────────────────────────────────
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"
OCR_ENGINE = os.getenv("OCR_ENGINE", "paddle")
OCR_MIN_TEXT_LENGTH = int(os.getenv("OCR_MIN_TEXT_LENGTH", "80"))
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "0.60"))
OCR_ON_CROPPED_IMAGES = os.getenv("OCR_ON_CROPPED_IMAGES", "false").lower() == "true"

# ── Storage Paths ──────────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(BASE_DIR, "..", "uploads")
DATA_FOLDER = os.path.join(BASE_DIR, "storage", "data")
RAW_FOLDER = os.path.join(BASE_DIR, "storage", "raw_text")
IMAGE_FOLDER = os.path.join(BASE_DIR, "storage", "images")

# ── Startup diagnostics (never print secrets) ─────────────────────────────────
logger.info("Configuration loaded from %s", _env_source)
logger.info("MongoDB configured=%s db=%s collection=%s", bool(MONGODB_URI), MONGODB_DB, MONGODB_COLLECTION)
logger.info("Qdrant configured=%s host=%s", bool(QDRANT_URL), _qdrant_host)
logger.info("OCR enabled=%s; upload_without_mongodb=%s", OCR_ENABLED, ALLOW_UPLOAD_WITHOUT_MONGODB)
logger.info("Models: llm=%s text=%s image=%s reranker=%s", LLM_MODEL, TEXT_EMBEDDING_MODEL, IMAGE_EMBEDDING_MODEL, VOYAGE_RERANK_MODEL)


