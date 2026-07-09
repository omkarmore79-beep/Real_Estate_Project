"""
Document Status Store — tracks RAG processing lifecycle per document.

Statuses (in order):
  uploaded → extracting_text → extracting_images → chunking →
  embedding_text → embedding_images → indexing_qdrant → ready | failed

Stored in MongoDB `document_status` collection (reuses the shared MongoClient
from mongo_store to avoid opening a new connection on every status write).
Falls back to an in-memory dict if MongoDB is unavailable.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── In-memory store (primary for speed; MongoDB is a durable mirror) ──────────
_memory_store: dict[str, dict] = {}

import json
import os

STATUS_FILE_PATH = os.path.join(os.path.dirname(__file__), "local_status.json")

def _save_status_locally():
    try:
        # Create directory if missing
        os.makedirs(os.path.dirname(STATUS_FILE_PATH), exist_ok=True)
        with open(STATUS_FILE_PATH, "w") as f:
            json.dump(_memory_store, f, indent=2)
    except Exception as exc:
        logger.debug("Failed to save status locally: %s", exc)

def _load_status_locally():
    global _memory_store
    if os.path.exists(STATUS_FILE_PATH):
        try:
            with open(STATUS_FILE_PATH, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Only override or merge keys
                    for k, v in data.items():
                        _memory_store[k] = v
        except Exception as exc:
            logger.debug("Failed to load local status: %s", exc)

# Pre-load local status
_load_status_locally()

# ── Status definitions ─────────────────────────────────────────────────────────
STATUSES = [
    "uploaded",
    "extracting_text",
    "extracting_images",
    "running_ocr",
    "chunking",
    "embedding_text",
    "embedding_images",
    "indexing_qdrant",
    "ready",
    "failed",
]

_STATUS_PROGRESS: dict[str, int] = {
    "uploaded": 5,
    "extracting_text": 15,
    "extracting_images": 25,
    "running_ocr": 35,
    "chunking": 45,
    "embedding_text": 60,
    "embedding_images": 75,
    "indexing_qdrant": 90,
    "ready": 100,
    "failed": 0,
}

_STATUS_MESSAGES: dict[str, str] = {
    "uploaded": "Document received. Starting processing.",
    "extracting_text": "Extracting text from PDF pages…",
    "extracting_images": "Extracting images from PDF pages…",
    "running_ocr": "Running OCR on page scans…",
    "chunking": "Splitting text into searchable chunks…",
    "embedding_text": "Generating text embeddings (bge-m3)…",
    "embedding_images": "Generating image embeddings (jina-clip-v2)…",
    "indexing_qdrant": "Indexing vectors into Qdrant Cloud…",
    "ready": "Document is ready for chat.",
    "failed": "Processing failed. Please try re-uploading.",
}



# ── MongoDB helper — reuses shared client from mongo_store ────────────────────

def _get_status_collection():
    """Return the MongoDB 'document_status' collection, or None on failure."""
    try:
        # Reuse the shared client already established by mongo_store
        from storage.mongo_store import _get_db
        db = _get_db()
        if db is None:
            return None
        return db["document_status"]
    except Exception as exc:
        logger.debug("Could not get status collection: %s", exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def set_status(
    document_id: str,
    status: str,
    *,
    message: str | None = None,
    text_chunks_indexed: int = 0,
    images_indexed: int = 0,
    total_pages: int = 0,
    error: str | None = None,
) -> None:
    """Persist processing status for a document (memory + JSON backup + MongoDB)."""
    now = datetime.now(timezone.utc).isoformat()

    # Preserve accumulated counts from previous calls
    prev = _memory_store.get(document_id, {})
    record: dict = {
        "document_id": document_id,
        "status": status,
        "progress": _STATUS_PROGRESS.get(status, 0),
        "message": message or _STATUS_MESSAGES.get(status, status),
        "text_chunks_indexed": text_chunks_indexed or prev.get("text_chunks_indexed", 0),
        "images_indexed": images_indexed or prev.get("images_indexed", 0),
        "total_pages": total_pages or prev.get("total_pages", 0),
        "updated_at": now,
        "created_at": prev.get("created_at", now),
        "error": error or "",
    }

    # Always update memory first (fast path for polling)
    _memory_store[document_id] = record
    _save_status_locally()
    
    logger.info("[STATUS] %s → %s (%d%%)", document_id[:8], status, record["progress"])

    # Persist to MongoDB asynchronously (best-effort)
    try:
        col = _get_status_collection()
        if col is not None:
            col.update_one(
                {"document_id": document_id},
                {"$set": record, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )
    except Exception as exc:
        logger.debug("MongoDB status write failed (non-fatal): %s", exc)


def get_status(document_id: str) -> dict | None:
    """Retrieve the current processing status for a document."""
    # Memory cache is always up-to-date for the current process
    if document_id in _memory_store:
        return _memory_store[document_id]

    # Try loading from local file
    _load_status_locally()
    if document_id in _memory_store:
        return _memory_store[document_id]

    # If not in memory, try MongoDB
    try:
        col = _get_status_collection()
        if col is not None:
            doc = col.find_one({"document_id": document_id}, {"_id": 0})
            if doc:
                _memory_store[document_id] = dict(doc)
                _save_status_locally()
                return _memory_store[document_id]
    except Exception as exc:
        logger.debug("MongoDB status read failed: %s", exc)

    return None


def is_ready(document_id: str) -> bool:
    """Return True if the document has been fully indexed."""
    s = get_status(document_id)
    return s is not None and s.get("status") == "ready"


def is_processing(document_id: str) -> bool:
    """Return True if the document is still being indexed."""
    s = get_status(document_id)
    if s is None:
        return False
    return s.get("status") not in ("ready", "failed")


def recover_interrupted_tasks() -> None:
    """
    Recover document tasks that were left in processing states (uploaded, extracting, etc.)
    and mark them as failed/interrupted due to server reload/restart.
    """
    _load_status_locally()
    updated = False
    for doc_id, record in list(_memory_store.items()):
        current_status = record.get("status")
        if current_status not in ("ready", "failed"):
            record["status"] = "failed"
            record["progress"] = 0
            record["message"] = "Processing was interrupted by server restart. Please reindex."
            record["error"] = "Server restart interrupted background task."
            record["updated_at"] = datetime.now(timezone.utc).isoformat()
            updated = True
            logger.info("Recovered and marked document %s as failed (interrupted by reload).", doc_id)
            
            # Mirror to MongoDB if available
            try:
                col = _get_status_collection()
                if col is not None:
                    col.update_one(
                        {"document_id": doc_id},
                        {"$set": record}
                    )
            except Exception:
                pass
                
    if updated:
        _save_status_locally()

