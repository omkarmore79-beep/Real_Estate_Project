"""
Collection creation script for Hyundai R215L Excavator RAG.
Sets up 6 Qdrant collections with indexes on:
  - document_id
  - machine_model
  - dtc_codes
  - component_tags
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))

from config import (
    IMAGE_VECTOR_DIM,
    QDRANT_API_KEY,
    QDRANT_URL,
    TEXT_VECTOR_DIM,
)
from retrieval.qdrant_client import client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# List of text collections (1024 dimensions)
TEXT_COLLECTIONS = [
    "im_manuals_text",
    "im_service_bulletins",
    "im_maintenance_logs",
    "im_parts_catalog",
    "im_field_reports",
]

# Image collections (512 dimensions)
IMAGE_COLLECTIONS = [
    "im_manuals_images",
]

def create_im_collections():
    from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
    
    logger.info("Initializing Hyundai R215L Qdrant Collections setup...")
    logger.info("Connecting to Qdrant URL: %s", QDRANT_URL)

    existing_collections = client.get_collections().collections
    existing = {c.name for c in existing_collections}

    def check_and_create_collection(col_name: str, target_dim: int):
        recreate = False
        if col_name in existing:
            try:
                col_info = client.get_collection(col_name)
                # Inspect vectors config
                params = col_info.config.params.vectors
                current_size = -1
                if hasattr(params, "size"):
                    current_size = params.size
                elif isinstance(params, dict) and "size" in params:
                    current_size = params["size"]
                
                if current_size != target_dim:
                    logger.warning(
                        "Collection %s exists with dimension %s but target is %d. Recreating...",
                        col_name, str(current_size), target_dim
                    )
                    client.delete_collection(col_name)
                    recreate = True
            except Exception as e:
                logger.warning("Failed to check collection config for %s: %s. Recreating...", col_name, e)
                try:
                    client.delete_collection(col_name)
                except Exception:
                    pass
                recreate = True

        if col_name not in existing or recreate:
            client.create_collection(
                collection_name=col_name,
                vectors_config=VectorParams(size=target_dim, distance=Distance.COSINE),
            )
            logger.info("Created collection: %s (dim=%d)", col_name, target_dim)
            create_payload_indexes(col_name)
        else:
            logger.info("Collection %s already exists with correct dimensions.", col_name)

    # ── 1. Setup Text Collections ─────────────────────────────────────────────
    for col in TEXT_COLLECTIONS:
        check_and_create_collection(col, TEXT_VECTOR_DIM)

    # ── 2. Setup Image Collections ────────────────────────────────────────────
    for col in IMAGE_COLLECTIONS:
        check_and_create_collection(col, IMAGE_VECTOR_DIM)

    logger.info("Excavator Qdrant Collections setup completed successfully.")

def create_payload_indexes(collection_name: str):
    from qdrant_client.models import PayloadSchemaType
    
    fields_to_index = [
        ("document_id", PayloadSchemaType.KEYWORD),
        ("machine_model", PayloadSchemaType.KEYWORD),
        ("dtc_codes", PayloadSchemaType.KEYWORD),
        ("component_tags", PayloadSchemaType.KEYWORD),
        ("doc_type", PayloadSchemaType.KEYWORD),
    ]
    
    for field, schema_type in fields_to_index:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=schema_type,
            )
            logger.debug("Created index on field '%s' in collection '%s'", field, collection_name)
        except Exception as exc:
            # Index might already exist, log warning but don't crash
            logger.warning("Could not create payload index on %s for %s: %s", field, collection_name, exc)

if __name__ == "__main__":
    create_im_collections()
