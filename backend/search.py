from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from retrieval.embeddings import get_image_embedder, get_text_embedder
from retrieval.qdrant_service import search_images, search_text_dense, search_text_keyword
from services.reranker import rerank_sync

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logger = logging.getLogger("search")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: dict[str, Any] | None = None
    max_price: float | None = None
    min_price: float | None = None
    city: str | None = None
    bedrooms: int | None = None
    include_images: bool = True


def search_listing(query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Retrieve up to 50 Qdrant candidates, then return Voyage-reranked top N."""
    text_embedder = get_text_embedder()
    query_vector = text_embedder.embed(query, input_type="query")

    dense_results = search_text_dense(query_vector, filters=filters, top_k=50)
    keyword_results = search_text_keyword(query, filters=filters, top_k=50)

    merged = dense_results + keyword_results
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in merged:
        item_id = str(item.get("id", ""))
        if item_id in seen:
            continue
        seen.add(item_id)
        deduped.append(item)

    # Image vectors use the same Voyage dimensionality.  Include them in the
    # candidate set so the reranker can compare visual captions/context with
    # text passages; payloads carry the image URL/path and structured metadata.
    try:
        image_query_vector = get_image_embedder().embed_text(query, input_type="query")
        image_results = search_images(image_query_vector, filters=filters, top_k=10)
    except Exception as exc:
        logger.warning("Image candidate search unavailable: %s", exc)
        image_results = []
    for item in image_results:
        item["source_type"] = "image"
        item["metadata"] = item.get("payload", {})
        item["content"] = item.get("content") or item["metadata"].get("caption", "")
    candidates = deduped[:50] + image_results
    reranked = rerank_sync(query, candidates, top_k=top_k)
    return reranked[:top_k]


if __name__ == "__main__":
    result = search_listing("luxury apartment with clubhouse and pool", top_k=5)
    for item in result:
        logger.info("result score=%s content=%s", item.get("rerank_score"), item.get("content", "")[:160])
