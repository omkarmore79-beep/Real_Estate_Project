"""
Embedding Service — Voyage AI Multimodal 3.5 Integration with local caching.

Uses voyage-multimodal-3.5:
  - Text-to-Vector (1024 dimensions)
  - Image-to-Vector (1024 dimensions)
  - Interleaved Text + Image-to-Vector (1024 dimensions)

All embeddings are cached to Redis/local memory.
"""

from __future__ import annotations

import logging
import os
import hashlib
import time
from typing import Any
from PIL import Image

from config import VOYAGE_API_KEY, TEXT_VECTOR_DIM, IMAGE_VECTOR_DIM

logger = logging.getLogger(__name__)

# ── Retry Helpers ─────────────────────────────────────────────────────────────

def _multimodal_embed_with_retry(client, inputs, model, input_type, max_retries=5, initial_backoff=2.0):
    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            res = client.multimodal_embed(
                inputs=inputs,
                model=model,
                input_type=input_type
            )
            return res
        except Exception as exc:
            err_msg = str(exc)
            is_rate_limit = "429" in err_msg or "rate limit" in err_msg.lower() or "limit" in err_msg.lower() or "quota" in err_msg.lower()
            if is_rate_limit and attempt < max_retries - 1:
                logger.warning("Voyage AI rate limit hit. Retrying in %.2fs (attempt %d/%d)...", backoff, attempt + 1, max_retries)
                time.sleep(backoff)
                backoff *= 2.0
            else:
                raise exc

# ── Lazy Voyage client ────────────────────────────────────────────────────────
_voyage_client = None

def get_voyage_client():
    global _voyage_client
    if _voyage_client is None:
        import voyageai
        api_key = VOYAGE_API_KEY or os.getenv("VOYAGE_API_KEY")
        if not api_key:
            logger.warning("VOYAGE_API_KEY is not configured in .env. Voyage requests may fail.")
        _voyage_client = voyageai.Client(api_key=api_key)
    return _voyage_client


# ═══════════════════════════════════════════════════════════════════════════════
#  Text Embedder — voyage-multimodal-3.5
# ═══════════════════════════════════════════════════════════════════════════════

class TextEmbedder:
    """Voyage AI Multimodal 3.5 Text Embedder wrapper."""

    def __init__(self, model_name: str = "voyage-multimodal-3.5"):
        self.model_name = model_name

    def embed(self, text: str, input_type: str = "document") -> list[float]:
        """Embed a single text string, returns a list of floats."""
        return self.embed_batch([text], input_type=input_type)[0]

    def embed_batch(self, texts: list[str], batch_size: int = 16, input_type: str = "document") -> list[list[float]]:
        """Embed a batch of texts using Voyage, with caching."""
        if not texts:
            return []

        from utils.cache import get_json_cache, set_json_cache

        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        # 1. Fetch from cache
        for idx, text in enumerate(texts):
            cleaned_text = text.strip() or " "
            cache_key = f"emb:voyage-3.5:txt:{input_type}:{cleaned_text}"
            cached_vector = get_json_cache(cache_key)
            if cached_vector is not None:
                results[idx] = cached_vector
            else:
                uncached_indices.append(idx)
                uncached_texts.append(cleaned_text)

        # 2. Embed missing items in batch
        if uncached_texts:
            client = get_voyage_client()
            try:
                # Wrap each string in a list because Voyage multimodal expects List[List[Union[str, Image]]]
                inputs = [[t] for t in uncached_texts]
                res = _multimodal_embed_with_retry(
                    client=client,
                    inputs=inputs,
                    model=self.model_name,
                    input_type=input_type
                )
                embedded_vectors = res.embeddings
            except Exception as exc:
                logger.error("Voyage text embedding failed permanently after retries: %s. Raising exception to prevent zero vector pollution.", exc)
                raise exc

            # 3. Populate results and write to cache
            for idx, vec in zip(uncached_indices, embedded_vectors):
                results[idx] = vec
                cleaned_text = uncached_texts[uncached_indices.index(idx)]
                cache_key = f"emb:voyage-3.5:txt:{input_type}:{cleaned_text}"
                set_json_cache(cache_key, vec, expire_seconds=86400 * 30) # 30 days

        return results

    @property
    def dim(self) -> int:
        return TEXT_VECTOR_DIM


# ═══════════════════════════════════════════════════════════════════════════════
#  Image Embedder — voyage-multimodal-3.5
# ═══════════════════════════════════════════════════════════════════════════════

class ImageEmbedder:
    """Voyage AI Multimodal 3.5 Image and Interleaved Embedder wrapper."""

    def __init__(self, model_name: str = "voyage-multimodal-3.5"):
        self.model_name = model_name

    def embed_text(self, text: str, input_type: str = "query") -> list[float]:
        """Embed a text query/caption using Voyage multimodal model."""
        return get_text_embedder().embed(text, input_type=input_type)

    def embed_image_file(self, image_path: str, input_type: str = "document") -> list[float] | None:
        """Embed an image file from path using Voyage, with caching."""
        if not image_path or not os.path.exists(image_path):
            return None

        from utils.cache import get_json_cache, set_json_cache

        # Calculate image file hash for cache lookup
        try:
            sha256 = hashlib.sha256()
            with open(image_path, "rb") as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            img_hash = sha256.hexdigest()
            cache_key = f"emb:voyage-3.5:img:{input_type}:{img_hash}"
            
            cached_vector = get_json_cache(cache_key)
            if cached_vector is not None:
                return cached_vector
        except Exception:
            img_hash = None
            cache_key = None

        try:
            img = Image.open(image_path).convert("RGB")
            client = get_voyage_client()
            res = _multimodal_embed_with_retry(
                client=client,
                inputs=[[img]],
                model=self.model_name,
                input_type=input_type
            )
            vec = res.embeddings[0]
            if vec and cache_key:
                set_json_cache(cache_key, vec, expire_seconds=86400 * 30)
            return vec
        except Exception as exc:
            logger.warning("Failed to embed image %s using Voyage: %s. Returning None for text fallback.", image_path, exc)
            return None

    def embed_interleaved(self, text: str, image_path: str, input_type: str = "document") -> list[float]:
        """Embed an interleaved text context and image together as a single multimodal vector."""
        if not image_path or not os.path.exists(image_path):
            # Fallback to plain text embedding
            return get_text_embedder().embed(text, input_type=input_type)

        from utils.cache import get_json_cache, set_json_cache
        
        try:
            # Hash text + image content
            sha256 = hashlib.sha256()
            sha256.update(text.encode("utf-8"))
            with open(image_path, "rb") as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            combined_hash = sha256.hexdigest()
            cache_key = f"emb:voyage-3.5:interleaved:{input_type}:{combined_hash}"
            
            cached_vector = get_json_cache(cache_key)
            if cached_vector is not None:
                return cached_vector
        except Exception:
            combined_hash = None
            cache_key = None

        try:
            img = Image.open(image_path).convert("RGB")
            client = get_voyage_client()
            # Voyage expects interleaved input inside list: [[text, PIL_image]]
            res = _multimodal_embed_with_retry(
                client=client,
                inputs=[[text, img]],
                model=self.model_name,
                input_type=input_type
            )
            vec = res.embeddings[0]
            if vec and cache_key:
                set_json_cache(cache_key, vec, expire_seconds=86400 * 30)
            return vec
        except Exception as exc:
            logger.warning("Failed to embed interleaved content for %s: %s. Falling back to text-only.", image_path, exc)
            # Fallback to text embedding
            return get_text_embedder().embed(text, input_type=input_type)

    def embed_images(self, images: list[Image.Image], input_type: str = "document") -> list[list[float]]:
        """Embed a batch of PIL images."""
        if not images:
            return []

        client = get_voyage_client()
        try:
            inputs = [[img] for img in images]
            res = _multimodal_embed_with_retry(
                client=client,
                inputs=inputs,
                model=self.model_name,
                input_type=input_type
            )
            return res.embeddings
        except Exception as exc:
            logger.error("Voyage batch image embedding failed permanently: %s. Raising exception to prevent zero vector pollution.", exc)
            raise exc

    @property
    def dim(self) -> int:
        return IMAGE_VECTOR_DIM

    @property
    def backend(self) -> str:
        return "voyage-multimodal"


# ═══════════════════════════════════════════════════════════════════════════════
#  Singletons
# ═══════════════════════════════════════════════════════════════════════════════

_text_embedder: TextEmbedder | None = None
_image_embedder: ImageEmbedder | None = None


def get_text_embedder() -> TextEmbedder:
    global _text_embedder
    if _text_embedder is None:
        _text_embedder = TextEmbedder()
    return _text_embedder


def get_image_embedder() -> ImageEmbedder:
    global _image_embedder
    if _image_embedder is None:
        _image_embedder = ImageEmbedder()
    return _image_embedder


def embedding_model_status() -> dict:
    """Return loading status of Voyage model."""
    client_loaded = _voyage_client is not None
    return {
        "text_embedder": {
            "model": "voyage-multimodal-3.5",
            "loaded": client_loaded,
        },
        "image_embedder": {
            "model": "voyage-multimodal-3.5",
            "loaded": client_loaded,
            "backend": "voyage-multimodal",
        },
    }
