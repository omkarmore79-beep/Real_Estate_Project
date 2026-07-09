"""
Embedding Service — lazy-loading text and image embedding models.

Text embedder:  BAAI/bge-m3      → 1024-dim dense vectors
Image embedder: jinaai/jina-clip-v2 (fallback: openai/clip-vit-base-patch32)
                → 512-dim dense vectors (for both images and text captions)

Models are loaded lazily on first use so the FastAPI app starts instantly.
"""

from __future__ import annotations

import logging
import os
from typing import Union

logger = logging.getLogger(__name__)

# ── Config (read from environment at import time) ──────────────────────────────
from config import (
    IMAGE_EMBEDDING_MODEL,
    IMAGE_VECTOR_DIM,
    TEXT_EMBEDDING_MODEL,
    TEXT_VECTOR_DIM,
)

_CLIP_FALLBACK_MODEL = "openai/clip-vit-base-patch32"


# ═══════════════════════════════════════════════════════════════════════════════
#  Text Embedder — BAAI/bge-m3
# ═══════════════════════════════════════════════════════════════════════════════

class TextEmbedder:
    """Lazy wrapper around BAAI/bge-m3 FlagEmbedding model."""

    def __init__(self, model_name: str = TEXT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        logger.info("Loading text embedding model: %s (this may take a while…)", self.model_name)
        try:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(self.model_name, use_fp16=False)
            logger.info("Text embedding model loaded: %s", self.model_name)
        except Exception as exc:
            logger.warning(
                "FlagEmbedding load failed (%s), falling back to sentence-transformers: %s",
                self.model_name,
                exc,
            )
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")

    def embed(self, text: str) -> list[float]:
        """Embed a single text string, returns a list of floats."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """Embed a batch of texts, returns list of float lists."""
        self._load()
        if not texts:
            return []

        # Normalise empties to avoid model crashes
        cleaned = [t.strip() or " " for t in texts]

        try:
            # FlagEmbedding path
            from FlagEmbedding import BGEM3FlagModel
            if isinstance(self._model, BGEM3FlagModel):
                output = self._model.encode(
                    cleaned,
                    batch_size=batch_size,
                    max_length=512,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
                vectors = output["dense_vecs"].tolist()
                return vectors
        except Exception:
            pass

        # sentence-transformers fallback
        import numpy as np
        vectors = self._model.encode(cleaned, batch_size=batch_size, show_progress_bar=False)
        return vectors.tolist()

    @property
    def dim(self) -> int:
        return TEXT_VECTOR_DIM


# ═══════════════════════════════════════════════════════════════════════════════
#  Image Embedder — jinaai/jina-clip-v2 (with CLIP fallback)
# ═══════════════════════════════════════════════════════════════════════════════

class ImageEmbedder:
    """
    Lazy wrapper for multimodal image+text embedding.

    Primary:  jinaai/jina-clip-v2
    Fallback: openai/clip-vit-base-patch32
    """

    def __init__(self, model_name: str = IMAGE_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._backend: str = "unknown"

    def _load(self):
        if self._model is not None:
            return

        # Try jina-clip-v2 first
        try:
            logger.info("Loading image embedding model: %s …", self.model_name)
            from transformers import AutoModel
            self._model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            ).eval()
            self._backend = "jina"
            logger.info("Jina CLIP model loaded: %s", self.model_name)
            return
        except Exception as exc:
            logger.warning(
                "Failed to load %s (%s). Falling back to CLIP.", self.model_name, exc
            )

        # CLIP fallback
        try:
            from transformers import CLIPModel, CLIPProcessor
            logger.info("Loading CLIP fallback: %s …", _CLIP_FALLBACK_MODEL)
            self._model = CLIPModel.from_pretrained(_CLIP_FALLBACK_MODEL).eval()
            self._processor = CLIPProcessor.from_pretrained(_CLIP_FALLBACK_MODEL)
            self._backend = "clip"
            logger.info("CLIP fallback model loaded.")
        except Exception as exc:
            logger.error("Both image embedding models failed to load: %s", exc)
            raise

    def embed_text(self, text: str) -> list[float]:
        """Embed a text string using the image-text model (for caption/query search)."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        self._load()
        if not texts:
            return []

        import torch

        if self._backend == "jina":
            with torch.no_grad():
                vectors = self._model.encode_text(texts)
            return _to_list(vectors)

        # CLIP fallback
        with torch.no_grad():
            inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            features = self._model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return _to_list(features)

    def embed_image_file(self, image_path: str) -> list[float] | None:
        """Embed an image from a file path. Returns None if file missing."""
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            return self.embed_images([img])[0]
        except Exception as exc:
            logger.warning("Failed to embed image %s: %s", image_path, exc)
            return None

    def embed_images(self, images) -> list[list[float]]:
        """Embed a batch of PIL images."""
        self._load()
        if not images:
            return []

        import torch

        if self._backend == "jina":
            with torch.no_grad():
                vectors = self._model.encode_image(images)
            return _to_list(vectors)

        # CLIP fallback
        with torch.no_grad():
            inputs = self._processor(images=images, return_tensors="pt")
            features = self._model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return _to_list(features)

    @property
    def dim(self) -> int:
        return IMAGE_VECTOR_DIM

    @property
    def backend(self) -> str:
        return self._backend


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
    """Return current loading status of embedding models (for health endpoint)."""
    text_loaded = _text_embedder is not None and _text_embedder._model is not None
    image_loaded = _image_embedder is not None and _image_embedder._model is not None
    return {
        "text_embedder": {
            "model": TEXT_EMBEDDING_MODEL,
            "loaded": text_loaded,
        },
        "image_embedder": {
            "model": IMAGE_EMBEDDING_MODEL,
            "loaded": image_loaded,
            "backend": _image_embedder.backend if image_loaded else "not_loaded",
        },
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_list(tensor_or_array) -> list[list[float]]:
    """Convert torch tensor or numpy array to a Python list of float lists."""
    try:
        import torch
        if isinstance(tensor_or_array, torch.Tensor):
            return tensor_or_array.detach().cpu().numpy().tolist()
    except ImportError:
        pass
    import numpy as np
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array.tolist()
    return list(tensor_or_array)
