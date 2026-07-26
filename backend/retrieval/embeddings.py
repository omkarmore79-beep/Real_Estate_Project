"""
Embedding Service — Voyage AI Integration.

Strategy:
  - Text chunks  → voyage-3        (1M TPM, client.embed(), fast batch API)
  - Image/Visual → voyage-multimodal-3.5  (client.multimodal_embed(), per-image fallback to text)

All embeddings are cached to local memory (30-day TTL).
"""

from __future__ import annotations

import logging
import os
import re
import hashlib
import time
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import VOYAGE_API_KEY, TEXT_VECTOR_DIM, IMAGE_VECTOR_DIM, TEXT_EMBEDDING_MODEL, VOYAGE_TPM_LIMIT

logger = logging.getLogger(__name__)

# ── Voyage client singleton ────────────────────────────────────────────────────
_voyage_client = None

def get_voyage_client():
    global _voyage_client
    if _voyage_client is None:
        import voyageai
        api_key = VOYAGE_API_KEY or os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY is not set. Add it to backend/.env and restart.")
        _voyage_client = voyageai.Client(api_key=api_key)
        logger.info("Voyage AI client initialized.")
    return _voyage_client


def _is_rate_limit_error(exc: Exception) -> bool:
    err = str(exc).lower()
    return "rate limit" in err or "429" in err or "tpm" in err or "too many requests" in err

def get_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After header or parse wait time from exception message."""
    try:
        if hasattr(exc, "response") and exc.response is not None:
            headers = exc.response.headers
            if "Retry-After" in headers:
                return float(headers["Retry-After"])
    except Exception:
        pass
    msg = str(exc)
    match = re.search(r"try again in (\d+(?:\.\d+)?) seconds", msg, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def _embed_texts_with_retry(client, texts: list[str], model: str, input_type: str, document_id: str | None = None) -> list:
    from storage.doc_status import set_status
    max_attempts = 5
    base_sleep = 2.0
    
    for attempt in range(1, max_attempts + 1):
        try:
            res = client.embed(texts=texts, model=model, input_type=input_type)
            return res.embeddings
        except Exception as exc:
            is_rate = _is_rate_limit_error(exc)
            retry_after = get_retry_after(exc)
            
            if attempt == max_attempts:
                logger.error("All text embedding retry attempts failed: %s", exc)
                raise
                
            sleep_time = retry_after if retry_after is not None else (base_sleep * (2 ** (attempt - 1)))
            logger.warning(
                "Text embedding failed (attempt %d/%d): %s. Sleeping %.2fs (rate_limit=%s)",
                attempt, max_attempts, exc, sleep_time, is_rate
            )
            
            if document_id:
                if is_rate:
                    set_status(document_id, "waiting_quota", message=f"Waiting for embedding quota (Rate Limit). Retrying in {int(sleep_time)}s...")
                else:
                    set_status(document_id, "retrying", message=f"Embedding failed. Retrying in {int(sleep_time)}s (Attempt {attempt+1}/5)...")
                    
            time.sleep(sleep_time)
            
    raise RuntimeError("Unreachable code in text embedding retry loop")

def _embed_multimodal_with_retry(client, inputs: list, model: str, input_type: str, document_id: str | None = None) -> list:
    from storage.doc_status import set_status
    max_attempts = 5
    base_sleep = 2.0
    
    for attempt in range(1, max_attempts + 1):
        try:
            res = client.multimodal_embed(inputs=inputs, model=model, input_type=input_type)
            return res.embeddings
        except Exception as exc:
            is_rate = _is_rate_limit_error(exc)
            retry_after = get_retry_after(exc)
            
            if attempt == max_attempts:
                logger.error("All multimodal embedding retry attempts failed: %s", exc)
                raise
                
            sleep_time = retry_after if retry_after is not None else (base_sleep * (2 ** (attempt - 1)))
            logger.warning(
                "Multimodal embedding failed (attempt %d/%d): %s. Sleeping %.2fs (rate_limit=%s)",
                attempt, max_attempts, exc, sleep_time, is_rate
            )
            
            if document_id:
                if is_rate:
                    set_status(document_id, "waiting_quota", message=f"Waiting for embedding quota (Rate Limit). Retrying in {int(sleep_time)}s...")
                else:
                    set_status(document_id, "retrying", message=f"Embedding failed. Retrying in {int(sleep_time)}s (Attempt {attempt+1}/5)...")
                    
            time.sleep(sleep_time)
            
    raise RuntimeError("Unreachable code in multimodal embedding retry loop")


# ═══════════════════════════════════════════════════════════════════════════════
#  Text Embedder
# ═══════════════════════════════════════════════════════════════════════════════

class TextEmbedder:
    """
    Embeds text using voyage-3 (or TEXT_EMBEDDING_MODEL from .env).
    Uses client.embed() — supports 1,000,000 TPM.
    Falls back to multimodal path only if model name contains 'multimodal'.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = (model_name or TEXT_EMBEDDING_MODEL or "voyage-3").strip()
        self._is_multimodal = "multimodal" in self.model_name.lower()
        logger.info("TextEmbedder initialized with model: %s (multimodal_path=%s)", self.model_name, self._is_multimodal)

    def embed(self, text: str, input_type: str = "document") -> list[float]:
        return self.embed_batch([text], input_type=input_type)[0]

    def embed_batch(self, texts: list[str], batch_size: int = 128, input_type: str = "document", document_id: str | None = None) -> list[list[float]]:
        if not texts:
            return []

        from utils.cache import get_json_cache, set_json_cache

        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # 1. Cache lookup
        for idx, text in enumerate(texts):
            cleaned = text.strip() or " "
            key = f"emb:{self.model_name}:{input_type}:{hashlib.md5(cleaned.encode()).hexdigest()}"
            hit = get_json_cache(key)
            if hit is not None:
                results[idx] = hit
            else:
                uncached_indices.append(idx)
                uncached_texts.append(cleaned)

        if not uncached_texts:
            return results

        client = get_voyage_client()
        all_vectors: list[list[float]] = []

        if not self._is_multimodal:
            # Fast path: voyage-3 — but with strict token rate limiting
            # Group into batches that don't exceed VOYAGE_TPM_LIMIT * 0.85
            safe_token_limit = int(VOYAGE_TPM_LIMIT * 0.85)
            # Ensure safe_token_limit is at least 300 so we can process single chunks
            safe_token_limit = max(safe_token_limit, 300)

            current_batch = []
            current_batch_tokens = 0
            
            for idx, text in enumerate(uncached_texts):
                tokens = len(text) // 4 + 20
                if current_batch and (current_batch_tokens + tokens > safe_token_limit or len(current_batch) >= batch_size):
                    logger.info("Sending text batch of %d items (~%d tokens)", len(current_batch), current_batch_tokens)
                    vecs = _embed_texts_with_retry(client, current_batch, self.model_name, input_type, document_id)
                    all_vectors.extend(vecs)
                    
                    sleep_time = (current_batch_tokens / VOYAGE_TPM_LIMIT) * 60.0
                    logger.info("Voyage rate limit preservation: sleeping %.2fs", sleep_time)
                    time.sleep(sleep_time)
                    
                    current_batch = []
                    current_batch_tokens = 0

                current_batch.append(text)
                current_batch_tokens += tokens

            if current_batch:
                logger.info("Sending final text batch of %d items (~%d tokens)", len(current_batch), current_batch_tokens)
                vecs = _embed_texts_with_retry(client, current_batch, self.model_name, input_type, document_id)
                all_vectors.extend(vecs)
                # Sleep briefly if this is a low-TPM environment to prevent immediate subsequent request failure
                if VOYAGE_TPM_LIMIT <= 10000:
                    sleep_time = (current_batch_tokens / VOYAGE_TPM_LIMIT) * 60.0
                    logger.info("Voyage final rate limit preservation sleep: %.2fs", sleep_time)
                    time.sleep(sleep_time)
        else:
            # Multimodal path (rare — only if TEXT_EMBEDDING_MODEL=voyage-multimodal-3.5)
            for i in range(0, len(uncached_texts), 8):
                batch = uncached_texts[i : i + 8]
                inputs = [[t] for t in batch]
                vecs = _embed_multimodal_with_retry(client, inputs, self.model_name, input_type, document_id)
                all_vectors.extend(vecs)
                if i + 8 < len(uncached_texts):
                    time.sleep(0.5)

        # 2. Write to cache and populate results
        for local_idx, (orig_idx, vec) in enumerate(zip(uncached_indices, all_vectors)):
            results[orig_idx] = vec
            cleaned = uncached_texts[local_idx]
            key = f"emb:{self.model_name}:{input_type}:{hashlib.md5(cleaned.encode()).hexdigest()}"
            set_json_cache(key, vec, expire_seconds=86400 * 30)

        return results

    @property
    def dim(self) -> int:
        return TEXT_VECTOR_DIM


# ═══════════════════════════════════════════════════════════════════════════════
#  Image Embedder
# ═══════════════════════════════════════════════════════════════════════════════

class ImageEmbedder:
    """
    Embeds images (and interleaved text+image) using voyage-multimodal-3.5.
    Falls back to text-only embedding (voyage-3) on rate limit or error.
    """

    def __init__(self, model_name: str = "voyage-multimodal-3.5"):
        self.model_name = model_name

    def embed_text(self, text: str, input_type: str = "document") -> list[float]:
        """Use the fast text embedder for caption/fallback text."""
        return get_text_embedder().embed(text, input_type=input_type)

    def embed_image_file(self, image_path: str, input_type: str = "document") -> list[float] | None:
        """Embed a single image file. Returns None on failure (caller uses text fallback)."""
        if not image_path or not os.path.exists(image_path):
            return None

        from utils.cache import get_json_cache, set_json_cache

        # Cache key from file hash
        cache_key = None
        try:
            sha = hashlib.sha256()
            with open(image_path, "rb") as f:
                while chunk := f.read(8192):
                    sha.update(chunk)
            cache_key = f"emb:img:{self.model_name}:{sha.hexdigest()}"
            hit = get_json_cache(cache_key)
            if hit is not None:
                return hit
        except Exception:
            pass

        try:
            img = Image.open(image_path).convert("RGB")
            client = get_voyage_client()
            vecs = _embed_multimodal_with_retry(client, [[img]], self.model_name, input_type)
            vec = vecs[0]
            if cache_key:
                set_json_cache(cache_key, vec, expire_seconds=86400 * 30)
            return vec
        except Exception as exc:
            logger.warning("Image embedding failed for %s: %s. Using text fallback.", image_path, exc)
            return None

    def embed_interleaved(self, text: str, image_path: str, input_type: str = "document") -> list[float]:
        """Embed text+image together. Gracefully falls back to text-only on any error."""
        if not image_path or not os.path.exists(image_path):
            return self.embed_text(text, input_type=input_type)

        from utils.cache import get_json_cache, set_json_cache

        cache_key = None
        try:
            sha = hashlib.sha256()
            sha.update(text.encode("utf-8"))
            with open(image_path, "rb") as f:
                while chunk := f.read(8192):
                    sha.update(chunk)
            cache_key = f"emb:interleaved:{self.model_name}:{sha.hexdigest()}"
            hit = get_json_cache(cache_key)
            if hit is not None:
                return hit
        except Exception:
            pass

        try:
            img = Image.open(image_path).convert("RGB")
            client = get_voyage_client()
            vecs = _embed_multimodal_with_retry(client, [[text, img]], self.model_name, input_type)
            vec = vecs[0]
            if cache_key:
                set_json_cache(cache_key, vec, expire_seconds=86400 * 30)
            return vec
        except Exception as exc:
            logger.warning("Interleaved embedding failed for %s: %s. Falling back to text.", image_path, exc)
            return self.embed_text(text, input_type=input_type)

    def embed_interleaved_batch(
        self,
        text_image_pairs: list[tuple[str, str | None]],
        batch_size: int = 8,
        input_type: str = "document",
        document_id: str | None = None,
    ) -> list[list[float]]:
        """Embed PDF images in bounded multimodal batches, respecting VOYAGE_TPM_LIMIT."""
        if not text_image_pairs:
            return []

        results: list[list[float] | None] = [None] * len(text_image_pairs)
        multimodal_inputs: list[list] = []
        multimodal_indices: list[int] = []
        fallback_indices: list[int] = []
        fallback_texts: list[str] = []

        for index, (text, image_path) in enumerate(text_image_pairs):
            cleaned_text = (text or " ").strip() or " "
            if not image_path or not os.path.exists(image_path):
                fallback_indices.append(index)
                fallback_texts.append(cleaned_text)
                continue
            try:
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                    # PDF source images can be huge. Downscale in memory before
                    # upload to reduce request size and remote processing time.
                    image.thumbnail((768, 768), Image.Resampling.LANCZOS)
                    multimodal_inputs.append([cleaned_text, image.copy()])
                multimodal_indices.append(index)
            except Exception as exc:
                logger.warning("Could not prepare image %s: %s. Using text fallback.", image_path, exc)
                fallback_indices.append(index)
                fallback_texts.append(cleaned_text)

        client = get_voyage_client()
        
        safe_token_limit = int(VOYAGE_TPM_LIMIT * 0.85)
        # 324 tokens per image + text tokens
        safe_token_limit = max(safe_token_limit, 400)

        current_batch_inputs = []
        current_batch_indices = []
        current_batch_tokens = 0

        for text_img_pair, index in zip(multimodal_inputs, multimodal_indices):
            text_part = text_img_pair[0]
            tokens = 324 + len(text_part) // 4 + 20
            
            if current_batch_inputs and (current_batch_tokens + tokens > safe_token_limit or len(current_batch_inputs) >= batch_size):
                try:
                    logger.info("Sending multimodal batch of %d items (~%d tokens)", len(current_batch_inputs), current_batch_tokens)
                    vectors = _embed_multimodal_with_retry(client, current_batch_inputs, self.model_name, input_type, document_id)
                    for idx, vector in zip(current_batch_indices, vectors):
                        results[idx] = vector
                except Exception as exc:
                    logger.warning("Multimodal batch failed: %s. Falling back to text embeddings.", exc)
                    for idx in current_batch_indices:
                        fallback_indices.append(idx)
                        fallback_texts.append((text_image_pairs[idx][0] or " ").strip() or " ")

                sleep_time = (current_batch_tokens / VOYAGE_TPM_LIMIT) * 60.0
                logger.info("Voyage rate limit preservation (multimodal): sleeping %.2fs", sleep_time)
                time.sleep(sleep_time)

                current_batch_inputs = []
                current_batch_indices = []
                current_batch_tokens = 0

            current_batch_inputs.append(text_img_pair)
            current_batch_indices.append(index)
            current_batch_tokens += tokens

        if current_batch_inputs:
            try:
                logger.info("Sending final multimodal batch of %d items (~%d tokens)", len(current_batch_inputs), current_batch_tokens)
                vectors = _embed_multimodal_with_retry(client, current_batch_inputs, self.model_name, input_type, document_id)
                for idx, vector in zip(current_batch_indices, vectors):
                    results[idx] = vector
            except Exception as exc:
                logger.warning("Multimodal final batch failed: %s. Falling back to text.", exc)
                for idx in current_batch_indices:
                    fallback_indices.append(idx)
                    fallback_texts.append((text_image_pairs[idx][0] or " ").strip() or " ")
            
            if VOYAGE_TPM_LIMIT <= 10000:
                sleep_time = (current_batch_tokens / VOYAGE_TPM_LIMIT) * 60.0
                logger.info("Voyage final multimodal rate limit preservation sleep: %.2fs", sleep_time)
                time.sleep(sleep_time)

        if fallback_texts:
            fallback_vectors = get_text_embedder().embed_batch(fallback_texts, input_type=input_type, document_id=document_id)
            for index, vector in zip(fallback_indices, fallback_vectors):
                results[index] = vector

        if any(vector is None for vector in results):
            raise RuntimeError("Image embedding batch returned an incomplete result set.")
        return results  # type: ignore[return-value]

    def embed_images(self, images: list[Image.Image], input_type: str = "document", document_id: str | None = None) -> list[list[float]]:
        """Batch embed PIL images."""
        if not images:
            return []
        client = get_voyage_client()
        inputs = [[img] for img in images]
        return _embed_multimodal_with_retry(client, inputs, self.model_name, input_type, document_id)

    def embed_image_paths(self, image_paths: list[str], batch_size: int = 8, input_type: str = "document", document_id: str | None = None) -> list[list[float]]:
        """Batch embed image files from disk with a small multimodal batch size to avoid upload bursts."""
        if not image_paths:
            return []

        client = get_voyage_client()
        results: list[list[float]] = []

        safe_token_limit = int(VOYAGE_TPM_LIMIT * 0.85)
        safe_token_limit = max(safe_token_limit, 350)

        current_batch_paths = []
        current_batch_tokens = 0

        for path in image_paths:
            if not path or not os.path.exists(path):
                continue
            
            tokens = 324
            if current_batch_paths and (current_batch_tokens + tokens > safe_token_limit or len(current_batch_paths) >= batch_size):
                inputs = []
                for p in current_batch_paths:
                    with Image.open(p) as img:
                        inputs.append([img.convert("RGB")])
                
                logger.info("Sending image paths batch of %d items (~%d tokens)", len(inputs), current_batch_tokens)
                vectors = _embed_multimodal_with_retry(client, inputs, self.model_name, input_type, document_id)
                results.extend(vectors)
                
                sleep_time = (current_batch_tokens / VOYAGE_TPM_LIMIT) * 60.0
                logger.info("Voyage rate limit preservation (image paths): sleeping %.2fs", sleep_time)
                time.sleep(sleep_time)
                
                current_batch_paths = []
                current_batch_tokens = 0

            current_batch_paths.append(path)
            current_batch_tokens += tokens

        if current_batch_paths:
            inputs = []
            for p in current_batch_paths:
                with Image.open(p) as img:
                    inputs.append([img.convert("RGB")])
            logger.info("Sending final image paths batch of %d items (~%d tokens)", len(inputs), current_batch_tokens)
            vectors = _embed_multimodal_with_retry(client, inputs, self.model_name, input_type, document_id)
            results.extend(vectors)
            
            if VOYAGE_TPM_LIMIT <= 10000:
                sleep_time = (current_batch_tokens / VOYAGE_TPM_LIMIT) * 60.0
                logger.info("Voyage final image paths rate limit preservation sleep: %.2fs", sleep_time)
                time.sleep(sleep_time)

        return results

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
    text_model = TEXT_EMBEDDING_MODEL or "voyage-3"
    return {
        "text_embedder": {"model": text_model, "loaded": _voyage_client is not None},
        "image_embedder": {"model": "voyage-multimodal-3.5", "loaded": _voyage_client is not None, "backend": "voyage-multimodal"},
    }
