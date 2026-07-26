from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, OptimizersConfigDiff
import voyageai

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ingest")

QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "").strip()
TEXT_EMBED_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "voyage-3")
IMAGE_EMBED_MODEL = os.getenv("IMAGE_EMBEDDING_MODEL", "voyage-multimodal-3.5")
TEXT_VECTOR_DIM = int(os.getenv("TEXT_VECTOR_DIM", "1024"))
IMAGE_VECTOR_DIM = int(os.getenv("IMAGE_VECTOR_DIM", "1024"))
TEXT_COLLECTION = os.getenv("QDRANT_COLLECTION_TEXT", "real_estate_text_chunks")
IMAGE_COLLECTION = os.getenv("QDRANT_COLLECTION_IMAGES", "real_estate_image_chunks")

MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "768"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "128"))
QDRANT_BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", "100"))


class ListingMetadata(BaseModel):
    project_name: str = ""
    builder: str = ""
    city: str = ""
    property_type: str = ""
    bedrooms: int | None = None
    price: float | None = None
    price_unit: str = ""
    document_id: str = ""
    source_file: str = ""


class RealEstateListing(BaseModel):
    id: str = Field(..., description="Stable document/listing identifier")
    title: str = ""
    content: str = ""
    metadata: ListingMetadata = Field(default_factory=ListingMetadata)


class IngestConfig(BaseModel):
    qdrant_url: str = QDRANT_URL
    qdrant_api_key: str = QDRANT_API_KEY
    voyage_api_key: str = VOYAGE_API_KEY
    text_model: str = TEXT_EMBED_MODEL
    image_model: str = IMAGE_EMBED_MODEL
    text_collection: str = TEXT_COLLECTION
    image_collection: str = IMAGE_COLLECTION
    batch_size: int = EMBED_BATCH_SIZE
    qdrant_upsert_batch: int = QDRANT_BATCH_SIZE


cfg = IngestConfig()


def _qdrant_id(value: str) -> str:
    """Convert a stable listing/chunk key into a Qdrant-compatible UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


async def _ensure_collection(async_client: AsyncQdrantClient, collection_name: str, vector_size: int) -> None:
    """Create a collection if needed, without blocking the event loop."""
    exists = False
    try:
        existing = await async_client.get_collection(collection_name=collection_name)
        exists = True
        vectors = existing.config.params.vectors
        current_size = getattr(vectors, "size", None)
        if isinstance(vectors, dict):
            current_size = vectors.get("size")
        if current_size != vector_size:
            logger.warning("Recreating collection %s due to vector-dimension mismatch.", collection_name)
            await async_client.delete_collection(collection_name=collection_name)
            exists = False
    except Exception:
        exists = False

    if not exists:
        await async_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
        )
        logger.info("Created Qdrant collection %s (indexing disabled for bulk load)", collection_name)

    try:
        from qdrant_client.models import PayloadSchemaType
        await async_client.create_payload_index(
            collection_name=collection_name,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    except Exception:
        logger.debug("Payload index already exists for %s.", collection_name)


async def _embed_texts_async(client: voyageai.Client, texts: Sequence[str]) -> list[list[float]]:
    """Embed text in bounded concurrent batches."""
    batches = [list(texts[i : i + cfg.batch_size]) for i in range(0, len(texts), cfg.batch_size)]
    if not batches:
        return []
    vectors = await asyncio.gather(*(
        asyncio.to_thread(_embed_texts, client, batch) for batch in batches
    ))
    return [vector for batch in vectors for vector in batch]


async def _embed_images_async(client: voyageai.Client, encoded_images: Sequence[str]) -> list[list[float]]:
    """Embed compressed images in small multimodal batches."""
    image_batch_size = min(8, cfg.batch_size)
    batches = [list(encoded_images[i : i + image_batch_size]) for i in range(0, len(encoded_images), image_batch_size)]
    if not batches:
        return []
    vectors = await asyncio.gather(*(
        asyncio.to_thread(_embed_images, client, batch) for batch in batches
    ))
    return [vector for batch in vectors for vector in batch]


async def _set_indexing_threshold(async_client: AsyncQdrantClient, threshold: int) -> None:
    for collection_name in (cfg.text_collection, cfg.image_collection):
        try:
            await async_client.update_collection(
                collection_name=collection_name,
                optimizer_config=OptimizersConfigDiff(indexing_threshold=threshold),
            )
            logger.info("Indexing threshold=%s applied to %s", threshold, collection_name)
        except Exception as exc:
            logger.warning("Could not update threshold for %s: %s", collection_name, exc)


def _compress_image_for_voyage(src_path: str, dst_path: str) -> bool:
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            if min(w, h) < 120:
                return False
            if max(w, h) > MAX_IMAGE_SIDE:
                ratio = MAX_IMAGE_SIDE / max(w, h)
                img = img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.LANCZOS)
            img.save(dst_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        return True
    except Exception as exc:
        logger.warning("Image compression failed for %s: %s", src_path, exc)
        return False


def _encode_image(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.0, min=1.0, max=30.0),
    reraise=True,
)
def _embed_texts(client: voyageai.Client, texts: list[str]) -> list[list[float]]:
    response = client.embed(texts=texts, model=cfg.text_model, input_type="document")
    return response.embeddings


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.0, min=1.0, max=30.0),
    reraise=True,
)
def _embed_images(client: voyageai.Client, encoded_images: list[str]) -> list[list[float]]:
    response = client.multimodal_embed(
        inputs=[[img] for img in encoded_images],
        model=cfg.image_model,
        input_type="document",
    )
    return response.embeddings


async def _upsert_points(async_client: AsyncQdrantClient, collection_name: str, points: list[PointStruct]) -> None:
    if not points:
        return
    await async_client.upsert(collection_name=collection_name, points=points)


async def ingest_listing(listing: RealEstateListing, asset_image_paths: list[str] | None = None) -> dict[str, Any]:
    listing_metadata = listing.metadata.model_copy(
        update={
            "document_id": listing.metadata.document_id or listing.id,
            "source_file": listing.metadata.source_file or listing.id,
        }
    )
    text_chunks = [
        {
            "chunk_id": f"{listing.id}-text-{idx}",
            "content": listing.content,
            "vector": [],
            "metadata": {"title": listing.title, **listing_metadata.model_dump()},
        }
        for idx in range(1)
    ]

    image_records: list[dict[str, Any]] = []

    if not VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY is required for ingestion.")

    client = voyageai.Client(api_key=VOYAGE_API_KEY)
    # Keep compressed files alive until all multimodal requests have finished.
    with tempfile.TemporaryDirectory(prefix=f"voyage_{listing.id}_") as tmpdir:
        encoded_images: list[str] = []
        if asset_image_paths:
            for idx, image_path in enumerate(asset_image_paths, start=1):
                comp_path = os.path.join(tmpdir, f"{idx}.jpg")
                if not _compress_image_for_voyage(image_path, comp_path):
                    continue
                image_id = f"{listing.id}-img-{idx}"
                payload = {"title": listing.title, **listing_metadata.model_dump()}
                payload.update({"image_id": image_id, "image_path": image_path, "caption": Path(image_path).stem})
                image_records.append({"image_id": image_id, "vector": [], "metadata": payload})
                # Voyage accepts data URIs for base64 multimodal inputs.
                encoded_images.append(f"data:image/jpeg;base64,{_encode_image(comp_path)}")

        text_vectors, image_vectors = await asyncio.gather(
            _embed_texts_async(client, [chunk["content"] for chunk in text_chunks]),
            _embed_images_async(client, encoded_images),
        )
        for chunk, vector in zip(text_chunks, text_vectors):
            chunk["vector"] = vector
        for record, vector in zip(image_records, image_vectors):
            record["vector"] = vector

    if not cfg.qdrant_url:
        raise RuntimeError("QDRANT_URL is required for ingestion.")
    async_client = AsyncQdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key or None, prefer_grpc=True, timeout=120)
    await _ensure_collection(async_client, cfg.text_collection, TEXT_VECTOR_DIM)
    await _ensure_collection(async_client, cfg.image_collection, IMAGE_VECTOR_DIM)
    await _set_indexing_threshold(async_client, threshold=0)
    try:
        text_points = [
            PointStruct(id=_qdrant_id(chunk["chunk_id"]), vector=chunk["vector"], payload={"content": chunk["content"], **chunk["metadata"]})
            for chunk in text_chunks if chunk.get("vector")
        ]
        image_points = [
            PointStruct(id=_qdrant_id(record["image_id"]), vector=record["vector"], payload={"image_id": record["image_id"], **record["metadata"]})
            for record in image_records if record.get("vector")
        ]
        for points, collection, label in ((text_points, cfg.text_collection, "text"), (image_points, cfg.image_collection, "image")):
            total_batches = (len(points) + cfg.qdrant_upsert_batch - 1) // cfg.qdrant_upsert_batch
            for offset in range(0, len(points), cfg.qdrant_upsert_batch):
                await _upsert_points(async_client, collection, points[offset : offset + cfg.qdrant_upsert_batch])
                logger.info("Upserted %s batch %d/%d into %s", label, offset // cfg.qdrant_upsert_batch + 1, total_batches, collection)
    finally:
        await _set_indexing_threshold(async_client, threshold=20_000)
        close = getattr(async_client, "close", None)
        if close:
            await close()

    return {
        "document_id": listing.id,
        "text_chunks_indexed": len(text_points),
        "image_chunks_indexed": len(image_points),
    }


async def main() -> None:
    if not cfg.qdrant_url:
        raise RuntimeError("QDRANT_URL is required.")

    listing = RealEstateListing(
        id="listing-001",
        title="Sunset Residency",
        content="Luxury 3 BHK apartment with clubhouse, gym, pool and close connectivity.",
        metadata=ListingMetadata(
            project_name="Sunset Residency",
            builder="Apex Builders",
            city="Mumbai",
            property_type="Apartment",
            bedrooms=3,
            price=12000000,
            price_unit="INR",
            document_id="listing-001",
            source_file="sunset_residency.pdf",
        ),
    )

    image_paths = [
        str(Path(__file__).resolve().parent.parent / "uploads" / "rag_images" / "sample.jpg"),
    ]
    t0 = time.perf_counter()
    result = await ingest_listing(listing, asset_image_paths=[p for p in image_paths if os.path.exists(p)])
    logger.info("Finished ingestion in %.2fs: %s", time.perf_counter() - t0, result)


if __name__ == "__main__":
    asyncio.run(main())
