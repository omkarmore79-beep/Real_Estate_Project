"""
Image Processor — text-context-based image metadata creator.

Does NOT run OCR or call a vision LLM.
Classifies images purely from:
  - Page text (nearby context)
  - Image filename / image_id patterns

Reuses the text-fallback classification logic already present in
``ingestion/image_analyzer.py`` (_fallback_page_metadata) to keep
classification consistent.
"""

from __future__ import annotations

import logging
import re
import uuid

logger = logging.getLogger(__name__)

# ── Image type labels ──────────────────────────────────────────────────────────
IMAGE_TYPES = (
    "floor_plan",
    "master_plan",
    "location_map",
    "amenity",
    "exterior",
    "interior",
    "logo",
    "table",
    "other",
)

# ── Classification rules: (image_type, phrases, keywords) ─────────────────────
_CLASSIFICATION_RULES: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    (
        "floor_plan",
        ("floor plan", "unit plan", "apartment layout", "carpet area", "super built-up"),
        ("bedroom", "kitchen", "balcony", "bhk", "living", "toilet", "dimensions"),
    ),
    (
        "master_plan",
        ("master plan", "master layout plan", "site layout", "township", "tower layout"),
        ("tower", "towers", "internal roads", "landscape", "entry", "podium"),
    ),
    (
        "location_map",
        ("location plan", "location map", "connectivity", "nearby landmarks", "map not to scale"),
        ("airport", "metro", "railway", "hospital", "school", "highway", "landmark", "distance"),
    ),
    (
        "amenity",
        ("amenities", "clubhouse", "gymnasium", "swimming pool", "jogging track", "indoor games"),
        ("gym", "pool", "kids", "play", "jogging", "games", "garden", "sports"),
    ),
    (
        "exterior",
        ("elevation", "building view", "tower view", "exterior", "project view"),
        ("elevation", "exterior", "render", "view", "facade"),
    ),
    (
        "interior",
        ("interior", "inside view", "room view", "lobby", "living space"),
        ("interior", "room", "lobby", "lounge"),
    ),
    (
        "logo",
        ("logo", "brand"),
        ("logo", "brand"),
    ),
    (
        "table",
        ("payment plan", "payment chart", "price chart", "cost sheet", "schedule"),
        ("payment", "installment", "schedule", "amount", "milestone"),
    ),
]

# ── Filename-based hints ───────────────────────────────────────────────────────
_FILENAME_HINTS: list[tuple[str, re.Pattern]] = [
    ("floor_plan", re.compile(r"floor|unit|layout|bhk", re.IGNORECASE)),
    ("master_plan", re.compile(r"master|site|township", re.IGNORECASE)),
    ("location_map", re.compile(r"location|map|connectivity", re.IGNORECASE)),
    ("amenity", re.compile(r"amenity|amenities|clubhouse|gym|pool", re.IGNORECASE)),
    ("exterior", re.compile(r"elevation|exterior|view|render|facade", re.IGNORECASE)),
    ("interior", re.compile(r"interior|room|lobby", re.IGNORECASE)),
    ("logo", re.compile(r"logo|brand", re.IGNORECASE)),
    ("table", re.compile(r"payment|price|chart|table|schedule", re.IGNORECASE)),
]


# ── Public API ─────────────────────────────────────────────────────────────────

def process_images(
    pages: list[dict],
    document_id: str,
    source_file: str,
    metadata: dict | None = None,
) -> list[dict]:
    """
    Build image metadata records from processed PDF pages.

    For every image in every page:
      - Classify image_type from page text and filename
      - Generate a short text caption
      - Return a complete image record

    Parameters
    ----------
    pages:        List of {page_number, text, ocr_text, images: [...]} from pdf_processor.
    document_id:  Document identifier.
    source_file:  Original PDF filename.
    metadata:     Optional extra metadata (project_name, builder, etc.).

    Returns
    -------
    List of image dicts ready for Qdrant upsert and MongoDB storage.
    """
    meta = metadata or {}
    records: list[dict] = []

    for page in pages:
        page_number = page.get("page_number", 0)
        page_text = (page.get("text") or "").strip()
        ocr_text = (page.get("ocr_text") or "").strip()

        for img in page.get("images", []):
            image_id = img.get("image_id") or f"page_{page_number}"
            image_path = img.get("image_path", "")
            local_path = img.get("local_path")
            image_url = image_path

            image_type = classify_image_type(page_text, image_id)
            caption = generate_caption(image_type, page_text, page_number)

            if meta.get("domain") == "excavator":
                payload_metadata = {
                    "doc_id": meta.get("doc_id", document_id),
                    "document_id": document_id,
                    "doc_type": meta.get("doc_type", ""),
                    "title": meta.get("title", ""),
                    "source_file": source_file,
                    "revision_date": meta.get("revision_date", ""),
                    "ingested_at": meta.get("ingested_at", ""),
                    "page_number": page_number,
                    "section_path": meta.get("section_path", ""),
                    "machine_model": meta.get("machine_model", "R215L"),
                    "component_tags": meta.get("component_tags", []),
                    "dtc_codes": meta.get("dtc_codes", []),
                    "supersedes_doc_id": meta.get("supersedes_doc_id", ""),
                    "confidence_weight": meta.get("confidence_weight", 1.0),
                    "domain": "excavator",
                    "image_id": image_id,
                    "image_path": image_path,
                    "image_url": image_url,
                    "image_type": image_type,
                    "caption": caption,
                    "nearby_text": page_text[:500],
                    "ocr_context": ocr_text[:500],
                }
            else:
                 payload_metadata = {
                    "document_id": document_id,
                    "page_number": page_number,
                    "image_id": image_id,
                    "image_path": image_path,
                    "image_url": image_url,
                    "image_type": image_type,
                    "caption": caption,
                    "source_file": source_file,
                    "project": meta.get("project_name", meta.get("project", "")),
                    "builder": meta.get("builder", ""),
                    "document_type": meta.get("document_type", ""),
                    "domain": meta.get("domain", "real_estate"),
                    "nearby_text": page_text[:500],
                    "ocr_context": ocr_text[:500],
                }

            records.append(
                {
                    "image_id": image_id,
                    "document_id": document_id,
                    "page_number": page_number,
                    "page": page_number,  # legacy field
                    "image_path": image_path,
                    "image_url": image_url,
                    "local_path": local_path,
                    "nearby_page_text": page_text[:1000],
                    "nearby_text": page_text[:1000],
                    "ocr_context": ocr_text[:1000],
                    "source_file": source_file,
                    "image_type": image_type,
                    "caption": caption,
                    "vector": [],  # filled by embedding service
                    "metadata": payload_metadata,
                }
            )

    logger.info(
        "Processed %d image records for document_id=%s", len(records), document_id
    )
    return records



def classify_image_type(page_text: str, filename: str = "") -> str:
    """
    Classify image type from page text and filename without OCR.

    Returns one of: floor_plan, master_plan, location_map, amenity,
    exterior, interior, logo, table, other.
    """
    # 1. Try filename hints first
    for image_type, pattern in _FILENAME_HINTS:
        if pattern.search(filename):
            return image_type

    if not page_text:
        return "other"

    haystack = page_text.lower()
    best_type = "other"
    best_score = 0

    for image_type, phrases, keywords in _CLASSIFICATION_RULES:
        phrase_score = sum(4 for phrase in phrases if phrase in haystack)
        keyword_score = sum(1 for kw in keywords if kw in haystack)
        score = phrase_score + keyword_score
        if score > best_score:
            best_score = score
            best_type = image_type

    return best_type


def generate_caption(image_type: str, page_text: str, page_number: int) -> str:
    """
    Generate a simple text caption from image type and page context.

    Never reads pixels or uses OCR — purely text-derived.
    """
    type_labels = {
        "floor_plan": "Floor plan",
        "master_plan": "Master plan",
        "location_map": "Location map",
        "amenity": "Amenity image",
        "exterior": "Exterior / elevation view",
        "interior": "Interior view",
        "logo": "Developer logo",
        "table": "Pricing / payment table",
        "other": "Document image",
    }
    label = type_labels.get(image_type, "Document image")

    # Extract first 120 chars of meaningful page text for context
    short_context = ""
    if page_text:
        clean = re.sub(r"\s+", " ", page_text[:300]).strip()
        short_context = clean[:120]

    if short_context:
        return f"{label} on page {page_number}. Context: {short_context}"
    return f"{label} on page {page_number}."


def build_image_embedding_text(record: dict) -> str:
    """
    Build a combined text string for image embedding.

    Combines caption + nearby page text so the image vector is
    searchable by natural language queries.
    """
    parts = [
        record.get("caption", ""),
        record.get("image_type", "").replace("_", " "),
        (record.get("nearby_page_text") or "")[:300],
    ]
    return " ".join(p for p in parts if p).strip()
