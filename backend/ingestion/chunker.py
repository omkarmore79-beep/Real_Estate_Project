"""
Text Chunker for the Hybrid Multimodal RAG pipeline.
Splits page-level text into overlapping chunks, preserves page-level OCR metadata,
and extracts real-estate-critical fields for downstream RAG retrieval.
"""

from __future__ import annotations

import logging
import re
import uuid

logger = logging.getLogger(__name__)

# Target chunk parameters (in words/tokens)
CHUNK_SIZE_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 100

_SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("RERA", re.compile(r"\b(?:rera|maharera|registration\s+number)\b", re.IGNORECASE)),
    ("Pricing", re.compile(r"\b(?:price|cost|rate|pricing|payment\s+plan|cost\s+sheet)\b", re.IGNORECASE)),
    ("Possession", re.compile(r"\b(?:possession\s+date|handover|completion\s+date)\b", re.IGNORECASE)),
    ("Floor Plan", re.compile(r"\b(?:floor\s+plan|unit\s+plan|apartment\s+layout|carpet\s+area|super\s+built[-\s]?up)\b", re.IGNORECASE)),
    ("Amenities", re.compile(r"\b(?:amenities|clubhouse|gymn|swimming\s+pool|jogging\s+track|landscap)\b", re.IGNORECASE)),
    ("Location", re.compile(r"\b(?:location|connectivity|landmark|highway|metro|railway|airport)\b", re.IGNORECASE)),
    ("Legal", re.compile(r"\b(?:legal|approval|approved\s+by|title|clearance|noc)\b", re.IGNORECASE)),
    ("Contact", re.compile(r"\b(?:contact|phone|mobile|email|website|sales\s+office)\b", re.IGNORECASE)),
    ("Master Plan", re.compile(r"\b(?:master\s+plan|site\s+layout|township|tower\s+layout)\b", re.IGNORECASE)),
    ("Configurations", re.compile(r"\b(?:configuration|bhk|unit\s+type|typology)\b", re.IGNORECASE)),
]

_TAG_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("rera", re.compile(r"\b(?:rera|maharera)\b", re.IGNORECASE)),
    ("pricing", re.compile(r"\b(?:price|cost|rate|lakh|crore|₹)\b", re.IGNORECASE)),
    ("floor_plan", re.compile(r"\b(?:floor\s+plan|carpet\s+area|super\s+built)\b", re.IGNORECASE)),
    ("amenities", re.compile(r"\b(?:amenities|clubhouse|gym|pool)\b", re.IGNORECASE)),
    ("possession", re.compile(r"\b(?:possession|handover|completion)\b", re.IGNORECASE)),
    ("location", re.compile(r"\b(?:location|landmark|highway|metro)\b", re.IGNORECASE)),
    ("legal", re.compile(r"\b(?:approval|approved|legal|noc|clearance)\b", re.IGNORECASE)),
    ("contact", re.compile(r"\b(?:contact|phone|email|website)\b", re.IGNORECASE)),
    ("master_plan", re.compile(r"\b(?:master\s+plan|township|site\s+layout)\b", re.IGNORECASE)),
    ("bhk", re.compile(r"\b\d\s*bhk\b", re.IGNORECASE)),
]

def chunk_text_pages(
    pages: list[dict],
    document_id: str,
    metadata: dict | None = None,
) -> list[dict]:
    """
    Chunk page dicts (containing text + OCR metadata) into overlapping RAG segments.
    """
    meta = metadata or {}
    chunks: list[dict] = []

    for page in pages:
        page_number = page.get("page_number", 0)
        text = (page.get("text") or "").strip()
        if not text:
            continue

        pdf_text = page.get("pdf_text") or ""
        ocr_text = page.get("ocr_text") or ""
        ocr_used = page.get("ocr_used", False)
        ocr_confidence = page.get("ocr_confidence", 1.0)

        # Classify the source type based on extraction method
        if ocr_used:
            source_type = "mixed" if pdf_text.strip() else "ocr"
        else:
            source_type = "pdf_text"

        page_chunks = _split_into_chunks(text)
        for chunk_index, chunk_text in enumerate(page_chunks):
            chunk_id = _make_chunk_id(document_id, page_number, chunk_index)
            section_title = _detect_section(chunk_text)
            tags = _extract_tags(chunk_text)
            re_details = _extract_real_estate_details(chunk_text)

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "content": chunk_text,
                    "vector": [],  # filled by embedding service
                    "metadata": {
                        "document_id": document_id,
                        "page_number": page_number,
                        "project": meta.get("project_name", meta.get("project", "")),
                        "builder": meta.get("builder", ""),
                        "document_type": meta.get("document_type", ""),
                        "source_file": meta.get("source_file", ""),
                        "chunk_index": chunk_index,
                        "section_title": section_title,
                        "source_type": source_type,
                        "ocr_used": ocr_used,
                        "ocr_confidence": ocr_confidence,
                        "tags": tags,
                        **re_details,
                    },
                }
            )

    logger.info(
        "Created %d text chunks for document_id=%s across %d pages",
        len(chunks),
        document_id,
        len(pages),
    )
    return chunks

def _split_into_chunks(text: str) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE_TOKENS, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS

    return chunks

def _detect_section(text: str) -> str:
    """Return the most likely real-estate section title for a chunk."""
    best: tuple[str, int] = ("General", 0)
    for label, pattern in _SECTION_PATTERNS:
        count = len(pattern.findall(text))
        if count > best[1]:
            best = (label, count)
    return best[0]

def _extract_tags(text: str) -> list[str]:
    """Return a list of real-estate topic tags found in a chunk."""
    return [tag for tag, pattern in _TAG_PATTERNS if pattern.search(text)]

def _extract_real_estate_details(text: str) -> dict:
    """Extract key real estate details from chunk text using regex patterns."""
    rera_pattern = re.compile(
        r"(?:RERA\s+(?:No|Reg|Registration)?[:\-#\s]*)((?:PRM|P|RER|P\-)[A-Z0-9\-/]+)", re.IGNORECASE
    )
    possession_pattern = re.compile(
        r"(?:possession|handover|completion|ready\s+to\s+move)\s*(?:by|date|expected|in)?[:\-#\s]*([A-Za-z0-9,\s\-/]{3,20})",
        re.IGNORECASE,
    )
    price_pattern = re.compile(
        r"(?:price|starting\s+at|starts\s+from|costing|price\s+range)[:\-#\s]*([A-Za-z0-9₹,\.\s\+]+(?:Lakh|Cr|Crore|Million)?)",
        re.IGNORECASE,
    )
    carpet_pattern = re.compile(
        r"(\d+(?:\.\d+)?\s*(?:sq\s*\.?\s*ft|sq\s*meters|sq\s*mt|carpet\s*area))", re.IGNORECASE
    )
    location_pattern = re.compile(
        r"(?:located\s+at|location|landmark|connectivity\s+to)[:\-#\s]*([A-Za-z0-9\s,\.\-]+(?:Street|Road|Nagar|Vihar|Phase|City|Sector)?)",
        re.IGNORECASE,
    )
    unit_pattern = re.compile(
        r"(\b\d\s*(?:BHK|Bedroom|Penthouse|Villa|Studio)\b)", re.IGNORECASE
    )

    rera_match = rera_pattern.search(text)
    possession_match = possession_pattern.search(text)
    price_match = price_pattern.search(text)
    carpet_match = carpet_pattern.search(text)
    location_match = location_pattern.search(text)
    unit_match = unit_pattern.search(text)

    return {
        "rera_number": rera_match.group(1).strip() if rera_match else "",
        "possession_date": possession_match.group(1).strip() if possession_match else "",
        "price_info": price_match.group(1).strip() if price_match else "",
        "carpet_area": carpet_match.group(1).strip() if carpet_match else "",
        "location": location_match.group(1).strip() if location_match else "",
        "unit_type": unit_match.group(1).strip() if unit_match else "",
    }

def _make_chunk_id(document_id: str, page_number: int, chunk_index: int) -> str:
    """Create a deterministic chunk ID."""
    raw = f"{document_id}_p{page_number}_c{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw))
