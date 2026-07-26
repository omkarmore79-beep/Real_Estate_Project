"""
Fast Sliding-Window Chunker for the Hybrid Multimodal RAG pipeline.

IMPORTANT: The previous "semantic chunker" called the Voyage embedding API on every
individual sentence to compute cosine similarity — this caused 300-500 extra API calls
per document BEFORE the actual embedding stage, completely overwhelming the Voyage TPM
rate limit.

This version uses a fast, pure-Python sliding window with token-aware splits on
heading boundaries, producing high-quality chunks without any API calls during chunking.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Attempt to use the enhanced chunker from repository root.
_CHUNKER_ENHANCED_AVAILABLE = False
try:
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    from chunker_enhanced import chunk_text_pages_enhanced
    _CHUNKER_ENHANCED_AVAILABLE = True
except Exception:
    chunk_text_pages_enhanced = None

# Target chunk size limits (in words)
MIN_CHUNK_WORDS = 40
MAX_CHUNK_WORDS = 320
OVERLAP_WORDS = 40  # Sliding-window overlap for context preservation

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

# Heading-like line pattern — used as natural split points
_HEADING_PATTERN = re.compile(
    r"^(?:[A-Z][A-Z\s]{3,40}|[0-9]+[\.\)]\s+[A-Z][^\n]{2,60}|\s*#{1,4}\s+.{3,60})$",
    re.MULTILINE
)


def _split_on_headings(text: str) -> list[str]:
    """Split text into sections by detecting heading-like lines."""
    # Find all heading positions
    boundaries = [0]
    for m in _HEADING_PATTERN.finditer(text):
        if m.start() > boundaries[-1] + 50:  # At least 50 chars between headings
            boundaries.append(m.start())
    boundaries.append(len(text))

    sections = []
    for i in range(len(boundaries) - 1):
        section = text[boundaries[i]:boundaries[i + 1]].strip()
        if section:
            sections.append(section)
    return sections if sections else [text]


def sliding_window_chunk(text: str) -> list[str]:
    """
    Fast sliding-window chunker with heading-aware splits.
    Zero API calls — pure Python.
    """
    if not text or not text.strip():
        return []

    # First try to split on headings for natural topic boundaries
    sections = _split_on_headings(text)
    chunks = []

    for section in sections:
        words = section.split()
        if not words:
            continue

        if len(words) <= MAX_CHUNK_WORDS:
            # Section fits in one chunk — keep it whole for best coherence
            # Short safety warnings, figure explanations, and table rows are often
            # the complete answer.  Dropping them made exact manual questions
            # impossible to retrieve.
            if len(words) >= 8:
                chunks.append(section.strip())
        else:
            # Slide through long sections with overlap
            start = 0
            while start < len(words):
                end = min(start + MAX_CHUNK_WORDS, len(words))
                chunk_words = words[start:end]
                # Prefer a sentence/table-row boundary near the end. This keeps
                # specifications and warnings intact instead of splitting a value
                # from its label.
                if end < len(words):
                    for cut in range(end, max(start + MIN_CHUNK_WORDS, end - 45), -1):
                        if re.search(r"[.!?;:]$|\|$", words[cut - 1]):
                            end = cut
                            chunk_words = words[start:end]
                            break
                chunk_text = " ".join(chunk_words).strip()
                if len(chunk_words) >= 8:
                    chunks.append(chunk_text)
                start = max(end - OVERLAP_WORDS, start + 1)
                if start >= len(words):
                    break

    return chunks


def chunk_text_pages(
    pages: list[dict],
    document_id: str,
    metadata: dict | None = None,
) -> list[dict]:
    """
    Chunk page dicts into structure-aware chunks, extracting section/subsection,
    identifying figure/table references, and linking chunks in a sequential list.
    """
    if _CHUNKER_ENHANCED_AVAILABLE and chunk_text_pages_enhanced is not None:
        try:
            chunks = chunk_text_pages_enhanced(pages, document_id, metadata)
            for idx, chunk in enumerate(chunks):
                meta = chunk.get("metadata", {})
                if "prev_chunk_id" not in meta:
                    meta["prev_chunk_id"] = None
                if "next_chunk_id" not in meta:
                    meta["next_chunk_id"] = None
                chunk["metadata"] = meta
            return chunks
        except Exception as exc:
            logger.warning(
                "Enhanced chunker failed for document_id=%s; falling back. %s",
                document_id,
                exc,
            )

    meta = metadata or {}
    chunks: list[dict] = []

    current_section = "General"
    current_subsection = ""

    # Compile heading pattern for subsections/sections
    # E.g. "1.2.3 Boom Cylinders" or "Sub-section A"
    subsect_pattern = re.compile(r'^(?:[0-9]+(?:\.[0-9]+){2,}|[A-Za-z]\.)\s+([A-Z][A-Za-z\s]{2,40})')

    for page in pages:
        page_number = page.get("page_number", 0)
        text = (page.get("text") or "").strip()
        if not text:
            continue

        pdf_text = page.get("pdf_text") or ""
        ocr_text = page.get("ocr_text") or ""
        ocr_used = page.get("ocr_used", False)
        ocr_confidence = page.get("ocr_confidence", 1.0)
        page_section = page.get("section", current_section)
        
        if page_section:
            current_section = page_section

        source_type = "mixed" if (ocr_used and pdf_text.strip()) else ("ocr" if ocr_used else "pdf_text")

        # Split page text using sliding window
        page_chunks = sliding_window_chunk(text)

        for chunk_index, chunk_text in enumerate(page_chunks):
            chunk_id = _make_chunk_id(document_id, page_number, chunk_index)

            # Detect subsection change in chunk text
            for line in chunk_text.split("\n"):
                line = line.strip()
                sub_match = subsect_pattern.match(line)
                if sub_match:
                    current_subsection = line
                    break

            # Extract Figure/Table ID if referenced
            fig_match = re.search(r'\b(Figure|Fig\.?|Diagram|Drawing)\s+(\d+[-.\d]*)\b', chunk_text, re.IGNORECASE)
            figure_id = f"{fig_match.group(1)} {fig_match.group(2)}" if fig_match else ""

            table_match = re.search(r'\[Table\]|\|\s*---\s*\|', chunk_text)
            table_id = f"table_page_{page_number}_chk_{chunk_index}" if table_match else ""

            if meta.get("domain") == "excavator":
                chunk_metadata = {
                    "doc_id": meta.get("doc_id", document_id),
                    "document_id": document_id,
                    "doc_type": meta.get("doc_type", ""),
                    "title": meta.get("title", ""),
                    "source_file": meta.get("source_file", ""),
                    "revision_date": meta.get("revision_date", ""),
                    "ingested_at": meta.get("ingested_at", ""),
                    "page_number": page_number,
                    "section_path": f"{current_section} > {current_subsection}".strip(" > "),
                    "section": current_section,
                    "subsection": current_subsection,
                    "machine_model": meta.get("machine_model", "R215L"),
                    "component_tags": meta.get("component_tags", []),
                    "dtc_codes": meta.get("dtc_codes", []),
                    "supersedes_doc_id": meta.get("supersedes_doc_id", ""),
                    "confidence_weight": meta.get("confidence_weight", 1.0),
                    "domain": "excavator",
                    "chunk_index": chunk_index,
                    "source_type": source_type,
                    "ocr_used": ocr_used,
                    "ocr_confidence": ocr_confidence,
                    "figure_id": figure_id,
                    "table_id": table_id,
                    "version": meta.get("version", "1.0"),
                    "prev_chunk_id": "",  # Linked below
                    "next_chunk_id": "",  # Linked below
                }
            else:
                section_title = _detect_section(chunk_text)
                tags = _extract_tags(chunk_text)
                re_details = _extract_real_estate_details(chunk_text)
                chunk_metadata = {
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
                    "domain": meta.get("domain", "generic"),
                    "figure_id": figure_id,
                    "table_id": table_id,
                    "prev_chunk_id": "",  # Linked below
                    "next_chunk_id": "",  # Linked below
                    **re_details,
                }

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "content": chunk_text,
                    "vector": [],  # filled by embedding service
                    "metadata": chunk_metadata,
                }
            )

    # Link chunks sequentially to form a linked list (Context Traversal)
    for idx in range(len(chunks)):
        if idx > 0:
            chunks[idx]["metadata"]["prev_chunk_id"] = chunks[idx - 1]["chunk_id"]
        if idx < len(chunks) - 1:
            chunks[idx]["metadata"]["next_chunk_id"] = chunks[idx + 1]["chunk_id"]

    logger.info(
        "Created %d chunks for document_id=%s across %d pages",
        len(chunks),
        document_id,
        len(pages),
    )
    return chunks


def _detect_section(text: str) -> str:
    best: tuple[str, int] = ("General", 0)
    for label, pattern in _SECTION_PATTERNS:
        count = len(pattern.findall(text))
        if count > best[1]:
            best = (label, count)
    return best[0]


def _extract_tags(text: str) -> list[str]:
    return [tag for tag, pattern in _TAG_PATTERNS if pattern.search(text)]


def _extract_real_estate_details(text: str) -> dict:
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
    raw = f"{document_id}_p{page_number}_c{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw))
