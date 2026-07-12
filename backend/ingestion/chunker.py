"""
Semantic Chunker for the Hybrid Multimodal RAG pipeline.
Splits page-level text into semantically cohesive chunks by analyzing the similarity
between sentence embeddings. Preserves page-level OCR metadata, headings,
and extracts real-estate-critical fields for downstream RAG retrieval.
"""

from __future__ import annotations

import logging
import re
import uuid
import math
from typing import Any
from retrieval.embeddings import get_text_embedder

logger = logging.getLogger(__name__)

# Target chunk size limits (in words)
MIN_CHUNK_WORDS = 150
MAX_CHUNK_WORDS = 800

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

def dot_product(v1: list[float], v2: list[float]) -> float:
    return sum(x * y for x, y in zip(v1, v2))

def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    n1, n2 = norm(v1), norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot_product(v1, v2) / (n1 * n2)

def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return d0 + d1

def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using standard regex boundaries."""
    # Matches . or ? or ! followed by space or newline, but ignores common abbreviations
    sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s')
    sentences = sentence_end.split(text)
    return [s.strip() for s in sentences if s.strip()]

def semantic_chunk_text(text: str, doc_id: str) -> list[str]:
    """
    Split text into semantically cohesive chunks.
    Embeds individual sentences, computes similarity between consecutive sentences,
    and splits where similarity drops below the threshold.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [sentences[0]]

    # 1. Embed sentences in a single batch
    try:
        embedder = get_text_embedder()
        embeddings = embedder.embed_batch(sentences)
    except Exception as exc:
        logger.warning("[%s] Failed to embed sentences for semantic chunking, falling back to word length splits: %s", doc_id, exc)
        return _split_fallback(text)

    # 2. Compute similarities between adjacent sentences
    similarities = []
    for i in range(len(sentences) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    # 3. Calculate threshold (e.g. 15th percentile of similarities)
    # Splits will occur at local minima of similarity
    threshold = percentile(similarities, 0.15) if similarities else 0.65
    threshold = max(0.5, min(threshold, 0.85)) # Keep threshold in a reasonable range

    chunks = []
    current_sentences = []
    current_word_count = 0

    for idx, sentence in enumerate(sentences):
        sentence_words = len(sentence.split())
        current_sentences.append(sentence)
        current_word_count += sentence_words

        # Check split conditions
        should_split = False
        
        # Condition A: Not the last sentence, and similarity is below threshold
        if idx < len(sentences) - 1:
            next_similarity = similarities[idx]
            if next_similarity < threshold and current_word_count >= MIN_CHUNK_WORDS:
                should_split = True

        # Condition B: Word count exceeds maximum hard limit
        if current_word_count >= MAX_CHUNK_WORDS:
            should_split = True

        if should_split:
            chunks.append(" ".join(current_sentences).strip())
            current_sentences = []
            current_word_count = 0

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return chunks

def _split_fallback(text: str) -> list[str]:
    """Word-length chunk split fallback."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + 400, len(words))
        chunks.append(" ".join(words[start:end]))
        start += 300
    return chunks

def chunk_text_pages(
    pages: list[dict],
    document_id: str,
    metadata: dict | None = None,
) -> list[dict]:
    """
    Chunk page dicts (containing text + OCR metadata) into overlapping semantic segments.
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

        page_chunks = semantic_chunk_text(text, document_id)
        
        for chunk_index, chunk_text in enumerate(page_chunks):
            chunk_id = _make_chunk_id(document_id, page_number, chunk_index)
            
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
                    "section_path": meta.get("section_path", ""),
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

    logger.info(
        "Created %d semantic chunks for document_id=%s across %d pages",
        len(chunks),
        document_id,
        len(pages),
    )
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
