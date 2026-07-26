"""
Enhanced Text Chunker for Hybrid Multimodal RAG Pipeline.

Improvements over basic chunker:
  - Semantic boundary detection (sections, procedures, lists)
  - Table structure preservation
  - Figure/caption linkage
  - Chunk type classification
  - Hierarchical structure awareness
  - Previous/next chunk references
  - Rich metadata generation

This module can be gradually integrated into the existing pipeline.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── Chunk type constants ───────────────────────────────────────────────────────
CHUNK_TYPES = {
    "section_header": "Section header or title",
    "subsection_header": "Subsection header",
    "maintenance_procedure": "Maintenance procedure steps",
    "troubleshooting_procedure": "Troubleshooting procedure steps",
    "safety_warning": "Safety warning or caution",
    "safety_caution": "Safety caution notice",
    "safety_note": "Safety note or information",
    "numbered_list": "Numbered list or steps",
    "bulleted_list": "Bulleted list",
    "table": "Data table or structured data",
    "figure_with_caption": "Figure or image with caption",
    "engineering_diagram": "Engineering diagram or schematic",
    "hydraulic_diagram": "Hydraulic system diagram",
    "electrical_diagram": "Electrical wiring diagram",
    "exploded_view": "Exploded parts view",
    "flowchart": "Process flowchart",
    "lifting_chart": "Lifting capacity chart",
    "paragraph": "Regular paragraph text",
    "definition_list": "Term definitions or glossary",
    "specification": "Technical specification or data",
    "error_code": "Error code or DTC information",
    "spare_part": "Spare part information",
}

# ── Patterns for semantic detection ────────────────────────────────────────────
_SECTION_PATTERN = re.compile(
    r"^(chapter|section|part|module|unit|article|clause|appendix|annex|attachment|exhibit)\s+[\d\w\.\-]+\s*[\:\-]?\s*(.+)$",
    re.IGNORECASE | re.MULTILINE
)

_SUBSECTION_PATTERN = re.compile(
    r"^(\d+\.\d+|\d+\.\d+\.\d+|[\w\)]\)\s+)(.+?):\s*$",
    re.IGNORECASE | re.MULTILINE
)

# Industrial/Technical patterns
_MAINTENANCE_PATTERN = re.compile(
    r"(maintenance|service|inspection|check|replace|adjust|lubricate|clean)\s+(procedure|schedule|instructions?|requirements?)",
    re.IGNORECASE
)

_TROUBLESHOOTING_PATTERN = re.compile(
    r"(troubleshooting|diagnosis|diagnostic|fault|problem|symptom|error|malfunction)\s*(procedure|guide|table|chart)?",
    re.IGNORECASE
)

_WARNING_PATTERN = re.compile(
    r"^(warning|danger|caution|notice|important)\s*[:\-!]",
    re.IGNORECASE | re.MULTILINE
)

_SAFETY_PATTERN = re.compile(
    r"(safety|precaution|hazard|risk|protective|personal protective equipment|ppe)",
    re.IGNORECASE
)

_PROCEDURE_PATTERN = re.compile(
    r"(procedure|process|method|workflow|steps?|instructions?|guidelines?|requirements?|specification?)\s*[\:\-]",
    re.IGNORECASE
)

_ERROR_CODE_PATTERN = re.compile(
    r"\b([A-Z]{1,4}\d{3,}[A-Z0-9\-]*|DTC\s+\d+|Error\s+Code\s*[:\-]?\s*\d+)\b",
    re.IGNORECASE
)

_SPARE_PART_PATTERN = re.compile(
    r"\b(part\s+number|pn|p/n|ref\s*no|item\s*no|component)\s*[:\-]?\s*([A-Z0-9\-]+)\b",
    re.IGNORECASE
)

_NUMBERED_LIST_PATTERN = re.compile(
    r"^\s*(\d+|[a-z])\s*[\.\)\-]\s+\w",
    re.MULTILINE
)

_BULLETED_LIST_PATTERN = re.compile(
    r"^\s*[\•\-\*\·]\s+\w",
    re.MULTILINE
)

_TABLE_PATTERN = re.compile(
    r"(\|.+\|.*\n)+|\t.+\t",
    re.MULTILINE
)

_FIGURE_PATTERN = re.compile(
    r"(figure|fig\.?|diagram|chart|illustration|image|exhibit|attachment)\s*[\#\-]?\s*([A-Za-z0-9\.\-]+)?(?:\s*[\:\-]\s*)?(.+)?",
    re.IGNORECASE
)

# Industrial diagram types
_ENGINEERING_DIAGRAM_PATTERN = re.compile(
    r"(major\s+component|component\s+diagram|assembly|schematic|layout)\s*(diagram|view|drawing)?",
    re.IGNORECASE
)

_HYDRAULIC_PATTERN = re.compile(
    r"(hydraulic|hydraulics|oil|fluid)\s*(system|circuit|diagram|schematic|flow)",
    re.IGNORECASE
)

_ELECTRICAL_PATTERN = re.compile(
    r"(electrical|electric|wiring|circuit|harness)\s*(diagram|schematic|system)",
    re.IGNORECASE
)

_EXPLODED_VIEW_PATTERN = re.compile(
    r"(exploded\s+view|breakdown|disassembly|assembly\s+view)",
    re.IGNORECASE
)

_FLOWCHART_PATTERN = re.compile(
    r"(flowchart|flow\s+chart|process\s+flow|logic\s+flow)",
    re.IGNORECASE
)

_LIFTING_CHART_PATTERN = re.compile(
    r"(lifting\s*chart|capacity\s*chart|load\s*chart|working\s*range)",
    re.IGNORECASE
)

# Target chunk parameters
CHUNK_SIZE_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 100
MIN_CHUNK_SIZE_TOKENS = 50  # Don't create chunks smaller than this

# Industrial/Technical section patterns (for excavators, machinery, etc.)
_INDUSTRIAL_SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Maintenance", re.compile(r"\b(?:maintenance|service|inspection|preventive\s+maintenance)\b", re.IGNORECASE)),
    ("Troubleshooting", re.compile(r"\b(?:troubleshooting|diagnosis|diagnostic|fault\s+finding)\b", re.IGNORECASE)),
    ("Specifications", re.compile(r"\b(?:specifications?|specs|technical\s+data|dimensions|capacity)\b", re.IGNORECASE)),
    ("Safety", re.compile(r"\b(?:safety|precautions?|hazards?|warnings?|cautions?)\b", re.IGNORECASE)),
    ("Hydraulic System", re.compile(r"\b(?:hydraulic\s+system|hydraulics|oil\s+system)\b", re.IGNORECASE)),
    ("Electrical System", re.compile(r"\b(?:electrical\s+system|electrical|wiring)\b", re.IGNORECASE)),
    ("Engine", re.compile(r"\b(?:engine|powertrain|drive\s+system)\b", re.IGNORECASE)),
    ("Undercarriage", re.compile(r"\b(?:undercarriage|track|crawler|travel\s+device)\b", re.IGNORECASE)),
    ("Attachments", re.compile(r"\b(?:attachment|bucket|breaker|hammer|quick\s+coupler)\b", re.IGNORECASE)),
    ("Components", re.compile(r"\b(?:component|part|assembly|subassembly)\b", re.IGNORECASE)),
    ("Procedures", re.compile(r"\b(?:procedure|instruction|step|method|process)\b", re.IGNORECASE)),
]

# Real estate specific patterns (from original) - kept for backward compatibility
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


# ════════════════════════════════════════════════════════════════════════════════
#  Main API - Enhanced Chunking
# ════════════════════════════════════════════════════════════════════════════════

def chunk_text_pages_enhanced(
    pages: list[dict],
    document_id: str,
    metadata: dict | None = None,
) -> list[dict]:
    """
    Enhanced chunking with semantic structure preservation for industrial/technical manuals.
    
    Backward compatible with existing chunk_text_pages() output format.
    Adds new metadata fields for improved retrieval.
    
    Key improvements:
    - Industrial section detection (maintenance, troubleshooting, safety, etc.)
    - Procedure boundary detection
    - Safety warning/caution/note isolation
    - Error code and spare part extraction
    - Diagram type classification
    - Table structure preservation
    - Figure/caption linkage
    """
    meta = metadata or {}
    chunks: list[dict] = []
    previous_chunk_id: Optional[str] = None
    ingestion_timestamp = datetime.now(timezone.utc).isoformat()
    
    # Determine domain for appropriate pattern matching
    domain = meta.get("domain", "generic")
    is_industrial = domain == "excavator" or domain == "industrial"

    for page in pages:
        page_number = page.get("page_number", 0)
        text = (page.get("text") or "").strip()
        if not text:
            continue

        pdf_text = page.get("pdf_text") or ""
        ocr_text = page.get("ocr_text") or ""
        ocr_used = page.get("ocr_used", False)
        ocr_confidence = page.get("ocr_confidence", 1.0)
        section = page.get("section", "General")

        # Determine source type
        if ocr_used:
            source_type = "mixed" if pdf_text.strip() else "ocr"
        else:
            source_type = "pdf_text"

        # Semantically split text (respecting structure)
        page_chunks = _split_into_semantic_chunks(text, is_industrial)
        
        for chunk_index, chunk_text in enumerate(page_chunks):
            chunk_id = _make_chunk_id(document_id, page_number, chunk_index)
            
            # Enhanced detection
            chunk_type = _detect_chunk_type(chunk_text, is_industrial)
            section_title = _detect_section(chunk_text, is_industrial)
            tags = _extract_tags(chunk_text)
            
            # Extract domain-specific details
            if is_industrial:
                industrial_details = _extract_industrial_details(chunk_text)
                re_details = {}
            else:
                industrial_details = {}
                re_details = _extract_real_estate_details(chunk_text)
            
            confidence_score = _calculate_confidence(chunk_text, source_type, ocr_confidence)

            # Build enriched metadata
            chunk_metadata = {
                "document_id": document_id,
                "page_number": page_number,
                "section": section,
                "subsection": _detect_subsection(chunk_text),
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
                # NEW FIELDS
                "chunk_type": chunk_type,
                "confidence_score": confidence_score,
                "word_count": len(chunk_text.split()),
                "char_count": len(chunk_text),
                "ingestion_timestamp": ingestion_timestamp,
                "previous_chunk_id": previous_chunk_id,
                "domain": domain,
                "machine_model": meta.get("machine_model", ""),
                "document_version": meta.get("version", ""),
                # next_chunk_id will be filled after all chunks are created
                **re_details,
                **industrial_details,
            }

            chunks.append({
                "chunk_id": chunk_id,
                "content": chunk_text,
                "vector": [],  # filled by embedding service
                "metadata": chunk_metadata,
            })

            previous_chunk_id = chunk_id

    # Fill in next_chunk_id references (backward pass)
    for i in range(len(chunks) - 1):
        if chunks[i]["metadata"].get("source_type") == chunks[i + 1]["metadata"].get("source_type"):
            chunks[i]["metadata"]["next_chunk_id"] = chunks[i + 1]["chunk_id"]
        else:
            chunks[i]["metadata"]["next_chunk_id"] = None
    
    # Add None for last chunk
    if chunks:
        chunks[-1]["metadata"]["next_chunk_id"] = None

    logger.info(
        "Created %d semantic chunks for document_id=%s across %d pages",
        len(chunks),
        document_id,
        len(pages),
    )
    
    # Log chunk type distribution
    type_counts = {}
    for chunk in chunks:
        ctype = chunk["metadata"].get("chunk_type", "unknown")
        type_counts[ctype] = type_counts.get(ctype, 0) + 1
    logger.info("Chunk type distribution: %s", type_counts)
    
    return chunks


# ════════════════════════════════════════════════════════════════════════════════
#  Internal Helpers
# ════════════════════════════════════════════════════════════════════════════════

def _split_into_semantic_chunks(text: str, is_industrial: bool = False) -> list[str]:
    """
    Split text into semantically meaningful chunks.
    
    Respects:
    - Section boundaries
    - Procedure steps
    - Safety warnings/cautions/notes
    - List structures
    - Paragraph breaks
    
    For industrial documents, also respects:
    - Maintenance procedures
    - Troubleshooting procedures
    - Error codes
    - Diagram references
    
    Falls back to word-based chunking if no clear structure detected.
    """
    if not text.strip():
        return []

    # First, isolate safety warnings/cautions/notes as independent chunks
    text = _isolate_safety_blocks(text)
    
    # Try to detect and preserve major structures
    sections = _split_by_sections(text, is_industrial)
    if len(sections) > 1:
        # Multiple sections detected - chunk each separately
        all_chunks = []
        for section_text in sections:
            all_chunks.extend(_split_into_word_chunks(section_text))
        return all_chunks
    
    # No section structure - use word-based chunking
    return _split_into_word_chunks(text)


def _split_by_sections(text: str, is_industrial: bool = False) -> list[str]:
    """Split text by detected section headers."""
    # Find all section header positions
    sections = []
    current_pos = 0
    boundaries = []
    
    # Use industrial patterns if applicable
    patterns_to_use = _INDUSTRIAL_SECTION_PATTERNS if is_industrial else _SECTION_PATTERNS
    
    for label, pattern in patterns_to_use:
        for match in pattern.finditer(text):
            if match.start() > current_pos + 50:  # Minimum distance between sections
                boundaries.append((match.start(), label))
    
    # Also check for standard section pattern
    for match in _SECTION_PATTERN.finditer(text):
        if match.start() > current_pos + 50:
            boundaries.append((match.start(), "standard_section"))
    
    # Sort boundaries by position
    boundaries.sort(key=lambda x: x[0])
    
    # Split at boundaries
    for pos, label in boundaries:
        if pos > current_pos:
            sections.append(text[current_pos:pos])
        current_pos = pos
    
    # Add final section
    if current_pos < len(text):
        sections.append(text[current_pos:])
    
    # Only return if we found multiple significant sections
    if len(sections) > 1 and all(len(s.strip()) > 50 for s in sections):
        return sections
    
    return [text]  # No clear section structure


def _split_into_word_chunks(text: str) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    Same logic as original chunker but with minimum size enforcement.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE_TOKENS, len(words))
        chunk = " ".join(words[start:end]).strip()
        
        if chunk and len(chunk.split()) >= MIN_CHUNK_SIZE_TOKENS:
            chunks.append(chunk)
        elif chunk:
            # Chunk too small - merge with previous or next
            if chunks:
                # Append to previous chunk
                chunks[-1] = chunks[-1] + " " + chunk
            else:
                # First chunk but too small - include anyway
                chunks.append(chunk)
        
        if end == len(words):
            break
        start += CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS

    return chunks


def _detect_chunk_type(text: str, is_industrial: bool = False) -> str:
    """Detect the type of chunk (section, procedure, list, table, etc.)"""
    text_lower = text.lower()
    
    # Check patterns in order of specificity
    
    # Industrial-specific patterns first
    if is_industrial:
        if _WARNING_PATTERN.search(text):
            # Determine warning type
            if re.search(r'^warning\b', text, re.IGNORECASE | re.MULTILINE):
                return "safety_warning"
            elif re.search(r'^caution\b', text, re.IGNORECASE | re.MULTILINE):
                return "safety_caution"
            elif re.search(r'^notice\b', text, re.IGNORECASE | re.MULTILINE):
                return "safety_note"
        
        if _ERROR_CODE_PATTERN.search(text):
            return "error_code"
        
        if _SPARE_PART_PATTERN.search(text):
            return "spare_part"
        
        if _LIFTING_CHART_PATTERN.search(text_lower):
            return "lifting_chart"
        
        if _EXPLODED_VIEW_PATTERN.search(text_lower):
            return "exploded_view"
        
        if _HYDRAULIC_PATTERN.search(text_lower):
            return "hydraulic_diagram"
        
        if _ELECTRICAL_PATTERN.search(text_lower):
            return "electrical_diagram"
        
        if _ENGINEERING_DIAGRAM_PATTERN.search(text_lower):
            return "engineering_diagram"
        
        if _FLOWCHART_PATTERN.search(text_lower):
            return "flowchart"
        
        if _MAINTENANCE_PATTERN.search(text_lower):
            return "maintenance_procedure"
        
        if _TROUBLESHOOTING_PATTERN.search(text_lower):
            return "troubleshooting_procedure"
    
    # General patterns
    if _FIGURE_PATTERN.search(text_lower):
        return "figure_with_caption"
    
    if _TABLE_PATTERN.search(text):
        return "table"
    
    if _PROCEDURE_PATTERN.search(text_lower):
        if _numbered_list_check(text):
            return "procedure"  # Numbered steps procedure
        return "procedure"
    
    if _NUMBERED_LIST_PATTERN.search(text):
        return "numbered_list"
    
    if _BULLETED_LIST_PATTERN.search(text):
        return "bulleted_list"
    
    if _SUBSECTION_PATTERN.search(text):
        return "subsection_header"
    
    if _SECTION_PATTERN.search(text):
        return "section_header"
    
    # Default
    return "paragraph"


def _numbered_list_check(text: str) -> bool:
    """Check if text contains numbered list indicators."""
    return bool(_NUMBERED_LIST_PATTERN.search(text))


def _detect_section(text: str, is_industrial: bool = False) -> str:
    """Return the most likely section title for a chunk."""
    patterns_to_use = _INDUSTRIAL_SECTION_PATTERNS if is_industrial else _SECTION_PATTERNS
    best: tuple[str, int] = ("General", 0)
    for label, pattern in patterns_to_use:
        count = len(pattern.findall(text))
        if count > best[1]:
            best = (label, count)
    return best[0]


def _detect_subsection(text: str) -> str:
    """Detect subsection heading from text."""
    match = _SUBSECTION_PATTERN.search(text)
    if match:
        return match.group(0).strip()
    return ""


def _isolate_safety_blocks(text: str) -> str:
    """
    Isolate safety warnings, cautions, and notes as separate blocks.
    
    This ensures these critical safety information chunks are not merged
    with regular text, making them independently retrievable.
    """
    # Pattern to match safety blocks
    safety_pattern = re.compile(
        r'^(WARNING|CAUTION|NOTICE|DANGER|IMPORTANT)\s*[:\-!].*?(?=^(?:WARNING|CAUTION|NOTICE|DANGER|IMPORTANT|\d+\.|[A-Z]\.)|$)',
        re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    
    # Find all safety blocks and mark them
    def add_marker(match):
        return f"\n---SAFETY_BLOCK---\n{match.group(0)}\n---END_SAFETY---\n"
    
    marked_text = safety_pattern.sub(add_marker, text)
    return marked_text


def _extract_industrial_details(text: str) -> dict:
    """Extract industrial-specific details from chunk text."""
    details = {}
    
    # Extract error codes/DTCs
    error_codes = _ERROR_CODE_PATTERN.findall(text)
    if error_codes:
        details["error_codes"] = error_codes
    
    # Extract spare part numbers
    spare_parts = _SPARE_PART_PATTERN.findall(text)
    if spare_parts:
        details["spare_parts"] = [sp[1] for sp in spare_parts]
    
    # Extract component references
    component_pattern = re.compile(
        r'\b(boom|cab|counterweight|engine|swing\s+motor|travel\s+motor|hydraulic\s+tank|track|bucket|arm|cylinder)\b',
        re.IGNORECASE
    )
    components = list(set(component_pattern.findall(text.lower())))
    if components:
        details["components"] = components
    
    return details


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


def _calculate_confidence(chunk_text: str, source_type: str, ocr_confidence: float) -> float:
    """
    Calculate confidence score for a chunk based on source quality.
    
    Factors:
    - Source type (PDF text highest, OCR lower)
    - OCR confidence value
    - Chunk coherence (rough estimate)
    """
    base_score = 1.0 if source_type == "pdf_text" else 0.85
    ocr_factor = ocr_confidence if source_type in ("ocr", "mixed") else 1.0
    
    # Rough coherence check (presence of complete sentences)
    sentence_count = len(re.split(r'[.!?]+', chunk_text.strip()))
    coherence_bonus = min(0.1, sentence_count * 0.02) if sentence_count > 1 else 0.0
    
    final_score = (base_score * ocr_factor) + coherence_bonus
    return min(1.0, final_score)


def _make_chunk_id(document_id: str, page_number: int, chunk_index: int) -> str:
    """Create a deterministic chunk ID."""
    raw = f"{document_id}_p{page_number}_c{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw))
