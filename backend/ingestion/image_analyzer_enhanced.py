"""
Enhanced Image Analyzer for Multimodal RAG.

Improvements over basic image processor:
  - OCR text extraction from images
  - Structured image metadata generation
  - Multimodal description building for embedding
  - Figure/caption parsing
  - Confidence scoring
  - Diagram-specific handling

Integrates with existing image extraction pipeline without breaking changes.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional
from datetime import datetime, timezone

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
    "diagram",
    "chart",
    # Industrial/Technical types
    "engineering_diagram",
    "exploded_view",
    "hydraulic_diagram",
    "electrical_diagram",
    "flowchart",
    "lifting_chart",
    "maintenance_image",
    "troubleshooting_image",
    "warning_label",
    "icon",
    "photograph",
    "graph",
    "other",
)

# ── Classification rules with confidence weights ─────────────────────────────
_CLASSIFICATION_RULES: list[tuple[str, tuple[str, ...], tuple[str, ...], int]] = [
    # Real estate types
    (
        "floor_plan",
        ("floor plan", "unit plan", "apartment layout", "carpet area", "super built-up"),
        ("bedroom", "kitchen", "balcony", "bhk", "living", "toilet", "dimensions"),
        3,  # confidence weight
    ),
    (
        "master_plan",
        ("master plan", "master layout plan", "site layout", "township", "tower layout"),
        ("tower", "towers", "internal roads", "landscape", "entry", "podium"),
        3,
    ),
    (
        "location_map",
        ("location plan", "location map", "connectivity", "nearby landmarks", "map not to scale"),
        ("airport", "metro", "railway", "hospital", "school", "highway", "landmark", "distance"),
        3,
    ),
    (
        "amenity",
        ("amenities", "clubhouse", "gymnasium", "swimming pool", "jogging track", "indoor games"),
        ("gym", "pool", "kids", "play", "jogging", "games", "garden", "sports"),
        2,
    ),
    (
        "exterior",
        ("elevation", "building view", "tower view", "exterior", "project view"),
        ("elevation", "exterior", "render", "view", "facade"),
        2,
    ),
    (
        "interior",
        ("interior", "inside view", "room view", "lobby", "living space"),
        ("interior", "room", "lobby", "lounge"),
        2,
    ),
    (
        "diagram",
        ("diagram", "schematic", "flowchart", "hydraulic", "electrical", "circuit"),
        ("flow", "connection", "link", "component", "system", "circuit", "line"),
        3,
    ),
    (
        "chart",
        ("chart", "graph", "plot", "curve", "trend", "comparison"),
        ("data", "value", "percentage", "trend", "axis", "scale"),
        2,
    ),
    (
        "table",
        ("payment plan", "payment chart", "price chart", "cost sheet", "schedule", "matrix"),
        ("payment", "installment", "schedule", "amount", "milestone", "row", "column"),
        2,
    ),
    (
        "logo",
        ("logo", "brand", "emblem"),
        ("logo", "brand"),
        1,
    ),
    # Industrial/Technical types
    (
        "engineering_diagram",
        ("major component", "component diagram", "assembly diagram", "schematic layout"),
        ("boom", "cab", "counterweight", "engine", "motor", "tank", "cylinder", "arm"),
        3,
    ),
    (
        "exploded_view",
        ("exploded view", "breakdown", "disassembly", "assembly view", "parts breakdown"),
        ("part", "component", "assembly", "subassembly", "reference number"),
        3,
    ),
    (
        "hydraulic_diagram",
        ("hydraulic system", "hydraulic circuit", "hydraulic schematic", "oil flow"),
        ("pump", "valve", "cylinder", "motor", "line", "hose", "pressure", "flow"),
        3,
    ),
    (
        "electrical_diagram",
        ("electrical system", "wiring diagram", "electrical schematic", "circuit diagram"),
        ("wire", "harness", "connector", "relay", "fuse", "switch", "ground", "power"),
        3,
    ),
    (
        "flowchart",
        ("flowchart", "process flow", "logic flow", "workflow", "decision tree"),
        ("decision", "process", "step", "action", "condition", "branch"),
        2,
    ),
    (
        "lifting_chart",
        ("lifting chart", "capacity chart", "load chart", "working range", "lifting capacity"),
        ("reach", "radius", "height", "capacity", "weight", "load", "boom length"),
        3,
    ),
    (
        "maintenance_image",
        ("maintenance", "service", "inspection", "check", "replace", "adjust"),
        ("procedure", "step", "instruction", "specification", "torque", "clearance"),
        2,
    ),
    (
        "troubleshooting_image",
        ("troubleshooting", "diagnosis", "diagnostic", "fault", "problem", "symptom"),
        ("error", "code", "dtc", "malfunction", "failure", "test"),
        2,
    ),
    (
        "warning_label",
        ("warning", "danger", "caution", "notice", "important", "alert"),
        ("safety", "hazard", "risk", "protective", "ppe", "precaution"),
        3,
    ),
]

# ── Filename-based hints ───────────────────────────────────────────────────────
_FILENAME_HINTS: list[tuple[str, re.Pattern]] = [
    ("floor_plan", re.compile(r"floor|unit|layout|bhk|plan", re.IGNORECASE)),
    ("master_plan", re.compile(r"master|site|township", re.IGNORECASE)),
    ("location_map", re.compile(r"location|map|connectivity|route", re.IGNORECASE)),
    ("amenity", re.compile(r"amenity|amenities|clubhouse|gym|pool|recreation", re.IGNORECASE)),
    ("exterior", re.compile(r"elevation|exterior|view|render|facade|building", re.IGNORECASE)),
    ("interior", re.compile(r"interior|room|lobby|inside", re.IGNORECASE)),
    ("diagram", re.compile(r"diagram|schematic|circuit|electrical|hydraulic", re.IGNORECASE)),
    ("chart", re.compile(r"chart|graph|plot|data", re.IGNORECASE)),
    ("table", re.compile(r"payment|price|chart|table|schedule|matrix", re.IGNORECASE)),
    ("logo", re.compile(r"logo|brand|emblem", re.IGNORECASE)),
    # Industrial filename hints
    ("engineering_diagram", re.compile(r"component|assembly|schematic|layout", re.IGNORECASE)),
    ("exploded_view", re.compile(r"exploded|breakdown|disassembly|parts", re.IGNORECASE)),
    ("hydraulic_diagram", re.compile(r"hydraulic|oil|fluid|circuit", re.IGNORECASE)),
    ("electrical_diagram", re.compile(r"electrical|wiring|harness|circuit", re.IGNORECASE)),
    ("flowchart", re.compile(r"flowchart|flow|process|logic", re.IGNORECASE)),
    ("lifting_chart", re.compile(r"lifting|capacity|load|range|chart", re.IGNORECASE)),
    ("maintenance_image", re.compile(r"maintenance|service|inspection|procedure", re.IGNORECASE)),
    ("troubleshooting_image", re.compile(r"troubleshoot|diagnostic|fault|error", re.IGNORECASE)),
    ("warning_label", re.compile(r"warning|danger|caution|safety|hazard", re.IGNORECASE)),
]

# ── Pattern detection for structured extraction ─────────────────────────────────
_FIGURE_NUMBER_PATTERN = re.compile(
    r"(?:figure|fig\.?|diagram|img\.?|image|exhibit|attachment|plate|chart)\s*[\#\-]?\s*([A-Za-z0-9\.\-_]+)?",
    re.IGNORECASE
)

_CAPTION_PATTERN = re.compile(
    r"(?:^|[\.:\-])\s*(figure|fig\.?|source|caption|description|note|illustration)\s*[\#\-]?\s*([A-Za-z0-9\.\-_]*)[\s\:]*(.+?)(?=[\.!?]|$)",
    re.IGNORECASE | re.MULTILINE
)

_TABLE_PATTERN = re.compile(
    r"(\|.+\|.*\n)+|(?:^\s*\|.+\|$\n?)+",
    re.MULTILINE
)


# ════════════════════════════════════════════════════════════════════════════════
#  Main API - Enhanced Image Processing
# ════════════════════════════════════════════════════════════════════════════════

def process_images_enhanced(
    pages: list[dict],
    document_id: str,
    source_file: str,
    metadata: dict | None = None,
) -> list[dict]:
    """
    Enhanced image processing with multimodal descriptions.
    
    For every image:
      1. Classify type with confidence scoring (industrial-aware)
      2. Extract OCR text (if available)
      3. Parse caption and figure number
      4. Extract nearby section/page context
      5. Generate multimodal description text
      6. Build rich metadata record with parent-child relationships
    
    Fully backward compatible with existing image records.
    Adds new fields without removing old ones.
    """
    meta = metadata or {}
    records: list[dict] = []
    ingestion_timestamp = datetime.now(timezone.utc).isoformat()
    
    # Determine domain for appropriate classification
    domain = meta.get("domain", "generic")
    is_industrial = domain == "excavator" or domain == "industrial"

    for page in pages:
        page_number = page.get("page_number", 0)
        page_text = (page.get("text") or "").strip()
        ocr_text = (page.get("ocr_text") or "").strip()
        section = page.get("section", "General")

        for img_index, img in enumerate(page.get("images", [])):
            image_id = img.get("image_id") or f"page_{page_number}"
            image_path = img.get("image_path", "")
            local_path = img.get("local_path")
            image_url = image_path

            # Enhanced classification with confidence (industrial-aware)
            image_type, type_confidence = classify_image_type_enhanced(
                page_text, image_id, is_industrial
            )
            
            # Extract figure number and caption
            figure_number = extract_figure_number(page_text)
            caption = generate_caption_enhanced(
                image_type, page_text, page_number, figure_number, is_industrial
            )
            
            # Extract OCR text if available
            ocr_from_image = extract_ocr_from_image_context(page_text, image_id)
            
            # Build multimodal description
            multimodal_description = build_multimodal_description(
                image_type=image_type,
                caption=caption,
                figure_number=figure_number,
                page_text=page_text,
                ocr_text=ocr_text,
                ocr_from_image=ocr_from_image,
                page_number=page_number,
                section_context=section or extract_section_context(page_text),
                nearby_paragraph=extract_nearby_paragraph(page_text, image_id),
                is_industrial=is_industrial,
            )
            
            # Calculate confidence for this image record
            image_confidence = calculate_image_confidence(
                type_confidence=type_confidence,
                has_caption=bool(caption and len(caption) > 10),
                has_ocr=bool(ocr_from_image or ocr_text),
                nearby_context=bool(page_text and len(page_text) > 100),
            )

            # Build parent-child metadata
            parent_chunk_id = f"{document_id}_page_{page_number}"
            
            # Build the image record
            record = {
                "image_id": image_id,
                "document_id": document_id,
                "page_number": page_number,
                "page": page_number,  # legacy compatibility
                "image_path": image_path,
                "image_url": image_url,
                "local_path": local_path,
                "nearby_page_text": page_text[:1000],
                "nearby_text": page_text[:1000],  # legacy field
                "ocr_context": ocr_text[:1000],
                "source_file": source_file,
                "image_type": image_type,
                "caption": caption,
                "vector": [],  # filled by embedding service
                # ENHANCED FIELDS
                "multimodal_description": multimodal_description,
                "figure_number": figure_number,
                "type_confidence": type_confidence,
                "image_confidence": image_confidence,
                "ocr_from_image": ocr_from_image,
                "section_title": section or extract_section_context(page_text),
                "ingestion_timestamp": ingestion_timestamp,
                "nearby_paragraph": extract_nearby_paragraph(page_text, image_id),
                # Parent-child relationships
                "parent_chunk_id": parent_chunk_id,
                "parent_section": section or extract_section_context(page_text),
                "parent_page": page_number,
                "parent_document": document_id,
                "domain": domain,
                "machine_model": meta.get("machine_model", ""),
                # Payload for Qdrant
                "metadata": {
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
                    "nearby_text": page_text[:500],
                    "ocr_context": ocr_text[:500],
                    # NEW: multimodal description for semantic search
                    "multimodal_description": multimodal_description,
                    "figure_number": figure_number,
                    "type_confidence": type_confidence,
                    "image_confidence": image_confidence,
                    # Parent-child relationships
                    "parent_chunk_id": parent_chunk_id,
                    "parent_section": section or extract_section_context(page_text),
                    "parent_page": page_number,
                    "parent_document": document_id,
                    "domain": domain,
                    "machine_model": meta.get("machine_model", ""),
                },
            }
            
            records.append(record)

    logger.info(
        "Enhanced processing: %d image records for document_id=%s",
        len(records), document_id
    )
    
    # Log type distribution with confidence
    type_dist = {}
    for record in records:
        itype = record.get("image_type", "unknown")
        conf = record.get("type_confidence", 0)
        if itype not in type_dist:
            type_dist[itype] = []
        type_dist[itype].append(conf)
    
    for itype, confs in type_dist.items():
        avg_conf = sum(confs) / len(confs) if confs else 0
        logger.info(f"  Image type '{itype}': {len(confs)} images, avg confidence {avg_conf:.2f}")
    
    return records


def build_image_embedding_text_enhanced(record: dict) -> str:
    """
    Build a rich text string for image embedding using multimodal description.
    
    This text will be embedded and used for semantic search.
    Combines all available context about the image into a single coherent description.
    """
    parts = []
    
    # Priority order: structured description > caption > type > context
    if record.get("multimodal_description"):
        parts.append(record["multimodal_description"])
    
    if record.get("caption") and not record.get("multimodal_description"):
        parts.append(record["caption"])
    
    # Add image type and confidence
    image_type = record.get("image_type", "").replace("_", " ")
    if image_type:
        confidence = record.get("type_confidence", 0)
        parts.append(f"Image type: {image_type} (confidence: {confidence:.2f})")
    
    # Add nearby context if available and space permits
    nearby = (record.get("nearby_page_text") or "")[:400]
    if nearby and len(" ".join(parts)) < 800:
        parts.append(f"Context: {nearby}")
    
    # Add OCR if present
    if record.get("ocr_from_image"):
        parts.append(f"Text in image: {record['ocr_from_image'][:200]}")
    
    combined = " ".join(p for p in parts if p).strip()
    return combined if combined else "Document image"


# ════════════════════════════════════════════════════════════════════════════════
#  Classification and Detection
# ════════════════════════════════════════════════════════════════════════════════

def classify_image_type_enhanced(
    page_text: str, 
    filename: str = "",
    is_industrial: bool = False
) -> tuple[str, float]:
    """
    Classify image type with confidence score.
    
    Args:
        page_text: Text surrounding the image
        filename: Image filename for hints
        is_industrial: Whether to prioritize industrial classification
    
    Returns:
        (image_type, confidence_score)
    """
    # 1. Try filename hints first (highest confidence)
    for image_type, pattern in _FILENAME_HINTS:
        if pattern.search(filename):
            return image_type, 0.95
    
    if not page_text:
        return "other", 0.5
    
    haystack = page_text.lower()
    scores: dict[str, tuple[float, int]] = {}  # (confidence, count)
    
    # Filter classification rules based on domain
    rules_to_use = _CLASSIFICATION_RULES
    if is_industrial:
        # Prioritize industrial types by filtering to industrial types first
        industrial_types = {
            "engineering_diagram", "exploded_view", "hydraulic_diagram", 
            "electrical_diagram", "flowchart", "lifting_chart",
            "maintenance_image", "troubleshooting_image", "warning_label"
        }
        industrial_rules = [r for r in _CLASSIFICATION_RULES if r[0] in industrial_types]
        general_rules = [r for r in _CLASSIFICATION_RULES if r[0] not in industrial_types]
        rules_to_use = industrial_rules + general_rules
    
    for image_type, phrases, keywords, weight in rules_to_use:
        phrase_matches = sum(weight for phrase in phrases if phrase in haystack)
        keyword_matches = sum(1 for kw in keywords if kw in haystack)
        
        if phrase_matches > 0 or keyword_matches > 0:
            # Confidence increases with more matches
            # Max 0.9 to leave room for filename-based scores
            raw_score = min(0.9, 0.5 + (phrase_matches + keyword_matches) * 0.15)
            scores[image_type] = (raw_score, phrase_matches + keyword_matches)
    
    if scores:
        best_type, (best_score, _) = max(scores.items(), key=lambda x: x[1][0])
        return best_type, best_score
    
    return "other", 0.3


def extract_figure_number(page_text: str) -> Optional[str]:
    """Extract figure number from page text (e.g., 'Figure 3.2', 'Fig. A1')."""
    match = _FIGURE_NUMBER_PATTERN.search(page_text)
    if match and match.group(1):
        return match.group(1).strip()
    return None


def extract_section_context(page_text: str) -> Optional[str]:
    """Extract section heading from page text."""
    section_pattern = re.compile(
        r"^(section|chapter|part|module|unit)\s+[\d\w\.\-]+\s*[\:\-]?\s*(.+)$",
        re.IGNORECASE | re.MULTILINE
    )
    match = section_pattern.search(page_text)
    if match:
        return match.group(2).strip()
    return None


def extract_nearby_paragraph(page_text: str, image_id: str = "") -> Optional[str]:
    """
    Extract a nearby paragraph of context text.
    This helps with semantic search by providing surrounding text.
    """
    if not page_text or len(page_text) < 100:
        return None
    
    # Get first meaningful paragraph
    paragraphs = page_text.split("\n\n")
    for para in paragraphs:
        if len(para.strip()) > 50:  # Meaningful paragraph
            return para.strip()[:500]
    
    return None


def extract_ocr_from_image_context(page_text: str, image_id: str = "") -> Optional[str]:
    """
    Extract OCR text hints from page context.
    
    In production, this would come from actual image OCR.
    For now, uses heuristics to find text likely within the image.
    """
    # This is a simplified version - in production you'd run OCR on actual image file
    # For now, we detect numbered/bulleted content which often appears in diagrams/charts
    
    ocr_indicators = []
    
    # Find bracketed content (often labels in diagrams)
    bracketed = re.findall(r"\[([^\]]+)\]", page_text)
    ocr_indicators.extend(bracketed[:3])
    
    # Find quoted text (often captions)
    quoted = re.findall(r'"([^"]+)"', page_text)
    ocr_indicators.extend(quoted[:2])
    
    # Find numbered items (often legend/key)
    numbered = re.findall(r"^\s*\d+\.\s+(.+)$", page_text, re.MULTILINE)
    ocr_indicators.extend(numbered[:3])
    
    # Find part numbers (industrial)
    part_numbers = re.findall(r"\b[A-Z]{1,4}\d{3,}[A-Z0-9\-]*\b", page_text)
    ocr_indicators.extend(part_numbers[:3])
    
    if ocr_indicators:
        return " ".join(ocr_indicators[:5])[:200]
    
    return None


def extract_components_from_text(page_text: str) -> list[str]:
    """
    Extract component names from text for industrial diagrams.
    
    Common excavator/machinery components.
    """
    component_patterns = [
        r"\b(boom|cab|counterweight|engine|swing\s+motor|travel\s+motor|hydraulic\s+tank|track|bucket|arm|cylinder|pump|valve|harness|relay|fuse)\b",
    ]
    
    components = set()
    for pattern in component_patterns:
        matches = re.findall(pattern, page_text, re.IGNORECASE)
        components.update(m.lower() for m in matches)
    
    return sorted(list(components))


# ════════════════════════════════════════════════════════════════════════════════
#  Description Building
# ════════════════════════════════════════════════════════════════════════════════

def build_multimodal_description(
    image_type: str,
    caption: str,
    figure_number: Optional[str],
    page_text: str,
    ocr_text: str,
    ocr_from_image: Optional[str],
    page_number: int,
    section_context: Optional[str],
    nearby_paragraph: Optional[str],
    is_industrial: bool = False,
) -> str:
    """
    Build a comprehensive multimodal description combining all available context.
    
    This single text will be embedded for semantic search and should contain:
    - What the image is (caption + type)
    - Any text visible in it (OCR)
    - Why it matters (section + nearby paragraph)
    - Where it is (page, figure number)
    
    For industrial diagrams, includes component-specific descriptions.
    """
    parts: list[str] = []
    
    # 1. Caption and figure number
    if figure_number:
        parts.append(f"Figure {figure_number}:")
    if caption and caption != "Document image":
        parts.append(caption)
    
    # 2. Image type description (industrial-aware)
    type_description = _describe_image_type(image_type, is_industrial)
    if type_description:
        parts.append(type_description)
    
    # 3. For industrial diagrams, extract component information
    if is_industrial and image_type in ("engineering_diagram", "exploded_view", "hydraulic_diagram", "electrical_diagram"):
        components = extract_components_from_text(page_text)
        if components:
            parts.append(f"Components: {', '.join(components[:8])}")
    
    # 4. Section context (why this image matters)
    if section_context:
        parts.append(f"In section: {section_context}")
    
    # 5. Nearby paragraph (surrounding context)
    if nearby_paragraph and len(nearby_paragraph) > 50:
        first_sentence = nearby_paragraph.split(".")[0][:150]
        if first_sentence:
            parts.append(f"Context: {first_sentence}")
    
    # 6. OCR text from image (if available)
    if ocr_from_image:
        parts.append(f"Contains text: {ocr_from_image[:100]}")
    
    # 7. Location metadata
    parts.append(f"Page {page_number}")
    
    # Combine all parts
    description = ". ".join(p.strip() for p in parts if p and p.strip())
    
    # Limit length for embedding
    return description[:1000] if description else "Document image"


def _describe_image_type(image_type: str, is_industrial: bool = False) -> str:
    """Get a human-readable description of image type."""
    descriptions = {
        # Real estate types
        "floor_plan": "Floor plan or unit layout showing room dimensions and arrangement",
        "master_plan": "Master site plan showing overall project layout and tower positions",
        "location_map": "Location map showing connectivity and nearby landmarks",
        "amenity": "Amenity image showing clubhouse, gym, pool or recreation facilities",
        "exterior": "Exterior elevation or building view",
        "interior": "Interior photograph showing room design or lobby",
        "diagram": "Technical diagram, schematic or flowchart",
        "chart": "Data chart, graph or visual comparison",
        "table": "Data table, pricing matrix or information table",
        "logo": "Developer or brand logo",
        # Industrial types
        "engineering_diagram": "Engineering diagram showing major components and their relationships",
        "exploded_view": "Exploded view showing component breakdown and assembly relationships",
        "hydraulic_diagram": "Hydraulic system diagram showing oil flow, pumps, valves and actuators",
        "electrical_diagram": "Electrical wiring diagram showing circuits, connectors and power distribution",
        "flowchart": "Process flowchart showing operational steps and decision points",
        "lifting_chart": "Lifting capacity chart showing working ranges and load limits",
        "maintenance_image": "Maintenance procedure image showing service steps or inspection points",
        "troubleshooting_image": "Troubleshooting image showing diagnostic procedures or fault locations",
        "warning_label": "Safety warning label showing hazard information or precautions",
        "icon": "Icon or symbol",
        "photograph": "Photograph",
        "graph": "Graph or plot",
        "other": "Document image",
    }
    return descriptions.get(image_type, "Document image")


def generate_caption_enhanced(
    image_type: str,
    page_text: str,
    page_number: int,
    figure_number: Optional[str] = None,
    is_industrial: bool = False,
) -> str:
    """
    Generate an enhanced caption for an image.
    
    Combines explicit caption detection with type-based generation.
    """
    # Try to extract explicit caption from page text
    explicit_caption = extract_explicit_caption(page_text)
    if explicit_caption and len(explicit_caption) > 20:
        return explicit_caption
    
    # Generate caption from type and context
    type_label = _describe_image_type(image_type, is_industrial)
    
    if figure_number:
        return f"Figure {figure_number}: {type_label}"
    
    return f"{type_label} on page {page_number}"


def extract_explicit_caption(page_text: str) -> Optional[str]:
    """Extract explicitly marked caption from page text."""
    caption_match = _CAPTION_PATTERN.search(page_text)
    if caption_match:
        # Group 3 contains the actual caption text
        caption_text = caption_match.group(3)
        if caption_text and len(caption_text) > 10:
            return caption_text.strip()
    return None


# ════════════════════════════════════════════════════════════════════════════════
#  Scoring and Confidence
# ════════════════════════════════════════════════════════════════════════════════

def calculate_image_confidence(
    type_confidence: float,
    has_caption: bool,
    has_ocr: bool,
    nearby_context: bool,
) -> float:
    """
    Calculate overall confidence score for image record.
    
    Factors:
    - Classification confidence (0-1)
    - Presence of caption (+0.1)
    - OCR data available (+0.1)
    - Nearby context available (+0.1)
    """
    score = type_confidence
    if has_caption:
        score += 0.1
    if has_ocr:
        score += 0.1
    if nearby_context:
        score += 0.1
    
    return min(1.0, score)
