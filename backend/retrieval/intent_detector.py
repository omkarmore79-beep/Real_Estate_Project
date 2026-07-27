"""
Query Intent Detection Module for Industrial Multimodal RAG.

Detects whether the query is about:
- Image/Diagram
- Table
- Troubleshooting
- Maintenance
- Error Code
- Spare Part
- Procedure
- General QA

Uses detected intent to prioritize correct retrieval objects.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# Intent patterns
_INTENT_PATTERNS = {
    "image": {
        "phrases": [
            "show", "display", "image", "photo", "picture", "visual", "diagram",
            "schematic", "chart", "graph", "illustration", "figure", "fig",
            "view", "layout", "plan", "map", "blueprint", "drawing"
        ],
        "keywords": [
            "visual", "graphic", "see", "look", "depict", "illustrate"
        ]
    },
    "diagram": {
        "phrases": [
            "diagram", "schematic", "circuit", "wiring", "hydraulic", "flowchart",
            "exploded", "breakdown", "assembly", "component", "layout"
        ],
        "keywords": [
            "component", "relationship", "connection", "system"
        ]
    },
    "table": {
        "phrases": [
            "table", "chart", "schedule", "matrix", "specification", "data",
            "capacity", "lifting", "load", "range", "dimension"
        ],
        "keywords": [
            "row", "column", "value", "spec", "rating"
        ]
    },
    "troubleshooting": {
        "phrases": [
            "troubleshoot", "diagnosis", "diagnostic", "fault", "problem", "issue",
            "malfunction", "failure", "error", "not working", "broken"
        ],
        "keywords": [
            "symptom", "cause", "fix", "repair", "resolve"
        ]
    },
    "maintenance": {
        "phrases": [
            "maintenance", "service", "inspection", "check", "replace", "adjust",
            "lubricate", "clean", "interval", "schedule", "procedure"
        ],
        "keywords": [
            "service", "preventive", "routine", "periodic"
        ]
    },
    "error_code": {
        "phrases": [
            "error code", "dtc", "diagnostic trouble code", "fault code",
            "warning", "alert", "indicator", "code"
        ],
        "keywords": [
            "code", "dtc", "error", "fault"
        ]
    },
    "spare_part": {
        "phrases": [
            "part", "spare", "component", "replacement", "item", "piece",
            "part number", "pn", "p/n", "reference", "catalog"
        ],
        "keywords": [
            "order", "buy", "purchase", "stock", "inventory"
        ]
    },
    "procedure": {
        "phrases": [
            "procedure", "step", "instruction", "how to", "guide", "manual",
            "method", "process", "workflow", "operation"
        ],
        "keywords": [
            "do", "perform", "execute", "carry out"
        ]
    },
}


def detect_query_intent(query: str) -> dict:
    """
    Detect the primary intent of a user query.
    
    Returns a dictionary with:
    - primary_intent: Main intent category
    - confidence: Confidence score (0-1)
    - secondary_intents: List of other detected intents
    - detected_types: Specific types to filter (for images)
    - priority_retrieval: Which retrieval type to prioritize
    """
    query_lower = query.lower()
    
    # Score each intent
    intent_scores = {}
    
    for intent, patterns in _INTENT_PATTERNS.items():
        phrase_score = sum(3 for phrase in patterns["phrases"] if phrase in query_lower)
        keyword_score = sum(1 for kw in patterns["keywords"] if kw in query_lower)
        total_score = phrase_score + keyword_score
        
        if total_score > 0:
            # Normalize score to 0-1 range
            intent_scores[intent] = min(1.0, total_score * 0.15)
    
    # Determine primary intent
    if intent_scores:
        primary_intent, confidence = max(intent_scores.items(), key=lambda x: x[1])
    else:
        primary_intent = "general"
        confidence = 0.5
    
    # Get secondary intents (within 0.1 of primary)
    secondary_intents = [
        intent for intent, score in intent_scores.items()
        if intent != primary_intent and score >= confidence - 0.1
    ]
    
    # Determine specific types for filtering
    detected_types = _get_detected_types(query_lower, primary_intent)
    
    # Determine priority retrieval
    priority_retrieval = _get_priority_retrieval(primary_intent, detected_types)
    
    # Extract error codes if present
    error_codes = _extract_error_codes(query)
    
    # Extract part numbers if present
    part_numbers = _extract_part_numbers(query)
    
    return {
        "primary_intent": primary_intent,
        "confidence": confidence,
        "secondary_intents": secondary_intents,
        "detected_types": detected_types,
        "priority_retrieval": priority_retrieval,
        "error_codes": error_codes,
        "part_numbers": part_numbers,
    }


def _get_detected_types(query_lower: str, primary_intent: str) -> list[str]:
    """
    Get specific types for filtering based on intent.
    
    For example, if intent is "diagram", return specific diagram types.
    """
    if primary_intent == "image" or primary_intent == "diagram":
        # Detect specific diagram types
        diagram_types = []
        
        if "major component" in query_lower or "component diagram" in query_lower:
            diagram_types.append("engineering_diagram")
        if "exploded" in query_lower or "breakdown" in query_lower:
            diagram_types.append("exploded_view")
        if "hydraulic" in query_lower:
            diagram_types.append("hydraulic_diagram")
        if "electrical" in query_lower or "wiring" in query_lower:
            diagram_types.append("electrical_diagram")
        if "flowchart" in query_lower or "flow" in query_lower:
            diagram_types.append("flowchart")
        if "lifting" in query_lower or "capacity" in query_lower:
            diagram_types.append("lifting_chart")
        if "floor plan" in query_lower or "unit plan" in query_lower:
            diagram_types.append("floor_plan")
        if "master plan" in query_lower or "site layout" in query_lower:
            diagram_types.append("master_plan")
        if "location" in query_lower or "map" in query_lower:
            diagram_types.append("location_map")
        
        return diagram_types if diagram_types else ["diagram"]
    
    elif primary_intent == "table":
        # Detect specific table types
        if "lifting" in query_lower or "capacity" in query_lower:
            return ["lifting_chart"]
        if "specification" in query_lower or "spec" in query_lower:
            return ["specification"]
        if "price" in query_lower or "payment" in query_lower:
            return ["pricing"]
        if "schedule" in query_lower or "timeline" in query_lower:
            return ["schedule"]
        
        return ["table"]
    
    return []


def _get_priority_retrieval(primary_intent: str, detected_types: list[str]) -> str:
    """
    Determine which retrieval type to prioritize.
    
    Returns: "text", "image", "table", "hybrid"
    """
    if primary_intent in ("image", "diagram"):
        return "image"
    elif primary_intent == "table":
        return "table"
    elif primary_intent in ("error_code", "spare_part"):
        return "text"  # These are usually in text chunks
    elif primary_intent in ("troubleshooting", "maintenance", "procedure"):
        return "hybrid"  # Need both text and possibly images
    else:
        return "hybrid"


def _extract_error_codes(query: str) -> list[str]:
    """
    Extract error codes/DTCs from query.
    
    Patterns: "E1234", "DTC 1234", "Error Code 1234"
    """
    patterns = [
        r'\b([A-Z]{1,4}\d{3,}[A-Z0-9\-]*)\b',  # E1234, ABC1234
        r'\b(DTC\s+\d+)\b',  # DTC 1234
        r'\b(Error\s+Code\s*\d+)\b',  # Error Code 1234
    ]
    
    codes = []
    for pattern in patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        codes.extend(matches)
    
    return codes


def _extract_part_numbers(query: str) -> list[str]:
    """
    Extract spare part numbers from query.
    
    Patterns: "123-4567", "PN 12345", "Part Number ABC-123"
    """
    patterns = [
        r'\b([A-Z0-9]{3,}[-/][A-Z0-9]{3,})\b',  # ABC-123, 123/456
        r'\b(PN\s*[A-Z0-9\-]+)\b',  # PN 12345
        r'\b(Part\s+Number\s*[A-Z0-9\-]+)\b',  # Part Number 12345
    ]
    
    parts = []
    for pattern in patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        parts.extend(matches)
    
    return parts


def classify_query_for_routing(query: str) -> dict:
    """
    Classify query for routing to appropriate retrieval pipeline.
    
    Returns routing information including:
    - collection: Which Qdrant collection to use
    - filters: Metadata filters to apply
    - top_k: How many results to retrieve
    - include_images: Whether to include image search
    """
    intent_info = detect_query_intent(query)
    primary_intent = intent_info["primary_intent"]
    
    routing = {
        "collection": "text",  # Default
        "filters": {},
        "top_k": 8,
        "include_images": False,
        "intent_info": intent_info,
    }
    
    # Set retrieval based on intent
    if primary_intent in ("image", "diagram"):
        routing["include_images"] = True
        routing["top_k"] = 6
        if intent_info["detected_types"]:
            routing["filters"]["image_type"] = intent_info["detected_types"]
    
    elif primary_intent == "table":
        routing["filters"]["chunk_type"] = "table"
        routing["top_k"] = 5
    
    elif primary_intent == "error_code":
        if intent_info["error_codes"]:
            routing["filters"]["error_codes"] = intent_info["error_codes"]
        routing["top_k"] = 5
    
    elif primary_intent == "spare_part":
        if intent_info["part_numbers"]:
            routing["filters"]["spare_parts"] = intent_info["part_numbers"]
        routing["top_k"] = 5
    
    elif primary_intent in ("troubleshooting", "maintenance"):
        routing["include_images"] = True
        routing["filters"]["chunk_type"] = ["troubleshooting_procedure", "maintenance_procedure"]
        routing["top_k"] = 8
    
    return routing


def get_retrieval_priority_order(intent_info: dict) -> list[str]:
    """
    Get the order of retrieval types to try based on intent.
    
    Returns list like ["image", "text"] or ["text", "image"]
    """
    primary_intent = intent_info["primary_intent"]
    priority = intent_info["priority_retrieval"]
    
    if priority == "image":
        return ["image", "text"]
    elif priority == "text":
        return ["text", "image"]
    elif priority == "table":
        return ["text"]  # Tables are stored as text chunks
    else:  # hybrid
        return ["text", "image"]


def should_prioritize_images(query: str) -> bool:
    """
    Quick check if images should be prioritized for this query.
    """
    intent_info = detect_query_intent(query)
    return intent_info["priority_retrieval"] == "image"


def get_metadata_filters_from_intent(intent_info: dict) -> dict:
    """
    Convert intent information into Qdrant metadata filters.
    """
    filters = {}
    
    # Add detected types as image_type filter
    if intent_info["detected_types"]:
        filters["image_type"] = intent_info["detected_types"]
    
    # Add error codes filter
    if intent_info["error_codes"]:
        filters["error_codes"] = intent_info["error_codes"]
    
    # Add part numbers filter
    if intent_info["part_numbers"]:
        filters["spare_parts"] = intent_info["part_numbers"]
    
    # Add chunk_type based on intent
    primary_intent = intent_info["primary_intent"]
    if primary_intent == "troubleshooting":
        filters["chunk_type"] = "troubleshooting_procedure"
    elif primary_intent == "maintenance":
        filters["chunk_type"] = "maintenance_procedure"
    elif primary_intent == "table":
        filters["chunk_type"] = "table"
    
    return filters
