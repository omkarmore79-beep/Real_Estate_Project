"""
Engineering Diagram Understanding Module for Industrial Multimodal RAG.

Generates semantic descriptions for engineering diagrams:
- Major Component Diagrams
- Exploded Views
- Hydraulic Schematics
- Electrical Diagrams
- Flowcharts
- Lifting Charts

Extracts component relationships and spatial layout information.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# Component patterns for excavators/heavy machinery
COMPONENT_PATTERNS = {
    "boom": r"\bboom\b",
    "cab": r"\bcab\b",
    "counterweight": r"\bcounterweight\b",
    "engine": r"\bengine\b",
    "swing_motor": r"\bswing\s+motor\b",
    "travel_motor": r"\btravel\s+motor\b",
    "hydraulic_tank": r"\bhydraulic\s+tank\b",
    "track": r"\btrack\b",
    "bucket": r"\bbucket\b",
    "arm": r"\barm\b",
    "cylinder": r"\bcylinder\b",
    "pump": r"\bpump\b",
    "valve": r"\bvalve\b",
    "harness": r"\bharness\b",
    "relay": r"\brelay\b",
    "fuse": r"\bfuse\b",
    "control": r"\bcontrol\b",
}


def generate_diagram_description(
    image_type: str,
    caption: str,
    page_text: str,
    figure_number: Optional[str] = None,
) -> str:
    """
    Generate a semantic description for an engineering diagram.
    
    Args:
        image_type: Type of diagram (engineering_diagram, exploded_view, etc.)
        caption: Image caption
        page_text: Surrounding text context
        figure_number: Figure identifier
    
    Returns:
        Semantic description suitable for embedding
    """
    components = extract_components(page_text)
    
    if image_type == "engineering_diagram":
        return describe_major_component_diagram(components, caption, figure_number)
    elif image_type == "exploded_view":
        return describe_exploded_view(components, caption, figure_number)
    elif image_type == "hydraulic_diagram":
        return describe_hydraulic_diagram(components, caption, figure_number)
    elif image_type == "electrical_diagram":
        return describe_electrical_diagram(components, caption, figure_number)
    elif image_type == "flowchart":
        return describe_flowchart(page_text, caption, figure_number)
    else:
        return describe_generic_diagram(components, caption, figure_number)


def extract_components(page_text: str) -> list[str]:
    """
    Extract component names from page text.
    
    Returns list of detected components.
    """
    detected = []
    
    for component_name, pattern in COMPONENT_PATTERNS.items():
        if re.search(pattern, page_text, re.IGNORECASE):
            detected.append(component_name.replace("_", " "))
    
    return detected


def describe_major_component_diagram(
    components: list[str],
    caption: str,
    figure_number: Optional[str] = None,
) -> str:
    """
    Generate description for a major component diagram.
    
    Example: "Major Component Diagram showing: Boom, Cab, Counterweight, Engine, 
    Swing Motor, Travel Motor, Hydraulic Tank. Counterweight located behind the 
    engine opposite the boom."
    """
    parts = []
    
    if figure_number:
        parts.append(f"Figure {figure_number}:")
    
    parts.append("Major Component Diagram showing:")
    
    if components:
        # List main components
        main_components = components[:8]
        parts.append(", ".join(main_components))
        
        # Add spatial relationships based on common excavator layout
        spatial_desc = generate_spatial_description(components)
        if spatial_desc:
            parts.append(spatial_desc)
    else:
        parts.append("major machine components and their relationships")
    
    if caption and caption != "Document image":
        parts.append(f"Caption: {caption}")
    
    return ". ".join(parts)


def describe_exploded_view(
    components: list[str],
    caption: str,
    figure_number: Optional[str] = None,
) -> str:
    """
    Generate description for an exploded view diagram.
    
    Example: "Exploded view showing component breakdown and assembly relationships.
    Components: Boom, Arm, Bucket, Cylinder, Pins. Shows disassembly sequence 
    and part relationships."
    """
    parts = []
    
    if figure_number:
        parts.append(f"Figure {figure_number}:")
    
    parts.append("Exploded view showing component breakdown and assembly relationships")
    
    if components:
        parts.append(f"Components: {', '.join(components[:8])}")
        parts.append("Shows disassembly sequence and part relationships")
    else:
        parts.append("Shows individual parts and assembly order")
    
    if caption and caption != "Document image":
        parts.append(f"Caption: {caption}")
    
    return ". ".join(parts)


def describe_hydraulic_diagram(
    components: list[str],
    caption: str,
    figure_number: Optional[str] = None,
) -> str:
    """
    Generate description for a hydraulic system diagram.
    
    Example: "Hydraulic system diagram showing oil flow, pumps, valves and actuators.
    Components: Pump, Valve, Cylinder, Motor, Tank, Line, Hose. Shows hydraulic 
    circuit and pressure flow paths."
    """
    parts = []
    
    if figure_number:
        parts.append(f"Figure {figure_number}:")
    
    parts.append("Hydraulic system diagram showing oil flow, pumps, valves and actuators")
    
    # Add hydraulic-specific components
    hydraulic_components = [c for c in components if c in ["pump", "valve", "cylinder", "motor", "tank"]]
    if hydraulic_components:
        parts.append(f"Components: {', '.join(hydraulic_components)}")
    
    parts.append("Shows hydraulic circuit and pressure flow paths")
    
    if caption and caption != "Document image":
        parts.append(f"Caption: {caption}")
    
    return ". ".join(parts)


def describe_electrical_diagram(
    components: list[str],
    caption: str,
    figure_number: Optional[str] = None,
) -> str:
    """
    Generate description for an electrical wiring diagram.
    
    Example: "Electrical wiring diagram showing circuits, connectors and power 
    distribution. Components: Wire, Harness, Connector, Relay, Fuse, Switch. 
    Shows electrical connections and grounding."
    """
    parts = []
    
    if figure_number:
        parts.append(f"Figure {figure_number}:")
    
    parts.append("Electrical wiring diagram showing circuits, connectors and power distribution")
    
    # Add electrical-specific components
    electrical_components = [c for c in components if c in ["harness", "connector", "relay", "fuse", "control"]]
    if electrical_components:
        parts.append(f"Components: {', '.join(electrical_components)}")
    
    parts.append("Shows electrical connections and grounding")
    
    if caption and caption != "Document image":
        parts.append(f"Caption: {caption}")
    
    return ". ".join(parts)


def describe_flowchart(
    page_text: str,
    caption: str,
    figure_number: Optional[str] = None,
) -> str:
    """
    Generate description for a process flowchart.
    
    Example: "Process flowchart showing operational steps and decision points.
    Shows workflow sequence with decision branches and action steps."
    """
    parts = []
    
    if figure_number:
        parts.append(f"Figure {figure_number}:")
    
    parts.append("Process flowchart showing operational steps and decision points")
    
    # Extract step indicators
    steps = re.findall(r"\bstep\s+\d+\b", page_text, re.IGNORECASE)
    if steps:
        parts.append(f"Contains {len(steps)} operational steps")
    
    parts.append("Shows workflow sequence with decision branches and action steps")
    
    if caption and caption != "Document image":
        parts.append(f"Caption: {caption}")
    
    return ". ".join(parts)


def describe_generic_diagram(
    components: list[str],
    caption: str,
    figure_number: Optional[str] = None,
) -> str:
    """
    Generate description for a generic technical diagram.
    """
    parts = []
    
    if figure_number:
        parts.append(f"Figure {figure_number}:")
    
    parts.append("Technical diagram showing system components and relationships")
    
    if components:
        parts.append(f"Components: {', '.join(components[:6])}")
    
    if caption and caption != "Document image":
        parts.append(f"Caption: {caption}")
    
    return ". ".join(parts)


def generate_spatial_description(components: list[str]) -> Optional[str]:
    """
    Generate spatial relationship description based on component list.
    
    Uses common excavator layout knowledge to describe component positions.
    """
    if not components:
        return None
    
    spatial_rules = [
        ("counterweight", "behind the engine opposite the boom"),
        ("cab", "on top of the upper structure"),
        ("engine", "in the upper structure"),
        ("hydraulic tank", "behind the cab"),
        ("boom", "at the front of the upper structure"),
        ("arm", "attached to the boom"),
        ("bucket", "at the end of the arm"),
        ("track", "at the base of the machine"),
    ]
    
    descriptions = []
    for component, location in spatial_rules:
        if component in components:
            descriptions.append(f"{component.replace('_', ' ').title()} {location}")
    
    if descriptions:
        return ". ".join(descriptions[:3])
    
    return None


def classify_diagram_subtype(page_text: str, image_type: str) -> str:
    """
    Classify the subtype of engineering diagram.
    
    Returns more specific type like "major_component", "hydraulic_circuit", 
    "electrical_wiring", etc.
    """
    if image_type == "engineering_diagram":
        if "major component" in page_text.lower() or "component diagram" in page_text.lower():
            return "major_component"
        elif "assembly" in page_text.lower():
            return "assembly"
    
    elif image_type == "hydraulic_diagram":
        if "circuit" in page_text.lower():
            return "hydraulic_circuit"
        elif "schematic" in page_text.lower():
            return "hydraulic_schematic"
    
    elif image_type == "electrical_diagram":
        if "wiring" in page_text.lower():
            return "electrical_wiring"
        elif "schematic" in page_text.lower():
            return "electrical_schematic"
    
    return image_type


def extract_diagram_relationships(page_text: str) -> list[str]:
    """
    Extract component relationships from text.
    
    Looks for phrases like "connected to", "attached to", "mounted on", etc.
    """
    relationship_patterns = [
        r"(\w+)\s+(?:is\s+)?(?:connected|attached|mounted|linked)\s+(?:to|on)\s+(\w+)",
        r"(\w+)\s+(?:drives|powers|controls)\s+(\w+)",
    ]
    
    relationships = []
    for pattern in relationship_patterns:
        matches = re.findall(pattern, page_text, re.IGNORECASE)
        for match in matches:
            relationships.append(f"{match[0]} -> {match[1]}")
    
    return relationships[:5]


def enhance_image_description_with_diagram_info(
    image_record: dict,
    page_text: str,
) -> dict:
    """
    Enhance an existing image record with diagram-specific information.
    
    Adds:
    - diagram_subtype
    - components
    - spatial_description
    - relationships
    """
    image_type = image_record.get("image_type", "")
    
    if image_type not in ("engineering_diagram", "exploded_view", "hydraulic_diagram", "electrical_diagram", "flowchart"):
        return image_record
    
    # Extract diagram information
    components = extract_components(page_text)
    diagram_subtype = classify_diagram_subtype(page_text, image_type)
    spatial_desc = generate_spatial_description(components)
    relationships = extract_diagram_relationships(page_text)
    
    # Generate enhanced description
    figure_number = image_record.get("figure_number")
    caption = image_record.get("caption", "")
    enhanced_description = generate_diagram_description(
        image_type, caption, page_text, figure_number
    )
    
    # Update image record
    image_record["diagram_subtype"] = diagram_subtype
    image_record["components"] = components
    image_record["spatial_description"] = spatial_desc
    image_record["relationships"] = relationships
    image_record["enhanced_description"] = enhanced_description
    
    # Update multimodal description if it exists
    if "multimodal_description" in image_record:
        # Use enhanced description for better retrieval
        image_record["multimodal_description"] = enhanced_description
    
    # Update metadata
    if "metadata" in image_record:
        image_record["metadata"]["diagram_subtype"] = diagram_subtype
        image_record["metadata"]["components"] = components
        image_record["metadata"]["spatial_description"] = spatial_desc
        image_record["metadata"]["relationships"] = relationships
    
    return image_record
