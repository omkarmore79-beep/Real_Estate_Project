"""
Citation Builder Module for Industrial Multimodal RAG.

Generates detailed citations for answers including:
- Manual/Document
- Page
- Section
- Figure
- Table
- Confidence
- Supporting Evidence

Citations reference the retrieved chunks.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def build_citation(retrieved_obj: dict) -> dict:
    """
    Build a comprehensive citation for a retrieved object.
    
    Returns a dictionary with all citation information.
    """
    metadata = retrieved_obj.get("metadata", {})
    
    citation = {
        "document_id": metadata.get("document_id", ""),
        "source_file": metadata.get("source_file", ""),
        "page": metadata.get("page_number"),
        "section": metadata.get("section", "General"),
        "subsection": metadata.get("subsection", ""),
        "confidence": retrieved_obj.get("combined_confidence", 0.0),
        "confidence_level": retrieved_obj.get("confidence_level", "unknown"),
        "chunk_id": retrieved_obj.get("id", ""),
        "source_type": retrieved_obj.get("source_type", "text"),
    }
    
    # Add figure/table information if present
    if metadata.get("figure_number"):
        citation["figure"] = metadata["figure_number"]
        citation["figure_title"] = metadata.get("figure_title", "")
        citation["caption"] = metadata.get("caption", "")
    
    if metadata.get("table_number"):
        citation["table"] = metadata["table_number"]
        citation["table_title"] = metadata.get("title", "")
    
    # Add image information if present
    if retrieved_obj.get("source_type") == "image":
        citation["image_id"] = metadata.get("image_id", "")
        citation["image_type"] = metadata.get("image_type", "")
        citation["image_caption"] = metadata.get("caption", "")
    
    # Add supporting evidence (content snippet)
    content = retrieved_obj.get("content", "")
    if content:
        citation["evidence_snippet"] = content[:200] + "..." if len(content) > 200 else content
    
    return citation


def format_citation(citation: dict) -> str:
    """
    Format a citation as a human-readable string.
    
    Example: "Manual: R215L Operator Manual, Page 45, Section: Maintenance, Figure 3.2, Confidence: 0.85"
    """
    parts = []
    
    # Document/Manual
    source_file = citation.get("source_file", "")
    if source_file:
        parts.append(f"Manual: {source_file}")
    
    # Page
    page = citation.get("page")
    if page is not None:
        parts.append(f"Page {page}")
    
    # Section
    section = citation.get("section")
    if section and section != "General":
        parts.append(f"Section: {section}")
    
    # Figure
    figure = citation.get("figure")
    if figure:
        parts.append(f"Figure {figure}")
    
    # Table
    table = citation.get("table")
    if table:
        parts.append(f"Table {table}")
    
    # Confidence
    confidence = citation.get("confidence", 0)
    if confidence > 0:
        parts.append(f"Confidence: {confidence:.2f}")
    
    return ", ".join(parts)


def build_citations_for_results(results: list[dict]) -> list[dict]:
    """
    Build citations for all retrieved results.
    
    Returns list of citation dictionaries.
    """
    citations = []
    
    for result in results:
        citation = build_citation(result)
        citations.append(citation)
    
    return citations


def format_citations_for_answer(results: list[dict], max_citations: int = 5) -> str:
    """
    Format citations for inclusion in an answer.
    
    Returns a formatted string with citations.
    """
    citations = build_citations_for_results(results)
    
    # Limit number of citations
    citations = citations[:max_citations]
    
    if not citations:
        return ""
    
    formatted = "\n\n**Sources:**\n"
    for i, citation in enumerate(citations, 1):
        formatted += f"{i}. {format_citation(citation)}\n"
    
    return formatted


def get_supporting_evidence(results: list[dict], max_evidence: int = 3) -> list[str]:
    """
    Extract supporting evidence snippets from results.
    
    Returns list of content snippets.
    """
    evidence = []
    
    for result in results[:max_evidence]:
        content = result.get("content", "")
        if content:
            # Get first meaningful sentence
            sentences = content.split(".")
            if sentences:
                snippet = sentences[0].strip()
                if len(snippet) > 50:
                    evidence.append(snippet + ".")
                elif len(sentences) > 1:
                    evidence.append(sentences[0].strip() + ". " + sentences[1].strip() + ".")
    
    return evidence


def build_answer_with_citations(
    answer: str,
    results: list[dict],
    include_evidence: bool = True,
) -> str:
    """
    Build a complete answer with citations and supporting evidence.
    
    Args:
        answer: The generated answer text
        results: Retrieved results used for the answer
        include_evidence: Whether to include supporting evidence snippets
    
    Returns:
        Complete answer with citations
    """
    final_answer = answer
    
    # Add supporting evidence if requested
    if include_evidence:
        evidence = get_supporting_evidence(results)
        if evidence:
            final_answer += "\n\n**Supporting Evidence:**\n"
            for i, snippet in enumerate(evidence, 1):
                final_answer += f"- {snippet}\n"
    
    # Add citations
    citations = format_citations_for_answer(results)
    if citations:
        final_answer += citations
    
    return final_answer


def get_citation_summary(results: list[dict]) -> dict:
    """
    Get a summary of citations from results.
    
    Returns:
        Dictionary with:
        - total_sources: Number of unique sources
        - pages: List of page numbers
        - sections: List of sections
        - figures: List of figure numbers
        - tables: List of table numbers
        - average_confidence: Average confidence
    """
    citations = build_citations_for_results(results)
    
    pages = sorted(set(c.get("page") for c in citations if c.get("page") is not None))
    sections = sorted(set(c.get("section") for c in citations if c.get("section")))
    figures = sorted(set(c.get("figure") for c in citations if c.get("figure")))
    tables = sorted(set(c.get("table") for c in citations if c.get("table")))
    
    confidences = [c.get("confidence", 0) for c in citations]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    return {
        "total_sources": len(citations),
        "pages": pages,
        "sections": sections,
        "figures": figures,
        "tables": tables,
        "average_confidence": avg_confidence,
    }


def format_citation_for_summary(citation: dict) -> str:
    """
    Format a citation for a summary (shorter format).
    
    Example: "[Page 45, Section: Maintenance, Fig 3.2]"
    """
    parts = []
    
    page = citation.get("page")
    if page is not None:
        parts.append(f"Page {page}")
    
    section = citation.get("section")
    if section and section != "General":
        parts.append(f"Section: {section}")
    
    figure = citation.get("figure")
    if figure:
        parts.append(f"Fig {figure}")
    
    table = citation.get("table")
    if table:
        parts.append(f"Table {table}")
    
    return f"[{', '.join(parts)}]" if parts else "[Unknown]"


def add_inline_citations(answer: str, results: list[dict]) -> str:
    """
    Add inline citations to an answer.
    
    Replaces placeholders like [1], [2] with actual citations.
    """
    citations = build_citations_for_results(results)
    
    for i, citation in enumerate(citations, 1):
        placeholder = f"[{i}]"
        formatted = format_citation_for_summary(citation)
        answer = answer.replace(placeholder, formatted)
    
    return answer


def validate_citation_completeness(citation: dict) -> dict:
    """
    Validate that a citation has all required fields.
    
    Returns:
        Dictionary with:
        - is_valid: Boolean
        - missing_fields: List of missing fields
    """
    required_fields = ["document_id", "page", "section", "confidence"]
    
    missing = []
    for field in required_fields:
        if field not in citation or citation[field] is None:
            missing.append(field)
    
    return {
        "is_valid": len(missing) == 0,
        "missing_fields": missing,
    }
