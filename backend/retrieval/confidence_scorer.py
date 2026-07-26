"""
Confidence Scoring Module for Industrial Multimodal RAG.

Calculates combined confidence scores for retrieved objects including:
- retrieval_score (from vector similarity)
- rerank_score (from cross-encoder reranking)
- OCR confidence (for text chunks)
- layout confidence (for chunk quality)
- combined confidence (weighted average)

If confidence is below threshold, respond: "I could not find sufficient supporting evidence."
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# Confidence threshold for considering a result reliable
CONFIDENCE_THRESHOLD = 0.35
LOW_CONFIDENCE_THRESHOLD = 0.20
HIGH_CONFIDENCE_THRESHOLD = 0.70


def calculate_combined_confidence(
    retrieval_score: float,
    rerank_score: Optional[float] = None,
    ocr_confidence: float = 1.0,
    layout_confidence: float = 1.0,
    source_type: str = "pdf_text",
) -> float:
    """
    Calculate combined confidence score from multiple sources.
    
    Args:
        retrieval_score: Vector similarity score (0-1)
        rerank_score: Cross-encoder rerank score (may be logits)
        ocr_confidence: OCR confidence (0-1)
        layout_confidence: Layout quality confidence (0-1)
        source_type: Source type (pdf_text, ocr, mixed)
    
    Returns:
        Combined confidence score (0-1)
    """
    # Normalize retrieval score (should already be 0-1)
    norm_retrieval = max(0.0, min(1.0, retrieval_score))
    
    # Normalize rerank score (may be logits from -inf to +inf)
    if rerank_score is not None:
        if rerank_score < 0:
            # Convert logits to probability using sigmoid
            norm_rerank = 1.0 / (1.0 + math.exp(-rerank_score))
        else:
            norm_rerank = max(0.0, min(1.0, rerank_score))
    else:
        norm_rerank = norm_retrieval  # Fallback to retrieval score
    
    # Weight the scores
    # Rerank score is most important (weight 0.5)
    # Retrieval score is secondary (weight 0.3)
    # OCR and layout confidence are quality factors (weight 0.1 each)
    combined = (
        norm_rerank * 0.5 +
        norm_retrieval * 0.3 +
        ocr_confidence * 0.1 +
        layout_confidence * 0.1
    )
    
    return max(0.0, min(1.0, combined))


def score_retrieved_object(obj: dict) -> dict:
    """
    Score a retrieved object with comprehensive confidence metrics.
    
    Adds the following fields to the object:
    - retrieval_score: Original vector similarity
    - rerank_score: Cross-encoder rerank score (normalized)
    - ocr_confidence: OCR quality score
    - layout_confidence: Layout quality score
    - combined_confidence: Weighted average
    - confidence_level: low, medium, or high
    """
    metadata = obj.get("metadata", {})
    
    # Get existing scores
    retrieval_score = obj.get("score", 0.0)
    rerank_score = obj.get("rerank_score")
    
    # Get confidence factors
    ocr_confidence = metadata.get("ocr_confidence", 1.0)
    layout_confidence = metadata.get("layout_confidence", 1.0)
    source_type = metadata.get("source_type", "pdf_text")
    
    # Adjust layout confidence based on source type
    if source_type == "ocr":
        layout_confidence = min(layout_confidence, 0.8)
    elif source_type == "mixed":
        layout_confidence = (layout_confidence + 0.95) / 2
    
    # Calculate combined confidence
    combined = calculate_combined_confidence(
        retrieval_score,
        rerank_score,
        ocr_confidence,
        layout_confidence,
        source_type,
    )
    
    # Determine confidence level
    if combined >= HIGH_CONFIDENCE_THRESHOLD:
        confidence_level = "high"
    elif combined >= CONFIDENCE_THRESHOLD:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    
    # Add scores to object
    obj["retrieval_score"] = retrieval_score
    obj["rerank_score"] = rerank_score
    obj["ocr_confidence"] = ocr_confidence
    obj["layout_confidence"] = layout_confidence
    obj["combined_confidence"] = combined
    obj["confidence_level"] = confidence_level
    
    return obj


def score_retrieved_objects(objects: list[dict]) -> list[dict]:
    """
    Score all retrieved objects.
    
    Returns list of objects with confidence scores added.
    """
    scored_objects = []
    
    for obj in objects:
        scored_obj = score_retrieved_object(obj)
        scored_objects.append(scored_obj)
    
    # Log confidence distribution
    high_count = sum(1 for o in scored_objects if o.get("confidence_level") == "high")
    medium_count = sum(1 for o in scored_objects if o.get("confidence_level") == "medium")
    low_count = sum(1 for o in scored_objects if o.get("confidence_level") == "low")
    
    logger.info(
        "Confidence distribution: %d high, %d medium, %d low",
        high_count, medium_count, low_count
    )
    
    return scored_objects


def filter_by_confidence(
    objects: list[dict],
    threshold: float = CONFIDENCE_THRESHOLD,
    min_results: int = 3,
) -> list[dict]:
    """
    Filter objects by confidence threshold.
    
    If too few objects pass the threshold, lower it to ensure minimum results.
    """
    # Filter by threshold
    filtered = [o for o in objects if o.get("combined_confidence", 0) >= threshold]
    
    # If too few results, lower threshold
    if len(filtered) < min_results and len(objects) >= min_results:
        # Sort by confidence
        sorted_objects = sorted(objects, key=lambda x: x.get("combined_confidence", 0), reverse=True)
        # Take top min_results
        filtered = sorted_objects[:min_results]
        
        logger.warning(
            "Only %d objects passed confidence threshold %f, lowered to include top %d",
            len([o for o in objects if o.get("combined_confidence", 0) >= threshold]),
            threshold,
            min_results
        )
    
    return filtered


def get_overall_confidence(objects: list[dict]) -> dict:
    """
    Calculate overall confidence metrics for a set of retrieved objects.
    
    Returns:
        Dictionary with:
        - average_confidence: Mean combined confidence
        - max_confidence: Highest confidence
        - min_confidence: Lowest confidence
        - high_confidence_count: Number of high-confidence results
        - sufficient_evidence: Whether there's sufficient evidence
    """
    if not objects:
        return {
            "average_confidence": 0.0,
            "max_confidence": 0.0,
            "min_confidence": 0.0,
            "high_confidence_count": 0,
            "sufficient_evidence": False,
        }
    
    confidences = [o.get("combined_confidence", 0) for o in objects]
    
    average = sum(confidences) / len(confidences)
    max_conf = max(confidences)
    min_conf = min(confidences)
    high_count = sum(1 for c in confidences if c >= HIGH_CONFIDENCE_THRESHOLD)
    
    # Sufficient evidence if at least 2 results above threshold
    sufficient_evidence = sum(1 for c in confidences if c >= CONFIDENCE_THRESHOLD) >= 2
    
    return {
        "average_confidence": average,
        "max_confidence": max_conf,
        "min_confidence": min_conf,
        "high_confidence_count": high_count,
        "sufficient_evidence": sufficient_evidence,
    }


def should_hallucinate_warning(objects: list[dict]) -> tuple[bool, str]:
    """
    Determine if a hallucination warning should be shown.
    
    Returns:
        (should_warn, warning_message)
    """
    overall = get_overall_confidence(objects)
    
    if not overall["sufficient_evidence"]:
        return True, "I could not find sufficient supporting evidence for this query."
    
    if overall["average_confidence"] < LOW_CONFIDENCE_THRESHOLD:
        return True, "The available information has low confidence. Please verify with the source documents."
    
    if overall["high_confidence_count"] == 0:
        return True, "The retrieved information has moderate confidence. Please verify important details."
    
    return False, ""


def calculate_image_confidence(
    image_record: dict,
    retrieval_score: float,
) -> float:
    """
    Calculate confidence for an image retrieval result.
    
    Factors:
    - Retrieval score (vector similarity)
    - Type classification confidence
    - Whether it has a caption
    - Whether it has OCR text
    """
    metadata = image_record.get("metadata", {})
    
    type_confidence = metadata.get("type_confidence", 0.5)
    has_caption = bool(metadata.get("caption") and len(metadata.get("caption", "")) > 10)
    has_ocr = bool(metadata.get("ocr_from_image") or metadata.get("ocr_context"))
    
    # Base score from retrieval
    base = max(0.0, min(1.0, retrieval_score))
    
    # Adjust based on type confidence
    adjusted = (base + type_confidence) / 2
    
    # Bonus for caption and OCR
    if has_caption:
        adjusted += 0.1
    if has_ocr:
        adjusted += 0.05
    
    return max(0.0, min(1.0, adjusted))


def calculate_table_confidence(
    table_record: dict,
    retrieval_score: float,
) -> float:
    """
    Calculate confidence for a table retrieval result.
    
    Factors:
    - Retrieval score
    - Table structure quality (row/column count)
    - Whether it has headers
    - Whether it has units
    """
    metadata = table_record.get("metadata", {})
    
    row_count = metadata.get("row_count", 0)
    column_count = metadata.get("column_count", 0)
    has_headers = bool(metadata.get("column_headers"))
    has_units = bool(metadata.get("units"))
    
    # Base score from retrieval
    base = max(0.0, min(1.0, retrieval_score))
    
    # Adjust based on structure quality
    structure_quality = 0.5
    if row_count > 0 and column_count > 0:
        structure_quality = 0.8
    if has_headers:
        structure_quality += 0.1
    if has_units:
        structure_quality += 0.1
    
    adjusted = (base + structure_quality) / 2
    
    return max(0.0, min(1.0, adjusted))


def normalize_score(score: float, score_type: str = "similarity") -> float:
    """
    Normalize a score to 0-1 range based on its type.
    
    Args:
        score: Raw score
        score_type: Type of score (similarity, logits, probability)
    
    Returns:
        Normalized score (0-1)
    """
    if score_type == "similarity":
        # Cosine similarity is already -1 to 1, map to 0-1
        return (score + 1) / 2
    elif score_type == "logits":
        # Convert logits to probability
        return 1.0 / (1.0 + math.exp(-score))
    elif score_type == "probability":
        # Already 0-1
        return max(0.0, min(1.0, score))
    else:
        # Default: clamp to 0-1
        return max(0.0, min(1.0, score))
