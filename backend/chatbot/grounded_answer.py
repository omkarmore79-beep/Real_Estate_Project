"""
Grounded Answer Generator.

Generates answers ONLY from retrieved evidence using the Groq LLM.
Never uses general knowledge. Every factual claim must be grounded in
the retrieved context.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from utils.observability import time_stage
from utils.evaluation import evaluate_rag_response

logger = logging.getLogger(__name__)

# ── Grounding system prompt ────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an expert real estate document analyst.

RULES — YOU MUST FOLLOW THESE EXACTLY:
1. Answer ONLY using the provided retrieved context. Do NOT use any external or general knowledge.
2. If the answer cannot be fully found in the retrieved context with high certainty, respond with EXACTLY:
   "Insufficient evidence found."
3. For every key claim (especially pricing, RERA numbers, possession dates, carpet area, amenities, or developer details), you must reference the source chunk/page.
4. NEVER invent, guess, or extrapolate. If details are missing, state that evidence is insufficient.
5. If images are referenced in the context, refer to them by their image_id and image_type.
6. Use plain text. Do not use Markdown headers. Keep responses concise and factual.
"""

def generate_grounded_answer(
    question: str,
    retrieved_results: list[dict],
    groq_client=None,
    llm_model: str | None = None,
) -> dict:
    """
    Generate a structured grounded answer from retrieved evidence.
    """
    if groq_client is None:
        groq_client = _get_groq_client()

    model = llm_model or _get_llm_model()

    # ── Separate text and image results ───────────────────────────────────────
    text_results = [r for r in retrieved_results if r.get("source_type") == "text"]
    image_results = [r for r in retrieved_results if r.get("source_type") == "image"]

    # ── Fallback: no retrieved context ─────────────────────────────────────────
    if not text_results and not image_results:
        return _insufficient_evidence_response(question)

    # ── Confidence assessment ──────────────────────────────────────────────────
    confidence = _assess_confidence(text_results)
    
    # If confidence is extremely low (e.g. no high-relevance chunks), abort early
    if confidence == "low":
        return _insufficient_evidence_response(question)

    # ── Build context string ───────────────────────────────────────────────────
    context_str = _build_context_string(text_results, image_results)

    # ── Call LLM with Observability Timing ─────────────────────────────────────
    user_message = f"""Retrieved Context:
{context_str}

Question: {question}

Answer based solely on the retrieved context above. If the answer is not fully supported, output "Insufficient evidence found."
"""

    answer_text = ""
    try:
        with time_stage("llm_generation_times"):
            response = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,      # Extremely low temperature to prevent hallucinations
                max_tokens=1024,
            )
            answer_text = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("LLM generation error: %s", exc)
        return _insufficient_evidence_response(question)

    # ── Post-process: check for refusal indicators ─────────────────────────────
    if _is_insufficient(answer_text):
        return _insufficient_evidence_response(question)

    # ── Build citations ────────────────────────────────────────────────────────
    citations = _build_citations(text_results)
    images = _build_image_refs(image_results)

    # ── Run RAG Evaluation on response ─────────────────────────────────────────
    try:
        evaluate_rag_response(
            question=question,
            answer=answer_text,
            retrieved_results=retrieved_results,
            citations=citations
        )
    except Exception as exc:
        logger.debug("Failed to record evaluation metrics: %s", exc)

    return {
        "question": question,
        "answer": answer_text,
        "citations": citations,
        "images": images,
        "confidence": confidence,
        "retrieved_context": [
            {
                "citation_id": r.get("citation_id"),
                "source_type": r.get("source_type"),
                "content": (r.get("content") or "")[:400],
                "document_id": r.get("document_id"),
                "page_number": r.get("page_number"),
                "score": round(r.get("score", 0.0), 4),
            }
            for r in retrieved_results
        ],
    }


# ── Private helpers ────────────────────────────────────────────────────────────

def _get_groq_client():
    from groq import Groq
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def _get_llm_model() -> str:
    from config import LLM_MODEL
    return LLM_MODEL


def _assess_confidence(text_results: list[dict]) -> str:
    """Assess retrieval confidence based on top rerank scores."""
    if not text_results:
        return "low"
    
    top_score = text_results[0].get("confidence_score", text_results[0].get("score", 0.0))
    
    # If using cross-encoder reranker, score is already sigmoid-mapped.
    # High confidence: top chunk score >= 0.70
    # Medium confidence: top chunk score >= 0.40
    # Low confidence: top chunk score < 0.40
    if top_score >= 0.65:
        return "high"
    elif top_score >= 0.40:
        return "medium"
    return "low"


def _build_context_string(text_results: list[dict], image_results: list[dict]) -> str:
    parts: list[str] = []

    for i, r in enumerate(text_results[:8]):
        meta = r.get("metadata", {})
        doc_id = r.get("document_id") or meta.get("document_id", "unknown")
        page = r.get("page_number") or meta.get("page_number", "?")
        source = meta.get("source_file", "")
        content = (r.get("content") or "").strip()
        parts.append(
            f"[TEXT {i+1}] chunk_id={r.get('id')} document_id={doc_id} page={page} file={source}\n{content}"
        )

    for i, r in enumerate(image_results[:4]):
        meta = r.get("metadata", {})
        doc_id = r.get("document_id") or meta.get("document_id", "unknown")
        page = r.get("page_number") or meta.get("page_number", "?")
        img_id = r.get("image_id") or meta.get("image_id", "")
        img_type = meta.get("image_type", "image")
        caption = meta.get("caption", "")
        img_path = r.get("image_path") or meta.get("image_path", "")
        parts.append(
            f"[IMAGE {i+1}] document_id={doc_id} page={page} image_id={img_id} "
            f"type={img_type} path={img_path}\nCaption: {caption}"
        )

    return "\n\n".join(parts)


def _is_insufficient(text: str) -> bool:
    lower = text.lower()
    markers = (
        "insufficient evidence",
        "data not available",
        "not found in",
        "not present in",
        "not mentioned in",
        "no information",
        "cannot find",
        "not provided",
    )
    return any(m in lower for m in markers) or len(text.strip()) < 10


def _build_citations(text_results: list[dict]) -> list[dict]:
    citations: list[dict] = []
    for r in text_results[:6]:
        meta = r.get("metadata", {})
        content = (r.get("content") or "").strip()
        ocr_used = bool(r.get("ocr_used", meta.get("ocr_used", False)))
        source_type = r.get("source_type", meta.get("source_type", "pdf_text"))
        citations.append(
            {
                "citation_id": r.get("citation_id", ""),
                "document_id": r.get("document_id") or meta.get("document_id", ""),
                "source_file": meta.get("source_file", ""),
                "page_number": r.get("page_number") or meta.get("page_number"),
                "source_type": source_type,
                "section": meta.get("section_title", "General"),
                "chunk_id": r.get("id", ""),
                "builder": meta.get("builder", ""),
                "project": meta.get("project", ""),
                "snippet": content[:300] if content else "",
                "ocr_used": ocr_used,
            }
        )
    return citations


def _build_image_refs(image_results: list[dict]) -> list[dict]:
    images: list[dict] = []
    for r in image_results[:4]:
        meta = r.get("metadata", {})
        doc_id = r.get("document_id") or meta.get("document_id", "")
        img_id = r.get("image_id") or meta.get("image_id", "")
        img_path = r.get("image_path") or meta.get("image_path", "")

        # Build accessible image URL from existing route
        image_url = f"/documents/{doc_id}/images/{img_id}" if doc_id and img_id else img_path

        images.append(
            {
                "document_id": doc_id,
                "image_id": img_id,
                "image_url": image_url,
                "page_number": r.get("page_number") or meta.get("page_number"),
                "image_type": meta.get("image_type", "other"),
                "caption": meta.get("caption", ""),
            }
        )
    return images


def _insufficient_evidence_response(question: str) -> dict:
    return {
        "question": question,
        "answer": "Insufficient evidence found.",
        "citations": [],
        "images": [],
        "confidence": "low",
        "retrieved_context": [],
    }
