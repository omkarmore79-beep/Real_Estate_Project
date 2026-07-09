"""
Grounded Answer Generator.

Generates answers ONLY from retrieved evidence using the Groq LLM.
Never uses general knowledge. Every factual claim must be grounded in
the retrieved context.

Critical real-estate fields that must NEVER be hallucinated:
  - Price / cost / rate
  - RERA number / registration ID
  - Possession / handover date
  - Carpet area / super built-up area
  - Legal approvals / NOC
  - Amenities list
  - Builder / developer claims
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Grounding system prompt ────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a strict real estate document analyst.

RULES — YOU MUST FOLLOW THESE EXACTLY:
1. Answer ONLY from the provided retrieved context. Do NOT use any outside knowledge.
2. If the answer is not present in the retrieved context, respond with:
   "Data not available in the uploaded documents."
3. For every factual claim, cite the source (document_id, page number).
4. NEVER invent or guess: prices, RERA numbers, possession dates, carpet area,
   super built-up area, legal approvals, amenity lists, or builder claims.
5. If images are in the context, reference them by image_id and image_type.
6. Comparison questions: compare ONLY documents present in the retrieved context.
7. Be concise. Do not add disclaimers unless information is missing.
8. Use plain language. Do not use markdown headers in your answer.
"""

# ── Confidence thresholds ──────────────────────────────────────────────────────
_HIGH_CONFIDENCE_MIN_CHUNKS = 3
_MEDIUM_CONFIDENCE_MIN_CHUNKS = 1


def generate_grounded_answer(
    question: str,
    retrieved_results: list[dict],
    groq_client=None,
    llm_model: str | None = None,
) -> dict:
    """
    Generate a structured grounded answer from retrieved evidence.

    Parameters
    ----------
    question:          User's question.
    retrieved_results: List of result dicts from hybrid_retriever.retrieve().
    groq_client:       Groq client instance (created lazily if None).
    llm_model:         Override LLM model name.

    Returns
    -------
    dict with keys: question, answer, citations, images, confidence, retrieved_context
    """
    if groq_client is None:
        groq_client = _get_groq_client()

    model = llm_model or _get_llm_model()

    # ── Separate text and image results ───────────────────────────────────────
    text_results = [r for r in retrieved_results if r.get("source_type") == "text"]
    image_results = [r for r in retrieved_results if r.get("source_type") == "image"]

    # ── Confidence assessment ──────────────────────────────────────────────────
    confidence = _assess_confidence(text_results)

    # ── Build context string ───────────────────────────────────────────────────
    context_str = _build_context_string(text_results, image_results)

    # ── Fallback: no content at all ───────────────────────────────────────────
    if not context_str.strip():
        return _no_data_response(question)

    # ── Call LLM ──────────────────────────────────────────────────────────────
    user_message = f"""Retrieved Context:
{context_str}

Question: {question}

Answer based solely on the retrieved context above. Cite document_id and page_number for every key claim. If the answer is not in the context, say: "Data not available in the uploaded documents."
"""

    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,      # low temperature for factual accuracy
            max_tokens=1024,
        )
        answer_text = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("LLM generation error: %s", exc)
        answer_text = "Data not available in the uploaded documents."

    # ── Post-process: detect "not available" answers ───────────────────────────
    if _is_not_found(answer_text):
        confidence = "low"
        answer_text = "Data not available in the uploaded documents."

    # ── Build citations ────────────────────────────────────────────────────────
    citations = _build_citations(text_results)
    images = _build_image_refs(image_results)

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
    n = len(text_results)
    if n >= _HIGH_CONFIDENCE_MIN_CHUNKS:
        return "high"
    if n >= _MEDIUM_CONFIDENCE_MIN_CHUNKS:
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
            f"[TEXT {i+1}] document_id={doc_id} page={page} file={source}\n{content}"
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


def _is_not_found(text: str) -> bool:
    lower = text.lower()
    markers = (
        "data not available",
        "not found in",
        "not present in",
        "not mentioned in",
        "no information",
        "cannot find",
        "not provided",
    )
    return any(m in lower for m in markers)


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
                "section": meta.get("section_title", ""),
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


def _no_data_response(question: str) -> dict:
    return {
        "question": question,
        "answer": "Data not available in the uploaded documents.",
        "citations": [],
        "images": [],
        "confidence": "low",
        "retrieved_context": [],
    }
