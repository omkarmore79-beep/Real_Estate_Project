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

# ── Domain-aware system prompts ───────────────────────────────────────────────
_SYSTEM_PROMPT_EXCAVATOR = """You are an expert Hyundai R215L Excavator Predictive Maintenance Copilot assisting qualified field engineers.

DOMAIN: Industrial Heavy Machinery — Hyundai R215L Hydraulic Excavator
PURPOSE: Fault diagnosis, root cause analysis, troubleshooting, preventive maintenance, and technical document search.

STRICT GROUNDING RULES — FOLLOW EXACTLY:
1. Answer ONLY from the retrieved context (manuals, service bulletins, maintenance logs, field reports, parts catalog).
   Do NOT use general mechanical or engineering knowledge not present in the retrieved text.
2. If the answer cannot be fully supported by the evidence, respond EXACTLY with:
   "I couldn't find sufficient information in the uploaded manuals for this query."
3. MANDATORY CITATIONS: For every technical claim (torque specs, clearances, pressures, part numbers, DTC codes,
   oil grades, intervals), you MUST cite:
   - Document: [source_file]
   - Page: [page_number]
   - Section: [section_path or section]
   - Figure: [figure_number or image_id] if applicable
4. DTC CODES: If a fault code appears in context, always explain: code meaning, affected component, probable cause, and recommended corrective action per the manual.
5. COMPONENT REFERENCES: Use exact Hyundai terminology and part numbers from the retrieved text.
6. DIAGRAMS: When a diagram or schematic is available in image context, reference it by its Figure number, page, and caption.
7. SAFETY: Prefix any safety-critical procedure with a ⚠️ WARNING label.
8. NEVER fabricate, extrapolate, or guess. State when evidence is insufficient.
9. Structure your response clearly: Diagnosis → Root Cause → Corrective Action → References.
"""

_SYSTEM_PROMPT_GENERIC = """You are an expert technical document analyst and assistant.

RULES — YOU MUST FOLLOW THESE EXACTLY:
1. Answer ONLY using the provided retrieved context. Do NOT use any external or general knowledge.
2. If the answer cannot be fully found in the retrieved context with high certainty, respond with EXACTLY:
   "I couldn't find enough information in the uploaded documents."
3. For every key claim, reference the source document, page, and section.
4. NEVER invent, guess, or extrapolate. If details are missing, state that evidence is insufficient.
5. If images/diagrams are referenced in the context, refer to them by their image_id, page, figure_number, and caption.
6. Keep responses concise, factual, and strictly grounded.
"""

# Legacy alias — defaults to generic
_SYSTEM_PROMPT = _SYSTEM_PROMPT_GENERIC

def generate_grounded_answer(
    question: str,
    retrieved_results: list[dict],
    groq_client=None,
    llm_model: str | None = None,
    domain: str | None = None,
) -> dict:
    """
    Generate a structured grounded answer from retrieved evidence.
    Uses domain-specific system prompts for excavator vs. real_estate.
    """
    if groq_client is None:
        groq_client = _get_groq_client()

    model = llm_model or _get_llm_model()

    # Auto-detect domain from first result if not provided
    if not domain:
        for r in retrieved_results:
            d = r.get("metadata", {}).get("domain") or r.get("domain")
            if d:
                domain = d
                break

    system_prompt = _SYSTEM_PROMPT_EXCAVATOR if domain == "excavator" else _SYSTEM_PROMPT_GENERIC

    # ── Separate text and image results ───────────────────────────────────────
    text_results = [r for r in retrieved_results if r.get("source_type") in ("text", "context_window")][:8]
    image_results = [r for r in retrieved_results if r.get("source_type") == "image"]

    # ── Fallback: no retrieved context ─────────────────────────────────────────
    if not text_results and not image_results:
        return _insufficient_evidence_response(question, domain)

    # ── Confidence assessment ──────────────────────────────────────────────────
    primary_text = [r for r in text_results if r.get("source_type") == "text"]
    confidence = _assess_confidence(primary_text if primary_text else text_results)
    
    # If confidence is extremely low (e.g. no high-relevance chunks), abort early
    if confidence == "low":
        return _insufficient_evidence_response(question, domain)

    # ── Build context string ───────────────────────────────────────────────────
    context_str = _build_context_string(text_results, image_results, domain=domain)

    # ── Call LLM with Observability Timing ─────────────────────────────────────
    not_found_msg = (
        "I couldn't find sufficient information in the uploaded manuals for this query."
        if domain == "excavator"
        else "I couldn't find enough information in the uploaded documents."
    )
    user_message = f"""Retrieved Context:
{context_str}

Question: {question}

Answer based solely on the retrieved context above. If the answer is not fully supported, respond with EXACTLY:
"{not_found_msg}"
"""

    answer_text = ""
    try:
        with time_stage("llm_generation_times"):
            response = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.05 if domain == "excavator" else 0.1,
                max_tokens=1536 if domain == "excavator" else 1024,
            )
            answer_text = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("LLM generation error: %s", exc)
        return _insufficient_evidence_response(question, domain)

    # ── Post-process: check for refusal indicators ─────────────────────────────
    if _is_insufficient(answer_text):
        return _insufficient_evidence_response(question, domain)

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


def _build_context_string(text_results: list[dict], image_results: list[dict], domain: str | None = None) -> str:
    parts: list[str] = []

    for i, r in enumerate(text_results[:8]):
        meta = r.get("metadata", {})
        doc_id = r.get("document_id") or meta.get("document_id", "unknown")
        page = r.get("page_number") or meta.get("page_number", "?")
        source = meta.get("source_file", "")
        content = (r.get("content") or "").strip()
        src_type = r.get("source_type", "text")
        label = "CONTEXT" if src_type == "context_window" else f"TEXT {i+1}"

        if domain == "excavator":
            section_path = meta.get("section_path") or meta.get("section", "")
            dtc_codes = ", ".join(meta.get("dtc_codes") or [])
            comp_tags = ", ".join(meta.get("component_tags") or [])
            figure_number = meta.get("figure_number", "")
            doc_type = meta.get("doc_type", "")
            version = meta.get("version", "")
            header = f"[{label}] doc_type={doc_type} version={version} page={page} file={source}"
            if section_path:
                header += f" section={section_path}"
            if comp_tags:
                header += f" components=[{comp_tags}]"
            if dtc_codes:
                header += f" dtc_codes=[{dtc_codes}]"
            if figure_number:
                header += f" figure={figure_number}"
            parts.append(f"{header}\n{content}")
        else:
            parts.append(f"[{label}] page={page} file={source}\n{content}")

    for i, r in enumerate(image_results[:4]):
        meta = r.get("metadata", {})
        doc_id = r.get("document_id") or meta.get("document_id", "unknown")
        page = r.get("page_number") or meta.get("page_number", "?")
        img_id = r.get("image_id") or meta.get("image_id", "")
        img_type = meta.get("image_type", "image")
        caption = meta.get("caption", "")
        img_path = r.get("image_path") or meta.get("image_path", "")
        figure_number = meta.get("figure_number", "")
        section_path = meta.get("section_path", "")
        ocr_labels = meta.get("ocr_labels", "")
        img_header = (
            f"[IMAGE {i+1}] document_id={doc_id} page={page} image_id={img_id} "
            f"type={img_type} path={img_path}"
        )
        if figure_number:
            img_header += f" figure={figure_number}"
        if section_path:
            img_header += f" section={section_path}"
        parts.append(f"{img_header}\nCaption: {caption}" + (f"\nOCR: {ocr_labels[:300]}" if ocr_labels else ""))

    return "\n\n".join(parts)


def _is_insufficient(text: str) -> bool:
    lower = text.lower()
    markers = (
        "insufficient evidence",
        "i couldn't find enough information",
        "i couldn't find sufficient information",
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
        
        # Extract section
        section = meta.get("section", meta.get("section_title", "General"))
        
        citations.append(
            {
                "citation_id": r.get("citation_id", ""),
                "document_id": r.get("document_id") or meta.get("document_id", ""),
                "source_file": meta.get("source_file", ""),
                "page_number": r.get("page_number") or meta.get("page_number"),
                "source_type": source_type,
                "section": section,
                "chunk_id": r.get("id", ""),
                "builder": meta.get("builder", ""),
                "project": meta.get("project", ""),
                
                # Excavator and Multimodal details
                "doc_type": meta.get("doc_type", ""),
                "machine_model": meta.get("machine_model", ""),
                "component_tags": meta.get("component_tags", []),
                "dtc_codes": meta.get("dtc_codes", []),
                "section_path": meta.get("section_path", ""),
                "figure_number": meta.get("figure_number", ""),
                "image_id": meta.get("image_id", ""),
                "version": meta.get("version", "1.0"),
                "hash": meta.get("hash", ""),
                "timestamp": meta.get("timestamp", ""),
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
                
                # Additional diagram context
                "figure_number": meta.get("figure_number", ""),
                "section": meta.get("section", "General"),
                "explanation": meta.get("surrounding_explanation", ""),
            }
        )
    return images


def _insufficient_evidence_response(question: str, domain: str | None = None) -> dict:
    msg = (
        "I couldn't find sufficient information in the uploaded manuals for this query."
        if domain == "excavator"
        else "I couldn't find enough information in the uploaded documents."
    )
    return {
        "question": question,
        "answer": msg,
        "citations": [],
        "images": [],
        "confidence": "low",
        "retrieved_context": [],
    }
