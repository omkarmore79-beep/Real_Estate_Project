"""
PDF Processor — extracts text (direct + OCR fallback) and full-page PNG renders using PyMuPDF and PaddleOCR.
"""

from __future__ import annotations

import logging
import os
import uuid
from config import OCR_ENABLED
from ingestion.ocr_service import should_run_ocr, run_ocr_on_page_pixmap, merge_extracted_and_ocr_text

logger = logging.getLogger(__name__)

# DPI for page renders
_DEFAULT_DPI = 160

def process_pdf(
    pdf_path: str,
    document_id: str,
    source_file: str,
    output_folder: str | None = None,
    image_base_path: str | None = None,
) -> dict:
    """
    Extract text and images from a PDF. If direct extraction results in poor text,
    it falls back to running OCR on the page render.

    Parameters
    ----------
    pdf_path:         Absolute path to the PDF file.
    document_id:      Unique document identifier.
    source_file:      Original filename (for metadata).
    output_folder:    Directory to save page-render PNGs.
                      Defaults to a temp ``images/`` sub-folder next to the PDF.
    image_base_path:  URL prefix for image paths stored in metadata.
                      Defaults to ``documents/{document_id}/images``.

    Returns
    -------
    dict with keys:
        document_id, source_file, total_pages, pages, full_text, warnings
    """
    import fitz  # PyMuPDF

    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(pdf_path), "images")
    if image_base_path is None:
        image_base_path = f"documents/{document_id}/images"

    os.makedirs(output_folder, exist_ok=True)

    pages: list[dict] = []
    full_text_parts: list[str] = []
    warnings: list[str] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", pdf_path, exc)
        raise

    with doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1

            # ── 1. Text extraction (direct) ───────────────────────────────────
            pdf_text = page.get_text("text").strip()

            # ── 2. Full-page image render ─────────────────────────────────────
            image_id = f"page_{page_number}"
            filename = f"page_{page_number}.png"
            output_path = os.path.join(output_folder, filename)

            image_saved = False
            try:
                pix = page.get_pixmap(dpi=_DEFAULT_DPI, alpha=False)
                pix.save(output_path)
                image_saved = True
            except Exception as exc:
                warn_msg = f"Failed to render page {page_number} image: {exc}"
                logger.warning(warn_msg)
                warnings.append(warn_msg)

            # ── 3. OCR (if text is empty or too short) ─────────────────────────
            ocr_text = ""
            ocr_confidence = 1.0
            ocr_used = False

            if should_run_ocr(pdf_text):
                logger.info("[%s] Page %d is image-heavy or scanned. Running OCR...", document_id, page_number)
                try:
                    from storage.doc_status import set_status
                    set_status(document_id, "running_ocr")
                except Exception:
                    pass
                ocr_result = run_ocr_on_page_pixmap(page)

                if ocr_result.text:
                    ocr_text = ocr_result.text
                    ocr_confidence = ocr_result.confidence
                    ocr_used = True
                if ocr_result.warnings:
                    warnings.extend([f"Page {page_number}: {w}" for w in ocr_result.warnings])

            # ── 4. Merge text ─────────────────────────────────────────────────
            merged_text = merge_extracted_and_ocr_text(pdf_text, ocr_text)
            full_text_parts.append(f"\n--- Page {page_number} ---\n{merged_text}")

            image_record = {
                "image_id": image_id,
                "page_number": page_number,
                "page": page_number,  # legacy compat
                "image_path": f"{image_base_path}/{image_id}",
                "local_path": output_path if image_saved else None,
            }

            pages.append(
                {
                    "page_number": page_number,
                    "text": merged_text,
                    "pdf_text": pdf_text,
                    "ocr_text": ocr_text,
                    "ocr_confidence": ocr_confidence,
                    "ocr_used": ocr_used,
                    "images": [image_record],
                }
            )

    full_text = "\n".join(full_text_parts)

    logger.info(
        "PDF processed: document_id=%s, pages=%d, source=%s, ocr_runs=%d",
        document_id,
        len(pages),
        source_file,
        sum(1 for p in pages if p["ocr_used"]),
    )

    return {
        "document_id": document_id,
        "source_file": source_file,
        "total_pages": len(pages),
        "pages": pages,
        "full_text": full_text,
        "warnings": warnings,
    }

def get_all_images(processed: dict) -> list[dict]:
    """Flatten all image records from a processed PDF dict."""
    images = []
    for page in processed.get("pages", []):
        images.extend(page.get("images", []))
    return images
