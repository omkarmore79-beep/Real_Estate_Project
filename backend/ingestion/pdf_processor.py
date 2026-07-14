"""
PDF Processor — extracts text and diagrams with surrounding context and figure numbers.
Supports page renders and embedded diagram extraction.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from PIL import Image

from config import OCR_ENABLED
from ingestion.ocr_service import should_run_ocr, run_ocr_on_page_pixmap, merge_extracted_and_ocr_text

logger = logging.getLogger(__name__)

# DPI for renders
_DEFAULT_DPI = 160

# Regex patterns for figures and sections
FIG_PATTERN = re.compile(
    r'\b(Figure|Fig\.?|Diagram|Table|Drawing)\s+(\d+[-.\d]*)\b[:.-]?\s*([^\n.]+)', 
    re.IGNORECASE
)
SECTION_PATTERN = re.compile(
    r'^(?:[0-9]+(?:\.[0-9]+)+|SECTION|Section)\s+[A-Za-z]', 
    re.IGNORECASE
)

def _detect_heading(text: str, current_heading: str) -> str:
    """Detect if any line in text looks like a heading."""
    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 80 and SECTION_PATTERN.match(line):
            return line
    return current_heading

def process_pdf(
    pdf_path: str,
    document_id: str,
    source_file: str,
    output_folder: str | None = None,
    image_base_path: str | None = None,
) -> dict:
    import fitz  # PyMuPDF

    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(pdf_path), "images")
    if image_base_path is None:
        image_base_path = f"documents/{document_id}/images"

    os.makedirs(output_folder, exist_ok=True)

    pages: list[dict] = []
    full_text_parts: list[str] = []
    warnings: list[str] = []
    
    current_section = "General"

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", pdf_path, exc)
        raise

    with doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1

            # 1. Direct text extraction
            pdf_text = page.get_text("text").strip()

            # 2. Section heading detection
            current_section = _detect_heading(pdf_text, current_heading=current_section)

            # 3. OCR (if text is empty or too short)
            ocr_text = ""
            ocr_confidence = 1.0
            ocr_used = False

            if should_run_ocr(pdf_text):
                logger.info("[%s] Page %d is image-heavy. Running OCR...", document_id, page_number)
                ocr_result = run_ocr_on_page_pixmap(page)
                if ocr_result.text:
                    ocr_text = ocr_result.text
                    ocr_confidence = ocr_result.confidence
                    ocr_used = True
                if ocr_result.warnings:
                    warnings.extend([f"Page {page_number}: {w}" for w in ocr_result.warnings])

            # Merge text
            merged_text = merge_extracted_and_ocr_text(pdf_text, ocr_text)
            full_text_parts.append(f"\n--- Page {page_number} ---\n{merged_text}")

            # 4. Extract figures/diagrams references on the page
            fig_matches = []
            for match in FIG_PATTERN.finditer(merged_text):
                fig_type = match.group(1)
                fig_num = match.group(2)
                caption = match.group(3).strip()
                
                # Context surrounding the figure mention
                start = max(0, match.start() - 300)
                end = min(len(merged_text), match.end() + 300)
                explanation = merged_text[start:end].strip()

                fig_matches.append({
                    "figure_number": f"{fig_type} {fig_num}",
                    "caption": caption,
                    "surrounding_explanation": explanation,
                })

            # 5. Extract images on the page
            image_records = []
            image_list = page.get_images(full=True)

            # Case A: Page has raster images
            if image_list:
                for img_idx, img in enumerate(image_list):
                    xref = img[0]
                    rects = page.get_image_rects(xref)
                    
                    for r_idx, rect in enumerate(rects):
                        image_id = f"page_{page_number}_img_{xref}_{img_idx}_{r_idx}"
                        filename = f"{image_id}.png"
                        output_path = os.path.join(output_folder, filename)
                        
                        image_saved = False
                        try:
                            # Crop and render the specific image area
                            pix = page.get_pixmap(clip=rect, dpi=_DEFAULT_DPI)
                            pix.save(output_path)
                            image_saved = True
                        except Exception as exc:
                            logger.warning("Failed to crop image on page %d: %s", page_number, exc)

                        if image_saved:
                            # Run OCR on the diagram to get inner labels
                            inner_ocr_labels = ""
                            if OCR_ENABLED:
                                try:
                                    from ingestion.ocr_service import get_ocr_engine
                                    engine = get_ocr_engine()
                                    if engine:
                                        ocr_res = engine.ocr(output_path, cls=False)
                                        if ocr_res and ocr_res[0]:
                                            inner_ocr_labels = " ".join([line[1][0] for line in ocr_res[0] if line[1][1] > 0.5])
                                except Exception as ocr_err:
                                    logger.debug("Failed OCR on cropped image %s: %s", image_id, ocr_err)

                            # Assign to a matched figure reference or default
                            fig_num = "Page Image"
                            caption = f"Embedded image on page {page_number}"
                            explanation = merged_text[:600]
                            
                            if img_idx < len(fig_matches):
                                match_info = fig_matches[img_idx]
                                fig_num = match_info["figure_number"]
                                caption = match_info["caption"]
                                explanation = match_info["surrounding_explanation"]

                            image_records.append({
                                "image_id": image_id,
                                "page_number": page_number,
                                "page": page_number,
                                "figure_number": fig_num,
                                "caption": caption,
                                "surrounding_explanation": explanation,
                                "ocr_labels": inner_ocr_labels,
                                "section": current_section,
                                "image_path": f"{image_base_path}/{image_id}",
                                "local_path": output_path,
                            })

            # Case B: Page has no raster images, but has figure references or could be a vector drawing/layout
            # Or if no raster images were successfully saved, use the full page render
            if not image_records:
                # Save full page render
                image_id = f"page_{page_number}"
                filename = f"page_{page_number}.png"
                output_path = os.path.join(output_folder, filename)
                
                image_saved = False
                try:
                    pix = page.get_pixmap(dpi=_DEFAULT_DPI, alpha=False)
                    pix.save(output_path)
                    image_saved = True
                except Exception as exc:
                    warnings.append(f"Failed to render page {page_number}: {exc}")

                if image_saved:
                    # Run OCR on page to extract labels if needed
                    inner_ocr_labels = ocr_text
                    
                    fig_num = "Page Image"
                    caption = f"Full page layout {page_number}"
                    explanation = merged_text[:600]

                    if fig_matches:
                        match_info = fig_matches[0]
                        fig_num = match_info["figure_number"]
                        caption = match_info["caption"]
                        explanation = match_info["surrounding_explanation"]

                    image_records.append({
                        "image_id": image_id,
                        "page_number": page_number,
                        "page": page_number,
                        "figure_number": fig_num,
                        "caption": caption,
                        "surrounding_explanation": explanation,
                        "ocr_labels": inner_ocr_labels,
                        "section": current_section,
                        "image_path": f"{image_base_path}/{image_id}",
                        "local_path": output_path,
                    })

            pages.append({
                "page_number": page_number,
                "text": merged_text,
                "pdf_text": pdf_text,
                "ocr_text": ocr_text,
                "ocr_confidence": ocr_confidence,
                "ocr_used": ocr_used,
                "section": current_section,
                "images": image_records,
            })

    full_text = "\n".join(full_text_parts)
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
