"""
PDF Processor — extracts text, tables, and diagrams with layout-aware sorting,
multi-column support, and proximal caption association.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from PIL import Image

from config import OCR_ENABLED, OCR_ON_CROPPED_IMAGES
from ingestion.ocr_service import should_run_ocr, run_ocr_on_page_pixmap, merge_extracted_and_ocr_text

logger = logging.getLogger(__name__)

# DPI for renders
_DEFAULT_DPI = 160

# Regex patterns for figures and sections
FIG_PATTERN = re.compile(
    r'\b(Figure|Fig\.?|Diagram|Table|Drawing)\s+([A-Z0-9][A-Z0-9.-]{2,})\b[:.-]?\s*([^\n.]+)',
    re.IGNORECASE
)
FIGURE_CODE_PATTERN = re.compile(r'\b[A-Z]{1,4}\d{3,}[A-Z0-9-]*\b')
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

def sort_blocks_layout_aware(blocks: list, page_width: float) -> list:
    """
    Sort fitz text blocks by reading order, detecting multi-column layout.
    """
    mid_x = page_width / 2
    
    # Classify blocks on left and right sides
    left_blocks = [b for b in blocks if b[2] <= mid_x + 15]
    right_blocks = [b for b in blocks if b[0] >= mid_x - 15]
    
    # If substantial vertical overlap between left and right blocks exists, it's multi-column
    if len(left_blocks) >= 2 and len(right_blocks) >= 2:
        def block_key(b):
            bx0, by0, bx1, by1 = b[0], b[1], b[2], b[3]
            center_x = (bx0 + bx1) / 2
            col = 0 if center_x < mid_x else 1
            return (col, by0)
        return sorted(blocks, key=block_key)
    else:
        return sorted(blocks, key=lambda b: b[1])

def find_image_caption(image_rect: tuple, text_blocks: list) -> tuple[str, str]:
    """
    Find figure captions or titles vertically proximal to the image bounds.
    """
    ix0, iy0, ix1, iy1 = image_rect
    min_dist = 999999.0
    best_caption = ""
    best_explanation = ""
    
    for b in text_blocks:
        bx0, by0, bx1, by1, text, _, _ = b
        text_clean = text.strip()
        if not text_clean:
            continue
            
        match = FIG_PATTERN.search(text_clean)
        code_match = FIGURE_CODE_PATTERN.search(text_clean)
        if match or code_match:
            dist = 999999.0
            if by1 <= iy0:  # block is above image
                dist = iy0 - by1
            elif by0 >= iy1:  # block is below image
                dist = by0 - iy1
            else:  # overlapping vertically
                dist = 0
                
            if dist < 150 and dist < min_dist:
                min_dist = dist
                if match:
                    best_caption = f"{match.group(1)} {match.group(2)}: {match.group(3).strip()}"
                else:
                    best_caption = f"Diagram {code_match.group(0)}: {text_clean[:180]}"
                best_explanation = text_clean
                
    return best_caption, best_explanation

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
    parents: dict[str, dict] = {}
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
            page_rect = page.rect
            page_width = page_rect.width
            page_height = page_rect.height

            # 1. Table extraction
            tables = []
            tables_markdown = []
            try:
                if hasattr(page, "find_tables"):
                    tabs = page.find_tables()
                    if tabs and tabs.tables:
                        for tab in tabs.tables:
                            df = tab.extract()
                            if df and len(df) > 0:
                                headers = [str(h or "").strip() for h in df[0]]
                                table_md = []
                                table_md.append("| " + " | ".join(headers) + " |")
                                table_md.append("| " + " | ".join(["---"] * len(headers)) + " |")
                                for row in df[1:]:
                                    table_md.append("| " + " | ".join([str(c or "").strip() for c in row]) + " |")
                                tables.append(tab)
                                tables_markdown.append((tab.bbox, "\n".join(table_md)))
            except Exception as tab_exc:
                logger.debug("PDF table extraction skipped/failed on page %d: %s", page_number, tab_exc)

            # 2. Text block extraction (avoiding overlap with tables)
            raw_blocks = page.get_text("blocks")
            content_blocks = []
            for b in raw_blocks:
                x0, y0, x1, y1, text, block_no, block_type = b
                # Skip header/footer noise
                if (y0 < 35 and len(text.strip()) < 50) or (y1 > page_height - 35 and len(text.strip()) < 30):
                    continue
                if not text.strip():
                    continue
                
                # Check if block center lies inside any extracted table bbox
                is_inside = False
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                for tab in tables:
                    tx0, ty0, tx1, ty1 = tab.bbox
                    if tx0 - 5 <= cx <= tx1 + 5 and ty0 - 5 <= cy <= ty1 + 5:
                        is_inside = True
                        break
                if is_inside:
                    continue
                content_blocks.append(b)

            # Sort content blocks layout-awarely
            sorted_text_blocks = sort_blocks_layout_aware(content_blocks, page_width)

            # Interleave text blocks and tables in vertical (reading) order
            elements = []
            for b in sorted_text_blocks:
                elements.append((b[1], "text", b[4].strip(), (b[0], b[1], b[2], b[3])))
            for bbox, md in tables_markdown:
                elements.append((bbox[1], "table", md, bbox))
            
            elements.sort(key=lambda x: x[0])
            
            pdf_text_parts = []
            for el in elements:
                if el[1] == "table":
                    pdf_text_parts.append(f"\n[Table]\n{el[2]}\n")
                else:
                    pdf_text_parts.append(el[2])
            
            pdf_text = "\n\n".join(pdf_text_parts).strip()

            # 3. Section heading detection
            current_section = _detect_heading(pdf_text, current_heading=current_section)

            # 4. OCR fallback (if page text is empty/image-heavy)
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

            # 5. Extract images on the page & assign captions
            image_records = []
            image_list = page.get_images(full=True)

            if image_list:
                for img_idx, img in enumerate(image_list):
                    xref = img[0]
                    rects = page.get_image_rects(xref)
                    
                    for r_idx, rect in enumerate(rects):
                        if abs(rect.width) < 100 or abs(rect.height) < 100 or (abs(rect.width) * abs(rect.height)) < 15000:
                            continue

                        image_id = f"page_{page_number}_img_{xref}_{img_idx}_{r_idx}"
                        filename = f"{image_id}.png"
                        output_path = os.path.join(output_folder, filename)
                        
                        image_saved = False
                        try:
                            pix = page.get_pixmap(clip=rect, dpi=_DEFAULT_DPI)
                            if pix.width < 120 or pix.height < 120 or (pix.width * pix.height) < 20000:
                                continue
                            pix.save(output_path)
                            image_saved = True
                        except Exception as exc:
                            logger.warning("Failed to crop image on page %d: %s", page_number, exc)

                        if image_saved:
                            # Run OCR on cropped images if large
                            inner_ocr_labels = ""
                            if OCR_ENABLED and OCR_ON_CROPPED_IMAGES and pix.width >= 350 and pix.height >= 350:
                                try:
                                    from ingestion.ocr_service import get_ocr_engine
                                    engine = get_ocr_engine()
                                    if engine:
                                        ocr_res = engine.ocr(output_path)
                                        if ocr_res and ocr_res[0]:
                                            inner_ocr_labels = " ".join([line[1][0] for line in ocr_res[0] if line[1][1] > 0.5])
                                except Exception as ocr_err:
                                    logger.debug("Failed OCR on cropped image %s: %s", image_id, ocr_err)

                            # Proximity-based figure caption matching
                            best_caption, best_explanation = find_image_caption((rect.x0, rect.y0, rect.x1, rect.y1), content_blocks)
                            
                            fig_num = "Page Image"
                            caption = f"Embedded image on page {page_number}"
                            explanation = merged_text[:600]
                            
                            if best_caption:
                                caption = best_caption
                                explanation = best_explanation
                                fig_match = re.search(r'\b(Figure|Fig\.?|Diagram|Table|Drawing)\s+(\d+[-.\d]*)\b', best_caption, re.IGNORECASE)
                                if fig_match:
                                    fig_num = f"{fig_match.group(1)} {fig_match.group(2)}"

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

            # Tables are searchable as text, but a user who asks to *show* a
            # schedule/specification needs the table region rather than its page.
            # Persist a crop for each detected table so it can be visually ranked.
            for table_index, (bbox, table_markdown) in enumerate(tables_markdown):
                image_id = f"page_{page_number}_table_{table_index + 1}"
                output_path = os.path.join(output_folder, f"{image_id}.png")
                try:
                    table_rect = fitz.Rect(bbox)
                    clip = fitz.Rect(table_rect.x0 - 8, table_rect.y0 - 8, table_rect.x1 + 8, table_rect.y1 + 8)
                    clip &= page.rect
                    pix = page.get_pixmap(clip=clip, dpi=_DEFAULT_DPI, alpha=False)
                    if pix.width < 120 or pix.height < 80:
                        continue
                    pix.save(output_path)
                    image_records.append({
                        "image_id": image_id,
                        "page_number": page_number,
                        "page": page_number,
                        "figure_number": f"Table {table_index + 1}",
                        "caption": f"Table on page {page_number}: {table_markdown.splitlines()[0][:180]}",
                        "surrounding_explanation": table_markdown[:1000],
                        "ocr_labels": "",
                        "section": current_section,
                        "image_type": "table",
                        "image_path": f"{image_base_path}/{image_id}",
                        "local_path": output_path,
                    })
                except Exception as exc:
                    logger.debug("Failed to render table crop on page %d: %s", page_number, exc)

            # Many technical manuals encode diagrams as vector drawings rather
            # than embedded raster images.  When a page names a figure/code,
            # render its drawing bounds as one image-only asset.
            has_figure_reference = bool(FIG_PATTERN.search(pdf_text) or FIGURE_CODE_PATTERN.search(pdf_text))
            if not image_records and has_figure_reference:
                try:
                    drawing_rects = [fitz.Rect(d["rect"]) for d in page.get_drawings() if d.get("rect")]
                    drawing_rects = [r for r in drawing_rects if r.get_area() >= 5_000]
                    if drawing_rects:
                        region = drawing_rects[0]
                        for rect in drawing_rects[1:]:
                            region |= rect
                        if region.get_area() < page.rect.get_area() * 0.85:
                            image_id = f"page_{page_number}_diagram"
                            output_path = os.path.join(output_folder, f"{image_id}.png")
                            clip = fitz.Rect(region.x0 - 12, region.y0 - 12, region.x1 + 12, region.y1 + 12) & page.rect
                            pix = page.get_pixmap(clip=clip, dpi=_DEFAULT_DPI, alpha=False)
                            if pix.width >= 120 and pix.height >= 120:
                                pix.save(output_path)
                                caption, explanation = find_image_caption(tuple(clip), content_blocks)
                                image_records.append({
                                    "image_id": image_id,
                                    "page_number": page_number,
                                    "page": page_number,
                                    "figure_number": "Diagram",
                                    "caption": caption or f"Technical diagram on page {page_number}",
                                    "surrounding_explanation": explanation or merged_text[:800],
                                    "ocr_labels": "",
                                    "section": current_section,
                                    "image_type": "diagram",
                                    "image_path": f"{image_base_path}/{image_id}",
                                    "local_path": output_path,
                                })
                except Exception as exc:
                    logger.debug("Vector diagram extraction skipped on page %d: %s", page_number, exc)

            # Save full page render as fallback if scanned or image-heavy without sub-images
            if not image_records and (ocr_used or len(pdf_text) < 100):
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
                    image_records.append({
                        "image_id": image_id,
                        "page_number": page_number,
                        "page": page_number,
                        "figure_number": "Page Image",
                        "caption": f"Full page layout {page_number}",
                        "image_type": "full_page",
                        "surrounding_explanation": merged_text[:600],
                        "ocr_labels": ocr_text,
                        "section": current_section,
                        "image_path": f"{image_base_path}/{image_id}",
                        "local_path": output_path,
                    })

            # Save page context
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

            # Save parent record
            parent_id = f"{document_id}_page_{page_number}"
            parents[parent_id] = {
                "parent_id": parent_id,
                "document_id": document_id,
                "page_number": page_number,
                "text": merged_text,
                "section": current_section,
                "source_file": source_file,
            }

    full_text = "\n".join(full_text_parts)
    return {
        "document_id": document_id,
        "source_file": source_file,
        "total_pages": len(pages),
        "pages": pages,
        "parents": parents,
        "full_text": full_text,
        "warnings": warnings,
    }

def get_all_images(processed: dict) -> list[dict]:
    """Flatten all image records from a processed PDF dict."""
    images = []
    for page in processed.get("pages", []):
        images.extend(page.get("images", []))
    return images
