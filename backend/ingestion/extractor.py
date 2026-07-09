"""
Legacy text extractor — uses PyMuPDF only, NO OCR.

This module is kept for backward compatibility with the LLM formatter
(format_with_llm) which uses the extracted text to produce structured JSON.
OCR has been completely removed. Pages with sparse text simply return
whatever the PDF text layer contains (may be empty for image-only pages).

For the RAG pipeline, use ingestion.pdf_processor.process_pdf() instead,
which returns structured per-page output.
"""

import fitz  # PyMuPDF


def extract_document(path: str) -> str:
    """
    Extract all text from a PDF using PyMuPDF's direct text layer.
    No OCR. No external dependencies.

    Returns a single string with page markers:
        --- Page N (TEXT) ---
        <page text>
    """
    doc = fitz.open(path)
    final_text = ""

    for i, page in enumerate(doc):
        text = page.get_text().strip()
        final_text += f"\n--- Page {i + 1} (TEXT) ---\n{text}"

    doc.close()
    return final_text