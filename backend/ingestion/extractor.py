from pdf2image import convert_from_path
import fitz
from ingestion.ocr_parser import extract_text_ocr_images

POPPLER_PATH = r"C:\poppler\Library\bin"   # 🔴 update if your path differs


def extract_document(path):
    doc = fitz.open(path)

    # ✅ Use poppler path explicitly
    images = convert_from_path(path, poppler_path=POPPLER_PATH)

    final_text = ""

    for i, page in enumerate(doc):
        text = page.get_text().strip()

        if len(text) > 50:
            final_text += f"\n--- Page {i+1} (TEXT) ---\n{text}"
        else:
            final_text += f"\n--- Page {i+1} (OCR) ---\n"
            final_text += extract_text_ocr_images([images[i]])

    return final_text