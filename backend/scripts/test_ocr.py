import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure we can load modules from backend/
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(BACKEND_DIR / ".env")

if len(sys.argv) < 2:
    print("Usage: python scripts/test_ocr.py <image_or_pdf_path>")
    sys.exit(1)

target_path = sys.argv[1]
if not os.path.exists(target_path):
    print(f"Error: Path '{target_path}' does not exist.")
    sys.exit(1)

print(f"Testing OCR on path: {target_path}")

from ingestion.ocr_service import run_ocr_on_image, get_ocr_engine

engine = get_ocr_engine()
if not engine:
    print("Error: PaddleOCR failed to initialize (or OCR_ENABLED=false).")
    sys.exit(1)

try:
    if target_path.lower().endswith(".pdf"):
        import fitz
        from PIL import Image
        from io import BytesIO
        
        doc = fitz.open(target_path)
        if len(doc) == 0:
            print("PDF has 0 pages.")
            sys.exit(1)
        print("Rendering first page for OCR...")
        page = doc[0]
        pix = page.get_pixmap(dpi=150)
        img = Image.open(BytesIO(pix.tobytes("png")))
        result = run_ocr_on_image(img)
    else:
        result = run_ocr_on_image(target_path)
        
    print("\n--- OCR RESULTS ---")
    print(f"Engine: {result.engine}")
    print(f"Text length: {len(result.text)} characters")
    print(f"Average confidence: {result.confidence:.4f}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
    print("\nSample text (first 300 chars):")
    print(result.text[:300])
except Exception as e:
    print("OCR run FAILED with exception:")
    print(str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
