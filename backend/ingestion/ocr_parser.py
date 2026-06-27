import pytesseract
from pdf2image import convert_from_path

# ✅ Force correct Tesseract path (no PATH issues anymore)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_ocr_images(images):
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text