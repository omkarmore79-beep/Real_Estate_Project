import logging
import os
import cv2
import numpy as np
from PIL import Image
from config import OCR_ENABLED, OCR_ENGINE, OCR_MIN_TEXT_LENGTH, OCR_MIN_CONFIDENCE

logger = logging.getLogger(__name__)

_ocr_engine = None

class OCRResult:
    """Structured container for OCR extraction results."""
    def __init__(self, text: str, confidence: float, engine: str, warnings: list[str] = None):
        self.text = text
        self.confidence = confidence
        self.engine = engine
        self.warnings = warnings or []

def get_ocr_engine():
    """Lazy initialize and return PaddleOCR engine."""
    global _ocr_engine
    if not OCR_ENABLED:
        return None
    if _ocr_engine is None:
        try:
            from paddleocr import PaddleOCR
            # Suppress excessive logging from PaddleOCR
            logging.getLogger("ppocr").setLevel(logging.WARNING)
            _ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            logger.info("PaddleOCR engine initialized successfully.")
        except Exception as exc:
            logger.error("Failed to initialize PaddleOCR engine: %s", exc)
            _ocr_engine = None
    return _ocr_engine

def preprocess_image_for_ocr(image_path_or_pil):
    """
    Preprocess image for better OCR accuracy:
      - Grayscale conversion
      - Upscale if width is small
      - Sharpening filter to boost contrast/definition
    """
    try:
        if isinstance(image_path_or_pil, str):
            img = cv2.imread(image_path_or_pil)
        elif isinstance(image_path_or_pil, Image.Image):
            img = cv2.cvtColor(np.array(image_path_or_pil), cv2.COLOR_RGB2BGR)
        else:
            img = np.array(image_path_or_pil)

        if img is None:
            return None

        # Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Upscale if low-res (width < 1000px)
        h, w = gray.shape[:2]
        if w < 1000:
            scale = 2.0
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # Sharpness / Contrast enhancement
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)

        # Convert back to BGR for PaddleOCR input compatibility
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        logger.warning("Error preprocessing image for OCR: %s. Using raw image.", e)
        if isinstance(image_path_or_pil, Image.Image):
            return cv2.cvtColor(np.array(image_path_or_pil), cv2.COLOR_RGB2BGR)
        return cv2.imread(image_path_or_pil) if isinstance(image_path_or_pil, str) else image_path_or_pil

def run_ocr_on_image(image_path_or_pil) -> OCRResult:
    """Execute OCR engine on a file path or PIL image."""
    engine = get_ocr_engine()
    if not engine:
        return OCRResult("", 0.0, "none", ["OCR engine is disabled or failed to initialize"])

    try:
        processed = preprocess_image_for_ocr(image_path_or_pil)
        if processed is None:
            return OCRResult("", 0.0, "paddle", ["Image preprocessing returned None"])

        # Execute PaddleOCR
        result = engine.ocr(processed, cls=True)
        
        if not result or not result[0]:
            return OCRResult("", 1.0, "paddle", ["No text detected on the page image"])

        texts = []
        confidences = []
        for line in result[0]:
            if not line:
                continue
            box, (text_val, conf) = line
            texts.append(text_val)
            confidences.append(conf)

        ocr_text = "\n".join(texts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 1.0
        return OCRResult(ocr_text, avg_conf, "paddle")
    except Exception as exc:
        logger.error("OCR execution failed: %s", exc)
        return OCRResult("", 0.0, "paddle", [f"OCR execution error: {exc}"])

def run_ocr_on_page_pixmap(page) -> OCRResult:
    """Render a PyMuPDF Page to a pixmap, then run OCR on the rendered image data."""
    try:
        # Render page at 150 DPI for optimal OCR balance
        pix = page.get_pixmap(dpi=150)
        img_data = pix.tobytes("png")
        
        from io import BytesIO
        pil_img = Image.open(BytesIO(img_data))
        return run_ocr_on_image(pil_img)
    except Exception as exc:
        logger.error("Failed to run OCR on page pixmap: %s", exc)
        return OCRResult("", 0.0, "paddle", [f"Pixmap OCR error: {exc}"])

def should_run_ocr(extracted_text: str, page_images_count: int = 0) -> bool:
    """
    Decide if OCR needs to run on this page:
      - extracted_text is empty
      - extracted_text length is smaller than OCR_MIN_TEXT_LENGTH
    """
    if not OCR_ENABLED:
        return False
    
    clean_text = (extracted_text or "").strip()
    if not clean_text:
        return True
    
    if len(clean_text) < OCR_MIN_TEXT_LENGTH:
        return True
        
    return False

def merge_extracted_and_ocr_text(extracted_text: str, ocr_text: str) -> str:
    """Concatenate normal extracted text with OCR text cleanly."""
    ext = (extracted_text or "").strip()
    ocr = (ocr_text or "").strip()
    if not ext:
        return ocr
    if not ocr:
        return ext
    return f"{ext}\n\n[OCR Extracted Text]\n{ocr}"
