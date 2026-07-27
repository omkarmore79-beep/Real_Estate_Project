import logging
import os
import cv2
import numpy as np
from PIL import Image
from config import (
    ENABLE_GROQ_VISION,
    GROQ_VISION_MODEL,
    OCR_ENABLED,
    OCR_ENGINE,
    OCR_MIN_TEXT_LENGTH,
    OCR_MIN_CONFIDENCE,
)

logger = logging.getLogger(__name__)

_ocr_engine = None
_groq_vision_unavailable_logged = False

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
            # Disable oneDNN/MKLDNN PIR executor on Windows CPU to prevent oneDNN instruction crash
            os.environ["FLAGS_use_onednn"] = "0"
            os.environ["FLAGS_use_mkldnn"] = "0"
            os.environ["PADDLE_DISABLE_PIR"] = "1"

            from paddleocr import PaddleOCR
            # Suppress excessive logging from PaddleOCR
            logging.getLogger("ppocr").setLevel(logging.WARNING)
            _ocr_engine = PaddleOCR(lang="en", use_angle_cls=False, enable_mkldnn=False)
            logger.info("PaddleOCR engine initialized successfully.")
        except Exception as exc:
            logger.error("Failed to initialize PaddleOCR engine: %s", exc)
            _ocr_engine = None
    return _ocr_engine


def _groq_vision_warning_once(reason: str):
    global _groq_vision_unavailable_logged
    if not _groq_vision_unavailable_logged:
        logger.warning(
            "Groq Vision unavailable. Switching to PaddleOCR.\nReason: %s",
            reason,
        )
        _groq_vision_unavailable_logged = True


def _get_groq_vision_reason(exc: Exception) -> str:
    reason = ""
    if hasattr(exc, "status_code"):
        reason += f"HTTP {getattr(exc, 'status_code')} "
    if hasattr(exc, "code"):
        reason += f"{getattr(exc, 'code')} "
    reason += str(exc)
    reason = reason.strip()
    if not reason:
        return "Unknown Groq Vision error"
    return reason


def _run_paddle_ocr(image_path_or_pil) -> OCRResult:
    engine = get_ocr_engine()
    if not engine:
        return OCRResult("", 0.0, "none", ["PaddleOCR engine failed to initialize"])

    try:
        processed = preprocess_image_for_ocr(image_path_or_pil)
        if processed is None:
            return OCRResult("", 0.0, "paddle", ["Image preprocessing returned None"])

        result = engine.ocr(processed)
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

def run_ocr_with_groq_vision(image_path_or_pil) -> OCRResult:
    """Run OCR on a page image using Groq Vision API when configured."""
    if not ENABLE_GROQ_VISION or not GROQ_VISION_MODEL:
        reason = "Groq Vision is disabled or GROQ_VISION_MODEL is not configured."
        _groq_vision_warning_once(reason)
        return OCRResult("", 0.0, "groq", [reason])

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        reason = "GROQ_API_KEY is not set."
        _groq_vision_warning_once(reason)
        return OCRResult("", 0.0, "groq", [reason])

    import base64
    from io import BytesIO
    from groq import Groq

    try:
        if isinstance(image_path_or_pil, str):
            with open(image_path_or_pil, "rb") as f:
                img_bytes = f.read()
        elif isinstance(image_path_or_pil, Image.Image):
            buffer = BytesIO()
            image_path_or_pil.convert("RGB").save(buffer, format="JPEG", quality=80)
            img_bytes = buffer.getvalue()
        else:
            pil_img = Image.fromarray(cv2.cvtColor(image_path_or_pil, cv2.COLOR_BGR2RGB))
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG", quality=80)
            img_bytes = buffer.getvalue()

        encoded = base64.b64encode(img_bytes).decode("ascii")
        data_url = f"data:image/jpeg;base64,{encoded}"

        client = Groq(api_key=api_key)
        logger.info("Attempting Groq Vision OCR with model: %s", GROQ_VISION_MODEL)
        response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Transcribe all readable text from this document image. "
                                "Output ONLY the transcribed text. Do not add any introduction, "
                                "meta-explanation, or markdown formatting outside of structural "
                                "formatting (like tables or lists) that exist in the image."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        ocr_text = response.choices[0].message.content or ""
        if not ocr_text.strip():
            reason = "Groq Vision returned empty OCR text."
            _groq_vision_warning_once(reason)
            return OCRResult("", 0.0, "groq", [reason])

        logger.info("Groq Vision OCR succeeded with model: %s", GROQ_VISION_MODEL)
        return OCRResult(ocr_text.strip(), 1.0, "groq")
    except Exception as exc:
        reason = _get_groq_vision_reason(exc)
        _groq_vision_warning_once(reason)
        return OCRResult("", 0.0, "groq", [reason])

def run_ocr_on_image(image_path_or_pil) -> OCRResult:
    """Execute OCR engine on a file path or PIL image."""
    if not OCR_ENABLED:
        return OCRResult("", 0.0, "none", ["OCR engine is disabled"])

    if OCR_ENGINE.lower() == "groq":
        groq_result = run_ocr_with_groq_vision(image_path_or_pil)
        if groq_result.text:
            return groq_result
        logger.info("Groq Vision OCR failed or unavailable; falling back to PaddleOCR.")
        return _run_paddle_ocr(image_path_or_pil)

    return _run_paddle_ocr(image_path_or_pil)


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
      - Returns True ONLY if direct text extraction returned fewer than 15 characters (scanned image page).
    """
    if not OCR_ENABLED:
        return False
    
    clean_text = (extracted_text or "").strip()
    return len(clean_text) < 15

def merge_extracted_and_ocr_text(extracted_text: str, ocr_text: str) -> str:
    """Concatenate normal extracted text with OCR text cleanly."""
    ext = (extracted_text or "").strip()
    ocr = (ocr_text or "").strip()
    if not ext:
        return ocr
    if not ocr:
        return ext
    return f"{ext}\n\n[OCR Extracted Text]\n{ocr}"
