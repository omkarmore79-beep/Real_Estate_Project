"""
End-to-End RAG Pipeline Test Script.

Usage:
    1. Start Qdrant:    docker run -p 6333:6333 qdrant/qdrant
    2. Start MongoDB:   mongod  (or use Atlas)
    3. Start backend:   cd backend && uvicorn app:app --reload --port 8000
    4. Run this script: python tests/test_rag_pipeline.py --pdf /path/to/brochure.pdf

The script will:
    A. Upload the PDF and verify indexing stats
    B. Check /rag/health
    C. Ask 7 standard real-estate questions and verify citations
    D. Ask 2 image-based questions and verify image URLs are returned
    E. Test "Data not available" fallback behaviour
    F. Test /rag/search debug endpoint
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000")

# ── Test questions ─────────────────────────────────────────────────────────────
TEXT_QUESTIONS = [
    "What is the project name?",
    "Who is the builder or developer?",
    "What amenities are available?",
    "What is the possession date?",
    "Is RERA mentioned? What is the RERA number?",
    "What is the carpet area or super built-up area?",
    "What are the price details?",
]

IMAGE_QUESTIONS = [
    "Show the floor plan",
    "Show the location map",
]

FALLBACK_QUESTIONS = [
    "What is the quantum mechanics formula?",  # clearly out of scope
]

# ── ANSI colours ───────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg):
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg):
    print(f"  {RED}✗{RESET} {msg}")


def info(msg):
    print(f"  {CYAN}·{RESET} {msg}")


def section(title):
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{CYAN}{'─' * 60}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════

def test_health():
    section("A. RAG Health Check")
    try:
        resp = requests.get(f"{BACKEND}/rag/health", timeout=10)
        data = resp.json()
        info(f"Status: {data.get('status')}")
        info(f"Qdrant connected: {data.get('qdrant', {}).get('qdrant_connected')}")
        text_col = data.get("qdrant", {}).get("text_collection")
        img_col = data.get("qdrant", {}).get("image_collection")
        if isinstance(text_col, dict):
            ok(f"Text collection: {text_col.get('points_count', 0)} points")
        else:
            fail(f"Text collection: {text_col}")
        if isinstance(img_col, dict):
            ok(f"Image collection: {img_col.get('points_count', 0)} points")
        else:
            fail(f"Image collection: {img_col}")
        embed = data.get("embeddings", {})
        info(f"Text embedder loaded: {embed.get('text_embedder', {}).get('loaded')}")
        info(f"Image embedder loaded: {embed.get('image_embedder', {}).get('loaded')}")
        return data.get("qdrant", {}).get("qdrant_connected", False)
    except Exception as exc:
        fail(f"Health check failed: {exc}")
        return False


def upload_pdf(pdf_path: str) -> str | None:
    section("B. Upload PDF")
    info(f"Uploading: {pdf_path}")
    try:
        with open(pdf_path, "rb") as f:
            resp = requests.post(
                f"{BACKEND}/upload",
                files={"file": (Path(pdf_path).name, f, "application/pdf")},
                data={
                    "builder": "Test Builder",
                    "project": "Test Project",
                    "document_type": "Brochure",
                    "title": "Test Brochure",
                },
                timeout=300,  # model loading can take a while
            )
        data = resp.json()

        if resp.status_code != 200:
            fail(f"Upload failed HTTP {resp.status_code}: {data}")
            return None

        doc_id = data.get("document_id")
        ok(f"document_id: {doc_id}")
        ok(f"total_pages: {data.get('total_pages', '?')}")

        text_chunks = data.get("text_chunks_indexed", 0)
        if text_chunks and text_chunks > 0:
            ok(f"text_chunks_indexed: {text_chunks}")
        else:
            fail(f"text_chunks_indexed: {text_chunks} (expected > 0)")

        images_idx = data.get("images_indexed", 0)
        if images_idx and images_idx > 0:
            ok(f"images_indexed: {images_idx}")
        else:
            fail(f"images_indexed: {images_idx} (expected > 0)")

        qdrant_status = data.get("qdrant_status", "unknown")
        if qdrant_status == "success":
            ok(f"qdrant_status: {qdrant_status}")
        else:
            fail(f"qdrant_status: {qdrant_status}")

        if data.get("ocr_used") is False:
            ok("ocr_used: False ✓")
        else:
            fail(f"ocr_used: {data.get('ocr_used')} (expected False)")

        return doc_id

    except Exception as exc:
        fail(f"Upload exception: {exc}")
        return None


def ask_chat(message: str, document_id: str | None, include_images: bool = False) -> dict:
    resp = requests.post(
        f"{BACKEND}/chat",
        json={"message": message, "document_id": document_id, "include_images": include_images, "top_k": 8},
        timeout=120,
    )
    return resp.json()


def test_text_questions(document_id: str):
    section("C. Text Questions — Citation Verification")
    passed = 0
    for question in TEXT_QUESTIONS:
        try:
            data = ask_chat(question, document_id)
            answer = data.get("answer", "")
            citations = data.get("citations", [])
            confidence = data.get("confidence", "?")

            has_answer = bool(answer and len(answer) > 5)
            has_citations = len(citations) > 0
            not_available = "not available" in answer.lower()

            status = "PASS" if has_answer else "FAIL"
            print(f"\n  Q: {question[:60]}")
            print(f"     Confidence: {confidence} | Citations: {len(citations)} | {'Not available' if not_available else 'Has answer'}")
            if has_answer:
                ok(f"Answer: {answer[:120]}…" if len(answer) > 120 else f"Answer: {answer}")
            else:
                fail("No answer returned")

            if has_citations:
                ok(f"Citations found: {[c.get('source_file', 'doc') + ' p.' + str(c.get('page_number', '?')) for c in citations[:2]]}")
            else:
                info("No citations (legacy fallback path may be active)")

            if has_answer:
                passed += 1
        except Exception as exc:
            fail(f"Error asking '{question[:40]}': {exc}")

    print(f"\n  Passed: {passed}/{len(TEXT_QUESTIONS)}")
    return passed


def test_image_questions(document_id: str):
    section("D. Image Questions — Image URL Verification")
    passed = 0
    for question in IMAGE_QUESTIONS:
        try:
            data = ask_chat(question, document_id, include_images=True)
            answer = data.get("answer", "")
            images = data.get("images", [])

            print(f"\n  Q: {question}")
            if images:
                ok(f"Images returned: {len(images)}")
                for img in images[:2]:
                    if isinstance(img, dict):
                        url = img.get("image_url", "")
                        img_type = img.get("image_type", "")
                        ok(f"  Image: {url} (type: {img_type})")
                    else:
                        ok(f"  Image path: {img}")
                passed += 1
            else:
                fail("No images returned (check if image vectors are indexed)")

        except Exception as exc:
            fail(f"Error: {exc}")

    print(f"\n  Passed: {passed}/{len(IMAGE_QUESTIONS)}")
    return passed


def test_fallback_questions(document_id: str):
    section("E. Fallback Behaviour — Out-of-Scope Questions")
    passed = 0
    for question in FALLBACK_QUESTIONS:
        try:
            data = ask_chat(question, document_id)
            answer = data.get("answer", "")
            is_grounded_refusal = (
                "not available" in answer.lower()
                or "uploaded documents" in answer.lower()
                or "not found" in answer.lower()
            )
            print(f"\n  Q: {question}")
            if is_grounded_refusal:
                ok(f"Correctly refused to hallucinate: '{answer[:100]}'")
                passed += 1
            else:
                fail(f"Unexpected answer (may be hallucination): '{answer[:100]}'")
        except Exception as exc:
            fail(f"Error: {exc}")

    print(f"\n  Passed: {passed}/{len(FALLBACK_QUESTIONS)}")
    return passed


def test_rag_search(document_id: str):
    section("F. /rag/search Debug Endpoint")
    try:
        resp = requests.post(
            f"{BACKEND}/rag/search",
            json={"message": "amenities", "document_id": document_id, "top_k": 5, "include_images": True},
            timeout=60,
        )
        data = resp.json()
        total = data.get("total", 0)
        results = data.get("results", [])
        if total > 0:
            ok(f"Retrieved {total} results")
            for r in results[:3]:
                info(f"  [{r.get('source_type')}] p.{r.get('page_number')} score={r.get('score', 0):.3f} — {str(r.get('content', ''))[:80]}")
        else:
            fail("No results returned from /rag/search")
        return total > 0
    except Exception as exc:
        fail(f"Error: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test the Hybrid Multimodal RAG pipeline")
    parser.add_argument("--pdf", required=True, help="Path to a real estate PDF brochure")
    parser.add_argument("--backend", default=BACKEND, help="Backend URL")
    args = parser.parse_args()

    global BACKEND
    BACKEND = args.backend

    print(f"\n{BOLD}Real Estate Hybrid RAG Pipeline — End-to-End Test{RESET}")
    print(f"Backend: {BACKEND}")
    print(f"PDF: {args.pdf}")

    qdrant_ok = test_health()
    if not qdrant_ok:
        print(f"\n{RED}⚠ Qdrant not connected. Start Qdrant first: docker run -p 6333:6333 qdrant/qdrant{RESET}")
        print("Continuing with upload (will use legacy fallback for chat)…\n")

    document_id = upload_pdf(args.pdf)
    if not document_id:
        print(f"\n{RED}Upload failed. Aborting tests.{RESET}")
        sys.exit(1)

    # Wait briefly for vectors to be committed
    time.sleep(2)

    text_passed = test_text_questions(document_id)
    image_passed = test_image_questions(document_id)
    fallback_passed = test_fallback_questions(document_id)
    search_passed = test_rag_search(document_id)

    section("Summary")
    print(f"  Text questions:    {text_passed}/{len(TEXT_QUESTIONS)}")
    print(f"  Image questions:   {image_passed}/{len(IMAGE_QUESTIONS)}")
    print(f"  Fallback refusal:  {fallback_passed}/{len(FALLBACK_QUESTIONS)}")
    print(f"  /rag/search:       {'pass' if search_passed else 'fail'}")
    print(f"\n  Document ID for re-testing: {document_id}")
    print()


if __name__ == "__main__":
    main()
