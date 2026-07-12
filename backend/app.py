"""
Real Estate RAG — FastAPI Application v3.1

Key fixes vs v3.0:
  - Fixed variable scope bug: saved_document referenced outside its with-block
  - Fixed LLM formatter model name (now reads LLM_MODEL env var)
  - Added PDF MIME type validation in /upload
  - Added full try/except wrapping around upload processing
  - Richer terminal debug logs throughout upload pipeline
  - /rag/health now explicitly tests Qdrant Cloud connectivity
  - All original routes preserved
"""

import json
import logging
import os
import tempfile
import uuid
import hashlib
from typing import Any

from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── Existing imports ──────────────────────────────────────────────────────────
from chatbot.chat_handler import answer_from_project_data, build_chat_context, generate_answer
from formatter.llm_formatter import format_with_llm
from formatter.project_normalizer import normalize_project_data
from ingestion.cleaner import clean_text
from ingestion.extractor import extract_document
from ingestion.image_analyzer import analyze_images, page_metadata_from_images
from ingestion.image_extractor import extract_images_from_pdf
from retrieval.intent_classifier import classify_intent
from retrieval.image_retriever import (
    find_matching_images,
    image_answer_text,
    should_prioritize_image,
)
from storage.mongo_store import (
    delete_project_from_mongo,
    load_builders,
    load_file_from_mongo,
    load_projects,
    save_file_to_mongo,
    save_project_to_mongo,
    check_mongo_health,
)


# ── RAG imports ───────────────────────────────────────────────────────────────
from ingestion.rag_worker import run_rag_indexing
from retrieval.embeddings import embedding_model_status
from retrieval.qdrant_service import (
    create_collections,
    delete_vectors_by_document_id,
    health_check as qdrant_health_check,
)
from retrieval.hybrid_retriever import detect_image_intent, retrieve, reranker_status
from chatbot.grounded_answer import generate_grounded_answer
from storage.doc_status import get_status, set_status
from storage.document_status import save_initial_status
from processing.task_queue import enqueue_document_processing


# ── Reindex-only imports ──────────────────────────────────────────────────────
from ingestion.pdf_processor import process_pdf
from ingestion.chunker import chunk_text_pages
from ingestion.image_processor import build_image_embedding_text, process_images
from retrieval.embeddings import get_image_embedder, get_text_embedder
from retrieval.qdrant_service import upsert_image_chunks, upsert_text_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Real Estate Hybrid Multimodal RAG API",
    description=(
        "Grounded answers from uploaded real estate brochures. "
        "Upload returns immediately; RAG indexing runs in the background."
    ),
    version="3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage_dir = os.path.join(os.path.dirname(__file__), "storage")
if os.path.isdir(storage_dir):
    app.mount("/storage", StaticFiles(directory=storage_dir), name="storage")


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI starting up — checking Qdrant Cloud connection…")
    try:
        result = create_collections()
        logger.info("Qdrant collections ready: %s", result)
    except Exception as exc:
        logger.warning("Qdrant not reachable at startup: %s", exc)

    # Recover any document indexing tasks that were interrupted by a server restart
    try:
        from storage.doc_status import recover_interrupted_tasks
        recover_interrupted_tasks()
    except Exception as exc:
        logger.error("Failed to run startup status recovery: %s", exc)



# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_llm_output(text: str) -> str:
    """Strip ```json fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last ``` lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def _is_supported_format(content_type: str | None, filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    supported_extensions = {
        ".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt",
        ".png", ".jpg", ".jpeg", ".webp"
    }
    if ext in supported_extensions:
        return True
    if content_type:
        lower_mime = content_type.lower()
        supported_mimes = {
            "application/pdf", 
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "text/csv", "text/plain", "image/png", "image/jpeg", "image/webp"
        }
        if lower_mime in supported_mimes:
            return True
    return False


# ════════════════════════════════════════════════════════════════════════════════
#  UPLOAD — returns quickly; RAG indexing happens in background
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    builder: str | None = Form(default=None),
    project: str | None = Form(default=None),
    document_type: str | None = Form(default=None),
    description: str | None = Form(default=None),
    tags: str | None = Form(default=None),
    # Excavator Domain Fields
    domain: str | None = Form(default="real_estate"),
    doc_type: str | None = Form(default=None),
    revision_date: str | None = Form(default=None),
    section_path: str | None = Form(default=None),
    component_tags: str | None = Form(default=None),
    dtc_codes: str | None = Form(default=None),
    supersedes_doc_id: str | None = Form(default=None),
):
    """
    Upload a real-estate or excavator document.
    """
    # ── Validate file ─────────────────────────────────────────────────────────
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    safe_filename = os.path.basename(file.filename)
    if not _is_supported_format(file.content_type, safe_filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: PDF, DOCX, PPTX, XLSX, CSV, TXT, PNG, JPEG, WEBP. Got: {file.content_type or safe_filename}",
        )

    document_id = uuid.uuid4().hex
    stored_filename = f"{document_id}_{safe_filename}"

    # ── Verify MongoDB availability first ─────────────────────────────────────
    from config import ALLOW_UPLOAD_WITHOUT_MONGODB
    from storage.mongo_client import ping_mongo, get_mongo_diagnostics
    
    mongo_ok = False
    try:
        ping_mongo()
        mongo_ok = True
    except Exception as exc:
        logger.error("[%s] MongoDB ping failed: %s", document_id, exc)

    if not mongo_ok:
        if ALLOW_UPLOAD_WITHOUT_MONGODB:
            logger.warning("[%s] MongoDB unreachable, but ALLOW_UPLOAD_WITHOUT_MONGODB is True. Proceeding with local fallback.", document_id)
        else:
            diag = get_mongo_diagnostics()
            set_status(document_id, "failed", error="MongoDB is not reachable or not configured.")
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "MongoDB connection failed. Upload cannot continue because metadata storage is unavailable.",
                    "safe_diagnostics": {
                        "mongodb_uri_configured": diag["mongodb_uri_configured"],
                        "database": diag["database"],
                        "collection": diag["collection"]
                    },
                    "possible_fixes": [
                        "Whitelist your current IP in MongoDB Atlas",
                        "Check username/password",
                        "URL encode password if needed",
                        "Use mobile hotspot if WiFi blocks MongoDB Atlas TLS",
                        "Upgrade pymongo, dnspython, and certifi"
                    ]
                }
            )


    if domain == "excavator":
        if not doc_type:
            raise HTTPException(status_code=400, detail="For excavator domain, 'doc_type' is required.")
        valid_types = {"manuals", "service_bulletins", "maintenance_logs", "parts_catalog", "field_reports"}
        if doc_type.lower() not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid doc_type for excavator. Must be one of: {list(valid_types)}")

    logger.info("=== UPLOAD START | doc=%s | file=%s ===", document_id, safe_filename)

    # Read bytes eagerly
    file_bytes = await file.read()
    if len(file_bytes) < 10:
        raise HTTPException(status_code=400, detail="Uploaded file is empty or corrupt.")

    # Eagerly compute hash and check for duplicates to save computing time
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    
    # ── Verify duplicates in MongoDB ─────────────────────────────────────────
    existing_doc = None
    try:
        from storage.mongo_store import _get_collection, _load_projects_locally
        col = _get_collection()
        if col is not None:
            existing_doc = col.find_one({"hash": file_hash})
        if not existing_doc:
            local_projects = _load_projects_locally(include_raw_text=True)
            for p in local_projects:
                if p.get("hash") == file_hash:
                    existing_doc = p
                    break
    except Exception as exc:
        logger.warning("[%s] Duplicate check exception: %s", document_id, exc)

    if existing_doc:
        # Re-save metadata with new doc_id so it can be queried immediately
        try:
            from storage.mongo_store import save_project_to_mongo
            reused_doc = dict(existing_doc)
            reused_doc["document_id"] = document_id
            reused_doc["source_file"] = safe_filename
            reused_doc["stored_file"] = f"{document_id}_{safe_filename}"
            reused_doc["domain"] = domain
            
            reused_meta = reused_doc.get("metadata", {})
            if title: reused_meta["title"] = title
            if builder: reused_meta["builder"] = builder
            if project: reused_meta["project"] = project
            if document_type: reused_meta["document_type"] = document_type
            if description: reused_meta["description"] = description
            if tags: reused_meta["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
            
            # Excavator specific
            if doc_type: reused_meta["doc_type"] = doc_type
            if revision_date: reused_meta["revision_date"] = revision_date
            if section_path: reused_meta["section_path"] = section_path
            if component_tags: reused_meta["component_tags"] = [t.strip() for t in component_tags.split(",") if t.strip()]
            if dtc_codes: reused_meta["dtc_codes"] = [t.strip() for t in dtc_codes.split(",") if t.strip()]
            if supersedes_doc_id: reused_meta["supersedes_doc_id"] = supersedes_doc_id
            reused_doc["metadata"] = reused_meta
            
            save_project_to_mongo(reused_doc)
            save_initial_status(document_id, status="ready", progress=100, filename=safe_filename)
            set_status(document_id, "ready", 
                       text_chunks_indexed=reused_doc.get("text_chunks_indexed", 0),
                       images_indexed=reused_doc.get("images_indexed", 0),
                       total_pages=len(reused_doc.get("pages", [])),
                       message="Duplicate document detected. Reused previous embeddings.")
            
            return {
                "document_id": document_id,
                "status": "ready",
                "progress": 100,
                "filename": safe_filename,
                "message": "Duplicate document detected. Embeddings successfully reused.",
                "saved_to_mongodb": mongo_ok,
                "ocr_used": False,
            }
        except Exception as exc:
            logger.warning("[%s] Failed to reuse deduplicated document: %s", document_id, exc)

    logger.info("[%s] File read: %d bytes", document_id, len(file_bytes))

    # Mark status immediately so frontend polling starts working
    save_initial_status(document_id, status="uploaded", progress=5, filename=safe_filename)

    # Save PDF file permanently
    from config import UPLOAD_FOLDER
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, stored_filename)
    try:
        with open(file_path, "wb") as buf:
            buf.write(file_bytes)
        logger.info("[%s] Saved uploaded file permanently to %s", document_id, file_path)
    except Exception as exc:
        logger.error("[%s] Failed to save uploaded file permanently: %s", document_id, exc)
        set_status(document_id, "failed", error="Failed to save uploaded file on backend.")
        raise HTTPException(status_code=500, detail="Failed to write uploaded file to disk.")

    # Enqueue background task
    metadata = {
        "domain": domain,
        "source_file": safe_filename,
        "content_type": file.content_type or "application/pdf",
        "title": title or "",
        "builder": builder or "",
        "project": project or "",
        "document_type": document_type or "",
        "description": description or "",
        "tags": tags or "",
        # Excavator specific
        "doc_type": doc_type or "",
        "revision_date": revision_date or "",
        "section_path": section_path or "",
        "component_tags": component_tags or "",
        "dtc_codes": dtc_codes or "",
        "supersedes_doc_id": supersedes_doc_id or "",
    }
    
    enqueue_document_processing(
        background_tasks=background_tasks,
        document_id=document_id,
        file_path=file_path,
        metadata=metadata
    )

    logger.info("[%s] Upload complete — RAG indexing enqueued in background", document_id)

    response_msg = "Document uploaded successfully. Multimodal hybrid RAG indexing started."
    if not mongo_ok and ALLOW_UPLOAD_WITHOUT_MONGODB:
        response_msg = "MongoDB unavailable. Using local JSON metadata fallback for development."

    return {
        "document_id": document_id,
        "status": "processing",
        "progress": 5,
        "filename": safe_filename,
        "message": response_msg,
        "saved_to_mongodb": mongo_ok,
        "ocr_used": False,
    }





# ════════════════════════════════════════════════════════════════════════════════
#  STATUS — polling endpoint
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/documents/{document_id}/status")
async def document_status(document_id: str):
    """
    Poll the RAG processing status of an uploaded document.

    Status lifecycle:
      uploaded → extracting_text → extracting_images → chunking →
      embedding_text → embedding_images → indexing_qdrant → ready | failed
    """
    status = get_status(document_id)

    if status is None:
        # Document may have been indexed before status tracking was added
        projects = load_projects(document_id=document_id)
        if projects:
            return {
                "document_id": document_id,
                "status": "ready",
                "progress": 100,
                "text_chunks_indexed": 0,
                "images_indexed": 0,
                "total_pages": len(projects[0].get("pages", [])),
                "message": "Document available.",
                "error": "",
            }
        raise HTTPException(status_code=404, detail="Document not found.")

    return {
        "document_id": document_id,
        "status": status.get("status", "unknown"),
        "progress": status.get("progress", 0),
        "text_chunks_indexed": status.get("text_chunks_indexed", 0),
        "images_indexed": status.get("images_indexed", 0),
        "total_pages": status.get("total_pages", 0),
        "message": status.get("message", ""),
        "error": status.get("error", ""),
    }


# ════════════════════════════════════════════════════════════════════════════════
#  CHAT — Hybrid RAG with readiness guard and legacy fallback
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/chat")
async def chat(query: Any = Body(...)):
    document_id = None
    top_k = 8
    include_images = False
    domain = "real_estate"

    if isinstance(query, dict):
        document_id = query.get("document_id")
        include_images = bool(query.get("include_images", False))
        top_k = int(query.get("top_k", 8))
        domain = query.get("domain", "real_estate")
        query = query.get("message") or query.get("query") or ""

    question = str(query).strip()
    if not question:
        return {
            "question": "",
            "answer": "Please ask a question.",
            "citations": [],
            "images": [],
            "confidence": "low",
        }

    # ── Excavator Scaffolding Route ───────────────────────────────────────────
    if domain == "excavator":
        from chatbot.query_router import classify_im_query
        routing_info = classify_im_query(question)
        return {
            "question": question,
            "answer": f"[SCANNED ROUTE: {routing_info['category'].upper()}] - Scaffolding for Hyundai excavator RAG active.",
            "citations": [],
            "images": [],
            "confidence": "high",
            "routing": routing_info,
        }

    # ── Guard: document must be ready ─────────────────────────────────────────
    if document_id:
        doc_status = get_status(document_id)
        if doc_status and doc_status.get("status") not in ("ready", None):
            status_val = doc_status.get("status", "processing")
            if status_val == "failed":
                return {
                    "question": question,
                    "answer": "Document processing failed. Please re-upload the brochure.",
                    "citations": [], "images": [], "confidence": "low", "status": "failed",
                }
            progress = doc_status.get("progress", 0)
            step_msg = doc_status.get("message", "Indexing in progress…")
            return {
                "question": question,
                "answer": (
                    f"The document is still being indexed ({progress}% complete). "
                    f"Current step: {step_msg} "
                    "Please ask your question again after processing is completed."
                ),
                "citations": [], "images": [], "confidence": "low",
                "status": "processing", "progress": progress,
            }

    # ── Intent detection ──────────────────────────────────────────────────────
    intent = classify_intent(question)
    image_intent = detect_image_intent(question)

    # ── Try Hybrid RAG ────────────────────────────────────────────────────────
    try:
        from utils.observability import time_stage
        with time_stage("total_response_times"):
            retrieved = retrieve(
                query=question,
                document_id=document_id,
                include_images=include_images or intent.get("requires_visual_response", False),
                top_k=top_k,
            )
            if retrieved:
                rag_response = generate_grounded_answer(question, retrieved)
                rag_response["intent"] = {
                    **intent,
                    "requires_image": image_intent["requires_image"],
                    "requires_text": True,
                    "detected_image_types": image_intent.get("detected_types", []),
                }
                return rag_response
    except Exception as exc:
        logger.warning("RAG pipeline failed, falling back to legacy path: %s", exc)

    # ── Legacy fallback ───────────────────────────────────────────────────────
    projects = load_projects(document_id=document_id, include_raw_text=True)
    if not projects:
        return {
            "question": question,
            "answer": "No project data found. Please upload a brochure first.",
            "citations": [], "images": [], "confidence": "low", "intent": intent,
        }

    matching_images = []
    if intent["requires_visual_response"]:
        matching_images = find_matching_images(
            question, projects, allowed_image_types=intent.get("image_types", [])
        )
    matching_image_paths = [img["image_path"] for img in matching_images]

    local_answer = answer_from_project_data(question, projects)
    if local_answer is not None:
        answer = local_answer
        if intent["requires_visual_response"] and matching_images:
            if should_prioritize_image(question) or "Data not available" in local_answer:
                answer = image_answer_text(question, matching_images) or local_answer
            else:
                answer = f"{local_answer}\n\nRelated image attached."
        return {
            "question": question, "answer": answer,
            "citations": [], "images": matching_image_paths,
            "confidence": "medium",
            "intent": {**intent, "requires_image": image_intent["requires_image"], "requires_text": True},
        }

    if intent["requires_visual_response"] and matching_images:
        return {
            "question": question,
            "answer": image_answer_text(question, matching_images),
            "citations": [], "images": matching_image_paths,
            "confidence": "medium",
            "intent": {**intent, "requires_image": True, "requires_text": False},
        }

    context = json.dumps(build_chat_context(projects, question), indent=2)
    prompt = (
        "You are a real estate assistant.\n"
        "Answer ONLY using the provided data.\n"
        "If answer not found, say \"Data not available in the uploaded documents.\"\n\n"
        f"Data:\n{context}\n\n"
        f"Intent:\n{json.dumps(intent)}\n\n"
        f"Question:\n{question}\n\nAnswer:"
    )
    answer = generate_answer(prompt)

    return {
        "question": question, "answer": answer,
        "citations": [],
        "images": matching_image_paths if intent["requires_visual_response"] else [],
        "confidence": "medium",
        "intent": {**intent, "requires_image": image_intent["requires_image"], "requires_text": True},
    }


# ════════════════════════════════════════════════════════════════════════════════
#  EXISTING ROUTES (unchanged)
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/documents/{document_id}/images/{image_id}")
async def document_image(document_id: str, image_id: str):
    file_obj = load_file_from_mongo(document_id, "image", image_id=image_id)
    if file_obj is None:
        raise HTTPException(status_code=404, detail="Image not found.")
    ct = file_obj.metadata.get("content_type") if getattr(file_obj, "metadata", None) else None
    return Response(content=file_obj.read(), media_type=ct or "image/png")


@app.get("/documents/{document_id}/pdf")
async def document_pdf(document_id: str):
    file_obj = load_file_from_mongo(document_id, "pdf")
    if file_obj is None:
        raise HTTPException(status_code=404, detail="PDF not found.")
    ct = file_obj.metadata.get("content_type") if getattr(file_obj, "metadata", None) else None
    return Response(content=file_obj.read(), media_type=ct or "application/pdf")


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    deleted = delete_project_from_mongo(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        delete_vectors_by_document_id(document_id)
    except Exception as exc:
        logger.warning("Failed to delete Qdrant vectors for %s: %s", document_id, exc)
    return {"message": "Document deleted", "document_id": document_id}


@app.get("/projects")
async def projects():
    return {"projects": load_projects()}


@app.get("/builders")
async def builders():
    return {"builders": load_builders()}


@app.get("/rag/metrics")
async def rag_metrics():
    """Retrieve observability metrics."""
    from utils.observability import get_observability_metrics
    return get_observability_metrics()


@app.get("/rag/evaluation")
async def rag_evaluation():
    """Retrieve evaluation scores and logs."""
    from utils.evaluation import get_summarized_evaluations
    return get_summarized_evaluations()


# ════════════════════════════════════════════════════════════════════════════════
#  RAG ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/health/mongo")
async def mongo_health():
    """Check MongoDB connection and return status or detailed error diagnostic."""
    health = check_mongo_health()
    if health.get("status") != "ok":
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content=health)
    return health




@app.get("/rag/health")
async def rag_health():

    """Check Qdrant Cloud connection, collections, and embedding model status."""
    qdrant_info = qdrant_health_check()
    embed_status = embedding_model_status()
    reranker = reranker_status()

    qdrant_ok = (
        qdrant_info.get("qdrant_connected", False)
        and isinstance(qdrant_info.get("text_collection"), dict)
        and isinstance(qdrant_info.get("image_collection"), dict)
    )

    return {
        "status": "ok" if qdrant_ok else "degraded",
        "qdrant": qdrant_info,
        "embeddings": embed_status,
        "reranker": reranker,
    }


@app.post("/rag/search")
async def rag_search(body: Any = Body(...)):
    """Debug endpoint: raw hybrid retrieval results before answer generation."""
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object.")
    question = body.get("message") or body.get("query") or ""
    if not question:
        raise HTTPException(status_code=422, detail="'message' field is required.")
    try:
        results = retrieve(
            query=question,
            document_id=body.get("document_id"),
            include_images=bool(body.get("include_images", True)),
            top_k=int(body.get("top_k", 10)),
        )
        return {"question": question, "results": results, "total": len(results)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc


@app.post("/rag/reindex/{document_id}")
async def rag_reindex(document_id: str, background_tasks: BackgroundTasks):
    """Rebuild Qdrant Cloud vectors for one document (loads PDF from MongoDB)."""
    pdf_file = load_file_from_mongo(document_id, "pdf")
    if pdf_file is None:
        raise HTTPException(status_code=404, detail="PDF not found in MongoDB.")

    projects = load_projects(document_id=document_id)
    if not projects:
        raise HTTPException(status_code=404, detail="Project metadata not found.")

    project = projects[0]
    meta = project.get("metadata") or {}
    doc_metadata = {
        "project_name": meta.get("project") or project.get("project_name", ""),
        "builder": meta.get("builder") or project.get("developer", ""),
        "document_type": meta.get("document_type", ""),
        "source_file": project.get("source_file", "document.pdf"),
    }

    pdf_bytes = pdf_file.read()
    try:
        delete_vectors_by_document_id(document_id)
    except Exception as exc:
        logger.warning("Failed to delete old vectors before reindex: %s", exc)

    set_status(document_id, "uploaded", message="Reindex requested.")
    background_tasks.add_task(
        run_rag_indexing,
        document_id=document_id,
        pdf_bytes=pdf_bytes,
        source_file=doc_metadata["source_file"],
        doc_metadata=doc_metadata,
    )

    return {
        "document_id": document_id,
        "status": "processing",
        "message": f"Reindexing started. Poll /documents/{document_id}/status.",
    }


@app.delete("/rag/index/{document_id}")
async def delete_rag_index(document_id: str):
    """Delete all Qdrant Cloud vectors for a document without touching MongoDB."""
    try:
        result = delete_vectors_by_document_id(document_id)
        return {"document_id": document_id, "status": "deleted", "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}") from exc


@app.get("/health/live")
async def health_live():
    """Liveness check for container orchestration or uptime monitoring."""
    return {"status": "ok", "live": true}


@app.get("/health/ready")
async def health_ready():
    """
    Readiness check validating all supporting services are available:
      - Qdrant Cloud reachable
      - MongoDB reachable (or ALLOW_UPLOAD_WITHOUT_MONGODB active)
      - OCR engine loaded if enabled
    """
    # 1. Qdrant check
    qdrant_ok = False
    try:
        qdrant_info = qdrant_health_check()
        qdrant_ok = (
            qdrant_info.get("qdrant_connected", False)
            and isinstance(qdrant_info.get("text_collection"), dict)
            and isinstance(qdrant_info.get("image_collection"), dict)
        )
    except Exception:
        pass

    # 2. MongoDB check
    from config import ALLOW_UPLOAD_WITHOUT_MONGODB
    mongo_ok = False
    try:
        health = check_mongo_health()
        mongo_ok = (health.get("status") == "ok")
    except Exception:
        pass

    db_ready = mongo_ok or ALLOW_UPLOAD_WITHOUT_MONGODB

    # 3. OCR check
    from config import OCR_ENABLED
    ocr_ready = True
    if OCR_ENABLED:
        from ingestion.ocr_service import get_ocr_engine
        try:
            engine = get_ocr_engine()
            ocr_ready = (engine is not None)
        except Exception:
            ocr_ready = False

    ready = qdrant_ok and db_ready and ocr_ready

    return {
        "status": "ready" if ready else "not_ready",
        "qdrant": "connected" if qdrant_ok else "disconnected",
        "mongodb": "connected" if mongo_ok else ("fallback_enabled" if ALLOW_UPLOAD_WITHOUT_MONGODB else "disconnected"),
        "ocr": "available" if (not OCR_ENABLED or ocr_ready) else "unavailable",
        "message": "System is ready for document processing and chat." if ready else "One or more components are not ready."
    }

