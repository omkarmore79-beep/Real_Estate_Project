import json
import os
import shutil
import tempfile
import uuid
from typing import Any

from fastapi import Body, FastAPI, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
)


app = FastAPI()
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


def clean_llm_output(text):
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1)
    return text.strip()


@app.post("/upload")
async def upload_pdf(
    file: UploadFile,
    title: str | None = Form(default=None),
    builder: str | None = Form(default=None),
    project: str | None = Form(default=None),
    document_type: str | None = Form(default=None),
    description: str | None = Form(default=None),
    tags: str | None = Form(default=None),
):
    document_id = uuid.uuid4().hex
    safe_filename = os.path.basename(file.filename)
    stored_filename = f"{document_id}_{safe_filename}"

    with tempfile.TemporaryDirectory(prefix=f"upload_{document_id}_") as temp_dir:
        file_path = os.path.join(temp_dir, stored_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        raw_text = extract_document(file_path)
        cleaned_text = clean_text(raw_text)

        formatted_output = format_with_llm(cleaned_text)
        cleaned_output = clean_llm_output(formatted_output)

        try:
            json_data = json.loads(cleaned_output)
        except json.JSONDecodeError:
            json_data = {}

        json_data = normalize_project_data(json_data, cleaned_text)
        metadata_project = (project or "").strip()
        metadata_builder = (builder or "").strip()

        if metadata_project:
            json_data["project_name"] = metadata_project
        if metadata_builder:
            json_data["developer"] = metadata_builder

        image_folder = os.path.join(temp_dir, "images")
        extracted_images = extract_images_from_pdf(
            file_path,
            output_folder=image_folder,
            image_base_path=f"documents/{document_id}/images",
        )
        image_metadata = analyze_images(extracted_images, cleaned_text)

        saved_pdf_id = save_file_to_mongo(
            document_id=document_id,
            file_path=file_path,
            filename=safe_filename,
            content_type=file.content_type or "application/pdf",
            file_kind="pdf",
        )

        if saved_pdf_id is None:
            raise HTTPException(
                status_code=500,
                detail="MongoDB is not configured. Set MONGODB_URI before uploading.",
            )

        for image in image_metadata:
            image_id = image.get("image_id")
            image_filename = f"{image_id}.png"
            local_path = image.pop("local_path", None)
            if not image_id or not local_path:
                continue

            save_file_to_mongo(
                document_id=document_id,
                file_path=local_path,
                filename=image_filename,
                content_type="image/png",
                file_kind="image",
                image_id=image_id,
            )
            image["image_path"] = f"documents/{document_id}/images/{image_id}"

        json_data["images"] = image_metadata
        json_data["pages"] = page_metadata_from_images(image_metadata)
        json_data["document_id"] = document_id
        json_data["source_file"] = safe_filename
        json_data["stored_file"] = stored_filename
        json_data["raw_text"] = cleaned_text
        json_data["pdf_path"] = f"documents/{document_id}/pdf"
        json_data["metadata"] = {
            "title": title or safe_filename,
            "builder": metadata_builder,
            "project": metadata_project,
            "document_type": document_type or "",
            "description": description or "",
            "tags": [tag.strip() for tag in (tags or "").split(",") if tag.strip()],
        }

        try:
            print("Attempting to save processed document to MongoDB...")
            saved_document = save_project_to_mongo(json_data)
            if saved_document:
                print("MONGODB SAVE: success", saved_document.get("_id"))
            else:
                raise RuntimeError("MongoDB save returned None")
        except HTTPException:
            raise
        except Exception as error:
            print("MONGODB SAVE ERROR:", error)
            raise HTTPException(
                status_code=500,
                detail="Could not save processed document to MongoDB.",
            ) from error

    response_data = dict(json_data)
    response_data.pop("raw_text", None)

    return {
        "message": "Processed successfully",
        "document_id": document_id,
        "saved_to_mongodb": saved_document is not None,
        "data": response_data,
    }


@app.get("/documents/{document_id}/images/{image_id}")
async def document_image(document_id: str, image_id: str):
    file_obj = load_file_from_mongo(document_id, "image", image_id=image_id)
    if file_obj is None:
        raise HTTPException(status_code=404, detail="Image not found.")

    content_type = (
        file_obj.metadata.get("content_type")
        if getattr(file_obj, "metadata", None)
        else None
    )
    return Response(content=file_obj.read(), media_type=content_type or "image/png")


@app.get("/documents/{document_id}/pdf")
async def document_pdf(document_id: str):
    file_obj = load_file_from_mongo(document_id, "pdf")
    if file_obj is None:
        raise HTTPException(status_code=404, detail="PDF not found.")

    content_type = (
        file_obj.metadata.get("content_type")
        if getattr(file_obj, "metadata", None)
        else None
    )
    return Response(content=file_obj.read(), media_type=content_type or "application/pdf")


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    deleted = delete_project_from_mongo(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")

    return {"message": "Document deleted", "document_id": document_id}


@app.post("/chat")
async def chat(query: Any = Body(...)):
    document_id = None
    if isinstance(query, dict):
        document_id = query.get("document_id")
        query = query.get("message") or query.get("query") or ""

    projects = load_projects(document_id=document_id, include_raw_text=True)

    if not projects:
        return {
            "question": query,
            "answer": "No project data available for this upload. Please upload a brochure first.",
            "images": [],
        }

    intent = classify_intent(query)
    matching_images = []
    if intent["requires_visual_response"]:
        matching_images = find_matching_images(
            query,
            projects,
            allowed_image_types=intent.get("image_types", []),
        )
    matching_image_paths = [image["image_path"] for image in matching_images]

    local_answer = answer_from_project_data(query, projects)
    if local_answer is not None:
        answer = local_answer

        if intent["requires_visual_response"] and matching_images and (
            should_prioritize_image(query) or "Data not available" in local_answer
        ):
            image_answer = image_answer_text(query, matching_images)
            answer = image_answer or local_answer
        elif intent["requires_visual_response"] and matching_images:
            answer = f"{local_answer}\n\nRelated image attached."

        return {
            "question": query,
            "answer": answer,
            "images": matching_image_paths,
            "intent": intent,
        }

    if intent["requires_visual_response"] and matching_images:
        return {
            "question": query,
            "answer": image_answer_text(query, matching_images),
            "images": matching_image_paths,
            "intent": intent,
        }

    context = json.dumps(build_chat_context(projects, query), indent=2)
    prompt = f"""
You are a real estate assistant.

Answer ONLY using the provided data.

Mention related images only when the provided intent requires a visual response
and relevant image metadata exists.

If answer not found, say "Data not available".

Data:
{context}

Intent:
{json.dumps(intent)}

Question:
{query}

Answer:
"""

    answer = generate_answer(prompt)

    return {
        "question": query,
        "answer": answer,
        "images": matching_image_paths if intent["requires_visual_response"] else [],
        "intent": intent,
    }


@app.get("/projects")
async def projects():
    return {"projects": load_projects()}


@app.get("/builders")
async def builders():
    return {"builders": load_builders()}
