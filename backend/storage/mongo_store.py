import json
import os
from datetime import datetime, timezone

from config import DATA_FOLDER, MONGODB_COLLECTION, MONGODB_DB, MONGODB_URI

try:
    from bson import ObjectId
except ImportError:
    ObjectId = None


_client = None


def _get_db():
    if not MONGODB_URI:
        return None

    global _client
    if _client is None:
        from pymongo import MongoClient

        try:
            _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            _client.admin.command("ping")
            print("MongoDB connected")
        except Exception as error:
            print("MongoDB connection failed:", error)
            raise

    return _client[MONGODB_DB]


def _get_collection():
    db = _get_db()
    if db is None:
        return None

    return db[MONGODB_COLLECTION]


def _get_gridfs():
    db = _get_db()
    if db is None:
        return None

    import gridfs

    return gridfs.GridFS(db)


def _serialize_document(document):
    if not document:
        return document

    serialized = dict(document)
    if ObjectId is not None and isinstance(serialized.get("_id"), ObjectId):
        serialized["_id"] = str(serialized["_id"])
    return serialized


def save_project_to_mongo(project):
    collection = _get_collection()
    if collection is None:
        return None

    now = datetime.now(timezone.utc).isoformat()
    document = dict(project)
    document["updated_at"] = now
    document.setdefault("uploaded_at", now)

    collection.update_one(
        {"document_id": document["document_id"]},
        {"$set": document},
        upsert=True,
    )
    saved = collection.find_one({"document_id": document["document_id"]})
    return _serialize_document(saved)


def delete_project_from_mongo(document_id):
    collection = _get_collection()
    fs = _get_gridfs()
    if collection is None or fs is None:
        return False

    deleted_files = 0
    for existing in fs.find({"metadata.document_id": document_id}):
        fs.delete(existing._id)
        deleted_files += 1

    result = collection.delete_one({"document_id": document_id})
    return result.deleted_count > 0 or deleted_files > 0


def save_file_to_mongo(
    *,
    document_id,
    file_path,
    filename,
    content_type,
    file_kind,
    image_id=None,
):
    fs = _get_gridfs()
    if fs is None:
        return None

    metadata = {
        "document_id": document_id,
        "file_kind": file_kind,
        "content_type": content_type,
        "filename": filename,
    }
    if image_id:
        metadata["image_id"] = image_id

    existing_query = {
        "metadata.document_id": document_id,
        "metadata.file_kind": file_kind,
        "metadata.filename": filename,
    }
    if image_id:
        existing_query["metadata.image_id"] = image_id

    for existing in fs.find(existing_query):
        fs.delete(existing._id)

    with open(file_path, "rb") as file_obj:
        return fs.put(
            file_obj,
            filename=filename,
            content_type=content_type,
            metadata=metadata,
        )


def load_file_from_mongo(document_id, file_kind, image_id=None):
    fs = _get_gridfs()
    if fs is None:
        return None

    query = {
        "metadata.document_id": document_id,
        "metadata.file_kind": file_kind,
    }
    if image_id:
        query["metadata.image_id"] = image_id

    files = list(fs.find(query).sort("uploadDate", -1).limit(1))
    return files[0] if files else None


def load_projects_from_mongo(document_id=None, include_raw_text=False):
    collection = _get_collection()
    if collection is None:
        return None

    query = {"document_id": document_id} if document_id else {}
    projection = None if include_raw_text else {"raw_text": 0}
    return [
        _serialize_document(item)
        for item in collection.find(query, projection).sort("uploaded_at", -1)
    ]


def load_projects_from_json(document_id=None, include_raw_text=False):
    data = []

    if not os.path.exists(DATA_FOLDER):
        return data

    for file in os.listdir(DATA_FOLDER):
        if not file.endswith(".json"):
            continue

        path = os.path.join(DATA_FOLDER, file)
        with open(path, "r", encoding="utf-8") as f:
            project = json.load(f)

        if not project.get("document_id"):
            continue

        if document_id and project.get("document_id") != document_id:
            continue

        if not include_raw_text:
            project.pop("raw_text", None)

        data.append(project)

    return data


def load_projects(document_id=None, include_raw_text=False):
    try:
        mongo_projects = load_projects_from_mongo(
            document_id,
            include_raw_text=include_raw_text,
        )
        if mongo_projects is not None:
            return mongo_projects
    except Exception as error:
        print("MONGODB LOAD ERROR:", error)

    return load_projects_from_json(document_id, include_raw_text=include_raw_text)


def load_builders():
    projects = load_projects()
    builders = {}

    for project in projects:
        metadata = project.get("metadata") or {}
        name = (
            metadata.get("builder")
            or project.get("developer")
            or "Builder not specified"
        )
        key = str(name).strip() or "Builder not specified"

        if key not in builders:
            builders[key] = {
                "name": key,
                "documents": [],
                "project_count": 0,
            }

        builders[key]["documents"].append(project)
        builders[key]["project_count"] = len(builders[key]["documents"])

    return sorted(builders.values(), key=lambda builder: builder["name"].lower())
