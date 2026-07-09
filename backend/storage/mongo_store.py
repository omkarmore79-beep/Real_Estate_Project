import json
import os
from datetime import datetime, timezone
from typing import Any


from config import DATA_FOLDER, MONGODB_COLLECTION, MONGODB_DB, MONGODB_URI

try:
    from bson import ObjectId
except ImportError:
    ObjectId = None


_client = None

# Strip surrounding quotes that python-dotenv sometimes preserves
# (e.g. MONGODB_URI="mongodb+srv://..." → value includes the literal quotes)
_MONGODB_URI = (MONGODB_URI or "").strip().strip('"').strip("'")


def _get_db():
    from storage.mongo_client import get_database
    try:
        return get_database()
    except Exception as exc:
        print(f"[MongoDB] Failed to retrieve database: {exc}")
        return None





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


# ── Local Document Fallback Emulation ──────────────────────────────────────────

def _save_project_locally(project):
    """Save document metadata locally to backend/storage/local_documents.json."""
    from config import BASE_DIR
    local_file = os.path.join(BASE_DIR, "storage", "local_documents.json")
    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    
    projects = []
    if os.path.exists(local_file):
        try:
            with open(local_file, "r", encoding="utf-8") as f:
                projects = json.load(f)
                if not isinstance(projects, list):
                    projects = []
        except Exception as exc:
            print(f"[Local Storage] Error reading {local_file}: {exc}")
            projects = []
            
    now = datetime.now(timezone.utc).isoformat()
    document = dict(project)
    document["updated_at"] = now
    document.setdefault("uploaded_at", now)
    document = _serialize_document(document)
    
    updated = False
    for i, p in enumerate(projects):
        if p.get("document_id") == document["document_id"]:
            projects[i] = document
            updated = True
            break
    if not updated:
        projects.append(document)
        
    try:
        with open(local_file, "w", encoding="utf-8") as f:
            json.dump(projects, f, indent=2, ensure_ascii=False)
        print(f"[Local Storage] Saved document {document['document_id']} successfully to local_documents.json ✓")
        return document
    except Exception as exc:
        print(f"[Local Storage] Error writing to {local_file}: {exc}")
        return None


def _load_projects_locally(document_id=None, include_raw_text=False):
    """Load documents from backend/storage/local_documents.json."""
    from config import BASE_DIR
    local_file = os.path.join(BASE_DIR, "storage", "local_documents.json")
    if not os.path.exists(local_file):
        return []
        
    try:
        with open(local_file, "r", encoding="utf-8") as f:
            projects = json.load(f)
            if not isinstance(projects, list):
                return []
    except Exception as exc:
        print(f"[Local Storage] Error reading {local_file}: {exc}")
        return []
        
    results = []
    for p in projects:
        if document_id and p.get("document_id") != document_id:
            continue
        item = dict(p)
        if not include_raw_text:
            item.pop("raw_text", None)
        results.append(item)
    return results


def _save_file_locally(document_id, file_path, filename, file_kind, image_id=None):
    """Copy uploaded file to local storage when MongoDB is not used/available."""
    from config import BASE_DIR
    dest_dir = os.path.join(BASE_DIR, "storage", "local_files", document_id, file_kind)
    if image_id:
        dest_dir = os.path.join(dest_dir, image_id)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    import shutil
    try:
        shutil.copy2(file_path, dest_path)
        print(f"[Local File Storage] Saved {filename} to {dest_path}")
        return f"local://{dest_path}"
    except Exception as exc:
        print(f"[Local File Storage] Error saving {filename}: {exc}")
        return None


def _load_file_locally(document_id, file_kind, image_id=None):
    """Load file from local storage instead of MongoDB GridFS."""
    from config import BASE_DIR
    src_dir = os.path.join(BASE_DIR, "storage", "local_files", document_id, file_kind)
    if image_id:
        src_dir = os.path.join(src_dir, image_id)
    if not os.path.exists(src_dir):
        return None
    try:
        files = os.listdir(src_dir)
    except Exception:
        return None
    if not files:
        return None
        
    filepath = os.path.join(src_dir, files[0])
    
    class LocalFileWrapper:
        def __init__(self, path):
            self.path = path
            self._file = open(path, "rb")
            self.metadata = {
                "content_type": "image/png" if path.endswith(".png") else "application/pdf",
                "filename": os.path.basename(path)
            }
        def read(self, *args, **kwargs):
            return self._file.read(*args, **kwargs)
        def close(self):
            try:
                self._file.close()
            except Exception:
                pass
            
    return LocalFileWrapper(filepath)


def _delete_project_locally(document_id):
    """Delete document metadata and local files for a document_id."""
    from config import BASE_DIR
    local_file = os.path.join(BASE_DIR, "storage", "local_documents.json")
    if os.path.exists(local_file):
        try:
            with open(local_file, "r", encoding="utf-8") as f:
                projects = json.load(f)
            if isinstance(projects, list):
                new_projects = [p for p in projects if p.get("document_id") != document_id]
                if len(new_projects) < len(projects):
                    with open(local_file, "w", encoding="utf-8") as f:
                        json.dump(new_projects, f, indent=2, ensure_ascii=False)
                    print(f"[Local Storage] Deleted document {document_id} from local_documents.json")
        except Exception as exc:
            print(f"[Local Storage] Error deleting from {local_file}: {exc}")

    import shutil
    local_dir = os.path.join(BASE_DIR, "storage", "local_files", document_id)
    if os.path.exists(local_dir):
        try:
            shutil.rmtree(local_dir)
            print(f"[Local Storage] Removed folder {local_dir}")
            return True
        except Exception as exc:
            print(f"[Local Storage] Error removing folder {local_dir}: {exc}")
    return False


# ── MongoDB Operations with Local Fallbacks ────────────────────────────────────

def save_project_to_mongo(project):
    from config import ALLOW_UPLOAD_WITHOUT_MONGODB
    try:
        collection = _get_collection()
        if collection is not None:
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
    except Exception as exc:
        print(f"MongoDB save failed: {exc}")

    if ALLOW_UPLOAD_WITHOUT_MONGODB:
        return _save_project_locally(project)
    return None


def delete_project_from_mongo(document_id):
    from config import ALLOW_UPLOAD_WITHOUT_MONGODB
    deleted_mongo = False
    try:
        collection = _get_collection()
        fs = _get_gridfs()
        if collection is not None and fs is not None:
            deleted_files = 0
            for existing in fs.find({"metadata.document_id": document_id}):
                fs.delete(existing._id)
                deleted_files += 1

            result = collection.delete_one({"document_id": document_id})
            deleted_mongo = result.deleted_count > 0 or deleted_files > 0
    except Exception as exc:
        print(f"MongoDB delete failed: {exc}")

    deleted_local = False
    if ALLOW_UPLOAD_WITHOUT_MONGODB:
        deleted_local = _delete_project_locally(document_id)

    return deleted_mongo or deleted_local


def save_file_to_mongo(
    *,
    document_id,
    file_path,
    filename,
    content_type,
    file_kind,
    image_id=None,
):
    from config import ALLOW_UPLOAD_WITHOUT_MONGODB
    try:
        fs = _get_gridfs()
        if fs is not None:
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
    except Exception as exc:
        print(f"MongoDB file save failed: {exc}")

    if ALLOW_UPLOAD_WITHOUT_MONGODB:
        return _save_file_locally(
            document_id=document_id,
            file_path=file_path,
            filename=filename,
            file_kind=file_kind,
            image_id=image_id
        )
    return None


def load_file_from_mongo(document_id, file_kind, image_id=None):
    from config import ALLOW_UPLOAD_WITHOUT_MONGODB
    try:
        fs = _get_gridfs()
        if fs is not None:
            query = {
                "metadata.document_id": document_id,
                "metadata.file_kind": file_kind,
            }
            if image_id:
                query["metadata.image_id"] = image_id

            files = list(fs.find(query).sort("uploadDate", -1).limit(1))
            if files:
                return files[0]
    except Exception as exc:
        print(f"MongoDB file load failed: {exc}")

    if ALLOW_UPLOAD_WITHOUT_MONGODB:
        return _load_file_locally(document_id, file_kind, image_id)
    return None


def load_projects_from_mongo(document_id=None, include_raw_text=False):
    try:
        collection = _get_collection()
        if collection is not None:
            query = {"document_id": document_id} if document_id else {}
            projection = None if include_raw_text else {"raw_text": 0}
            return [
                _serialize_document(item)
                for item in collection.find(query, projection).sort("uploaded_at", -1)
            ]
    except Exception as exc:
        print(f"MongoDB load failed: {exc}")
    return None


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
    from config import ALLOW_UPLOAD_WITHOUT_MONGODB
    
    # Try MongoDB
    try:
        mongo_projects = load_projects_from_mongo(
            document_id,
            include_raw_text=include_raw_text,
        )
        if mongo_projects is not None:
            return mongo_projects
    except Exception as error:
        print("MONGODB LOAD ERROR:", error)

    # Try local storage fallback if allowed
    if ALLOW_UPLOAD_WITHOUT_MONGODB:
        local_projects = _load_projects_locally(document_id, include_raw_text)
        if local_projects:
            return local_projects

    # Legacy JSON fallback
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


def check_mongo_health() -> dict[str, Any]:
    """Test MongoDB connection and return connectivity details/diagnostics."""
    from storage.mongo_client import ping_mongo, get_mongo_diagnostics
    diag = get_mongo_diagnostics()
    if not diag["mongodb_uri_configured"]:
        return {
            "status": "error",
            "mongodb_configured": False,
            "database": diag["database"],
            "collection": diag["collection"],
            "message": "MONGODB_URI is missing in backend/.env",
            "possible_causes": [
                "MONGODB_URI environment variable is missing"
            ]
        }
    
    try:
        ping_mongo()
        return {
            "status": "ok",
            "mongodb_configured": True,
            "database": diag["database"],
            "collection": diag["collection"],
            "message": "MongoDB connection successful"
        }
    except Exception as error:
        print(f"[MongoDB Health Check Error] Detailed log: {error}")
        return {
            "status": "error",
            "mongodb_configured": True,
            "database": diag["database"],
            "collection": diag["collection"],
            "message": "MongoDB connection failed",
            "possible_causes": [
                "MongoDB Atlas IP whitelist does not include your current IP",
                "Wrong database username or password",
                "Password contains special characters and must be URL encoded",
                "Database user does not have readWrite permission",
                "Cluster is paused or unavailable",
                "College WiFi, VPN, antivirus, or firewall is blocking MongoDB Atlas TLS",
                "Python certificate issue; certifi CA bundle is required"
            ]
        }






