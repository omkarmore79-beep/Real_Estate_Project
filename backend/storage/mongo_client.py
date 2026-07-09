"""
Centralized MongoDB Client Service.
Manages connections to MongoDB Atlas using certifi CA certificates.
Provides diagnostics and pings for health checks.
"""

import os
from urllib.parse import urlparse
import certifi
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConfigurationError, OperationFailure

# ── Load environment variables ────────────────────────────────────────────────
# Env is already loaded in config.py, but we read from os.environ
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "real_estate_chatbot")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "projects")

def get_mongo_client():
    """Retrieve configured MongoClient with certifi CA certificates and 30s timeouts."""
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI is missing in backend/.env")
    
    # Strip any surrounding quotes that may have been preserved by dotenv
    clean_uri = MONGODB_URI.strip().strip('"').strip("'")
    
    return MongoClient(
        clean_uri,
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=30000,
        socketTimeoutMS=30000,
        tls=True,
        tlsCAFile=certifi.where(),
        retryWrites=True,
    )

def get_database():
    """Get database object."""
    client = get_mongo_client()
    return client[MONGODB_DB]

def get_collection():
    """Get main collection object."""
    db = get_database()
    return db[MONGODB_COLLECTION]

def ping_mongo():
    """Ping MongoDB deployment to verify connection."""
    client = get_mongo_client()
    return client.admin.command("ping")

def get_mongo_diagnostics() -> dict:
    """Return sanitized connection diagnostics without exposing credentials."""
    uri = os.getenv("MONGODB_URI")
    clean_uri = (uri or "").strip().strip('"').strip("'")
    
    safe_host = "unknown"
    if clean_uri:
        try:
            # Handle schemes safely
            parse_uri = clean_uri if "://" in clean_uri else f"mongodb://{clean_uri}"
            parsed = urlparse(parse_uri)
            if parsed.netloc:
                safe_host = parsed.netloc.split("@")[-1].split("/")[0]
            else:
                safe_host = clean_uri.split("@")[-1].split("/")[0] if "@" in clean_uri else "unknown"
        except Exception:
            safe_host = "unknown"
            
    return {
        "mongodb_uri_configured": bool(uri),
        "database": MONGODB_DB,
        "collection": MONGODB_COLLECTION,
        "host": safe_host
    }
