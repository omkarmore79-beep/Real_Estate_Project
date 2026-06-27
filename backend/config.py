import os

from dotenv import load_dotenv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "real_estate_chatbot")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "projects")

print(f"MongoDB URI configured: {'yes' if MONGODB_URI else 'no'}")
print(f"MongoDB database: {MONGODB_DB}, collection: {MONGODB_COLLECTION}")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "..", "uploads")
DATA_FOLDER = os.path.join(BASE_DIR, "storage", "data")
RAW_FOLDER = os.path.join(BASE_DIR, "storage", "raw_text")
IMAGE_FOLDER = os.path.join(BASE_DIR, "storage", "images")
