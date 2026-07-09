from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
import os
import certifi
from urllib.parse import urlparse

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")

uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB", "real_estate_chatbot")

print("MongoDB URI configured:", bool(uri))

if not uri:
    raise SystemExit("MONGODB_URI missing in backend/.env")

# Clean surrounding quotes from environmental variable if needed
uri = uri.strip().strip('"').strip("'")

safe_host = uri.split("@")[-1].split("/")[0] if "@" in uri else "unknown"
print("MongoDB host:", safe_host)
print("Using certifi:", certifi.where())

client = MongoClient(
    uri,
    serverSelectionTimeoutMS=30000,
    connectTimeoutMS=30000,
    socketTimeoutMS=30000,
    tls=True,
    tlsCAFile=certifi.where(),
)

try:
    print("Pinging MongoDB...")
    result = client.admin.command("ping")
    print("MongoDB ping OK:", result)
    print("Selected database:", db_name)
except Exception as e:
    print("MongoDB connection failed:")
    print(type(e).__name__)
    print(str(e))
    print()
    print("Likely fixes:")
    print("1. Whitelist current IP in MongoDB Atlas Network Access or temporarily allow 0.0.0.0/0.")
    print("2. Check username/password.")
    print("3. URL encode password if it contains special characters.")
    print("4. Give database user readWrite permission.")
    print("5. Try mobile hotspot if college WiFi/firewall blocks TLS.")
    print("6. Make sure certifi, pymongo, and dnspython are upgraded.")
