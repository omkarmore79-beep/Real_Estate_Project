import json
import os
from config import DATA_FOLDER


def save_json(data, filename):
    document_id = data.get("document_id")
    if document_id:
        name = f"{document_id}.json"
    else:
        name = filename.replace(".pdf", ".json")
    path = os.path.join(DATA_FOLDER, name)

    os.makedirs(DATA_FOLDER, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
