import json
import os
from config import DATA_FOLDER


def load_all_data():
    all_data = []

    for file in os.listdir(DATA_FOLDER):
        with open(os.path.join(DATA_FOLDER, file)) as f:
            all_data.append(json.load(f))

    return all_data