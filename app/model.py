import os
import joblib
import json
from datetime import datetime

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CURRENT_MODEL_FILE = os.path.join(MODEL_DIR, "current_model.json")


def save_model(model, version):
    filename = f"churn_model_{version}.pkl"
    path = os.path.join(MODEL_DIR, filename)

    joblib.dump(model, path)

    metadata = {
        "active_model": filename,
        "version": version,
        "last_updated": datetime.utcnow().isoformat()
    }

    with open(CURRENT_MODEL_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

    return path


def load_model():
    if not os.path.exists(CURRENT_MODEL_FILE):
        raise FileNotFoundError("No active model metadata found.")

    with open(CURRENT_MODEL_FILE, "r") as f:
        metadata = json.load(f)

    model_path = os.path.join(MODEL_DIR, metadata["active_model"])
    return joblib.load(model_path)


def get_model_info():
    if not os.path.exists(CURRENT_MODEL_FILE):
        return {"error": "No model metadata found."}

    with open(CURRENT_MODEL_FILE, "r") as f:
        return json.load(f)