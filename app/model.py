import os
import joblib
import json

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_DIR = os.path.abspath(MODEL_DIR)

CURRENT_MODEL_FILE = os.path.join(MODEL_DIR, "current_model.json")


def load_model():
    with open(CURRENT_MODEL_FILE, "r") as f:
        metadata = json.load(f)

    model_path = os.path.join(MODEL_DIR, metadata["active_model"])
    return joblib.load(model_path)


def save_model(model, version):
    os.makedirs(MODEL_DIR, exist_ok=True)

    filename = f"churn_model_{version}.pkl"
    path = os.path.join(MODEL_DIR, filename)

    joblib.dump(model, path)

    metadata = {
        "active_model": filename,
        "version": version
    }

    with open(CURRENT_MODEL_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

    return path


def get_model_info():
    with open(CURRENT_MODEL_FILE, "r") as f:
        return json.load(f)