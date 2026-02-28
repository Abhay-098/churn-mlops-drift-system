import os
import joblib
import json
from datetime import datetime

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_DIR = os.path.abspath(MODEL_DIR)

CURRENT_MODEL_FILE = os.path.join(MODEL_DIR, "current_model.json")


def save_model(model, version):
    os.makedirs(MODEL_DIR, exist_ok=True)

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

    print(f"✅ Model saved as {filename}")

    return path


def load_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # If no metadata exists → auto-train initial model
    if not os.path.exists(CURRENT_MODEL_FILE):
        print("⚠ No model metadata found. Training initial model...")

        from retraining.retrain import retrain_model

        model = retrain_model(version="v1")
        save_model(model, version="v1")

        return model

    with open(CURRENT_MODEL_FILE, "r") as f:
        metadata = json.load(f)

    model_path = os.path.join(MODEL_DIR, metadata["active_model"])

    if not os.path.exists(model_path):
        print("⚠ Model file missing. Retraining...")

        from retraining.retrain import retrain_model

        model = retrain_model(version="v1")
        save_model(model, version="v1")

        return model

    print(f"🚀 Loading model: {metadata['active_model']}")

    return joblib.load(model_path)


def get_model_info():
    if not os.path.exists(CURRENT_MODEL_FILE):
        return {
            "status": "No model initialized yet"
        }

    with open(CURRENT_MODEL_FILE, "r") as f:
        return json.load(f)