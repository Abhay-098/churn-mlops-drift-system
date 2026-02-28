from app.model import load_model, save_model, get_model_info
from fastapi import FastAPI
import pandas as pd
import os
import sys

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from monitoring.drift import detect_drift
from retraining.retrain import retrain_model
from app.model import load_model, save_model

app = FastAPI()

# Load model using model manager
model = load_model()

# Load baseline training distribution
TRAIN_DIST_PATH = os.path.join(BASE_DIR, "monitoring", "training_distribution.csv")
baseline_df = pd.read_csv(TRAIN_DIST_PATH)

# Store incoming prediction data
prediction_log = []


@app.get("/")
def home():
    return {"message": "Churn MLOps API is running 🚀"}


@app.post("/predict")
def predict(data: dict):

    global model
    global prediction_log

    df = pd.DataFrame([data])

    # =========================
    # Prediction
    # =========================
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    # =========================
    # Log prediction input
    # =========================
    prediction_log.append(df)

    # =========================
    # Drift Detection
    # =========================
    if len(prediction_log) >= 20:

        new_data = pd.concat(prediction_log, ignore_index=True)

        drift_flag, drift_scores = detect_drift(
            baseline_df,
            new_data.select_dtypes(include=["int64", "float64"])
        )

        if drift_flag:
            print("⚠ Drift detected! Retraining model...")

            # Retrain model
            new_model = retrain_model(version="v_auto")

            # Save via model manager
            save_model(new_model, version="v_auto")

            # Reload latest model
            model = load_model()

            # Reset prediction log
            prediction_log = []

            print("✅ Model retrained and activated.")

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability)
    }

@app.get("/model-info")
def model_info():
    return get_model_info()