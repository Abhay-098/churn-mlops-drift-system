from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from prometheus_client import Counter, generate_latest
from fastapi.responses import Response
from app.model import load_model, save_model, get_model_info
from fastapi import FastAPI
from app.model import rollback_model
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

prediction_counter = Counter(
    "churn_predictions_total",
    "Total number of predictions made"
)

drift_counter = Counter(
    "drift_events_total",
    "Total number of drift detections"
)

retrain_counter = Counter(
    "model_retraining_total",
    "Total number of model retraining events"
)

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
    prediction_counter.inc()

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
            drift_counter.inc()
            print("⚠ Drift detected! Retraining model...")
             # Retrain model
            new_model = retrain_model(version="v_auto")

            retrain_counter.inc()
           

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

@app.post("/rollback/{version}")
def rollback(version: str):

    result = rollback_model(version)

    global model
    model = load_model()

    return result

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.get("/evaluate")
def evaluate_model():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "churn.csv")

    df = pd.read_csv(DATA_PATH)

    # Preprocess same as training
    df.drop("customerID", axis=1, inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Predict
    preds = model.predict(X)

    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }