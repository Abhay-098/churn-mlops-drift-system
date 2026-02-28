from xgboost import XGBClassifier
import os
import pandas as pd


def retrain_model(version="v_auto"):

    # Load original training dataset
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "churn.csv")

    df = pd.read_csv(DATA_PATH)

    df.drop("customerID", axis=1, inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Handle edge case safely
    if sum(y == 1) == 0:
        scale_pos_weight = 1
    else:
        scale_pos_weight = sum(y == 0) / sum(y == 1)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X, y)

    print("✅ Model retrained successfully.")

    return model