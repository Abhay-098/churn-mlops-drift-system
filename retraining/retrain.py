import joblib
from xgboost import XGBClassifier


def retrain_model(X_train, y_train, version="v3"):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=(sum(y_train == 0) / sum(y_train == 1)),
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    model_path = f"../models/churn_model_{version}.pkl"
    joblib.dump(model, model_path)

    print(f"Model retrained and saved as {model_path}")

    return model