import json
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


MODEL_PATH = Path("breast_cancer_model.pkl")
METRICS_PATH = Path("metrics.json")


def train_model():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "model": "RandomForestClassifier",
        "dataset": "Breast Cancer Wisconsin",
        "test_accuracy": round(float(accuracy), 4),
        "minimum_required_accuracy": 0.90,
        "test_samples": int(len(y_test)),
        "features": int(X.shape[1])
    }

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8"
    )

    print("Breast Cancer model trained successfully.")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    train_model()
