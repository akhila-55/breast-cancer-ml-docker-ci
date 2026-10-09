from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


app = Flask(__name__)

MODEL_PATH = Path("breast_cancer_model.pkl")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "breast_cancer_model.pkl was not found. "
            "Run the training pipeline first."
        )

    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "breast-cancer-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    model = load_model()

    # Get the feature names used when training the model
    features = list(model.feature_names_in_)

    missing_fields = [
        feature for feature in features
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    sample = pd.DataFrame([{
        feature: data[feature]
        for feature in features
    }])

    prediction_code = int(model.predict(sample)[0])

    # The original dataset uses 0/1 encoded target values.
    prediction = f"CLASS_{prediction_code}"

    return jsonify({
        "prediction": prediction,
        "prediction_code": prediction_code
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
