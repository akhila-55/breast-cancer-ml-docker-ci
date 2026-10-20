
from pathlib import Path
import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

# Configure logging to standard output for Docker and Render
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger("breast-cancer-api")

app = Flask(__name__)

MODEL_PATH = Path("breast_cancer_model.pkl")


def load_model():
    """Load the trained model and log success or failure."""
    if not MODEL_PATH.exists():
        logger.error(
            "event=model_load_failed reason=file_not_found model_file=%s",
            MODEL_PATH.name,
        )
        raise FileNotFoundError(
            "breast_cancer_model.pkl was not found. "
            "Run the training pipeline first."
        )

    try:
        model = joblib.load(MODEL_PATH)
        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH.name,
        )
        return model
    except Exception:
        logger.exception(
            "event=model_load_failed model_file=%s",
            MODEL_PATH.name,
        )
        raise


@app.before_request
def start_request_tracking():
    """Assign a request ID and record the start time."""
    request.request_id = (
        request.headers.get("X-Request-ID") or str(uuid.uuid4())
    )
    request.start_time = time.perf_counter()


@app.after_request
def log_request(response):
    """Log HTTP status and duration, and return the request ID."""
    request_id = getattr(request, "request_id", str(uuid.uuid4()))
    start_time = getattr(request, "start_time", time.perf_counter())
    duration_ms = (time.perf_counter() - start_time) * 1000

    response.headers["X-Request-ID"] = request_id

    level = logging.WARNING if response.status_code >= 400 else logging.INFO

    logger.log(
        level,
        "event=http_request request_id=%s method=%s path=%s "
        "status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )

    return response


@app.errorhandler(HTTPException)
def handle_http_error(error):
    """Log HTTP errors such as 404 without exposing request data."""
    logger.warning(
        "event=http_error request_id=%s path=%s status=%s",
        getattr(request, "request_id", "unknown"),
        request.path,
        error.code,
    )
    return error


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    """Log unexpected server errors without returning internal details."""
    logger.exception(
        "event=unhandled_exception request_id=%s path=%s error_type=%s",
        getattr(request, "request_id", "unknown"),
        request.path,
        type(error).__name__,
    )
    return jsonify({"error": "Internal server error"}), 500


@app.get("/")
def health_check():
    """Basic service health endpoint."""
    logger.info("event=health_check status=ok")
    return jsonify({
        "status": "ok",
        "service": "breast-cancer-prediction",
    })


@app.post("/predict")
def predict():
    """Generate a prediction using the existing trained model."""
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or not data:
        logger.warning(
            "event=invalid_prediction_request request_id=%s reason=missing_or_invalid_json",
            request.request_id,
        )
        return jsonify({
            "error": "A non-empty JSON object is required"
        }), 400

    try:
        model = load_model()

        # Keep the feature names and ordering from the trained model.
        features = list(model.feature_names_in_)

        missing_fields = [
            feature for feature in features if feature not in data
        ]

        if missing_fields:
            logger.warning(
                "event=validation_error request_id=%s "
                "missing_field_count=%s",
                request.request_id,
                len(missing_fields),
            )
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields,
            }), 400

        # Do not log the submitted feature values.
        logger.info(
            "event=prediction_request request_id=%s feature_count=%s",
            request.request_id,
            len(features),
        )

        sample = pd.DataFrame([{
            feature: data[feature] for feature in features
        }])

        prediction_code = int(model.predict(sample)[0])
        prediction = f"CLASS_{prediction_code}"

        logger.info(
            "event=prediction_success request_id=%s prediction=%s",
            request.request_id,
            prediction,
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code,
        })

    except Exception:
        logger.exception(
            "event=prediction_failed request_id=%s",
            request.request_id,
        )
        return jsonify({
            "error": "Prediction failed"
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
