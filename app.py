from pathlib import Path
import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, request
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
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


# ============================================================
# PROMETHEUS METRICS
# ============================================================

HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests received by the Breast Cancer Prediction API",
    ["method", "path", "status"],
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total Breast Cancer prediction attempts",
    ["result"],
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the Breast Cancer Prediction model is loaded",
)

# Model is initially not loaded
MODEL_LOADED.set(0)


MODEL_PATH = Path("breast_cancer_model.pkl")


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    """Load the trained model and log success or failure."""

    if not MODEL_PATH.exists():
        MODEL_LOADED.set(0)

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

        MODEL_LOADED.set(1)

        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH.name,
        )

        return model

    except Exception:
        MODEL_LOADED.set(0)

        logger.exception(
            "event=model_load_failed model_file=%s",
            MODEL_PATH.name,
        )

        raise


# ============================================================
# REQUEST TRACKING
# ============================================================

@app.before_request
def start_request_tracking():
    """Assign a request ID and record the start time."""

    request.request_id = (
        request.headers.get("X-Request-ID") or str(uuid.uuid4())
    )

    request.start_time = time.perf_counter()


@app.after_request
def log_request(response):
    """Log HTTP status and duration and record Prometheus metrics."""

    request_id = getattr(
        request,
        "request_id",
        str(uuid.uuid4()),
    )

    start_time = getattr(
        request,
        "start_time",
        time.perf_counter(),
    )

    duration_ms = (
        time.perf_counter() - start_time
    ) * 1000

    response.headers["X-Request-ID"] = request_id

    level = (
        logging.WARNING
        if response.status_code >= 400
        else logging.INFO
    )

    # Existing Practical-11 logging
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

    # Prometheus metrics
    # Do not count the /metrics endpoint itself.
    if request.path != "/metrics":

        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.path,
            status=str(response.status_code),
        ).inc()

        HTTP_LATENCY.labels(
            method=request.method,
            path=request.path,
        ).observe(duration_ms / 1000)

    return response


# ============================================================
# ERROR HANDLING
# ============================================================

@app.errorhandler(HTTPException)
def handle_http_error(error):
    """Log HTTP errors such as 404."""

    logger.warning(
        "event=http_error request_id=%s path=%s status=%s",
        getattr(request, "request_id", "unknown"),
        request.path,
        error.code,
    )

    return error


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    """Log unexpected server errors."""

    logger.exception(
        "event=unhandled_exception request_id=%s path=%s error_type=%s",
        getattr(request, "request_id", "unknown"),
        request.path,
        type(error).__name__,
    )

    return jsonify({
        "error": "Internal server error"
    }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def health_check():
    """Basic service health endpoint."""

    logger.info(
        "event=health_check status=ok"
    )

    return jsonify({
        "status": "ok",
        "service": "breast-cancer-prediction",
    })


# ============================================================
# PROMETHEUS METRICS ENDPOINT
# ============================================================

@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""

    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST,
    )


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict():
    """Generate a prediction using the existing trained model."""

    data = request.get_json(silent=True)

    # Validate JSON request
    if not isinstance(data, dict) or not data:

        logger.warning(
            "event=invalid_prediction_request "
            "request_id=%s reason=missing_or_invalid_json",
            request.request_id,
        )

        return jsonify({
            "error": "A non-empty JSON object is required"
        }), 400

    try:

        # Load trained model
        model = load_model()

        # Keep feature names and ordering from trained model
        features = list(
            model.feature_names_in_
        )

        # Check for missing fields
        missing_fields = [
            feature
            for feature in features
            if feature not in data
        ]

        if missing_fields:

            logger.warning(
                "event=validation_error "
                "request_id=%s missing_field_count=%s",
                request.request_id,
                len(missing_fields),
            )

            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields,
            }), 400

        # Do not log submitted feature values
        logger.info(
            "event=prediction_request "
            "request_id=%s feature_count=%s",
            request.request_id,
            len(features),
        )

        # Create input DataFrame
        sample = pd.DataFrame([
            {
                feature: data[feature]
                for feature in features
            }
        ])

        # Generate prediction
        prediction_code = int(
            model.predict(sample)[0]
        )

        prediction = f"CLASS_{prediction_code}"

        # Prometheus prediction success counter
        PREDICTIONS.labels(
            result="success"
        ).inc()

        logger.info(
            "event=prediction_success "
            "request_id=%s prediction=%s",
            request.request_id,
            prediction,
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code,
        })

    except Exception:

        # Prometheus prediction error counter
        PREDICTIONS.labels(
            result="error"
        ).inc()

        logger.exception(
            "event=prediction_failed "
            "request_id=%s",
            request.request_id,
        )

        return jsonify({
            "error": "Prediction failed"
        }), 500


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
    )
