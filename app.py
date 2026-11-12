
from pathlib import Path
import os

from flask import Flask, jsonify, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model from the same folder as app.py
MODEL_PATH = Path(__file__).resolve().parent / "random_forest_model_com.joblib"

try:
    model_data = joblib.load(MODEL_PATH)

    model = model_data["model"]
    target_encoder = model_data["target_encoder"]
    features = list(model_data["features"])

    print("Model loaded successfully.")
    print("Expected features:", features)

except Exception as exc:
    raise RuntimeError(
        f"Could not load model file: {MODEL_PATH}. Error: {exc}"
    ) from exc


@app.get("/")
def home():
    return jsonify({
        "message": "Country Prediction API is active.",
        "expected_features": features
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@app.post("/predict")
def predict():

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Send a valid JSON object."
            }), 400

        # Supported input features
        supported_inputs = {
            "Quantity": data.get("Quantity"),
            "UnitPrice": data.get("UnitPrice"),
            "CustomerID": data.get("CustomerID")
        }

        # Check whether the model uses unsupported features
        unsupported = [
            name for name in features
            if name not in supported_inputs
        ]

        if unsupported:
            return jsonify({
                "error": "Model contains unsupported features.",
                "unsupported_features": unsupported,
                "expected_features": features
            }), 500

        # Check missing values
        missing_values = [
            name for name in features
            if supported_inputs[name] is None
        ]

        if missing_values:
            return jsonify({
                "error": "Missing required input fields.",
                "missing_fields": missing_values
            }), 400

        # Convert input values into numbers
        row = {}

        for name in features:
            try:
                value = float(supported_inputs[name])

                if not np.isfinite(value):
                    raise ValueError("Value must be finite.")

                row[name] = value

            except (TypeError, ValueError):
                return jsonify({
                    "error": f"Invalid numeric value for {name}."
                }), 400

        # Create DataFrame in training feature order
        input_frame = pd.DataFrame(
            [row],
            columns=features
        )

        # Predict country
        prediction_encoded = model.predict(input_frame)[0]

        predicted_country = target_encoder.inverse_transform(
            np.asarray([prediction_encoded])
        )[0]

        result = {
            "prediction": str(predicted_country)
        }

        # Return confidence if available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_frame)[0]

            result["confidence"] = float(
                np.max(probabilities)
            )

        return jsonify(result), 200

    except Exception as exc:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": str(exc)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

