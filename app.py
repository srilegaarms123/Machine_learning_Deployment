 from pathlib import Path

from flask import Flask, jsonify, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Keep this file in the same directory as app.py.
MODEL_PATH = Path(__file__).resolve().parent / "random_forest_model_com.joblib"

model = None
target_encoder = None
features = None

try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    target_encoder = model_data["target_encoder"]
    features = list(model_data["features"])
    print(f"Model loaded successfully. Expected features: {features}")
except Exception as exc:
    # Fail at startup rather than silently serving an unusable API.
    raise RuntimeError(f"Could not load model file at {MODEL_PATH}: {exc}") from exc


@app.get("/")
def home():
    return jsonify({
        "message": "Country Prediction API is active.",
        "expected_features": features
    })


@app.post("/predict")
def predict():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Send a JSON object in the request body."}), 400

        # Values are read by feature name so the input column order matches training.
        supported_inputs = {
            "Quantity": data.get("Quantity"),
            "UnitPrice": data.get("UnitPrice"),
            "CustomerID": data.get("CustomerID"),
        }

        missing = [name for name in features if name not in supported_inputs]
        if missing:
            return jsonify({
                "error": "The model expects feature names not handled by this API.",
                "missing_features": missing,
                "expected_features": features
            }), 500

        missing_values = [
            name for name in features
            if supported_inputs[name] is None
        ]
        if missing_values:
            return jsonify({
                "error": "Missing required input fields.",
                "missing_fields": missing_values
            }), 400               "missing_fields": missing_values
            }), 400

        row = {
            name: float(supported_inputs[name])
            for name in features
        }
        input_frame = pd.DataFrame([row], columns=features)

        prediction_encoded = model.predict(input_frame)[0]
        predicted_country = target_encoder.inverse_transform(
            np.asarray([prediction_encoded])
        )[0]

        result = {
            "prediction": str(predicted_country)
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_frame)[0]
            result["confidence"] = float(np.max(probabilities))

        return jsonify(result)

    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input value: {exc}"}), 400
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    # Local development only. Use a production WSGI server when deploying.
    app.run(host="0.0.0.0", port=5000, debug=False)
