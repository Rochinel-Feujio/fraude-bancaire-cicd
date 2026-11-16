"""
API de scoring (Flask) : charge le modèle entraîné et expose un endpoint
POST /predict — représente la "mise en production" du modèle.
"""
import os
import sys

import joblib
import numpy as np
from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import FEATURE_COLUMNS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")

app = Flask(__name__)
_model = None
_scaler = None


def get_model_and_scaler():
    global _model, _scaler
    if _model is None or _scaler is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Modèle introuvable — lance d'abord `python3 src/train.py`."
            )
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
    return _model, _scaler


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    model, scaler = get_model_and_scaler()
    payload = request.get_json(force=True)

    missing = [c for c in FEATURE_COLUMNS if c not in payload]
    if missing:
        return jsonify({"error": f"Champs manquants : {missing}"}), 400

    x = np.array([[payload[c] for c in FEATURE_COLUMNS]])
    x_scaled = scaler.transform(x)

    prediction = int(model.predict(x_scaled)[0])
    proba = float(model.predict_proba(x_scaled)[0][1])

    return jsonify({
        "is_fraud_prediction": prediction,
        "fraud_probability": round(proba, 4),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
