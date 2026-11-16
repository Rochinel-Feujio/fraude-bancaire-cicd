"""
Tests de RECETTE (acceptance tests) : exécutés contre l'API déjà démarrée
(http://localhost:5000), pas contre les fonctions Python directement.
Objectif : valider que le comportement observable de l'application correspond
à ce qu'attend le métier, avant la mise en production.

Pré-requis : l'API doit tourner (voir étape "recette" du workflow CI/CD).
"""
import os

import pytest
import requests

API_URL = os.environ.get("API_URL", "http://localhost:5000")


def _predict(payload):
    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
    resp.raise_for_status()
    return resp.json()


def test_health_endpoint_available():
    resp = requests.get(f"{API_URL}/health", timeout=5)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_high_risk_transaction_is_flagged():
    # Transaction de nuit, loin du domicile, montant très supérieur à l'habitude,
    # sans puce ni code PIN, en ligne : scénario typiquement frauduleux.
    payload = {
        "distance_from_home": 80,
        "distance_from_last_transaction": 60,
        "ratio_to_median_purchase_price": 6,
        "amount": 950,
        "repeat_retailer": 0,
        "used_chip": 0,
        "used_pin_number": 0,
        "online_order": 1,
        "transaction_hour": 3,
    }
    result = _predict(payload)
    assert result["is_fraud_prediction"] == 1
    assert result["fraud_probability"] > 0.5


def test_low_risk_transaction_is_not_flagged():
    # Transaction habituelle, en journée, chez un commerçant connu, avec puce et PIN.
    payload = {
        "distance_from_home": 2,
        "distance_from_last_transaction": 1,
        "ratio_to_median_purchase_price": 1.0,
        "amount": 35,
        "repeat_retailer": 1,
        "used_chip": 1,
        "used_pin_number": 1,
        "online_order": 0,
        "transaction_hour": 14,
    }
    result = _predict(payload)
    assert result["is_fraud_prediction"] == 0
    assert result["fraud_probability"] < 0.5


def test_missing_field_returns_400():
    resp = requests.post(f"{API_URL}/predict", json={"amount": 100}, timeout=5)
    assert resp.status_code == 400
