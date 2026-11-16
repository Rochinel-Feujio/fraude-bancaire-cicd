"""Évaluation du modèle : precision, recall, F1-score — sauvegardés comme preuve."""
import json
import os
import sys

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import train

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

# Seuil minimal de F1-score accepté : sert de garde-fou en CI (non-régression du modèle)
MIN_F1_THRESHOLD = 0.75


def evaluate(min_f1_threshold=MIN_F1_THRESHOLD, metrics_path=METRICS_PATH):
    model, scaler, (X_test, y_test) = train()
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "n_test_samples": int(len(y_test)),
        "n_fraud_in_test": int(y_test.sum()),
        "min_f1_threshold": min_f1_threshold,
        "passed_threshold": bool(f1 >= min_f1_threshold),
    }

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics, report


if __name__ == "__main__":
    metrics, report = evaluate()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if not metrics["passed_threshold"]:
        print(f"ÉCHEC : F1-score ({metrics['f1_score']}) en dessous du seuil ({MIN_F1_THRESHOLD})")
        sys.exit(1)
    print("Seuil de performance respecté.")
