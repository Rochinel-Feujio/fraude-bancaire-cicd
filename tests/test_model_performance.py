"""
Test de non-régression du modèle : c'est ce test qui, dans le pipeline CI/CD,
bloque l'intégration si le F1-score du modèle retombe sous le seuil acceptable.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import evaluate, MIN_F1_THRESHOLD


def test_model_meets_minimum_f1_threshold():
    metrics, _ = evaluate()
    assert metrics["f1_score"] >= MIN_F1_THRESHOLD, (
        f"F1-score {metrics['f1_score']} en dessous du seuil minimal {MIN_F1_THRESHOLD}"
    )


def test_model_precision_and_recall_are_reasonable():
    metrics, _ = evaluate()
    assert metrics["precision"] > 0.5
    assert metrics["recall"] > 0.5
