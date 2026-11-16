"""Entraînement du modèle de détection de fraude."""
import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import prepare_from_csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "transactions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")


def train(csv_path=DATA_PATH, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    X_train, X_test, y_train, y_test, scaler = prepare_from_csv(csv_path)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return model, scaler, (X_test, y_test)


if __name__ == "__main__":
    train()
    print(f"Modèle entraîné et sauvegardé : {MODEL_PATH}")
