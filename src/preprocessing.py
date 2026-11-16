"""Préparation des données : chargement, split train/test, mise à l'échelle."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "amount",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
    "transaction_hour",
]
TARGET_COLUMN = "is_fraud"


def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Jeu de données introuvable : {csv_path}. "
            "Lance d'abord `python3 data/generate_data.py`."
        )
    return pd.read_csv(csv_path)


def split_and_scale(df, test_size=0.2, random_state=42):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def prepare_from_csv(csv_path, test_size=0.2, random_state=42):
    df = load_data(csv_path)
    return split_and_scale(df, test_size=test_size, random_state=random_state)
