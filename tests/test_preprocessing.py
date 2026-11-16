import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import load_data, split_and_scale, prepare_from_csv, FEATURE_COLUMNS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "transactions.csv")


def test_load_data_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_data("ce_fichier_n_existe_pas.csv")


def test_load_data_has_expected_columns():
    df = load_data(DATA_PATH)
    for col in FEATURE_COLUMNS + ["is_fraud"]:
        assert col in df.columns


def test_split_and_scale_shapes():
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df, test_size=0.2)

    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] == len(FEATURE_COLUMNS)
    # Le split est stratifié : le taux de fraude doit être comparable entre train et test
    fraud_rate_train = y_train.mean()
    fraud_rate_test = y_test.mean()
    assert abs(fraud_rate_train - fraud_rate_test) < 0.02


def test_prepare_from_csv_end_to_end():
    X_train, X_test, y_train, y_test, scaler = prepare_from_csv(DATA_PATH)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
