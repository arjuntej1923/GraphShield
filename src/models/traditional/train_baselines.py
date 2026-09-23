from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
EXPERIMENTS_DIR = ROOT / "experiments" / "baseline"

TRAIN_FILE = DATA_DIR / "train_temporal.csv"
VAL_FILE = DATA_DIR / "validation_temporal.csv"
TEST_FILE = DATA_DIR / "test_temporal.csv"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# We exclude transaction ID, time step and the label.
FEATURE_COLUMNS = [
    f"feature_{i}"
    for i in range(1, 166)
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("GRAPHSHIELD - TRADITIONAL ML BASELINES")
    print("=" * 70)

    print("\nLoading temporal datasets...")

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VAL_FILE)
    test = pd.read_csv(TEST_FILE)

    print(f"Training samples   : {len(train):,}")
    print(f"Validation samples : {len(validation):,}")
    print(f"Test samples       : {len(test):,}")

    return train, validation, test


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_data(train, validation, test):

    print("\n" + "=" * 70)
    print("PREPARING FEATURES")
    print("=" * 70)

    X_train = train[FEATURE_COLUMNS].copy()
    X_val = validation[FEATURE_COLUMNS].copy()
    X_test = test[FEATURE_COLUMNS].copy()

    # Convert labels:
    # 1 = illicit
    # 2 = licit
    #
    # We convert them to:
    # 1 = illicit
    # 0 = licit

    y_train = (train["class"].astype(str) == "1").astype(int)
    y_val = (validation["class"].astype(str) == "1").astype(int)
    y_test = (test["class"].astype(str) == "1").astype(int)

    print("\nFeature dimensions:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val  : {X_val.shape}")
    print(f"X_test : {X_test.shape}")

    print("\nClass distribution:")

    print(
        f"Train      - Licit: {(y_train == 0).sum():,} | "
        f"Illicit: {(y_train == 1).sum():,}"
    )

    print(
        f"Validation - Licit: {(y_val == 0).sum():,} | "
        f"Illicit: {(y_val == 1).sum():,}"
    )

    print(
        f"Test       - Licit: {(y_test == 0).sum():,} | "
        f"Illicit: {(y_test == 1).sum():,}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# SCALE FEATURES
# ============================================================

def scale_features(X_train, X_val, X_test):

    print("\n" + "=" * 70)
    print("FEATURE SCALING")
    print("=" * 70)

    print("\nFitting StandardScaler on TRAINING data only...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    # IMPORTANT:
    # Validation and test are transformed using the
    # scaler fitted only on training data.

    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("[PASS] No validation/test data used to fit scaler.")

    return X_train_scaled, X_val_scaled, X_test_scaled


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(name, model, X, y):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "model": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    train, validation, test = load_data()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = prepare_data(
        train,
        validation,
        test
    )

    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
    ) = scale_features(
        X_train,
        X_val,
        X_test
    )

    # ========================================================
    # MODELS
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING BASELINE MODELS")
    print("=" * 70)

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    validation_results = []
    test_results = []

    # ========================================================
    # TRAIN + VALIDATE
    # ========================================================

    for name, model in models.items():

        print("\n" + "=" * 70)
        print(f"TRAINING: {name}")
        print("=" * 70)

        model.fit(
            X_train_scaled,
            y_train
        )

        print("\nValidation performance:")

        val_result = evaluate_model(
            name,
            model,
            X_val_scaled,
            y_val
        )

        validation_results.append(val_result)

        print("\nTest performance:")

        test_result = evaluate_model(
            name,
            model,
            X_test_scaled,
            y_test
        )

        test_results.append(test_result)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    validation_df = pd.DataFrame(
        validation_results
    )

    test_df = pd.DataFrame(
        test_results
    )

    validation_file = (
        EXPERIMENTS_DIR /
        "validation_results.csv"
    )

    test_file = (
        EXPERIMENTS_DIR /
        "test_results.csv"
    )

    validation_df.to_csv(
        validation_file,
        index=False
    )

    test_df.to_csv(
        test_file,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(
        validation_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        test_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("RESULT FILES")
    print("=" * 70)

    print(f"\nSaved: {validation_file}")
    print(f"Saved: {test_file}")

    print("\n" + "=" * 70)
    print("BASELINE EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()