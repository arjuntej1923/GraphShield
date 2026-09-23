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

ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT / "results"

TRAIN_FILE = RESULTS_DIR / "graph_structure_train.csv"
VAL_FILE = RESULTS_DIR / "graph_structure_validation.csv"
TEST_FILE = RESULTS_DIR / "graph_structure_test.csv"

OUTPUT_DIR = RESULTS_DIR / "degree_baseline"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

DEGREE_FEATURES = [
    "in_degree",
    "out_degree",
    "total_degree",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("GRAPHSHIELD - DEGREE-ONLY BASELINE")
    print("=" * 70)

    print("\nLoading graph structure statistics...")

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VAL_FILE)
    test = pd.read_csv(TEST_FILE)

    print(
        f"\nTraining samples   : {len(train):,}"
    )

    print(
        f"Validation samples : {len(validation):,}"
    )

    print(
        f"Test samples       : {len(test):,}"
    )

    return train, validation, test


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(train, validation, test):

    print("\n" + "=" * 70)
    print("PREPARING DEGREE FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # Keep only labeled transactions
    # --------------------------------------------------------

    train = train[
        train["labeled"] == True
    ].copy()

    validation = validation[
        validation["labeled"] == True
    ].copy()

    test = test[
        test["labeled"] == True
    ].copy()

    # --------------------------------------------------------
    # Target
    #
    # 0 = licit
    # 1 = illicit
    # --------------------------------------------------------

    y_train = train["label"].astype(int).to_numpy()
    y_validation = validation["label"].astype(int).to_numpy()
    y_test = test["label"].astype(int).to_numpy()

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X_train = train[
        DEGREE_FEATURES
    ].astype(float).to_numpy()

    X_validation = validation[
        DEGREE_FEATURES
    ].astype(float).to_numpy()

    X_test = test[
        DEGREE_FEATURES
    ].astype(float).to_numpy()

    print("\nFeatures used:")

    for feature in DEGREE_FEATURES:
        print(f"[PASS] {feature}")

    print(
        f"\nTraining illicit rate     : "
        f"{y_train.mean():.4f}"
    )

    print(
        f"Validation illicit rate   : "
        f"{y_validation.mean():.4f}"
    )

    print(
        f"Test illicit rate         : "
        f"{y_test.mean():.4f}"
    )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    print(
        "\nFitting StandardScaler "
        "on training data only..."
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_validation = scaler.transform(
        X_validation
    )

    X_test = scaler.transform(
        X_test
    )

    print(
        "[PASS] No validation/test data "
        "used to fit scaler."
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


# ============================================================
# MODELS
# ============================================================

def create_models():

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_SEED
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
    }

    return models


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    y_true,
    probabilities
):

    thresholds = np.arange(
        0.05,
        0.951,
        0.01
    )

    best_threshold = 0.5
    best_f1 = -1

    best_precision = 0
    best_recall = 0

    all_results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        all_results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    threshold_df = pd.DataFrame(
        all_results
    )

    return (
        best_threshold,
        best_precision,
        best_recall,
        best_f1,
        threshold_df
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model_name,
    model,
    X_train,
    X_validation,
    X_test,
    y_train,
    y_validation,
    y_test
):

    print("\n" + "-" * 70)
    print(f"TRAINING: {model_name}")
    print("-" * 70)

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Validation probabilities
    # --------------------------------------------------------

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    # --------------------------------------------------------
    # Threshold selection
    # --------------------------------------------------------

    (
        best_threshold,
        val_precision,
        val_recall,
        val_f1,
        threshold_df
    ) = find_best_threshold(
        y_validation,
        validation_probabilities
    )

    print("\nBest validation threshold:")
    print(
        f"Threshold : {best_threshold:.2f}"
    )

    print(
        f"Precision : {val_precision:.4f}"
    )

    print(
        f"Recall    : {val_recall:.4f}"
    )

    print(
        f"F1        : {val_f1:.4f}"
    )

    # --------------------------------------------------------
    # Save threshold curve
    # --------------------------------------------------------

    threshold_file = (
        OUTPUT_DIR
        / f"{model_name.lower().replace(' ', '_')}_thresholds.csv"
    )

    threshold_df.to_csv(
        threshold_file,
        index=False
    )

    # --------------------------------------------------------
    # Test probabilities
    # --------------------------------------------------------

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    # --------------------------------------------------------
    # Locked threshold
    # --------------------------------------------------------

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    test_precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    test_roc_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    test_pr_auc = average_precision_score(
        y_test,
        test_probabilities
    )

    cm = confusion_matrix(
        y_test,
        test_predictions
    )

    print(
        "\nFINAL TEST WITH "
        "VALIDATION-SELECTED THRESHOLD"
    )

    print(
        f"Threshold : {best_threshold:.2f}"
    )

    print(
        f"Precision : {test_precision:.4f}"
    )

    print(
        f"Recall    : {test_recall:.4f}"
    )

    print(
        f"F1        : {test_f1:.4f}"
    )

    print(
        f"ROC-AUC   : {test_roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {test_pr_auc:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "y_true": y_test,
            "predicted_probability":
                test_probabilities,
            "predicted_label":
                test_predictions
        }
    )

    prediction_file = (
        OUTPUT_DIR
        / f"{model_name.lower().replace(' ', '_')}_test_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_file,
        index=False
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "model": model_name,
        "threshold": best_threshold,
        "validation_precision": val_precision,
        "validation_recall": val_recall,
        "validation_f1": val_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "roc_auc": test_roc_auc,
        "pr_auc": test_pr_auc
    }


# ============================================================
# MAIN
# ============================================================

def main():

    train, validation, test = load_data()

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    ) = prepare_data(
        train,
        validation,
        test
    )

    models = create_models()

    results = []

    for model_name, model in models.items():

        result = evaluate_model(
            model_name,
            model,
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            y_test
        )

        results.append(result)

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    output_file = (
        OUTPUT_DIR
        / "degree_baseline_comparison.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("DEGREE-ONLY BASELINE RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print(
        f"\n[PASS] Saved results:"
    )

    print(output_file)

    print("\n" + "=" * 70)
    print("DEGREE-ONLY BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
