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

DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
EXPERIMENTS_DIR = ROOT / "experiments" / "baseline"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

TRAIN_FILE = DATA_DIR / "train_temporal.csv"
VAL_FILE = DATA_DIR / "validation_temporal.csv"
TEST_FILE = DATA_DIR / "test_temporal.csv"

GNN_THRESHOLD_FILE = RESULTS_DIR / "gnn_threshold_results.csv"

FINAL_FILE = RESULTS_DIR / "all_models_comparison.csv"
BASELINE_THRESHOLD_FILE = (
    RESULTS_DIR / "baseline_threshold_results.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    f"feature_{i}"
    for i in range(1, 166)
]

THRESHOLDS = np.arange(
    0.05,
    0.951,
    0.01
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("GRAPHSHIELD - SIX MODEL COMPARISON")
    print("=" * 70)

    print("\nLoading temporal datasets...")

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VAL_FILE)
    test = pd.read_csv(TEST_FILE)

    print(
        f"Training samples   : {len(train):,}"
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

def prepare_data(
    train,
    validation,
    test
):

    X_train = train[
        FEATURE_COLUMNS
    ].copy()

    X_val = validation[
        FEATURE_COLUMNS
    ].copy()

    X_test = test[
        FEATURE_COLUMNS
    ].copy()

    y_train = (
        train["class"]
        .astype(str)
        == "1"
    ).astype(int)

    y_val = (
        validation["class"]
        .astype(str)
        == "1"
    ).astype(int)

    y_test = (
        test["class"]
        .astype(str)
        == "1"
    ).astype(int)

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ============================================================
# SCALE FEATURES
# ============================================================

def scale_features(
    X_train,
    X_val,
    X_test
):

    print(
        "\nFitting StandardScaler "
        "on training data only..."
    )

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(X_train)
    )

    X_val_scaled = (
        scaler.transform(X_val)
    )

    X_test_scaled = (
        scaler.transform(X_test)
    )

    print(
        "[PASS] No validation/test "
        "data used to fit scaler."
    )

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    model_name,
    split_name,
    y_true,
    probabilities
):

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    output_file = (
        PREDICTIONS_DIR
        / f"{safe_name}_{split_name}.csv"
    )

    prediction_df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true),
            "predicted_probability": np.asarray(
                probabilities
            ),
        }
    )

    prediction_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"[PASS] Saved predictions: "
        f"{output_file}"
    )


# ============================================================
# THRESHOLD METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold
):

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

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# SELECT VALIDATION THRESHOLD
# ============================================================

def select_best_threshold(
    y_val,
    val_probabilities
):

    threshold_records = []

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            y_val,
            val_probabilities,
            threshold
        )

        threshold_records.append(
            {
                "threshold": threshold,
                **metrics,
            }
        )

    threshold_df = pd.DataFrame(
        threshold_records
    )

    best_index = (
        threshold_df["f1"].idxmax()
    )

    best_row = threshold_df.loc[
        best_index
    ]

    return (
        float(best_row["threshold"]),
        float(best_row["precision"]),
        float(best_row["recall"]),
        float(best_row["f1"]),
        threshold_df,
    )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

def evaluate_test(
    y_test,
    test_probabilities,
    threshold
):

    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        test_probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
    }


# ============================================================
# CREATE MODELS
# ============================================================

def create_models():

    return {

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
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
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
    }


# ============================================================
# RUN TRADITIONAL MODELS
# ============================================================

def run_traditional_models(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):

    print("\n")
    print("=" * 70)
    print(
        "TRADITIONAL ML - "
        "VALIDATION THRESHOLD SELECTION"
    )
    print("=" * 70)

    models = create_models()

    results = []
    all_thresholds = []

    for name, model in models.items():

        print("\n" + "-" * 70)
        print(f"TRAINING: {name}")
        print("-" * 70)

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # VALIDATION PROBABILITIES
        # ----------------------------------------------------

        val_probabilities = (
            model.predict_proba(X_val)[:, 1]
        )

        save_predictions(
            name,
            "validation",
            y_val,
            val_probabilities
        )

        # ----------------------------------------------------
        # SELECT VALIDATION THRESHOLD
        # ----------------------------------------------------

        (
            best_threshold,
            val_precision,
            val_recall,
            val_f1,
            threshold_df,
        ) = select_best_threshold(
            y_val,
            val_probabilities
        )

        threshold_df.insert(
            0,
            "model",
            name
        )

        all_thresholds.append(
            threshold_df
        )

        print(
            "\nBest validation threshold:"
        )

        print(
            f"Threshold : "
            f"{best_threshold:.2f}"
        )

        print(
            f"Precision : "
            f"{val_precision:.4f}"
        )

        print(
            f"Recall    : "
            f"{val_recall:.4f}"
        )

        print(
            f"F1        : "
            f"{val_f1:.4f}"
        )

        # ----------------------------------------------------
        # TEST PROBABILITIES
        # ----------------------------------------------------

        test_probabilities = (
            model.predict_proba(X_test)[:, 1]
        )

        save_predictions(
            name,
            "test",
            y_test,
            test_probabilities
        )

        # ----------------------------------------------------
        # FINAL TEST
        # ----------------------------------------------------

        test_metrics = evaluate_test(
            y_test,
            test_probabilities,
            best_threshold
        )

        print(
            "\nFINAL TEST WITH "
            "VALIDATION-SELECTED THRESHOLD"
        )

        print(
            f"Threshold : "
            f"{best_threshold:.2f}"
        )

        print(
            f"Precision : "
            f"{test_metrics['precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{test_metrics['recall']:.4f}"
        )

        print(
            f"F1        : "
            f"{test_metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC   : "
            f"{test_metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC    : "
            f"{test_metrics['pr_auc']:.4f}"
        )

        print("\nConfusion Matrix:")

        print(
            test_metrics[
                "confusion_matrix"
            ]
        )

        results.append(
            {
                "model": name,
                "threshold": best_threshold,

                "validation_precision":
                    val_precision,

                "validation_recall":
                    val_recall,

                "validation_f1":
                    val_f1,

                "test_precision":
                    test_metrics[
                        "precision"
                    ],

                "test_recall":
                    test_metrics[
                        "recall"
                    ],

                "test_f1":
                    test_metrics[
                        "f1"
                    ],

                "roc_auc":
                    test_metrics[
                        "roc_auc"
                    ],

                "pr_auc":
                    test_metrics[
                        "pr_auc"
                    ],
            }
        )

    return (
        pd.DataFrame(results),
        pd.concat(
            all_thresholds,
            ignore_index=True
        ),
    )


# ============================================================
# LOAD GNN RESULTS
# ============================================================

def load_gnn_results():

    print("\n")
    print("=" * 70)
    print("LOADING GNN THRESHOLD RESULTS")
    print("=" * 70)

    if not GNN_THRESHOLD_FILE.exists():

        raise FileNotFoundError(
            f"\nMissing file:\n"
            f"{GNN_THRESHOLD_FILE}\n\n"
            "Run:\n"
            "python "
            "src/evaluation/threshold_analysis.py"
        )

    gnn_df = pd.read_csv(
        GNN_THRESHOLD_FILE
    )

    print(
        f"\nLoaded: "
        f"{GNN_THRESHOLD_FILE}"
    )

    return gnn_df


# ============================================================
# STANDARDIZE GNN RESULTS
# ============================================================

def standardize_gnn_results(
    gnn_df
):

    records = []

    for _, row in gnn_df.iterrows():

        records.append(
            {
                "model": row["model"],

                "threshold":
                    row["threshold"],

                "validation_precision":
                    row[
                        "validation_precision"
                    ],

                "validation_recall":
                    row[
                        "validation_recall"
                    ],

                "validation_f1":
                    row[
                        "validation_f1"
                    ],

                "test_precision":
                    row["precision"],

                "test_recall":
                    row["recall"],

                "test_f1":
                    row["f1"],

                "roc_auc":
                    row["roc_auc"],

                "pr_auc":
                    row["pr_auc"],
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        train,
        validation,
        test
    ) = load_data()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = prepare_data(
        train,
        validation,
        test
    )

    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled
    ) = scale_features(
        X_train,
        X_val,
        X_test
    )

    (
        traditional_results,
        all_thresholds
    ) = run_traditional_models(
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    traditional_results.to_csv(
        BASELINE_THRESHOLD_FILE,
        index=False
    )

    all_thresholds.to_csv(
        RESULTS_DIR /
        "baseline_all_thresholds.csv",
        index=False
    )

    print(
        "\nSaved baseline threshold results:"
    )

    print(
        BASELINE_THRESHOLD_FILE
    )

    gnn_df = load_gnn_results()

    gnn_results = (
        standardize_gnn_results(
            gnn_df
        )
    )

    final_df = pd.concat(
        [
            traditional_results,
            gnn_results,
        ],
        ignore_index=True
    )

    model_order = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "GCN",
        "GraphSAGE",
        "GAT",
    ]

    final_df["model"] = pd.Categorical(
        final_df["model"],
        categories=model_order,
        ordered=True
    )

    final_df = (
        final_df
        .sort_values("model")
        .reset_index(drop=True)
    )

    final_df.to_csv(
        FINAL_FILE,
        index=False
    )

    print("\n")
    print("=" * 70)
    print(
        "GRAPHSHIELD - "
        "FINAL SIX MODEL COMPARISON"
    )
    print("=" * 70)

    display_columns = [
        "model",
        "threshold",
        "test_precision",
        "test_recall",
        "test_f1",
        "roc_auc",
        "pr_auc",
    ]

    print(
        "\nFINAL TEST RESULTS:"
    )

    print(
        final_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    print("\n")
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"\nFinal comparison:"
    )

    print(
        FINAL_FILE
    )

    print(
        "\nPrediction files:"
    )

    print(
        PREDICTIONS_DIR
    )

    print("\n")
    print("=" * 70)
    print(
        "SIX MODEL COMPARISON COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
