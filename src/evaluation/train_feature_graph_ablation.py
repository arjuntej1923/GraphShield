from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
RESULTS_DIR = ROOT / "results" / "feature_graph_ablation"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

FEATURE_COLUMNS = [
    f"feature_{i}"
    for i in range(1, 166)
]

GRAPH_COLUMNS = [
    "in_degree",
    "out_degree",
    "total_degree"
]


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph(path):

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )


# ============================================================
# CREATE DEGREE DATAFRAME FROM GRAPH
# ============================================================

def create_degree_dataframe(graph):

    edge_index = graph.edge_index.cpu()

    source = edge_index[0].numpy()
    target = edge_index[1].numpy()

    num_nodes = graph.num_nodes

    in_degree = np.bincount(
        target,
        minlength=num_nodes
    )

    out_degree = np.bincount(
        source,
        minlength=num_nodes
    )

    total_degree = (
        in_degree
        + out_degree
    )

    tx_ids = graph.tx_ids

    if torch.is_tensor(tx_ids):

        tx_ids = (
            tx_ids.cpu().numpy()
        )

    return pd.DataFrame(
        {
            "txId": tx_ids,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "total_degree": total_degree
        }
    )


# ============================================================
# LOAD FEATURE DATA
# ============================================================

def load_feature_split(
    filename
):

    path = (
        DATA_DIR
        / filename
    )

    df = pd.read_csv(path)

    return df


# ============================================================
# ADD GRAPH FEATURES
# ============================================================

def prepare_split(
    split_df,
    graph
):

    degree_df = (
        create_degree_dataframe(
            graph
        )
    )

    # --------------------------------------------------------
    # Merge using transaction ID
    # --------------------------------------------------------

    merged = split_df.merge(
        degree_df,
        on="txId",
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------------
    # Check merge
    # --------------------------------------------------------

    missing_graph_features = (
        merged[GRAPH_COLUMNS]
        .isna()
        .any(axis=1)
        .sum()
    )

    if missing_graph_features > 0:

        raise ValueError(
            f"Missing graph features for "
            f"{missing_graph_features} transactions"
        )

    # --------------------------------------------------------
    # Convert label
    #
    # Original dataset:
    # 1 = illicit
    # 2 = licit
    # --------------------------------------------------------

    merged["target"] = (
        merged["class"] == 1
    ).astype(int)

    return merged


# ============================================================
# BUILD FEATURES
# ============================================================

def build_feature_matrices(
    train_df,
    val_df,
    test_df
):

    # --------------------------------------------------------
    # Feature-only
    # --------------------------------------------------------

    X_train_feature = (
        train_df[
            FEATURE_COLUMNS
        ].to_numpy()
    )

    X_val_feature = (
        val_df[
            FEATURE_COLUMNS
        ].to_numpy()
    )

    X_test_feature = (
        test_df[
            FEATURE_COLUMNS
        ].to_numpy()
    )

    # --------------------------------------------------------
    # Feature + graph
    # --------------------------------------------------------

    combined_columns = (
        FEATURE_COLUMNS
        + GRAPH_COLUMNS
    )

    X_train_combined = (
        train_df[
            combined_columns
        ].to_numpy()
    )

    X_val_combined = (
        val_df[
            combined_columns
        ].to_numpy()
    )

    X_test_combined = (
        test_df[
            combined_columns
        ].to_numpy()
    )

    y_train = (
        train_df["target"]
        .to_numpy()
    )

    y_val = (
        val_df["target"]
        .to_numpy()
    )

    y_test = (
        test_df["target"]
        .to_numpy()
    )

    return (
        X_train_feature,
        X_val_feature,
        X_test_feature,
        X_train_combined,
        X_val_combined,
        X_test_combined,
        y_train,
        y_val,
        y_test
    )


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    y_true,
    probabilities
):

    thresholds = np.arange(
        0.01,
        1.00,
        0.01
    )

    best_threshold = 0.50
    best_f1 = -1.0

    rows = []

    for threshold in thresholds:

        predictions = (
            probabilities
            >= threshold
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

        rows.append(
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

    return (
        best_threshold,
        pd.DataFrame(rows)
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model_name,
    split_name,
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities
        >= threshold
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

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities
    )

    cm = confusion_matrix(
        y_true,
        predictions
    )

    return {
        "model": model_name,
        "split": split_name,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1]
    }


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    model_name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):

    print("\n" + "-" * 70)

    print(
        f"TRAINING: {model_name}"
    )

    print("-" * 70)

    model.fit(
        X_train,
        y_train
    )

    val_probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    # --------------------------------------------------------
    # Select threshold ONLY on validation
    # --------------------------------------------------------

    threshold, threshold_table = (
        find_best_threshold(
            y_val,
            val_probabilities
        )
    )

    threshold_filename = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("+", "plus")
        + "_thresholds.csv"
    )

    threshold_table.to_csv(
        RESULTS_DIR
        / threshold_filename,
        index=False
    )

    # --------------------------------------------------------
    # Evaluate validation
    # --------------------------------------------------------

    val_result = evaluate_model(
        model_name,
        "validation",
        y_val,
        val_probabilities,
        threshold
    )

    # --------------------------------------------------------
    # Evaluate test
    # --------------------------------------------------------

    test_result = evaluate_model(
        model_name,
        "test",
        y_test,
        test_probabilities,
        threshold
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"\nSelected threshold: "
        f"{threshold:.2f}"
    )

    print(
        "\nValidation:"
    )

    print(
        f"Precision : "
        f"{val_result['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{val_result['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{val_result['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{val_result['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{val_result['pr_auc']:.4f}"
    )

    print(
        "\nTest:"
    )

    print(
        f"Precision : "
        f"{test_result['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{test_result['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{test_result['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{test_result['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{test_result['pr_auc']:.4f}"
    )

    # --------------------------------------------------------
    # Save probabilities
    # --------------------------------------------------------

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("+", "plus")
    )

    pd.DataFrame(
        {
            "y_true": y_val,
            "predicted_probability":
                val_probabilities
        }
    ).to_csv(
        RESULTS_DIR
        / f"{safe_name}_validation_predictions.csv",
        index=False
    )

    pd.DataFrame(
        {
            "y_true": y_test,
            "predicted_probability":
                test_probabilities
        }
    ).to_csv(
        RESULTS_DIR
        / f"{safe_name}_test_predictions.csv",
        index=False
    )

    return [
        val_result,
        test_result
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print(
        "GRAPHSHIELD - FEATURE + GRAPH ABLATION"
    )
    print("=" * 70)

    # ========================================================
    # LOAD LABELED DATA
    # ========================================================

    print(
        "\nLoading temporal datasets..."
    )

    train_df = load_feature_split(
        "train_temporal.csv"
    )

    val_df = load_feature_split(
        "validation_temporal.csv"
    )

    test_df = load_feature_split(
        "test_temporal.csv"
    )

    print(
        f"Train samples : "
        f"{len(train_df):,}"
    )

    print(
        f"Validation samples : "
        f"{len(val_df):,}"
    )

    print(
        f"Test samples : "
        f"{len(test_df):,}"
    )

    # ========================================================
    # LOAD GRAPHS
    # ========================================================

    print(
        "\nLoading graphs..."
    )

    train_graph = load_graph(
        DATA_DIR / "graph_train.pt"
    )

    val_graph = load_graph(
        DATA_DIR / "graph_validation.pt"
    )

    test_graph = load_graph(
        DATA_DIR / "graph_test.pt"
    )

    # ========================================================
    # ADD GRAPH STRUCTURE
    # ========================================================

    print(
        "\nAdding graph structural features..."
    )

    train_df = prepare_split(
        train_df,
        train_graph
    )

    val_df = prepare_split(
        val_df,
        val_graph
    )

    test_df = prepare_split(
        test_df,
        test_graph
    )

    print(
        "\n[PASS] Graph features added:"
    )

    for column in GRAPH_COLUMNS:

        print(
            f"[PASS] {column}"
        )

    # ========================================================
    # BUILD MATRICES
    # ========================================================

    (
        X_train_feature,
        X_val_feature,
        X_test_feature,
        X_train_combined,
        X_val_combined,
        X_test_combined,
        y_train,
        y_val,
        y_test
    ) = build_feature_matrices(
        train_df,
        val_df,
        test_df
    )

    # ========================================================
    # SCALE FEATURE-ONLY
    # ========================================================

    print(
        "\nFitting scaler for feature-only model..."
    )

    scaler_feature = StandardScaler()

    X_train_feature = (
        scaler_feature.fit_transform(
            X_train_feature
        )
    )

    X_val_feature = (
        scaler_feature.transform(
            X_val_feature
        )
    )

    X_test_feature = (
        scaler_feature.transform(
            X_test_feature
        )
    )

    # ========================================================
    # SCALE FEATURE + GRAPH
    # ========================================================

    print(
        "\nFitting scaler for feature + graph model..."
    )

    scaler_combined = StandardScaler()

    X_train_combined = (
        scaler_combined.fit_transform(
            X_train_combined
        )
    )

    X_val_combined = (
        scaler_combined.transform(
            X_val_combined
        )
    )

    X_test_combined = (
        scaler_combined.transform(
            X_test_combined
        )
    )

    print(
        "\n[PASS] Scalers fitted on training data only."
    )

    # ========================================================
    # MODEL DEFINITIONS
    # ========================================================

    models = {

        "Feature-only Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1
            ),

        "Feature+Graph Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1
            ),

        "Feature-only Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1
            ),

        "Feature+Graph Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1
            ),

        "Feature-only XGBoost":
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=-1
            ),

        "Feature+Graph XGBoost":
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=-1
            )
    }

    # ========================================================
    # TRAIN
    # ========================================================

    all_results = []

    # --------------------------------------------------------
    # Feature-only Logistic Regression
    # --------------------------------------------------------

    all_results.extend(
        train_model(
            "Feature-only Logistic Regression",
            models[
                "Feature-only Logistic Regression"
            ],
            X_train_feature,
            y_train,
            X_val_feature,
            y_val,
            X_test_feature,
            y_test
        )
    )

    # --------------------------------------------------------
    # Feature + Graph Logistic Regression
    # --------------------------------------------------------

    all_results.extend(
        train_model(
            "Feature+Graph Logistic Regression",
            models[
                "Feature+Graph Logistic Regression"
            ],
            X_train_combined,
            y_train,
            X_val_combined,
            y_val,
            X_test_combined,
            y_test
        )
    )

    # --------------------------------------------------------
    # Feature-only Random Forest
    # --------------------------------------------------------

    all_results.extend(
        train_model(
            "Feature-only Random Forest",
            models[
                "Feature-only Random Forest"
            ],
            X_train_feature,
            y_train,
            X_val_feature,
            y_val,
            X_test_feature,
            y_test
        )
    )

    # --------------------------------------------------------
    # Feature + Graph Random Forest
    # --------------------------------------------------------

    all_results.extend(
        train_model(
            "Feature+Graph Random Forest",
            models[
                "Feature+Graph Random Forest"
            ],
            X_train_combined,
            y_train,
            X_val_combined,
            y_val,
            X_test_combined,
            y_test
        )
    )

    # --------------------------------------------------------
    # Feature-only XGBoost
    # --------------------------------------------------------

    all_results.extend(
        train_model(
            "Feature-only XGBoost",
            models[
                "Feature-only XGBoost"
            ],
            X_train_feature,
            y_train,
            X_val_feature,
            y_val,
            X_test_feature,
            y_test
        )
    )

    # --------------------------------------------------------
    # Feature + Graph XGBoost
    # --------------------------------------------------------

    all_results.extend(
        train_model(
            "Feature+Graph XGBoost",
            models[
                "Feature+Graph XGBoost"
            ],
            X_train_combined,
            y_train,
            X_val_combined,
            y_val,
            X_test_combined,
            y_test
        )
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_df = results_df[
        [
            "model",
            "split",
            "threshold",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "tn",
            "fp",
            "fn",
            "tp"
        ]
    ]

    output_file = (
        RESULTS_DIR
        / "feature_graph_ablation_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # PRINT FINAL TABLE
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "FEATURE + GRAPH ABLATION RESULTS"
    )
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    print(
        "\n[PASS] Results saved:"
    )

    print(output_file)

    print("\n" + "=" * 70)
    print(
        "FEATURE + GRAPH ABLATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
