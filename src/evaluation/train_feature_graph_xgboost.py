from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
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
# CONFIG
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
# LOAD
# ============================================================

def load_graph(path):

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )


def load_split(filename):

    return pd.read_csv(
        DATA_DIR / filename
    )


# ============================================================
# GRAPH FEATURES
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
        in_degree + out_degree
    )

    tx_ids = graph.tx_ids

    if torch.is_tensor(tx_ids):

        tx_ids = (
            tx_ids.cpu().numpy()
        )

    return pd.DataFrame({
        "txId": tx_ids,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "total_degree": total_degree
    })


def prepare_split(df, graph):

    degree_df = create_degree_dataframe(
        graph
    )

    merged = df.merge(
        degree_df,
        on="txId",
        how="left",
        validate="one_to_one"
    )

    if merged[
        GRAPH_COLUMNS
    ].isna().any().any():

        raise ValueError(
            "Missing graph features after merge."
        )

    merged = merged.copy()

    merged["target"] = (
        merged["class"] == 1
    ).astype(int)

    return merged


# ============================================================
# THRESHOLD
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
    best_f1 = -1

    rows = []

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

        rows.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    return (
        best_threshold,
        pd.DataFrame(rows)
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    split_name,
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

    result = {
        "model":
            "Feature+Graph XGBoost",
        "split":
            split_name,
        "threshold":
            threshold,
        "precision":
            precision,
        "recall":
            recall,
        "f1":
            f1,
        "roc_auc":
            roc_auc,
        "pr_auc":
            pr_auc,
        "tn":
            cm[0, 0],
        "fp":
            cm[0, 1],
        "fn":
            cm[1, 0],
        "tp":
            cm[1, 1]
    }

    print(
        f"\n{split_name.upper()}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print(
        "GRAPHSHIELD - FEATURE + GRAPH XGBOOST"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load temporal data
    # --------------------------------------------------------

    print(
        "\nLoading temporal datasets..."
    )

    train_df = load_split(
        "train_temporal.csv"
    )

    val_df = load_split(
        "validation_temporal.csv"
    )

    test_df = load_split(
        "test_temporal.csv"
    )

    print(
        f"Train      : {len(train_df):,}"
    )

    print(
        f"Validation : {len(val_df):,}"
    )

    print(
        f"Test       : {len(test_df):,}"
    )

    # --------------------------------------------------------
    # Load graphs
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Add graph features
    # --------------------------------------------------------

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
        "[PASS] in_degree"
    )

    print(
        "[PASS] out_degree"
    )

    print(
        "[PASS] total_degree"
    )

    # --------------------------------------------------------
    # Build matrices
    # --------------------------------------------------------

    combined_columns = (
        FEATURE_COLUMNS
        + GRAPH_COLUMNS
    )

    X_train = train_df[
        combined_columns
    ].to_numpy()

    X_val = val_df[
        combined_columns
    ].to_numpy()

    X_test = test_df[
        combined_columns
    ].to_numpy()

    y_train = train_df[
        "target"
    ].to_numpy()

    y_val = val_df[
        "target"
    ].to_numpy()

    y_test = test_df[
        "target"
    ].to_numpy()

    print(
        f"\nInput features: "
        f"{X_train.shape[1]}"
    )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    print(
        "\nFitting scaler on training data only..."
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_val = scaler.transform(
        X_val
    )

    X_test = scaler.transform(
        X_test
    )

    print(
        "[PASS] Scaling complete."
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    print(
        "\nTraining Feature+Graph XGBoost..."
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=1,
        tree_method="hist"
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "[PASS] XGBoost training complete."
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

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
    # Validation threshold
    # --------------------------------------------------------

    (
        best_threshold,
        threshold_table
    ) = find_best_threshold(
        y_val,
        val_probabilities
    )

    print(
        f"\nBest validation threshold: "
        f"{best_threshold:.2f}"
    )

    threshold_table.to_csv(
        RESULTS_DIR
        / "feature_graph_xgboost_thresholds.csv",
        index=False
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    val_result = evaluate(
        "validation",
        y_val,
        val_probabilities,
        best_threshold
    )

    test_result = evaluate(
        "test",
        y_test,
        test_probabilities,
        best_threshold
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = pd.DataFrame(
        [
            val_result,
            test_result
        ]
    )

    results_file = (
        RESULTS_DIR
        / "feature_graph_xgboost_results.csv"
    )

    results.to_csv(
        results_file,
        index=False
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    pd.DataFrame({
        "y_true": y_val,
        "predicted_probability":
            val_probabilities
    }).to_csv(
        RESULTS_DIR
        / "feature_graph_xgboost_validation_predictions.csv",
        index=False
    )

    pd.DataFrame({
        "y_true": y_test,
        "predicted_probability":
            test_probabilities
    }).to_csv(
        RESULTS_DIR
        / "feature_graph_xgboost_test_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "FEATURE + GRAPH XGBOOST COMPLETE"
    )
    print("=" * 70)

    print(
        "\nResults saved to:"
    )

    print(results_file)


if __name__ == "__main__":
    main()
