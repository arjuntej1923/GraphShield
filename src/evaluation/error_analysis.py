from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# GRAPHSHIELD - ERROR ANALYSIS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
PREDICTION_DIR = ROOT / "results" / "predictions"
RESULTS_DIR = ROOT / "results" / "error_analysis"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Thresholds selected previously using validation F1
THRESHOLDS = {
    "logistic_regression": 0.93,
    "random_forest": 0.57,
    "xgboost": 0.63,
    "gcn": 0.53,
    "graphsage": 0.90,
    "gat": 0.88,
}


MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "gcn",
    "graphsage",
    "gat",
]


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 70)
print("GRAPHSHIELD - ERROR ANALYSIS")
print("=" * 70)

TEST_FILE = DATA_DIR / "test_temporal.csv"
GRAPH_FILE = DATA_DIR / "graph_test.pt"

test_df = pd.read_csv(TEST_FILE)

print("\nTest dataset:")
print(f"Rows    : {len(test_df):,}")
print(f"Columns : {len(test_df.columns):,}")


# ============================================================
# LOAD TEST GRAPH
# ============================================================

print("\nLoading test graph...")

graph = torch.load(
    GRAPH_FILE,
    map_location="cpu",
    weights_only=False,
)

graph_tx_ids = np.asarray(graph.tx_ids)

labeled_mask = (
    graph.labeled_mask
    .detach()
    .cpu()
    .numpy()
    .astype(bool)
)

gnn_tx_ids = graph_tx_ids[labeled_mask]

print(f"Graph nodes   : {graph.num_nodes:,}")
print(f"Labeled nodes: {len(gnn_tx_ids):,}")


# ============================================================
# VALIDATE TXID ALIGNMENT
# ============================================================

if "txId" not in test_df.columns:
    raise ValueError(
        "test_temporal.csv does not contain txId."
    )

test_tx_ids = test_df["txId"].to_numpy()

if len(np.unique(test_tx_ids)) != len(test_tx_ids):
    raise ValueError(
        "Duplicate txId values found in test_temporal.csv."
    )

if len(np.unique(gnn_tx_ids)) != len(gnn_tx_ids):
    raise ValueError(
        "Duplicate txId values found in graph_test.pt."
    )


test_id_set = set(test_tx_ids)
gnn_id_set = set(gnn_tx_ids)


if test_id_set != gnn_id_set:
    missing_from_graph = test_id_set - gnn_id_set
    missing_from_test = gnn_id_set - test_id_set

    raise ValueError(
        "GNN/test txId sets do not match.\n"
        f"Missing from graph: {len(missing_from_graph)}\n"
        f"Missing from test : {len(missing_from_test)}"
    )


print(
    "[PASS] test_temporal.csv and graph_test.pt "
    "contain the same 9,973 labeled transaction IDs."
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

prediction_data = {}


for model in MODELS:

    path = PREDICTION_DIR / f"{model}_test.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "y_true",
        "predicted_probability",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{path} must contain "
            f"{required_columns}"
        )

    if len(df) != len(test_df):
        raise ValueError(
            f"{model}: prediction count {len(df):,} "
            f"does not match test count {len(test_df):,}"
        )

    prediction_data[model] = df[
        "predicted_probability"
    ].to_numpy()

    print(
        f"[PASS] Loaded {model}: "
        f"{len(df):,} predictions"
    )


# ============================================================
# BUILD MASTER DATAFRAME
# ============================================================

master = test_df.copy()

# Keep only useful metadata first
master = master[
    [
        "txId",
        "time_step",
        "class",
    ]
].copy()


# Binary label
master["y_true"] = (
    master["class"].astype(str) == "1"
).astype(int)


# ============================================================
# ADD TRADITIONAL MODEL PREDICTIONS
# ============================================================

traditional_models = [
    "logistic_regression",
    "random_forest",
    "xgboost",
]


for model in traditional_models:

    master[f"{model}_probability"] = (
        prediction_data[model]
    )


# ============================================================
# ADD GNN PREDICTIONS
#
# GNN predictions are ordered according to:
#
# graph.tx_ids[graph.labeled_mask]
#
# Therefore we attach them by txId.
# ============================================================

gnn_models = [
    "gcn",
    "graphsage",
    "gat",
]


gnn_lookup = pd.DataFrame({
    "txId": gnn_tx_ids,
})


for model in gnn_models:

    probabilities = prediction_data[model]

    gnn_lookup[f"{model}_probability"] = (
        probabilities
    )


master = master.merge(
    gnn_lookup,
    on="txId",
    how="left",
    validate="one_to_one",
)


# ============================================================
# VERIFY GNN ALIGNMENT
# ============================================================

for model in gnn_models:

    missing = master[
        f"{model}_probability"
    ].isna().sum()

    if missing > 0:
        raise ValueError(
            f"{model}: {missing} missing predictions "
            "after txId alignment."
        )


print(
    "\n[PASS] All GNN predictions aligned by txId."
)


# ============================================================
# CALCULATE MODEL METRICS
# ============================================================

summary_rows = []


for model in MODELS:

    probabilities = master[
        f"{model}_probability"
    ].to_numpy()

    y_true = master["y_true"].to_numpy()

    threshold = THRESHOLDS[model]

    predictions = (
        probabilities >= threshold
    ).astype(int)


    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()


    summary_rows.append({
        "model": model,
        "threshold": threshold,

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),

        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),

        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,

        "total_errors": fp + fn,
    })


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    RESULTS_DIR / "model_error_summary.csv",
    index=False,
)


# ============================================================
# DETAILED ERROR TYPES
# ============================================================

for model in MODELS:

    probabilities = master[
        f"{model}_probability"
    ].to_numpy()

    threshold = THRESHOLDS[model]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    y_true = master["y_true"].to_numpy()


    error_type = np.full(
        len(y_true),
        "correct",
        dtype=object,
    )


    error_type[
        (predictions == 1)
        & (y_true == 0)
    ] = "false_positive"


    error_type[
        (predictions == 0)
        & (y_true == 1)
    ] = "false_negative"


    master[
        f"{model}_prediction"
    ] = predictions

    master[
        f"{model}_error_type"
    ] = error_type


# Save complete aligned dataset
master.to_csv(
    RESULTS_DIR / "detailed_test_errors.csv",
    index=False,
)


# ============================================================
# ERROR COUNTS
# ============================================================

error_rows = []


for model in MODELS:

    error_type = master[
        f"{model}_error_type"
    ]

    error_rows.append({
        "model": model,

        "true_negatives": int(
            (error_type == "correct")
            .sum()
        ),

        "false_positives": int(
            (
                error_type
                == "false_positive"
            ).sum()
        ),

        "false_negatives": int(
            (
                error_type
                == "false_negative"
            ).sum()
        ),

        "total_errors": int(
            (
                error_type
                != "correct"
            ).sum()
        ),
    })


error_counts_df = pd.DataFrame(
    error_rows
)


# Correct TN/TP counts separately
for model in MODELS:

    y = master["y_true"].to_numpy()
    p = master[
        f"{model}_prediction"
    ].to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        y,
        p,
        labels=[0, 1],
    ).ravel()

    error_counts_df.loc[
        error_counts_df["model"] == model,
        "true_negatives"
    ] = tn

    error_counts_df.loc[
        error_counts_df["model"] == model,
        "true_positives"
    ] = tp


error_counts_df.to_csv(
    RESULTS_DIR / "error_counts.csv",
    index=False,
)


# ============================================================
# TEMPORAL ERROR ANALYSIS
# ============================================================

temporal_rows = []


for model in MODELS:

    for time_step, group in master.groupby(
        "time_step"
    ):

        y = group["y_true"].to_numpy()

        p = group[
            f"{model}_prediction"
        ].to_numpy()


        tp = int(
            ((p == 1) & (y == 1)).sum()
        )

        fp = int(
            ((p == 1) & (y == 0)).sum()
        )

        fn = int(
            ((p == 0) & (y == 1)).sum()
        )

        positives = int(
            (y == 1).sum()
        )


        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )


        temporal_rows.append({
            "model": model,
            "time_step": time_step,
            "samples": len(group),
            "illicit_transactions": positives,
            "precision": precision,
            "recall": recall,
            "false_positives": fp,
            "false_negatives": fn,
        })


temporal_df = pd.DataFrame(
    temporal_rows
)


temporal_df.to_csv(
    RESULTS_DIR / "errors_by_time_step.csv",
    index=False,
)


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

confidence_rows = []


for model in MODELS:

    probabilities = master[
        f"{model}_probability"
    ].to_numpy()

    predictions = master[
        f"{model}_prediction"
    ].to_numpy()

    y_true = master[
        "y_true"
    ].to_numpy()


    correct = predictions == y_true
    incorrect = ~correct

    false_positive = (
        (predictions == 1)
        & (y_true == 0)
    )

    false_negative = (
        (predictions == 0)
        & (y_true == 1)
    )


    confidence_rows.append({

        "model": model,

        "correct_count":
            int(correct.sum()),

        "incorrect_count":
            int(incorrect.sum()),

        "mean_probability_correct":
            float(
                probabilities[correct].mean()
            ),

        "mean_probability_incorrect":
            float(
                probabilities[incorrect].mean()
            ),

        "mean_probability_true_illicit":
            float(
                probabilities[
                    y_true == 1
                ].mean()
            ),

        "mean_probability_true_licit":
            float(
                probabilities[
                    y_true == 0
                ].mean()
            ),

        "mean_probability_false_positive":
            float(
                probabilities[
                    false_positive
                ].mean()
            )
            if false_positive.any()
            else 0.0,

        "mean_probability_false_negative":
            float(
                probabilities[
                    false_negative
                ].mean()
            )
            if false_negative.any()
            else 0.0,
    })


confidence_df = pd.DataFrame(
    confidence_rows
)


confidence_df.to_csv(
    RESULTS_DIR / "confidence_analysis.csv",
    index=False,
)


# ============================================================
# MODEL DISAGREEMENT
# ============================================================

prediction_matrix = pd.DataFrame({
    "y_true": master["y_true"].to_numpy()
})


for model in MODELS:

    prediction_matrix[model] = master[
        f"{model}_prediction"
    ].to_numpy()


prediction_matrix[
    "models_predicting_illicit"
] = prediction_matrix[
    MODELS
].sum(axis=1)


prediction_matrix[
    "all_models_agree"
] = (
    prediction_matrix[
        MODELS
    ].nunique(axis=1) == 1
)


prediction_matrix[
    "correct_model_count"
] = (
    prediction_matrix[
        MODELS
    ]
    .eq(
        prediction_matrix["y_true"],
        axis=0,
    )
    .sum(axis=1)
)


# Distribution of model votes
disagreement_df = (
    prediction_matrix[
        "models_predicting_illicit"
    ]
    .value_counts()
    .sort_index()
    .reset_index()
)


disagreement_df.columns = [
    "models_predicting_illicit",
    "transactions",
]


disagreement_df.to_csv(
    RESULTS_DIR
    / "model_disagreement_distribution.csv",
    index=False,
)


# ============================================================
# AGREEMENT VS ACTUAL LABEL
# ============================================================

agreement_rows = []


for count in range(len(MODELS) + 1):

    subset = prediction_matrix[
        prediction_matrix[
            "models_predicting_illicit"
        ] == count
    ]


    if len(subset) == 0:
        continue


    agreement_rows.append({
        "models_predicting_illicit": count,
        "transactions": len(subset),
        "actual_illicit_rate":
            float(
                subset["y_true"].mean()
            ),
    })


agreement_df = pd.DataFrame(
    agreement_rows
)


agreement_df.to_csv(
    RESULTS_DIR
    / "prediction_agreement_vs_truth.csv",
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL ERROR SUMMARY")
print("=" * 70)

print(
    summary_df[
        [
            "model",
            "threshold",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "false_positives",
            "false_negatives",
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 70)
print("ERROR COUNTS")
print("=" * 70)

print(
    error_counts_df.to_string(index=False)
)


print("\n" + "=" * 70)
print("CONFIDENCE ANALYSIS")
print("=" * 70)

print(
    confidence_df.to_string(index=False)
)


print("\n" + "=" * 70)
print("MODEL DISAGREEMENT")
print("=" * 70)

print(
    disagreement_df.to_string(index=False)
)


print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

for path in sorted(
    RESULTS_DIR.glob("*.csv")
):
    print(path)


print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)
