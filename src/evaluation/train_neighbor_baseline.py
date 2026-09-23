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
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "neighbor_baseline"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


GRAPHS = {
    "train": DATA_DIR / "graph_train.pt",
    "validation": DATA_DIR / "graph_validation.pt",
    "test": DATA_DIR / "graph_test.pt"
}


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
# CREATE NEIGHBOR FEATURES
# ============================================================

def create_neighbor_features(graph):

    edge_index = graph.edge_index.cpu()

    source = edge_index[0].numpy()
    target = edge_index[1].numpy()

    num_nodes = graph.num_nodes

    y = graph.y.cpu().numpy()

    labeled_mask = (
        graph.labeled_mask.cpu().numpy()
    )

    # --------------------------------------------------------
    # Build undirected neighborhood
    # --------------------------------------------------------

    neighbors = [
        set()
        for _ in range(num_nodes)
    ]

    for s, t in zip(source, target):

        neighbors[s].add(t)
        neighbors[t].add(s)

    # --------------------------------------------------------
    # Generate features
    # --------------------------------------------------------

    records = []

    for node in np.flatnonzero(
        labeled_mask
    ):

        node_neighbors = neighbors[node]

        total_neighbors = len(
            node_neighbors
        )

        if total_neighbors == 0:

            illicit_neighbors = 0
            licit_neighbors = 0
            unknown_neighbors = 0

        else:

            neighbor_array = np.fromiter(
                node_neighbors,
                dtype=np.int64
            )

            neighbor_labeled = (
                labeled_mask[
                    neighbor_array
                ]
            )

            neighbor_labels = (
                y[
                    neighbor_array
                ]
            )

            illicit_neighbors = np.sum(
                neighbor_labeled
                & (neighbor_labels == 1)
            )

            licit_neighbors = np.sum(
                neighbor_labeled
                & (neighbor_labels == 0)
            )

            unknown_neighbors = np.sum(
                ~neighbor_labeled
            )

        # ----------------------------------------------------
        # Ratios
        # ----------------------------------------------------

        labeled_neighbors = (
            illicit_neighbors
            + licit_neighbors
        )

        if total_neighbors > 0:

            illicit_ratio_all = (
                illicit_neighbors
                / total_neighbors
            )

            licit_ratio_all = (
                licit_neighbors
                / total_neighbors
            )

            unknown_ratio = (
                unknown_neighbors
                / total_neighbors
            )

        else:

            illicit_ratio_all = 0.0
            licit_ratio_all = 0.0
            unknown_ratio = 0.0

        if labeled_neighbors > 0:

            illicit_ratio_labeled = (
                illicit_neighbors
                / labeled_neighbors
            )

        else:

            illicit_ratio_labeled = 0.0

        records.append(
            {
                "node_index": node,
                "total_neighbors":
                    total_neighbors,
                "labeled_neighbors":
                    labeled_neighbors,
                "illicit_neighbors":
                    int(illicit_neighbors),
                "licit_neighbors":
                    int(licit_neighbors),
                "unknown_neighbors":
                    int(unknown_neighbors),
                "illicit_ratio_all":
                    illicit_ratio_all,
                "licit_ratio_all":
                    licit_ratio_all,
                "unknown_ratio":
                    unknown_ratio,
                "illicit_ratio_labeled":
                    illicit_ratio_labeled,
                "y_true":
                    int(y[node])
            }
        )

    return pd.DataFrame(records)


# ============================================================
# LABEL PROPAGATION SCORE
# ============================================================

def calculate_scores(df):

    # --------------------------------------------------------
    # Main relational score
    #
    # Fraction of labeled neighbors that are illicit.
    # --------------------------------------------------------

    scores = (
        df["illicit_ratio_labeled"]
        .to_numpy()
    )

    return scores


# ============================================================
# FIND BEST VALIDATION THRESHOLD
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

    best_threshold = 0.5
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
# EVALUATION
# ============================================================

def evaluate(
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

    return {
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
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("GRAPHSHIELD - RELATIONAL LABEL-PROPAGATION BASELINE")
    print("=" * 70)

    graphs = {}

    for split in [
        "train",
        "validation",
        "test"
    ]:

        print(
            f"\nLoading {split} graph..."
        )

        graph = load_graph(
            GRAPHS[split]
        )

        graphs[split] = (
            create_neighbor_features(
                graph
            )
        )

        print(
            f"{split.capitalize()} labeled nodes: "
            f"{len(graphs[split]):,}"
        )

    # ========================================================
    # CALCULATE RELATIONAL SCORES
    # ========================================================

    for split in graphs:

        graphs[split]["score"] = (
            calculate_scores(
                graphs[split]
            )
        )

    # ========================================================
    # VALIDATION THRESHOLD
    # ========================================================

    y_val = (
        graphs["validation"]["y_true"]
        .to_numpy()
    )

    val_scores = (
        graphs["validation"]["score"]
        .to_numpy()
    )

    best_threshold, threshold_table = (
        find_best_threshold(
            y_val,
            val_scores
        )
    )

    threshold_table.to_csv(
        RESULTS_DIR
        / "neighbor_thresholds.csv",
        index=False
    )

    print(
        f"\nBest validation threshold: "
        f"{best_threshold:.2f}"
    )

    # ========================================================
    # EVALUATE ALL SPLITS
    # ========================================================

    results = []

    for split in [
        "validation",
        "test"
    ]:

        df = graphs[split]

        y_true = (
            df["y_true"]
            .to_numpy()
        )

        scores = (
            df["score"]
            .to_numpy()
        )

        metrics = evaluate(
            y_true,
            scores,
            best_threshold
        )

        metrics["model"] = (
            "Neighbor Label Propagation"
        )

        metrics["split"] = split

        results.append(
            metrics
        )

        print(
            f"\n{split.upper()}"
        )

        print(
            f"Precision : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1        : "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC   : "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC    : "
            f"{metrics['pr_auc']:.4f}"
        )

        print(
            "\nConfusion Matrix:"
        )

        print(
            np.array(
                [
                    [
                        metrics["tn"],
                        metrics["fp"]
                    ],
                    [
                        metrics["fn"],
                        metrics["tp"]
                    ]
                ]
            )
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
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

    results_file = (
        RESULTS_DIR
        / "neighbor_baseline_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    for split in [
        "validation",
        "test"
    ]:

        prediction_df = graphs[split][
            [
                "node_index",
                "y_true",
                "score"
            ]
        ].copy()

        prediction_df[
            "predicted_label"
        ] = (
            prediction_df["score"]
            >= best_threshold
        ).astype(int)

        prediction_df.to_csv(
            RESULTS_DIR
            / f"{split}_predictions.csv",
            index=False
        )

    # ========================================================
    # PRINT FINAL RESULT
    # ========================================================

    print("\n" + "=" * 70)
    print("RELATIONAL BASELINE COMPLETE")
    print("=" * 70)

    print(
        "\nFinal comparison:"
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    print(
        "\nResults saved to:"
    )

    print(results_file)


if __name__ == "__main__":
    main()
