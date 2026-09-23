from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

TRAIN_GRAPH_FILE = DATA_DIR / "graph_train.pt"
VAL_GRAPH_FILE = DATA_DIR / "graph_validation.pt"
TEST_GRAPH_FILE = DATA_DIR / "graph_test.pt"

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
# NEIGHBOR ANALYSIS
# ============================================================

def analyze_neighbors(graph, split_name):

    print("\n" + "=" * 70)
    print(
        f"NEIGHBOR STRUCTURE ANALYSIS - "
        f"{split_name.upper()}"
    )
    print("=" * 70)

    edge_index = graph.edge_index.cpu()

    source_nodes = edge_index[0].numpy()
    target_nodes = edge_index[1].numpy()

    num_nodes = graph.num_nodes

    y = graph.y.cpu().numpy()

    labeled_mask = (
        graph.labeled_mask.cpu().numpy()
    )

    # --------------------------------------------------------
    # Node labels
    # --------------------------------------------------------

    illicit_mask = (
        labeled_mask & (y == 1)
    )

    licit_mask = (
        labeled_mask & (y == 0)
    )

    unknown_mask = (
        ~labeled_mask
    )

    # --------------------------------------------------------
    # Neighbor lists
    #
    # The graph edges are directed. For neighborhood analysis
    # we treat both incoming and outgoing connections as
    # neighbors.
    # --------------------------------------------------------

    neighbors = [
        set()
        for _ in range(num_nodes)
    ]

    for source, target in zip(
        source_nodes,
        target_nodes
    ):

        neighbors[source].add(target)
        neighbors[target].add(source)

    # --------------------------------------------------------
    # Calculate neighbor statistics
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

            neighbor_labels = y[
                neighbor_array
            ]

            neighbor_labeled = labeled_mask[
                neighbor_array
            ]

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

        if total_neighbors > 0:

            illicit_ratio = (
                illicit_neighbors
                / total_neighbors
            )

            licit_ratio = (
                licit_neighbors
                / total_neighbors
            )

            unknown_ratio = (
                unknown_neighbors
                / total_neighbors
            )

        else:

            illicit_ratio = 0.0
            licit_ratio = 0.0
            unknown_ratio = 0.0

        records.append(
            {
                "node_index": node,
                "label": int(y[node]),
                "total_neighbors":
                    total_neighbors,
                "illicit_neighbors":
                    int(illicit_neighbors),
                "licit_neighbors":
                    int(licit_neighbors),
                "unknown_neighbors":
                    int(unknown_neighbors),
                "illicit_neighbor_ratio":
                    illicit_ratio,
                "licit_neighbor_ratio":
                    licit_ratio,
                "unknown_neighbor_ratio":
                    unknown_ratio
            }
        )

    records = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # Class names
    # --------------------------------------------------------

    records["class_name"] = np.where(
        records["label"] == 1,
        "illicit",
        "licit"
    )

    # --------------------------------------------------------
    # Transaction IDs
    # --------------------------------------------------------

    tx_ids = graph.tx_ids

    if torch.is_tensor(tx_ids):

        tx_ids = (
            tx_ids.cpu().numpy()
        )

    records["txId"] = (
        records["node_index"]
        .map(
            lambda index:
            tx_ids[index]
        )
    )

    # --------------------------------------------------------
    # Time steps
    # --------------------------------------------------------

    time_steps = graph.time_steps

    if torch.is_tensor(time_steps):

        time_steps = (
            time_steps.cpu().numpy()
        )

    records["time_step"] = (
        records["node_index"]
        .map(
            lambda index:
            time_steps[index]
        )
    )

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    records = records[
        [
            "txId",
            "time_step",
            "node_index",
            "label",
            "class_name",
            "total_neighbors",
            "illicit_neighbors",
            "licit_neighbors",
            "unknown_neighbors",
            "illicit_neighbor_ratio",
            "licit_neighbor_ratio",
            "unknown_neighbor_ratio"
        ]
    ]

    # ========================================================
    # PRINT BASIC STATISTICS
    # ========================================================

    print(
        f"\nLabeled transactions : "
        f"{len(records):,}"
    )

    print(
        f"Illicit transactions : "
        f"{np.sum(records['label'] == 1):,}"
    )

    print(
        f"Licit transactions   : "
        f"{np.sum(records['label'] == 0):,}"
    )

    # --------------------------------------------------------
    # Summary by class
    # --------------------------------------------------------

    summary = (
        records
        .groupby("class_name")
        [
            [
                "total_neighbors",
                "illicit_neighbors",
                "licit_neighbors",
                "unknown_neighbors",
                "illicit_neighbor_ratio",
                "licit_neighbor_ratio",
                "unknown_neighbor_ratio"
            ]
        ]
        .agg(
            [
                "mean",
                "median",
                "std"
            ]
        )
    )

    print("\nNeighbor statistics by class:")

    print(
        summary.to_string(
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Save transaction-level data
    # --------------------------------------------------------

    output_file = (
        RESULTS_DIR
        / f"neighbor_structure_{split_name}.csv"
    )

    records.to_csv(
        output_file,
        index=False
    )

    print(
        f"\n[PASS] Saved transaction-level "
        f"neighbor statistics:"
    )

    print(output_file)

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_file = (
        RESULTS_DIR
        / f"neighbor_summary_{split_name}.csv"
    )

    summary.to_csv(
        summary_file
    )

    print(
        f"[PASS] Saved neighbor summary:"
    )

    print(summary_file)

    # ========================================================
    # PLOT 1: ILLICIT NEIGHBOR RATIO
    # ========================================================

    plt.figure(
        figsize=(10, 7)
    )

    illicit_values = records.loc[
        records["class_name"] == "illicit",
        "illicit_neighbor_ratio"
    ]

    licit_values = records.loc[
        records["class_name"] == "licit",
        "illicit_neighbor_ratio"
    ]

    plt.hist(
        licit_values,
        bins=50,
        alpha=0.6,
        label="Licit"
    )

    plt.hist(
        illicit_values,
        bins=50,
        alpha=0.6,
        label="Illicit"
    )

    plt.xlabel(
        "Illicit Neighbor Ratio",
        fontsize=12
    )

    plt.ylabel(
        "Number of Transactions",
        fontsize=12
    )

    plt.title(
        f"Illicit Neighbor Ratio - "
        f"{split_name.capitalize()}",
        fontsize=15,
        fontweight="bold"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    plot_file = (
        PLOTS_DIR
        / f"illicit_neighbor_ratio_{split_name}.png"
    )

    plt.savefig(
        plot_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[PASS] Saved illicit-neighbor plot:"
    )

    print(plot_file)

    # ========================================================
    # PLOT 2: TOTAL NEIGHBORS
    # ========================================================

    plt.figure(
        figsize=(10, 7)
    )

    licit_neighbors = records.loc[
        records["class_name"] == "licit",
        "total_neighbors"
    ]

    illicit_neighbors = records.loc[
        records["class_name"] == "illicit",
        "total_neighbors"
    ]

    plt.hist(
        np.log1p(licit_neighbors),
        bins=50,
        alpha=0.6,
        label="Licit"
    )

    plt.hist(
        np.log1p(illicit_neighbors),
        bins=50,
        alpha=0.6,
        label="Illicit"
    )

    plt.xlabel(
        "log(1 + Number of Neighbors)",
        fontsize=12
    )

    plt.ylabel(
        "Number of Transactions",
        fontsize=12
    )

    plt.title(
        f"Neighbor Count Distribution - "
        f"{split_name.capitalize()}",
        fontsize=15,
        fontweight="bold"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    plot_file = (
        PLOTS_DIR
        / f"neighbor_count_distribution_{split_name}.png"
    )

    plt.savefig(
        plot_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[PASS] Saved neighbor-count plot:"
    )

    print(plot_file)

    return records


# ============================================================
# CROSS-TEMPORAL SUMMARY
# ============================================================

def create_cross_temporal_summary(
    train_records,
    val_records,
    test_records
):

    print("\n" + "=" * 70)
    print("CROSS-TEMPORAL NEIGHBOR ANALYSIS")
    print("=" * 70)

    datasets = [
        ("train", train_records),
        ("validation", val_records),
        ("test", test_records)
    ]

    rows = []

    for split_name, records in datasets:

        for class_name in [
            "licit",
            "illicit"
        ]:

            subset = records[
                records["class_name"] == class_name
            ]

            rows.append(
                {
                    "split": split_name,
                    "class": class_name,
                    "count": len(subset),
                    "mean_neighbors":
                        subset[
                            "total_neighbors"
                        ].mean(),
                    "median_neighbors":
                        subset[
                            "total_neighbors"
                        ].median(),
                    "mean_illicit_neighbors":
                        subset[
                            "illicit_neighbors"
                        ].mean(),
                    "mean_licit_neighbors":
                        subset[
                            "licit_neighbors"
                        ].mean(),
                    "mean_unknown_neighbors":
                        subset[
                            "unknown_neighbors"
                        ].mean(),
                    "mean_illicit_neighbor_ratio":
                        subset[
                            "illicit_neighbor_ratio"
                        ].mean(),
                    "mean_licit_neighbor_ratio":
                        subset[
                            "licit_neighbor_ratio"
                        ].mean(),
                    "mean_unknown_neighbor_ratio":
                        subset[
                            "unknown_neighbor_ratio"
                        ].mean()
                }
            )

    comparison = pd.DataFrame(
        rows
    )

    print(
        "\n"
        + comparison.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    output_file = (
        RESULTS_DIR
        / "neighbor_cross_temporal_comparison.csv"
    )

    comparison.to_csv(
        output_file,
        index=False
    )

    print(
        f"\n[PASS] Saved cross-temporal summary:"
    )

    print(output_file)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("GRAPHSHIELD - NEIGHBOR STRUCTURE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nLoading training graph...")

    train_graph = load_graph(
        TRAIN_GRAPH_FILE
    )

    train_records = analyze_neighbors(
        train_graph,
        "train"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\nLoading validation graph...")

    val_graph = load_graph(
        VAL_GRAPH_FILE
    )

    val_records = analyze_neighbors(
        val_graph,
        "validation"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print("\nLoading test graph...")

    test_graph = load_graph(
        TEST_GRAPH_FILE
    )

    test_records = analyze_neighbors(
        test_graph,
        "test"
    )

    # --------------------------------------------------------
    # Cross-temporal analysis
    # --------------------------------------------------------

    create_cross_temporal_summary(
        train_records,
        val_records,
        test_records
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("NEIGHBOR STRUCTURE ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")

    expected_files = [
        "neighbor_structure_train.csv",
        "neighbor_structure_validation.csv",
        "neighbor_structure_test.csv",
        "neighbor_summary_train.csv",
        "neighbor_summary_validation.csv",
        "neighbor_summary_test.csv",
        "neighbor_cross_temporal_comparison.csv"
    ]

    for filename in expected_files:

        path = RESULTS_DIR / filename

        if path.exists():
            print(f"[PASS] {filename}")
        else:
            print(f"[FAIL] {filename}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
