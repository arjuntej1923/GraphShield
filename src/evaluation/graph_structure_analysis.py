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
# GRAPH STRUCTURE ANALYSIS
# ============================================================

def analyze_graph(graph, split_name):

    print("\n" + "=" * 70)
    print(f"GRAPH STRUCTURE ANALYSIS - {split_name.upper()}")
    print("=" * 70)

    edge_index = graph.edge_index.cpu()

    num_nodes = graph.num_nodes

    source_nodes = edge_index[0].numpy()
    target_nodes = edge_index[1].numpy()

    # --------------------------------------------------------
    # Degree calculations
    # --------------------------------------------------------

    out_degree = np.bincount(
        source_nodes,
        minlength=num_nodes
    )

    in_degree = np.bincount(
        target_nodes,
        minlength=num_nodes
    )

    total_degree = (
        in_degree + out_degree
    )

    # --------------------------------------------------------
    # Node labels
    # --------------------------------------------------------

    y = graph.y.cpu().numpy()

    labeled_mask = graph.labeled_mask.cpu().numpy()

    illicit_mask = (
        labeled_mask
        & (y == 1)
    )

    licit_mask = (
        labeled_mask
        & (y == 0)
    )

    unknown_mask = (
        ~labeled_mask
    )

    # --------------------------------------------------------
    # Basic graph statistics
    # --------------------------------------------------------

    print(
        f"\nNodes              : {num_nodes:,}"
    )

    print(
        f"Edges              : {graph.num_edges:,}"
    )

    print(
        f"Average in-degree  : {in_degree.mean():.4f}"
    )

    print(
        f"Average out-degree : {out_degree.mean():.4f}"
    )

    print(
        f"Average degree     : {total_degree.mean():.4f}"
    )

    print(
        f"Maximum degree     : {total_degree.max():,}"
    )

    print(
        f"Isolated nodes     : "
        f"{np.sum(total_degree == 0):,}"
    )

    print(
        f"Labeled nodes      : "
        f"{labeled_mask.sum():,}"
    )

    print(
        f"Illicit nodes      : "
        f"{illicit_mask.sum():,}"
    )

    print(
        f"Licit nodes        : "
        f"{licit_mask.sum():,}"
    )

    print(
        f"Unknown nodes      : "
        f"{unknown_mask.sum():,}"
    )

    # --------------------------------------------------------
    # Degree statistics by class
    # --------------------------------------------------------

    def degree_statistics(mask):

        values = total_degree[mask]

        if len(values) == 0:
            return {
                "count": 0,
                "mean_degree": np.nan,
                "median_degree": np.nan,
                "std_degree": np.nan,
                "min_degree": np.nan,
                "max_degree": np.nan
            }

        return {
            "count": len(values),
            "mean_degree": values.mean(),
            "median_degree": np.median(values),
            "std_degree": values.std(),
            "min_degree": values.min(),
            "max_degree": values.max()
        }

    illicit_stats = degree_statistics(
        illicit_mask
    )

    licit_stats = degree_statistics(
        licit_mask
    )

    unknown_stats = degree_statistics(
        unknown_mask
    )

    degree_table = pd.DataFrame(
        [
            {
                "class": "Illicit",
                **illicit_stats
            },
            {
                "class": "Licit",
                **licit_stats
            },
            {
                "class": "Unknown",
                **unknown_stats
            }
        ]
    )

    print("\nDegree statistics:")
    print(
        degree_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Transaction-level table
    # --------------------------------------------------------

    tx_ids = graph.tx_ids

    if torch.is_tensor(tx_ids):
        tx_ids = tx_ids.cpu().numpy()

    time_steps = graph.time_steps

    if torch.is_tensor(time_steps):
        time_steps = time_steps.cpu().numpy()

    records = pd.DataFrame(
        {
            "txId": tx_ids,
            "time_step": time_steps,
            "label": y,
            "labeled": labeled_mask,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "total_degree": total_degree
        }
    )

    records["class_name"] = np.select(
        [
            records["label"] == 1,
            records["label"] == 0,
            records["label"] == -1
        ],
        [
            "illicit",
            "licit",
            "unknown"
        ],
        default="unknown"
    )

    # --------------------------------------------------------
    # Save transaction-level graph statistics
    # --------------------------------------------------------

    output_file = (
        RESULTS_DIR
        / f"graph_structure_{split_name}.csv"
    )

    records.to_csv(
        output_file,
        index=False
    )

    print(
        f"\n[PASS] Saved transaction-level statistics:"
    )

    print(output_file)

    # --------------------------------------------------------
    # Save degree summary
    # --------------------------------------------------------

    summary_file = (
        RESULTS_DIR
        / f"graph_degree_summary_{split_name}.csv"
    )

    degree_table.to_csv(
        summary_file,
        index=False
    )

    print(
        f"[PASS] Saved degree summary:"
    )

    print(summary_file)

    # --------------------------------------------------------
    # Degree distribution plot
    # --------------------------------------------------------

    labeled_records = records[
        records["labeled"] == True
    ]

    illicit_degrees = labeled_records.loc[
        labeled_records["label"] == 1,
        "total_degree"
    ]

    licit_degrees = labeled_records.loc[
        labeled_records["label"] == 0,
        "total_degree"
    ]

    plt.figure(
        figsize=(10, 7)
    )

    plt.hist(
        np.log1p(licit_degrees),
        bins=50,
        alpha=0.6,
        label="Licit"
    )

    plt.hist(
        np.log1p(illicit_degrees),
        bins=50,
        alpha=0.6,
        label="Illicit"
    )

    plt.xlabel(
        "log(1 + total degree)",
        fontsize=12
    )

    plt.ylabel(
        "Number of transactions",
        fontsize=12
    )

    plt.title(
        f"Transaction Degree Distribution - "
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
        / f"degree_distribution_{split_name}.png"
    )

    plt.savefig(
        plot_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[PASS] Saved degree distribution:"
    )

    print(plot_file)

    return records


# ============================================================
# CLASS DEGREE COMPARISON
# ============================================================

def create_class_comparison(
    train_records,
    val_records,
    test_records
):

    print("\n" + "=" * 70)
    print("CLASS-LEVEL GRAPH STRUCTURE COMPARISON")
    print("=" * 70)

    datasets = [
        ("train", train_records),
        ("validation", val_records),
        ("test", test_records)
    ]

    rows = []

    for split_name, records in datasets:

        labeled = records[
            records["labeled"] == True
        ]

        for class_name, class_value in [
            ("licit", 0),
            ("illicit", 1)
        ]:

            subset = labeled[
                labeled["label"] == class_value
            ]

            rows.append(
                {
                    "split": split_name,
                    "class": class_name,
                    "count": len(subset),
                    "mean_in_degree":
                        subset["in_degree"].mean(),
                    "mean_out_degree":
                        subset["out_degree"].mean(),
                    "mean_total_degree":
                        subset["total_degree"].mean(),
                    "median_total_degree":
                        subset["total_degree"].median()
                }
            )

    comparison = pd.DataFrame(rows)

    print(
        "\n"
        + comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    output_file = (
        RESULTS_DIR
        / "graph_class_degree_comparison.csv"
    )

    comparison.to_csv(
        output_file,
        index=False
    )

    print(
        f"\n[PASS] Saved class comparison:"
    )

    print(output_file)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("GRAPHSHIELD - GRAPH STRUCTURE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nLoading training graph...")

    train_graph = load_graph(
        TRAIN_GRAPH_FILE
    )

    train_records = analyze_graph(
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

    val_records = analyze_graph(
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

    test_records = analyze_graph(
        test_graph,
        "test"
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    create_class_comparison(
        train_records,
        val_records,
        test_records
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GRAPH STRUCTURE ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")

    expected_files = [
        "graph_structure_train.csv",
        "graph_structure_validation.csv",
        "graph_structure_test.csv",
        "graph_degree_summary_train.csv",
        "graph_degree_summary_validation.csv",
        "graph_degree_summary_test.csv",
        "graph_class_degree_comparison.csv"
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
