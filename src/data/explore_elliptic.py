from pathlib import Path
import pandas as pd


# ============================================================
# GraphShield - Elliptic Bitcoin Dataset Exploration
# ============================================================

# Project root
ROOT = Path(__file__).resolve().parents[2]

# Dataset directory
DATA_DIR = ROOT / "data" / "raw" / "elliptic"

# Dataset files
FEATURES_FILE = DATA_DIR / "elliptic_txs_features.csv"
EDGES_FILE = DATA_DIR / "elliptic_txs_edgelist.csv"
CLASSES_FILE = DATA_DIR / "elliptic_txs_classes.csv"


def main():

    print("=" * 70)
    print("GRAPHSHIELD - ELLIPTIC DATASET EXPLORATION")
    print("=" * 70)

    # ========================================================
    # 1. Load class labels
    # ========================================================

    print("\n[1] Loading class labels...")

    classes = pd.read_csv(CLASSES_FILE)

    print("Shape:", classes.shape)

    print("\nColumns:")
    print(classes.columns.tolist())

    print("\nFirst 5 rows:")
    print(classes.head())

    print("\nClass distribution:")
    print(classes["class"].value_counts(dropna=False))

    print("\nClass distribution (%):")
    print(
        (classes["class"].value_counts(normalize=True, dropna=False) * 100)
        .round(2)
    )

    # ========================================================
    # 2. Load graph edges
    # ========================================================

    print("\n" + "=" * 70)
    print("[2] Loading graph edges...")
    print("=" * 70)

    edges = pd.read_csv(EDGES_FILE)

    print("Shape:", edges.shape)

    print("\nColumns:")
    print(edges.columns.tolist())

    print("\nFirst 5 edges:")
    print(edges.head())

    # ========================================================
    # 3. Load transaction features
    # ========================================================

    print("\n" + "=" * 70)
    print("[3] Loading transaction features...")
    print("=" * 70)

    # IMPORTANT:
    # The original Elliptic feature file has NO CSV header.
    # Therefore we must use header=None.
    features = pd.read_csv(
        FEATURES_FILE,
        header=None
    )

    # The dataset contains:
    # 1 transaction ID
    # 1 time step
    # 165 feature columns
    #
    # Total = 167 columns

    features.columns = (
        ["txId", "time_step"]
        + [f"feature_{i}" for i in range(1, 166)]
    )

    print("Shape:", features.shape)

    print("\nNumber of columns:", len(features.columns))

    print("\nFirst 5 rows:")
    print(features.head())

    # ========================================================
    # 4. Basic dataset statistics
    # ========================================================

    print("\n" + "=" * 70)
    print("[4] BASIC DATASET STATISTICS")
    print("=" * 70)

    print("\nTransactions/features:", len(features))

    print("Edges:", len(edges))

    print("Class records:", len(classes))

    print("Feature columns:", len(features.columns))

    print("Actual numerical features:", len(features.columns) - 2)

    # ========================================================
    # 5. Missing values
    # ========================================================

    print("\n" + "=" * 70)
    print("[5] MISSING VALUES")
    print("=" * 70)

    feature_missing = features.isna().sum().sum()
    edge_missing = edges.isna().sum().sum()
    class_missing = classes.isna().sum().sum()

    print("Missing values in features:", feature_missing)

    print("Missing values in edges:", edge_missing)

    print("Missing values in classes:", class_missing)

    # ========================================================
    # 6. Duplicate transaction IDs
    # ========================================================

    print("\n" + "=" * 70)
    print("[6] TRANSACTION ID VALIDATION")
    print("=" * 70)

    unique_feature_ids = features["txId"].nunique()
    duplicate_feature_ids = features["txId"].duplicated().sum()

    unique_class_ids = classes["txId"].nunique()
    duplicate_class_ids = classes["txId"].duplicated().sum()

    print("Feature transaction records:", len(features))

    print("Unique feature transaction IDs:", unique_feature_ids)

    print("Duplicate feature transaction IDs:", duplicate_feature_ids)

    print("\nClass transaction records:", len(classes))

    print("Unique class transaction IDs:", unique_class_ids)

    print("Duplicate class transaction IDs:", duplicate_class_ids)

    # ========================================================
    # 7. Check whether feature IDs and class IDs match
    # ========================================================

    print("\n" + "=" * 70)
    print("[7] FEATURE / CLASS ID CONSISTENCY")
    print("=" * 70)

    feature_ids = set(features["txId"])
    class_ids = set(classes["txId"])

    features_not_in_classes = feature_ids - class_ids
    classes_not_in_features = class_ids - feature_ids

    print(
        "Feature IDs missing from classes:",
        len(features_not_in_classes)
    )

    print(
        "Class IDs missing from features:",
        len(classes_not_in_features)
    )

    # ========================================================
    # 8. Graph node statistics
    # ========================================================

    print("\n" + "=" * 70)
    print("[8] GRAPH NODE STATISTICS")
    print("=" * 70)

    source_col = "txId1"
    target_col = "txId2"

    unique_sources = edges[source_col].nunique()
    unique_targets = edges[target_col].nunique()

    graph_nodes = set(edges[source_col]).union(
        set(edges[target_col])
    )

    print("Unique source nodes:", unique_sources)

    print("Unique target nodes:", unique_targets)

    print("Unique graph nodes:", len(graph_nodes))

    # ========================================================
    # 9. Check graph nodes against feature nodes
    # ========================================================

    print("\n" + "=" * 70)
    print("[9] GRAPH / FEATURE CONSISTENCY")
    print("=" * 70)

    graph_nodes_not_in_features = graph_nodes - feature_ids

    print(
        "Graph nodes missing from features:",
        len(graph_nodes_not_in_features)
    )

    if len(graph_nodes_not_in_features) > 0:
        print(
            "WARNING: Some graph nodes do not have feature vectors."
        )
    else:
        print(
            "All graph nodes have corresponding feature vectors."
        )

    # ========================================================
    # 10. Graph connectivity
    # ========================================================

    print("\n" + "=" * 70)
    print("[10] GRAPH CONNECTIVITY")
    print("=" * 70)

    out_degree = edges[source_col].value_counts()

    in_degree = edges[target_col].value_counts()

    print(
        "Average outgoing edges:",
        round(out_degree.mean(), 4)
    )

    print(
        "Maximum outgoing edges:",
        out_degree.max()
    )

    print(
        "Average incoming edges:",
        round(in_degree.mean(), 4)
    )

    print(
        "Maximum incoming edges:",
        in_degree.max()
    )

    # ========================================================
    # 11. Self-loop detection
    # ========================================================

    print("\n" + "=" * 70)
    print("[11] SELF-LOOP CHECK")
    print("=" * 70)

    self_loops = (
        edges[source_col] == edges[target_col]
    ).sum()

    print("Self-loops:", self_loops)

    # ========================================================
    # 12. Time-step information
    # ========================================================

    print("\n" + "=" * 70)
    print("[12] TIME INFORMATION")
    print("=" * 70)

    print("Time column:", "time_step")

    number_of_time_steps = features["time_step"].nunique()

    print(
        "Number of time steps:",
        number_of_time_steps
    )

    print(
        "Time steps:",
        sorted(features["time_step"].unique())
    )

    print("\nTransactions per time step:")

    print(
        features["time_step"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ========================================================
    # 13. Class distribution by time
    # ========================================================

    print("\n" + "=" * 70)
    print("[13] CLASS DISTRIBUTION BY TIME")
    print("=" * 70)

    # Merge only the information needed for analysis.
    analysis_df = features[
        ["txId", "time_step"]
    ].merge(
        classes,
        on="txId",
        how="left"
    )

    class_by_time = pd.crosstab(
        analysis_df["time_step"],
        analysis_df["class"]
    )

    print(class_by_time.to_string())

    # ========================================================
    # 14. Labeled vs unknown transactions
    # ========================================================

    print("\n" + "=" * 70)
    print("[14] LABELED VS UNKNOWN")
    print("=" * 70)

    labeled_count = (
        classes["class"] != "unknown"
    ).sum()

    unknown_count = (
        classes["class"] == "unknown"
    ).sum()

    print("Labeled transactions:", labeled_count)

    print("Unknown transactions:", unknown_count)

    print(
        "Labeled percentage:",
        round(labeled_count / len(classes) * 100, 2),
        "%"
    )

    print(
        "Unknown percentage:",
        round(unknown_count / len(classes) * 100, 2),
        "%"
    )

    # ========================================================
    # 15. Final summary
    # ========================================================

    print("\n" + "=" * 70)
    print("GRAPHSHIELD EDA SUMMARY")
    print("=" * 70)

    print(f"""
Transactions          : {len(features):,}
Graph edges           : {len(edges):,}
Features per node     : {len(features.columns) - 2}
Time steps            : {number_of_time_steps}
Licit transactions    : {(classes["class"] == "2").sum():,}
Illicit transactions  : {(classes["class"] == "1").sum():,}
Unknown transactions  : {unknown_count:,}
Graph nodes           : {len(graph_nodes):,}
Self-loops            : {self_loops:,}
Missing feature vals  : {feature_missing:,}
Missing edge vals     : {edge_missing:,}
Missing class vals    : {class_missing:,}
""")

    print("=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":
    main()