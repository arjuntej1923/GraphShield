from pathlib import Path
import pandas as pd


# ============================================================
# GraphShield - Dataset Validation
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw" / "elliptic"

FEATURES_FILE = DATA_DIR / "elliptic_txs_features.csv"
EDGES_FILE = DATA_DIR / "elliptic_txs_edgelist.csv"
CLASSES_FILE = DATA_DIR / "elliptic_txs_classes.csv"


def main():

    print("=" * 70)
    print("GRAPHSHIELD - DATASET VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading classes...")

    classes = pd.read_csv(CLASSES_FILE)

    print("Loading edges...")

    edges = pd.read_csv(EDGES_FILE)

    print("Loading features...")

    features = pd.read_csv(
        FEATURES_FILE,
        header=None
    )

    features.columns = (
        ["txId", "time_step"]
        + [f"feature_{i}" for i in range(1, 166)]
    )

    # --------------------------------------------------------
    # 1. Basic sizes
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[1] BASIC DATASET SIZES")
    print("=" * 70)

    print(f"Features : {len(features):,}")
    print(f"Classes  : {len(classes):,}")
    print(f"Edges    : {len(edges):,}")

    # --------------------------------------------------------
    # 2. Duplicate IDs
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[2] DUPLICATE CHECK")
    print("=" * 70)

    duplicate_features = features["txId"].duplicated().sum()
    duplicate_classes = classes["txId"].duplicated().sum()

    print("Duplicate feature IDs:", duplicate_features)
    print("Duplicate class IDs:", duplicate_classes)

    # --------------------------------------------------------
    # 3. Feature / class consistency
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[3] FEATURE / CLASS CONSISTENCY")
    print("=" * 70)

    feature_ids = set(features["txId"])
    class_ids = set(classes["txId"])

    missing_from_classes = feature_ids - class_ids
    missing_from_features = class_ids - feature_ids

    print(
        "Feature IDs missing from classes:",
        len(missing_from_classes)
    )

    print(
        "Class IDs missing from features:",
        len(missing_from_features)
    )

    # --------------------------------------------------------
    # 4. Graph endpoint consistency
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[4] GRAPH ENDPOINT VALIDATION")
    print("=" * 70)

    graph_source_ids = set(edges["txId1"])
    graph_target_ids = set(edges["txId2"])

    graph_nodes = graph_source_ids.union(graph_target_ids)

    missing_graph_features = graph_nodes - feature_ids

    print(
        "Unique graph nodes:",
        len(graph_nodes)
    )

    print(
        "Graph nodes without features:",
        len(missing_graph_features)
    )

    # --------------------------------------------------------
    # 5. Self loops
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[5] SELF-LOOP VALIDATION")
    print("=" * 70)

    self_loops = (
        edges["txId1"] == edges["txId2"]
    ).sum()

    print("Self-loops:", self_loops)

    # --------------------------------------------------------
    # 6. Labels
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[6] LABEL VALIDATION")
    print("=" * 70)

    valid_labels = {"1", "2", "unknown"}

    actual_labels = set(
        classes["class"].astype(str).unique()
    )

    print("Actual labels:", actual_labels)

    unexpected_labels = actual_labels - valid_labels

    print(
        "Unexpected labels:",
        unexpected_labels
    )

    # --------------------------------------------------------
    # 7. Labeled dataset
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[7] LABELED DATASET")
    print("=" * 70)

    labeled = classes[
        classes["class"].astype(str) != "unknown"
    ].copy()

    illicit = (
        labeled["class"].astype(str) == "1"
    ).sum()

    licit = (
        labeled["class"].astype(str) == "2"
    ).sum()

    total_labeled = len(labeled)

    print("Labeled transactions:", total_labeled)

    print("Illicit:", illicit)

    print("Licit:", licit)

    if total_labeled > 0:

        print(
            "Illicit percentage:",
            round(illicit / total_labeled * 100, 2),
            "%"
        )

        print(
            "Licit percentage:",
            round(licit / total_labeled * 100, 2),
            "%"
        )

    # --------------------------------------------------------
    # 8. Temporal coverage
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[8] TEMPORAL VALIDATION")
    print("=" * 70)

    time_steps = sorted(
        features["time_step"].unique()
    )

    print("Number of time steps:", len(time_steps))

    print(
        "First time step:",
        time_steps[0]
    )

    print(
        "Last time step:",
        time_steps[-1]
    )

    # --------------------------------------------------------
    # 9. Labeled transactions per time step
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[9] LABELED TRANSACTIONS BY TIME")
    print("=" * 70)

    temporal = features[
        ["txId", "time_step"]
    ].merge(
        classes,
        on="txId",
        how="left"
    )

    temporal_labeled = temporal[
        temporal["class"].astype(str) != "unknown"
    ]

    labeled_by_time = (
        temporal_labeled
        .groupby("time_step")
        .size()
    )

    print(labeled_by_time.to_string())

    # --------------------------------------------------------
    # 10. Illicit transactions per time step
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[10] ILLICIT TRANSACTIONS BY TIME")
    print("=" * 70)

    illicit_by_time = (
        temporal[
            temporal["class"].astype(str) == "1"
        ]
        .groupby("time_step")
        .size()
    )

    print(illicit_by_time.to_string())

    # --------------------------------------------------------
    # 11. Final validation status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION RESULT")
    print("=" * 70)

    checks = {
        "Feature/class counts match":
            len(features) == len(classes),

        "No duplicate feature IDs":
            duplicate_features == 0,

        "No duplicate class IDs":
            duplicate_classes == 0,

        "All feature IDs have labels":
            len(missing_from_classes) == 0,

        "All class IDs have features":
            len(missing_from_features) == 0,

        "All graph nodes have features":
            len(missing_graph_features) == 0,

        "No self-loops":
            self_loops == 0,

        "No unexpected labels":
            len(unexpected_labels) == 0,

        "49 time steps":
            len(time_steps) == 49,
    }

    print()

    all_passed = True

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"[{status}] {name}")

        if not passed:
            all_passed = False

    print("\n" + "=" * 70)

    if all_passed:
        print("ALL DATASET VALIDATION CHECKS PASSED")
    else:
        print("SOME DATASET VALIDATION CHECKS FAILED")

    print("=" * 70)


if __name__ == "__main__":
    main()