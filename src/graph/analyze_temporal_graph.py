from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw" / "elliptic"

FEATURES_FILE = DATA_DIR / "elliptic_txs_features.csv"
EDGES_FILE = DATA_DIR / "elliptic_txs_edgelist.csv"


# ============================================================
# LOAD DATA
# ============================================================

def main():

    print("=" * 70)
    print("GRAPHSHIELD - TEMPORAL GRAPH ANALYSIS")
    print("=" * 70)

    print("\nLoading transaction features...")

    features = pd.read_csv(
        FEATURES_FILE,
        header=None
    )

    features.columns = (
        ["txId", "time_step"]
        + [f"feature_{i}" for i in range(1, 166)]
    )

    print(f"Transactions: {len(features):,}")

    print("\nLoading graph edges...")

    edges = pd.read_csv(
        EDGES_FILE
    )

    print(f"Edges: {len(edges):,}")

    # --------------------------------------------------------
    # Create transaction -> time lookup
    # --------------------------------------------------------

    tx_time = dict(
        zip(
            features["txId"],
            features["time_step"]
        )
    )

    print("\n[PASS] Transaction-time mapping created.")


    # ========================================================
    # MAP EDGE ENDPOINTS TO TIME
    # ========================================================

    print("\n" + "=" * 70)
    print("ANALYZING EDGE TEMPORALITY")
    print("=" * 70)

    edges["time_1"] = edges["txId1"].map(tx_time)
    edges["time_2"] = edges["txId2"].map(tx_time)

    missing_time_1 = edges["time_1"].isna().sum()
    missing_time_2 = edges["time_2"].isna().sum()

    print(
        f"\nEdges with missing source time: {missing_time_1:,}"
    )

    print(
        f"Edges with missing target time: {missing_time_2:,}"
    )

    if missing_time_1 == 0 and missing_time_2 == 0:
        print("[PASS] Every graph endpoint has a valid time step.")


    # ========================================================
    # EDGE TIME RELATIONSHIP
    # ========================================================

    edges["time_difference"] = (
        edges["time_2"] - edges["time_1"]
    )

    print("\n" + "=" * 70)
    print("EDGE TIME DIFFERENCE")
    print("=" * 70)

    print("\nTime difference = target_time - source_time")

    print("\nSummary:")
    print(
        edges["time_difference"].describe()
    )

    same_time = (
        edges["time_difference"] == 0
    ).sum()

    forward_edges = (
        edges["time_difference"] > 0
    ).sum()

    backward_edges = (
        edges["time_difference"] < 0
    ).sum()

    print(
        f"\nSame-time edges     : {same_time:,}"
    )

    print(
        f"Forward-time edges  : {forward_edges:,}"
    )

    print(
        f"Backward-time edges : {backward_edges:,}"
    )


    # ========================================================
    # EDGE TIME RELATIONSHIP PERCENTAGES
    # ========================================================

    total_edges = len(edges)

    print("\nPercentages:")

    print(
        f"Same-time     : {same_time / total_edges * 100:.2f}%"
    )

    print(
        f"Forward-time  : {forward_edges / total_edges * 100:.2f}%"
    )

    print(
        f"Backward-time : {backward_edges / total_edges * 100:.2f}%"
    )


    # ========================================================
    # MAXIMUM TIME GAP
    # ========================================================

    print("\n" + "=" * 70)
    print("TIME GAP ANALYSIS")
    print("=" * 70)

    print(
        "\nMaximum positive time difference:",
        edges["time_difference"].max()
    )

    print(
        "Maximum negative time difference:",
        edges["time_difference"].min()
    )


    # ========================================================
    # CROSS-TEMPORAL EDGES
    # ========================================================

    cross_temporal = (
        edges["time_difference"] != 0
    ).sum()

    print(
        f"\nCross-temporal edges: {cross_temporal:,}"
    )


    # ========================================================
    # TIME-STEP EDGE COUNTS
    # ========================================================

    print("\n" + "=" * 70)
    print("EDGES BY SOURCE TIME")
    print("=" * 70)

    source_time_counts = (
        edges["time_1"]
        .value_counts()
        .sort_index()
    )

    print(
        source_time_counts.to_string()
    )


    # ========================================================
    # TEMPORAL GRAPH SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("TEMPORAL GRAPH SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal nodes        : {len(features):,}"
    )

    print(
        f"Total edges        : {len(edges):,}"
    )

    print(
        f"Time steps         : {features['time_step'].nunique()}"
    )

    print(
        f"Same-time edges    : {same_time:,}"
    )

    print(
        f"Cross-time edges   : {cross_temporal:,}"
    )

    print(
        f"Forward-time edges : {forward_edges:,}"
    )

    print(
        f"Backward-time edges: {backward_edges:,}"
    )

    print("\n" + "=" * 70)
    print("TEMPORAL GRAPH ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()