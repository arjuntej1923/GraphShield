from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw" / "elliptic"
PROCESSED_DIR = ROOT / "data" / "processed"

FEATURES_FILE = DATA_DIR / "elliptic_txs_features.csv"
CLASSES_FILE = DATA_DIR / "elliptic_txs_classes.csv"
EDGES_FILE = DATA_DIR / "elliptic_txs_edgelist.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("GRAPHSHIELD - BUILD TEMPORAL GRAPHS")
    print("=" * 70)

    print("\nLoading features...")

    features = pd.read_csv(
        FEATURES_FILE,
        header=None
    )

    features.columns = (
        ["txId", "time_step"]
        + [f"feature_{i}" for i in range(1, 166)]
    )

    print(f"Transactions: {len(features):,}")

    print("\nLoading classes...")

    classes = pd.read_csv(
        CLASSES_FILE
    )

    print(f"Class records: {len(classes):,}")

    print("\nLoading edges...")

    edges = pd.read_csv(
        EDGES_FILE
    )

    print(f"Edges: {len(edges):,}")

    return features, classes, edges


# ============================================================
# PREPARE LABELS
# ============================================================

def prepare_labels(features, classes):

    print("\n" + "=" * 70)
    print("PREPARING LABELS")
    print("=" * 70)

    data = features.merge(
        classes,
        on="txId",
        how="left"
    )

    # Label encoding:
    #
    # unknown = -1
    # licit   = 0
    # illicit = 1

    data["label"] = -1

    data.loc[
        data["class"].astype(str) == "2",
        "label"
    ] = 0

    data.loc[
        data["class"].astype(str) == "1",
        "label"
    ] = 1

    print("\nLabel distribution:")

    print(
        f"Unknown : {(data['label'] == -1).sum():,}"
    )

    print(
        f"Licit   : {(data['label'] == 0).sum():,}"
    )

    print(
        f"Illicit : {(data['label'] == 1).sum():,}"
    )

    return data


# ============================================================
# BUILD ONE TEMPORAL GRAPH
# ============================================================

def build_graph(
    data,
    edges,
    start_time,
    end_time,
    graph_name
):

    print("\n" + "=" * 70)
    print(f"BUILDING {graph_name.upper()} GRAPH")
    print("=" * 70)

    # --------------------------------------------------------
    # Select nodes belonging to this temporal period
    # --------------------------------------------------------

    period_data = data[
        (data["time_step"] >= start_time)
        & (data["time_step"] <= end_time)
    ].copy()

    print(
        f"\nTime range: {start_time}-{end_time}"
    )

    print(
        f"Nodes: {len(period_data):,}"
    )

    # --------------------------------------------------------
    # Create local node index
    # --------------------------------------------------------

    tx_to_local = {
        tx_id: idx
        for idx, tx_id
        in enumerate(period_data["txId"])
    }

    # --------------------------------------------------------
    # Select edges whose endpoints both belong
    # to this temporal graph
    # --------------------------------------------------------

    period_edges = edges[
        edges["txId1"].isin(tx_to_local)
        & edges["txId2"].isin(tx_to_local)
    ].copy()

    print(
        f"Edges: {len(period_edges):,}"
    )

    # --------------------------------------------------------
    # Convert transaction IDs to local node indices
    # --------------------------------------------------------

    source = period_edges["txId1"].map(
        tx_to_local
    ).to_numpy()

    target = period_edges["txId2"].map(
        tx_to_local
    ).to_numpy()

    edge_index = torch.tensor(
        [source, target],
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Node features
    # --------------------------------------------------------

    feature_columns = [
        f"feature_{i}"
        for i in range(1, 166)
    ]

    x = torch.tensor(
        period_data[feature_columns].to_numpy(
            dtype="float32"
        ),
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    y = torch.tensor(
        period_data["label"].to_numpy(),
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Masks
    # --------------------------------------------------------

    labeled_mask = y >= 0

    illicit_mask = y == 1

    licit_mask = y == 0

    # --------------------------------------------------------
    # Create PyTorch Geometric Data object
    # --------------------------------------------------------

    graph = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        labeled_mask=labeled_mask,
        illicit_mask=illicit_mask,
        licit_mask=licit_mask
    )

    # Store useful metadata separately
    graph.tx_ids = period_data["txId"].to_numpy()
    graph.time_steps = torch.tensor(
        period_data["time_step"].to_numpy(),
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\nGraph statistics:")

    print(
        f"Nodes           : {graph.num_nodes:,}"
    )

    print(
        f"Edges           : {graph.num_edges:,}"
    )

    print(
        f"Node features   : {graph.num_node_features}"
    )

    print(
        f"Labeled nodes   : {labeled_mask.sum().item():,}"
    )

    print(
        f"Unknown nodes   : {(~labeled_mask).sum().item():,}"
    )

    print(
        f"Licit nodes     : {licit_mask.sum().item():,}"
    )

    print(
        f"Illicit nodes   : {illicit_mask.sum().item():,}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if graph.num_nodes == len(period_data):
        print(
            "[PASS] Node count matches feature records."
        )
    else:
        print(
            "[FAIL] Node count mismatch."
        )

    if graph.num_edges == len(period_edges):
        print(
            "[PASS] Edge count matches filtered edges."
        )
    else:
        print(
            "[FAIL] Edge count mismatch."
        )

    if torch.all(
        graph.edge_index >= 0
    ):
        print(
            "[PASS] No negative edge indices."
        )

    if graph.edge_index.numel() > 0:

        if graph.edge_index.max() < graph.num_nodes:
            print(
                "[PASS] All edge indices are valid."
            )
        else:
            print(
                "[FAIL] Invalid edge index detected."
            )

    # --------------------------------------------------------
    # Save graph
    # --------------------------------------------------------

    output_file = (
        PROCESSED_DIR /
        f"{graph_name}.pt"
    )

    torch.save(
        graph,
        output_file
    )

    print(
        f"\nSaved: {output_file}"
    )

    return graph


# ============================================================
# MAIN
# ============================================================

def main():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    features, classes, edges = load_data()

    data = prepare_labels(
        features,
        classes
    )

    # ========================================================
    # BUILD TRAIN GRAPH
    # ========================================================

    train_graph = build_graph(
        data,
        edges,
        start_time=1,
        end_time=34,
        graph_name="graph_train"
    )

    # ========================================================
    # BUILD VALIDATION GRAPH
    # ========================================================

    validation_graph = build_graph(
        data,
        edges,
        start_time=35,
        end_time=40,
        graph_name="graph_validation"
    )

    # ========================================================
    # BUILD TEST GRAPH
    # ========================================================

    test_graph = build_graph(
        data,
        edges,
        start_time=41,
        end_time=49,
        graph_name="graph_test"
    )

    # ========================================================
    # FINAL CHECK
    # ========================================================

    print("\n" + "=" * 70)
    print("GRAPH DATASET SUMMARY")
    print("=" * 70)

    print(
        f"\nTraining graph:"
    )

    print(
        f"  Nodes: {train_graph.num_nodes:,}"
    )

    print(
        f"  Edges: {train_graph.num_edges:,}"
    )

    print(
        f"\nValidation graph:"
    )

    print(
        f"  Nodes: {validation_graph.num_nodes:,}"
    )

    print(
        f"  Edges: {validation_graph.num_edges:,}"
    )

    print(
        f"\nTest graph:"
    )

    print(
        f"  Nodes: {test_graph.num_nodes:,}"
    )

    print(
        f"  Edges: {test_graph.num_edges:,}"
    )

    print("\n" + "=" * 70)
    print("TEMPORAL GRAPH CONSTRUCTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()