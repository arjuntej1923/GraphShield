from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.nn import GCNConv, SAGEConv, GATConv


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
PREDICTIONS_DIR = ROOT / "results" / "predictions"

GCN_CHECKPOINT = ROOT / "experiments" / "gcn" / "gcn_best.pt"
GRAPHSAGE_CHECKPOINT = (
    ROOT / "experiments" / "graphsage" / "graphsage_best.pt"
)
GAT_CHECKPOINT = ROOT / "experiments" / "gat" / "gat_best.pt"

TRAIN_GRAPH_FILE = DATA_DIR / "graph_train.pt"
VAL_GRAPH_FILE = DATA_DIR / "graph_validation.pt"
TEST_GRAPH_FILE = DATA_DIR / "graph_test.pt"


# ============================================================
# CONFIGURATION
# ============================================================

GCN_HIDDEN = 64
GCN_DROPOUT = 0.3

GRAPHSAGE_HIDDEN = 64
GRAPHSAGE_DROPOUT = 0.3

GAT_HIDDEN = 32
GAT_HEADS = 4
GAT_DROPOUT = 0.3


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# LOAD GRAPHS
# ============================================================

def load_graphs():
    print("=" * 70)
    print("GRAPHSHIELD - LOAD GRAPHS")
    print("=" * 70)

    train_graph = torch.load(
        TRAIN_GRAPH_FILE,
        map_location="cpu",
        weights_only=False
    )

    val_graph = torch.load(
        VAL_GRAPH_FILE,
        map_location="cpu",
        weights_only=False
    )

    test_graph = torch.load(
        TEST_GRAPH_FILE,
        map_location="cpu",
        weights_only=False
    )

    print(f"\nTraining graph   : {train_graph.num_nodes:,} nodes")
    print(f"Validation graph : {val_graph.num_nodes:,} nodes")
    print(f"Test graph       : {test_graph.num_nodes:,} nodes")

    return train_graph, val_graph, test_graph


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_features(train_graph, val_graph, test_graph):
    print("\n" + "=" * 70)
    print("FEATURE NORMALIZATION")
    print("=" * 70)

    print("\nFitting normalization statistics on training graph only...")

    mean = train_graph.x.mean(
        dim=0,
        keepdim=True
    )

    std = train_graph.x.std(
        dim=0,
        keepdim=True
    )

    std = torch.where(
        std < 1e-8,
        torch.ones_like(std),
        std
    )

    train_graph.x = (train_graph.x - mean) / std
    val_graph.x = (val_graph.x - mean) / std
    test_graph.x = (test_graph.x - mean) / std

    print("[PASS] Training-only normalization applied.")

    return train_graph, val_graph, test_graph


# ============================================================
# GCN MODEL
# ============================================================

class GCN(nn.Module):

    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        dropout
    ):
        super().__init__()

        self.conv1 = GCNConv(
            input_channels,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        self.classifier = nn.Linear(
            hidden_channels,
            output_channels
        )

        self.dropout = dropout

    def forward(self, x, edge_index):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.classifier(x)

        return x


# ============================================================
# GRAPHSAGE MODEL
# ============================================================

class GraphSAGE(nn.Module):

    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        dropout
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            input_channels,
            hidden_channels
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            hidden_channels
        )

        self.classifier = nn.Linear(
            hidden_channels,
            output_channels
        )

        self.dropout = dropout

    def forward(self, x, edge_index):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.classifier(x)

        return x


# ============================================================
# GAT MODEL
# ============================================================

class GAT(nn.Module):

    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        heads,
        dropout
    ):
        super().__init__()

        self.gat1 = GATConv(
            input_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout
        )

        self.gat2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout
        )

        self.classifier = nn.Linear(
            hidden_channels,
            output_channels
        )

        self.dropout = dropout

    def forward(self, x, edge_index):

        x = self.gat1(
            x,
            edge_index
        )

        x = F.elu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.gat2(
            x,
            edge_index
        )

        x = F.elu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.classifier(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model, checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(checkpoint)

    model = model.to(device)

    model.eval()

    print(
        f"[PASS] Loaded checkpoint: "
        f"{checkpoint_path.name}"
    )

    return model


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

@torch.no_grad()
def generate_predictions(
    model,
    graph,
    device,
    model_name,
    split_name
):

    graph = graph.to(device)

    logits = model(
        graph.x,
        graph.edge_index
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )[:, 1]

    mask = graph.labeled_mask

    y_true = graph.y[mask].detach().cpu().numpy()
    probabilities = probabilities[mask].detach().cpu().numpy()

    output = pd.DataFrame(
        {
            "y_true": y_true.astype(int),
            "predicted_probability": probabilities
        }
    )

    output_path = (
        PREDICTIONS_DIR
        / f"{model_name}_{split_name}.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    print(
        f"[PASS] Saved {model_name} "
        f"{split_name} predictions:"
    )

    print(f"       {output_path}")
    print(f"       Samples: {len(output):,}")

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("GRAPHSHIELD - SAVE GNN PREDICTIONS")
    print("=" * 70)

    device = get_device()

    print(f"\nDevice: {device}")

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load graphs
    # --------------------------------------------------------

    train_graph, val_graph, test_graph = load_graphs()

    # --------------------------------------------------------
    # Normalize exactly as during training
    # --------------------------------------------------------

    (
        train_graph,
        val_graph,
        test_graph
    ) = normalize_features(
        train_graph,
        val_graph,
        test_graph
    )

    input_channels = train_graph.x.shape[1]

    print(
        f"\nInput features: {input_channels}"
    )

    # ========================================================
    # GCN
    # ========================================================

    print("\n" + "=" * 70)
    print("GCN")
    print("=" * 70)

    gcn = GCN(
        input_channels=input_channels,
        hidden_channels=GCN_HIDDEN,
        output_channels=2,
        dropout=GCN_DROPOUT
    )

    gcn = load_model(
        gcn,
        GCN_CHECKPOINT,
        device
    )

    generate_predictions(
        gcn,
        val_graph,
        device,
        "gcn",
        "validation"
    )

    generate_predictions(
        gcn,
        test_graph,
        device,
        "gcn",
        "test"
    )

    # ========================================================
    # GRAPHSAGE
    # ========================================================

    print("\n" + "=" * 70)
    print("GRAPHSAGE")
    print("=" * 70)

    graphsage = GraphSAGE(
        input_channels=input_channels,
        hidden_channels=GRAPHSAGE_HIDDEN,
        output_channels=2,
        dropout=GRAPHSAGE_DROPOUT
    )

    graphsage = load_model(
        graphsage,
        GRAPHSAGE_CHECKPOINT,
        device
    )

    generate_predictions(
        graphsage,
        val_graph,
        device,
        "graphsage",
        "validation"
    )

    generate_predictions(
        graphsage,
        test_graph,
        device,
        "graphsage",
        "test"
    )

    # ========================================================
    # GAT
    # ========================================================

    print("\n" + "=" * 70)
    print("GAT")
    print("=" * 70)

    gat = GAT(
        input_channels=input_channels,
        hidden_channels=GAT_HIDDEN,
        output_channels=2,
        heads=GAT_HEADS,
        dropout=GAT_DROPOUT
    )

    gat = load_model(
        gat,
        GAT_CHECKPOINT,
        device
    )

    generate_predictions(
        gat,
        val_graph,
        device,
        "gat",
        "validation"
    )

    generate_predictions(
        gat,
        test_graph,
        device,
        "gat",
        "test"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("ALL GNN PREDICTIONS SAVED")
    print("=" * 70)

    print("\nPrediction directory:")
    print(PREDICTIONS_DIR)

    print("\nExpected files:")

    expected_files = [
        "gcn_validation.csv",
        "gcn_test.csv",
        "graphsage_validation.csv",
        "graphsage_test.csv",
        "gat_validation.csv",
        "gat_test.csv",
    ]

    for filename in expected_files:
        path = PREDICTIONS_DIR / filename

        if path.exists():
            print(f"[PASS] {filename}")
        else:
            print(f"[FAIL] {filename}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
