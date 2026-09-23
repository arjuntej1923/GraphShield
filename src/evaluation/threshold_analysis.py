from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"

EXPERIMENTS_DIR = ROOT / "experiments"

RESULTS_DIR = ROOT / "results"


# ============================================================
# MODEL FILES
# ============================================================

GCN_MODEL = (
    EXPERIMENTS_DIR /
    "gcn" /
    "gcn_best.pt"
)

GRAPHSAGE_MODEL = (
    EXPERIMENTS_DIR /
    "graphsage" /
    "graphsage_best.pt"
)

GAT_MODEL = (
    EXPERIMENTS_DIR /
    "gat" /
    "gat_best.pt"
)


# ============================================================
# GRAPH FILES
# ============================================================

TRAIN_GRAPH_FILE = (
    DATA_DIR /
    "graph_train.pt"
)

VAL_GRAPH_FILE = (
    DATA_DIR /
    "graph_validation.pt"
)

TEST_GRAPH_FILE = (
    DATA_DIR /
    "graph_test.pt"
)


# ============================================================
# IMPORT GNN LAYERS
# ============================================================

from torch import nn

from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
)


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
    print("GRAPHSHIELD - THRESHOLD ANALYSIS")
    print("=" * 70)

    print("\nLoading graphs...")

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

    print(
        f"Training graph   : "
        f"{train_graph.num_nodes:,} nodes"
    )

    print(
        f"Validation graph : "
        f"{val_graph.num_nodes:,} nodes"
    )

    print(
        f"Test graph       : "
        f"{test_graph.num_nodes:,} nodes"
    )

    return (
        train_graph,
        val_graph,
        test_graph
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_features(
    train_graph,
    val_graph,
    test_graph
):

    print("\nNormalizing features...")

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

    train_graph.x = (
        train_graph.x - mean
    ) / std

    val_graph.x = (
        val_graph.x - mean
    ) / std

    test_graph.x = (
        test_graph.x - mean
    ) / std

    print(
        "[PASS] Training statistics "
        "used for all graphs."
    )

    return (
        train_graph,
        val_graph,
        test_graph
    )


# ============================================================
# GCN
# ============================================================

class GCN(nn.Module):

    def __init__(
        self,
        input_channels,
        hidden_channels=64,
        output_channels=2,
        dropout=0.3
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

    def forward(
        self,
        x,
        edge_index
    ):

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

        return self.classifier(x)


# ============================================================
# GRAPHSAGE
# ============================================================

class GraphSAGE(nn.Module):

    def __init__(
        self,
        input_channels,
        hidden_channels=64,
        output_channels=2,
        dropout=0.3
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

    def forward(
        self,
        x,
        edge_index
    ):

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

        return self.classifier(x)


# ============================================================
# GAT
# ============================================================

class GAT(nn.Module):

    def __init__(
        self,
        input_channels,
        hidden_channels=32,
        output_channels=2,
        heads=4,
        dropout=0.3
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

    def forward(
        self,
        x,
        edge_index
    ):

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

        return self.classifier(x)


# ============================================================
# GET MODEL
# ============================================================

def create_model(
    model_name,
    num_features
):

    if model_name == "GCN":

        model = GCN(
            input_channels=num_features
        )

    elif model_name == "GraphSAGE":

        model = GraphSAGE(
            input_channels=num_features
        )

    elif model_name == "GAT":

        model = GAT(
            input_channels=num_features
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model_name,
    model_file,
    num_features,
    device
):

    print(
        f"\nLoading {model_name} model..."
    )

    model = create_model(
        model_name,
        num_features
    )

    state_dict = torch.load(
        model_file,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        f"[PASS] {model_name} loaded."
    )

    return model


# ============================================================
# GET PREDICTIONS
# ============================================================

def get_predictions(
    model,
    graph,
    device
):

    model.eval()

    with torch.no_grad():

        logits = model(
            graph.x,
            graph.edge_index
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[:, 1]

    mask = graph.labeled_mask

    y_true = (
        graph.y[mask]
        .detach()
        .cpu()
        .numpy()
    )

    y_prob = (
        probabilities[mask]
        .detach()
        .cpu()
        .numpy()
    )

    return (
        y_true,
        y_prob
    )


# ============================================================
# THRESHOLD METRICS
# ============================================================

def calculate_threshold_metrics(
    y_true,
    y_prob,
    threshold
):

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# FIND BEST THRESHOLD
# ============================================================

def find_best_threshold(
    y_true,
    y_prob
):

    thresholds = np.arange(
        0.05,
        0.96,
        0.01
    )

    results = []

    for threshold in thresholds:

        metrics = calculate_threshold_metrics(
            y_true,
            y_prob,
            threshold
        )

        results.append(
            metrics
        )

    df = pd.DataFrame(
        results
    )

    # Select threshold using VALIDATION F1.
    best_index = df["f1"].idxmax()

    best_row = df.loc[
        best_index
    ]

    return (
        best_row,
        df
    )


# ============================================================
# EVALUATE TEST
# ============================================================

def evaluate_test(
    y_true,
    y_prob,
    threshold
):

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    pr_auc = average_precision_score(
        y_true,
        y_prob
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


# ============================================================
# MAIN
# ============================================================

def main():

    device = get_device()

    print(
        "\nDevice:",
        device
    )

    if device.type == "mps":

        print(
            "[INFO] Apple Metal "
            "Performance Shaders enabled."
        )

    # --------------------------------------------------------
    # Load graphs
    # --------------------------------------------------------

    (
        train_graph,
        val_graph,
        test_graph
    ) = load_graphs()

    # --------------------------------------------------------
    # Normalize
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

    # --------------------------------------------------------
    # Move graphs
    # --------------------------------------------------------

    train_graph = train_graph.to(
        device
    )

    val_graph = val_graph.to(
        device
    )

    test_graph = test_graph.to(
        device
    )

    num_features = (
        train_graph.num_node_features
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {

        "GCN": GCN_MODEL,

        "GraphSAGE": GRAPHSAGE_MODEL,

        "GAT": GAT_MODEL

    }

    all_results = []

    all_threshold_results = []

    # ========================================================
    # PROCESS EACH GNN
    # ========================================================

    for model_name, model_file in models.items():

        print("\n")
        print("=" * 70)

        print(
            f"THRESHOLD ANALYSIS - "
            f"{model_name}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model = load_model(
            model_name,
            model_file,
            num_features,
            device
        )

        # ----------------------------------------------------
        # Validation predictions
        # ----------------------------------------------------

        (
            y_val,
            val_prob
        ) = get_predictions(
            model,
            val_graph,
            device
        )

        # ----------------------------------------------------
        # Test predictions
        # ----------------------------------------------------

        (
            y_test,
            test_prob
        ) = get_predictions(
            model,
            test_graph,
            device
        )

        # ----------------------------------------------------
        # Find threshold using VALIDATION ONLY
        # ----------------------------------------------------

        (
            best_threshold_row,
            threshold_df
        ) = find_best_threshold(
            y_val,
            val_prob
        )

        best_threshold = float(
            best_threshold_row[
                "threshold"
            ]
        )

        best_val_precision = float(
            best_threshold_row[
                "precision"
            ]
        )

        best_val_recall = float(
            best_threshold_row[
                "recall"
            ]
        )

        best_val_f1 = float(
            best_threshold_row[
                "f1"
            ]
        )

        # Add model name.
        threshold_df.insert(
            0,
            "model",
            model_name
        )

        all_threshold_results.append(
            threshold_df
        )

        print(
            "\nBest validation threshold:"
        )

        print(
            f"Threshold : "
            f"{best_threshold:.2f}"
        )

        print(
            f"Precision : "
            f"{best_val_precision:.4f}"
        )

        print(
            f"Recall    : "
            f"{best_val_recall:.4f}"
        )

        print(
            f"F1        : "
            f"{best_val_f1:.4f}"
        )

        # ----------------------------------------------------
        # Apply locked threshold to TEST
        # ----------------------------------------------------

        test_metrics = evaluate_test(
            y_test,
            test_prob,
            best_threshold
        )

        test_metrics[
            "model"
        ] = model_name

        test_metrics[
            "validation_precision"
        ] = best_val_precision

        test_metrics[
            "validation_recall"
        ] = best_val_recall

        test_metrics[
            "validation_f1"
        ] = best_val_f1

        all_results.append(
            test_metrics
        )

        print(
            "\nFINAL TEST WITH "
            "VALIDATION-SELECTED THRESHOLD"
        )

        print(
            f"Threshold : "
            f"{best_threshold:.2f}"
        )

        print(
            f"Precision : "
            f"{test_metrics['precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{test_metrics['recall']:.4f}"
        )

        print(
            f"F1        : "
            f"{test_metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC   : "
            f"{test_metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC    : "
            f"{test_metrics['pr_auc']:.4f}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Final threshold-adjusted results
    # --------------------------------------------------------

    final_df = pd.DataFrame(
        all_results
    )

    final_df = final_df[
        [
            "model",
            "threshold",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "validation_precision",
            "validation_recall",
            "validation_f1"
        ]
    ]

    final_file = (
        RESULTS_DIR /
        "gnn_threshold_results.csv"
    )

    final_df.to_csv(
        final_file,
        index=False
    )

    # --------------------------------------------------------
    # All threshold results
    # --------------------------------------------------------

    threshold_all_df = pd.concat(
        all_threshold_results,
        ignore_index=True
    )

    threshold_file = (
        RESULTS_DIR /
        "gnn_all_thresholds.csv"
    )

    threshold_all_df.to_csv(
        threshold_file,
        index=False
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("THRESHOLD ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        "\nFinal threshold-adjusted "
        "test results:"
    )

    print(
        final_df.to_string(
            index=False
        )
    )

    print("\nSaved:")

    print(
        final_file
    )

    print(
        threshold_file
    )

    print("\n" + "=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()  