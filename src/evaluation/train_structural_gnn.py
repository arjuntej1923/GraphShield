from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "structural_gnn"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

HIDDEN = 32
DROPOUT = 0.30

LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4

EPOCHS = 100
PATIENCE = 15

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(
    f"\nUsing device: {DEVICE}"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


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
# CREATE STRUCTURAL FEATURES
# ============================================================

def create_structural_features(graph):

    edge_index = graph.edge_index.cpu()

    source = edge_index[0].numpy()
    target = edge_index[1].numpy()

    num_nodes = graph.num_nodes

    in_degree = np.bincount(
        target,
        minlength=num_nodes
    )

    out_degree = np.bincount(
        source,
        minlength=num_nodes
    )

    total_degree = (
        in_degree
        + out_degree
    )

    features = np.column_stack(
        [
            in_degree,
            out_degree,
            total_degree
        ]
    ).astype(
        np.float32
    )

    return torch.tensor(
        features,
        dtype=torch.float32
    )


# ============================================================
# NORMALIZE STRUCTURAL FEATURES
# ============================================================

def fit_normalization(train_x):

    mean = train_x.mean(
        dim=0,
        keepdim=True
    )

    std = train_x.std(
        dim=0,
        keepdim=True
    )

    std = torch.where(
        std < 1e-8,
        torch.ones_like(std),
        std
    )

    return mean, std


def normalize(
    x,
    mean,
    std
):

    return (
        x - mean
    ) / std


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(graph):

    mask = graph.labeled_mask.cpu()

    labels = graph.y.cpu()[mask]

    class_counts = torch.bincount(
        labels,
        minlength=2
    ).float()

    total = class_counts.sum()

    weights = total / (
        2.0 * class_counts
    )

    return weights


# ============================================================
# GCN
# ============================================================

class StructuralGCN(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout
    ):

        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim
        )

        self.classifier = nn.Linear(
            hidden_dim,
            2
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

        x = torch.relu(x)

        x = torch.dropout(
            x,
            p=self.dropout,
            train=self.training
        )

        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = torch.dropout(
            x,
            p=self.dropout,
            train=self.training
        )

        return self.classifier(x)


# ============================================================
# GRAPHSAGE
# ============================================================

class StructuralGraphSAGE(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout
    ):

        super().__init__()

        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim
        )

        self.classifier = nn.Linear(
            hidden_dim,
            2
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

        x = torch.relu(x)

        x = torch.dropout(
            x,
            p=self.dropout,
            train=self.training
        )

        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = torch.dropout(
            x,
            p=self.dropout,
            train=self.training
        )

        return self.classifier(x)


# ============================================================
# GAT
# ============================================================

class StructuralGAT(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        heads,
        dropout
    ):

        super().__init__()

        self.gat1 = GATConv(
            input_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout
        )

        self.gat2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout
        )

        self.classifier = nn.Linear(
            hidden_dim,
            2
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

        x = torch.dropout(
            x,
            p=self.dropout,
            train=self.training
        )

        x = self.gat2(
            x,
            edge_index
        )

        x = F.elu(x)

        x = torch.dropout(
            x,
            p=self.dropout,
            train=self.training
        )

        return self.classifier(x)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    graph,
    mask
):

    model.eval()

    logits = model(
        graph.x,
        graph.edge_index
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )[:, 1]

    y_true = (
        graph.y[mask]
        .cpu()
        .numpy()
    )

    probabilities = (
        probabilities[mask]
        .cpu()
        .numpy()
    )

    return (
        y_true,
        probabilities
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
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
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    y_true,
    probabilities
):

    thresholds = np.arange(
        0.05,
        0.96,
        0.01
    )

    best_threshold = 0.50

    best_f1 = -1

    rows = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold
        )

        rows.append(
            {
                "threshold":
                    threshold,
                "precision":
                    metrics["precision"],
                "recall":
                    metrics["recall"],
                "f1":
                    metrics["f1"]
            }
        )

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]

            best_threshold = threshold

    return (
        best_threshold,
        pd.DataFrame(rows)
    )


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    model_name,
    model,
    train_graph,
    val_graph,
    test_graph,
    class_weights
):

    print("\n" + "=" * 70)

    print(
        f"TRAINING STRUCTURAL {model_name}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Move model and graphs
    # --------------------------------------------------------

    model = model.to(
        DEVICE
    )

    train_graph = train_graph.to(
        DEVICE
    )

    val_graph = val_graph.to(
        DEVICE
    )

    test_graph = test_graph.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(
            DEVICE
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    train_mask = (
        train_graph.labeled_mask
    )

    val_mask = (
        val_graph.labeled_mask
    )

    best_val_pr_auc = -1

    best_state = None

    best_epoch = 0

    patience_counter = 0

    history = []

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        optimizer.zero_grad()

        logits = model(
            train_graph.x,
            train_graph.edge_index
        )

        loss = criterion(
            logits[train_mask],
            train_graph.y[
                train_mask
            ]
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        y_val, val_probabilities = (
            evaluate(
                model,
                val_graph,
                val_mask
            )
        )

        val_pr_auc = (
            average_precision_score(
                y_val,
                val_probabilities
            )
        )

        val_roc_auc = (
            roc_auc_score(
                y_val,
                val_probabilities
            )
        )

        history.append(
            {
                "epoch": epoch,
                "loss": loss.item(),
                "validation_pr_auc":
                    val_pr_auc,
                "validation_roc_auc":
                    val_roc_auc
            }
        )

        if val_pr_auc > best_val_pr_auc:

            best_val_pr_auc = (
                val_pr_auc
            )

            best_epoch = epoch

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict()
                .items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:

            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss.item():.4f} | "
                f"Val PR-AUC "
                f"{val_pr_auc:.4f}"
            )

        if patience_counter >= PATIENCE:

            print(
                f"\nEarly stopping at "
                f"epoch {epoch}"
            )

            break

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    model.load_state_dict(
        best_state
    )

    print(
        f"\nBest epoch: {best_epoch}"
    )

    print(
        f"Best validation PR-AUC: "
        f"{best_val_pr_auc:.4f}"
    )

    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    torch.save(
        model.state_dict(),
        RESULTS_DIR
        / f"{safe_name}_best.pt"
    )

    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    y_val, val_probabilities = (
        evaluate(
            model,
            val_graph,
            val_mask
        )
    )

    # --------------------------------------------------------
    # Threshold selected on validation
    # --------------------------------------------------------

    (
        best_threshold,
        threshold_table
    ) = find_best_threshold(
        y_val,
        val_probabilities
    )

    threshold_table.to_csv(
        RESULTS_DIR
        / f"{safe_name}_thresholds.csv",
        index=False
    )

    # ========================================================
    # TEST
    # ========================================================

    y_test, test_probabilities = (
        evaluate(
            model,
            test_graph,
            test_graph.labeled_mask
        )
    )

    val_metrics = calculate_metrics(
        y_val,
        val_probabilities,
        best_threshold
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        best_threshold
    )

    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"\nSelected threshold: "
        f"{best_threshold:.2f}"
    )

    print("\nVALIDATION")

    print(
        f"Precision : "
        f"{val_metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{val_metrics['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{val_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{val_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{val_metrics['pr_auc']:.4f}"
    )

    print("\nTEST")

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

    print("\nTest confusion matrix:")

    print(
        np.array(
            [
                [
                    test_metrics["tn"],
                    test_metrics["fp"]
                ],
                [
                    test_metrics["fn"],
                    test_metrics["tp"]
                ]
            ]
        )
    )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    pd.DataFrame(
        history
    ).to_csv(
        RESULTS_DIR
        / f"{safe_name}_training_history.csv",
        index=False
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    pd.DataFrame(
        {
            "y_true": y_val,
            "predicted_probability":
                val_probabilities
        }
    ).to_csv(
        RESULTS_DIR
        / f"{safe_name}_validation_predictions.csv",
        index=False
    )

    pd.DataFrame(
        {
            "y_true": y_test,
            "predicted_probability":
                test_probabilities
        }
    ).to_csv(
        RESULTS_DIR
        / f"{safe_name}_test_predictions.csv",
        index=False
    )

    return {
        "model": model_name,
        "threshold": best_threshold,
        "validation_precision":
            val_metrics["precision"],
        "validation_recall":
            val_metrics["recall"],
        "validation_f1":
            val_metrics["f1"],
        "validation_roc_auc":
            val_metrics["roc_auc"],
        "validation_pr_auc":
            val_metrics["pr_auc"],
        "test_precision":
            test_metrics["precision"],
        "test_recall":
            test_metrics["recall"],
        "test_f1":
            test_metrics["f1"],
        "test_roc_auc":
            test_metrics["roc_auc"],
        "test_pr_auc":
            test_metrics["pr_auc"]
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "GRAPHSHIELD - STRUCTURAL-ONLY GNN ABLATION"
    )

    print("=" * 70)

    # ========================================================
    # LOAD
    # ========================================================

    print(
        "\nLoading graphs..."
    )

    train_graph = load_graph(
        DATA_DIR / "graph_train.pt"
    )

    val_graph = load_graph(
        DATA_DIR / "graph_validation.pt"
    )

    test_graph = load_graph(
        DATA_DIR / "graph_test.pt"
    )

    # ========================================================
    # STRUCTURAL FEATURES
    # ========================================================

    print(
        "\nCreating structural node features..."
    )

    train_x = create_structural_features(
        train_graph
    )

    val_x = create_structural_features(
        val_graph
    )

    test_x = create_structural_features(
        test_graph
    )

    print(
        f"Structural features: "
        f"{train_x.shape[1]}"
    )

    # ========================================================
    # TRAIN-ONLY NORMALIZATION
    # ========================================================

    print(
        "\nFitting normalization on training graph only..."
    )

    mean, std = fit_normalization(
        train_x
    )

    train_graph.x = normalize(
        train_x,
        mean,
        std
    )

    val_graph.x = normalize(
        val_x,
        mean,
        std
    )

    test_graph.x = normalize(
        test_x,
        mean,
        std
    )

    print(
        "[PASS] Normalization complete."
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = calculate_class_weights(
        train_graph
    )

    print(
        "\nClass weights:"
    )

    print(
        class_weights
    )

    # ========================================================
    # MODELS
    # ========================================================

    models = [
        (
            "GCN",
            StructuralGCN(
                input_dim=3,
                hidden_dim=HIDDEN,
                dropout=DROPOUT
            )
        ),

        (
            "GraphSAGE",
            StructuralGraphSAGE(
                input_dim=3,
                hidden_dim=HIDDEN,
                dropout=DROPOUT
            )
        ),

        (
            "GAT",
            StructuralGAT(
                input_dim=3,
                hidden_dim=HIDDEN,
                heads=4,
                dropout=DROPOUT
            )
        )
    ]

    # ========================================================
    # TRAIN
    # ========================================================

    results = []

    for model_name, model in models:

        result = train_model(
            model_name,
            model,
            train_graph,
            val_graph,
            test_graph,
            class_weights
        )

        results.append(
            result
        )

    # ========================================================
    # SAVE COMPARISON
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    output_file = (
        RESULTS_DIR
        / "structural_gnn_comparison.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "STRUCTURAL-ONLY GNN RESULTS"
    )

    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    print(
        "\n[PASS] Results saved:"
    )

    print(
        output_file
    )

    print("\n" + "=" * 70)

    print(
        "STRUCTURAL GNN ABLATION COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
