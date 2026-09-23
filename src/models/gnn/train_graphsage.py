from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import SAGEConv

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

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data" / "processed"
EXPERIMENTS_DIR = ROOT / "experiments" / "graphsage"

TRAIN_GRAPH_FILE = DATA_DIR / "graph_train.pt"
VAL_GRAPH_FILE = DATA_DIR / "graph_validation.pt"
TEST_GRAPH_FILE = DATA_DIR / "graph_test.pt"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

HIDDEN_CHANNELS = 64

DROPOUT = 0.3

LEARNING_RATE = 0.01

WEIGHT_DECAY = 5e-4

EPOCHS = 100

PATIENCE = 15


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    print("GRAPHSHIELD - GRAPHSAGE")
    print("=" * 70)

    print("\nLoading training graph...")

    train_graph = torch.load(
        TRAIN_GRAPH_FILE,
        map_location="cpu",
        weights_only=False
    )

    print("Loading validation graph...")

    val_graph = torch.load(
        VAL_GRAPH_FILE,
        map_location="cpu",
        weights_only=False
    )

    print("Loading test graph...")

    test_graph = torch.load(
        TEST_GRAPH_FILE,
        map_location="cpu",
        weights_only=False
    )

    print("\nGraph sizes:")

    print(
        f"Training   : "
        f"{train_graph.num_nodes:,} nodes, "
        f"{train_graph.num_edges:,} edges"
    )

    print(
        f"Validation : "
        f"{val_graph.num_nodes:,} nodes, "
        f"{val_graph.num_edges:,} edges"
    )

    print(
        f"Test       : "
        f"{test_graph.num_nodes:,} nodes, "
        f"{test_graph.num_edges:,} edges"
    )

    return train_graph, val_graph, test_graph


# ============================================================
# NORMALIZE FEATURES
# ============================================================

def normalize_features(
    train_graph,
    val_graph,
    test_graph
):

    print("\n" + "=" * 70)
    print("FEATURE NORMALIZATION")
    print("=" * 70)

    print(
        "\nFitting normalization statistics "
        "on training graph only..."
    )

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
        "[PASS] Validation and test graphs "
        "were transformed using training statistics."
    )

    return (
        train_graph,
        val_graph,
        test_graph
    )


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


    def forward(
        self,
        x,
        edge_index
    ):

        # First GraphSAGE layer
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

        # Second GraphSAGE layer
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

        # Final classifier
        x = self.classifier(x)

        return x


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(graph):

    labels = graph.y[
        graph.labeled_mask
    ]

    licit_count = (
        labels == 0
    ).sum().item()

    illicit_count = (
        labels == 1
    ).sum().item()

    total = (
        licit_count +
        illicit_count
    )

    weight_licit = (
        total /
        (2.0 * licit_count)
    )

    weight_illicit = (
        total /
        (2.0 * illicit_count)
    )

    weights = torch.tensor(
        [
            weight_licit,
            weight_illicit
        ],
        dtype=torch.float32
    )

    print("\n" + "=" * 70)
    print("CLASS WEIGHTS")
    print("=" * 70)

    print(
        f"Licit count   : {licit_count:,}"
    )

    print(
        f"Illicit count : {illicit_count:,}"
    )

    print(
        f"Licit weight  : {weight_licit:.4f}"
    )

    print(
        f"Illicit weight: {weight_illicit:.4f}"
    )

    return weights


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    graph,
    device,
    split_name
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

        predictions = (
            probabilities >= 0.5
        ).long()

    mask = graph.labeled_mask

    y_true = graph.y[mask]

    y_prob = probabilities[mask]

    y_pred = predictions[mask]

    y_true_np = (
        y_true.detach()
        .cpu()
        .numpy()
    )

    y_prob_np = (
        y_prob.detach()
        .cpu()
        .numpy()
    )

    y_pred_np = (
        y_pred.detach()
        .cpu()
        .numpy()
    )

    precision = precision_score(
        y_true_np,
        y_pred_np,
        zero_division=0
    )

    recall = recall_score(
        y_true_np,
        y_pred_np,
        zero_division=0
    )

    f1 = f1_score(
        y_true_np,
        y_pred_np,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true_np,
        y_prob_np
    )

    pr_auc = average_precision_score(
        y_true_np,
        y_prob_np
    )

    cm = confusion_matrix(
        y_true_np,
        y_pred_np
    )

    print("\n" + "-" * 70)
    print(f"{split_name} PERFORMANCE")
    print("-" * 70)

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    return {
        "split": split_name,
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

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print("\nDevice:", device)

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
    # Normalize features
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
    # Calculate class weights BEFORE moving
    # graphs to MPS.
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_graph
    )

    # --------------------------------------------------------
    # Move graphs to device
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MOVING GRAPHS TO DEVICE")
    print("=" * 70)

    train_graph = train_graph.to(
        device
    )

    val_graph = val_graph.to(
        device
    )

    test_graph = test_graph.to(
        device
    )

    class_weights = class_weights.to(
        device
    )

    print(
        "[PASS] Graphs moved to device."
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = GraphSAGE(
        input_channels=(
            train_graph.num_node_features
        ),
        hidden_channels=HIDDEN_CHANNELS,
        output_channels=2,
        dropout=DROPOUT
    ).to(device)

    print("\n" + "=" * 70)
    print("GRAPHSAGE MODEL")
    print("=" * 70)

    print(
        f"Input features : "
        f"{train_graph.num_node_features}"
    )

    print(
        f"Hidden channels: "
        f"{HIDDEN_CHANNELS}"
    )

    print(
        "Output classes : 2"
    )

    print(
        f"Dropout        : "
        f"{DROPOUT}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING GRAPHSAGE")
    print("=" * 70)

    best_val_pr_auc = -np.inf

    best_state = None

    patience_counter = 0

    history = []

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

        train_mask = (
            train_graph.labeled_mask
        )

        loss = criterion(
            logits[train_mask],
            train_graph.y[train_mask]
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_result = evaluate(
            model,
            val_graph,
            device,
            "VALIDATION"
        )

        history.append(
            {
                "epoch": epoch,
                "loss": loss.item(),
                "val_pr_auc": (
                    val_result["pr_auc"]
                ),
                "val_f1": (
                    val_result["f1"]
                )
            }
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if (
            val_result["pr_auc"]
            > best_val_pr_auc
        ):

            best_val_pr_auc = (
                val_result["pr_auc"]
            )

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

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(
                f"\nEpoch {epoch:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Val PR-AUC: "
                f"{val_result['pr_auc']:.4f}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                f"\nEarly stopping at "
                f"epoch {epoch}."
            )

            break

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print("RESTORING BEST MODEL")
    print("=" * 70)

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

        model = model.to(
            device
        )

        print(
            f"[PASS] Best validation "
            f"PR-AUC: "
            f"{best_val_pr_auc:.4f}"
        )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    final_val_result = evaluate(
        model,
        val_graph,
        device,
        "FINAL VALIDATION"
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    final_test_result = evaluate(
        model,
        test_graph,
        device,
        "FINAL TEST"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    validation_df = pd.DataFrame(
        [final_val_result]
    )

    test_df = pd.DataFrame(
        [final_test_result]
    )

    history_df = pd.DataFrame(
        history
    )

    validation_file = (
        EXPERIMENTS_DIR /
        "validation_results.csv"
    )

    test_file = (
        EXPERIMENTS_DIR /
        "test_results.csv"
    )

    history_file = (
        EXPERIMENTS_DIR /
        "training_history.csv"
    )

    validation_df.to_csv(
        validation_file,
        index=False
    )

    test_df.to_csv(
        test_file,
        index=False
    )

    history_df.to_csv(
        history_file,
        index=False
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_file = (
        EXPERIMENTS_DIR /
        "graphsage_best.pt"
    )

    torch.save(
        model.state_dict(),
        model_file
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("GRAPHSAGE FINAL RESULTS")
    print("=" * 70)

    print("\nValidation:")

    print(
        validation_df.to_string(
            index=False
        )
    )

    print("\nTest:")

    print(
        test_df.to_string(
            index=False
        )
    )

    print("\nSaved files:")

    print(validation_file)

    print(test_file)

    print(history_file)

    print(model_file)

    print("\n" + "=" * 70)
    print("GRAPHSAGE EXPERIMENT COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()