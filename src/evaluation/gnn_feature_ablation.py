import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TOP_K_VALUES = [10, 20, 30, 50, 165]

EPOCHS = 100
PATIENCE = 15

HIDDEN_GCN = 64
HIDDEN_SAGE = 64
HIDDEN_GAT = 32
GAT_HEADS = 4

DROPOUT = 0.30

LR_GCN = 0.01
LR_SAGE = 0.01
LR_GAT = 0.005

WEIGHT_DECAY = 5e-4

RESULTS_DIR = "results/gnn_feature_ablation"
IMPORTANCE_DIR = "results/feature_importance"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 70)
print("GRAPHSHIELD - GNN FEATURE ABLATION")
print("=" * 70)

print(f"\nDevice: {DEVICE}")


# ============================================================
# LOAD GRAPH DATA
# ============================================================

print("\nLoading temporal graphs...")

train_graph = torch.load(
    "data/processed/graph_train.pt",
    map_location="cpu",
    weights_only=False
)

val_graph = torch.load(
    "data/processed/graph_validation.pt",
    map_location="cpu",
    weights_only=False
)

test_graph = torch.load(
    "data/processed/graph_test.pt",
    map_location="cpu",
    weights_only=False
)


# ============================================================
# LOAD TEMPORAL CSVs
# ============================================================

train_df = pd.read_csv(
    "data/processed/train_temporal.csv"
)

val_df = pd.read_csv(
    "data/processed/validation_temporal.csv"
)

test_df = pd.read_csv(
    "data/processed/test_temporal.csv"
)


# ============================================================
# FEATURE RANKING
# ============================================================

importance_path = os.path.join(
    IMPORTANCE_DIR,
    "combined_feature_importance.csv"
)

importance_df = pd.read_csv(
    importance_path
)

importance_df = importance_df.sort_values(
    "mean_importance",
    ascending=False
).reset_index(drop=True)

ranked_features = importance_df["feature"].tolist()

all_feature_names = [
    c for c in train_df.columns
    if c.startswith("feature_")
]

feature_to_index = {
    feature: i
    for i, feature in enumerate(all_feature_names)
}


# ============================================================
# VERIFY FEATURE COUNT
# ============================================================

print("\nFeature information:")

print(
    f"Total transaction features: "
    f"{len(all_feature_names)}"
)

print(
    f"Top feature: "
    f"{ranked_features[0]}"
)


# ============================================================
# GRAPH INFORMATION
# ============================================================

print("\nGraph information:")

print(
    f"Train nodes: "
    f"{train_graph.x.shape[0]:,}"
)

print(
    f"Validation nodes: "
    f"{val_graph.x.shape[0]:,}"
)

print(
    f"Test nodes: "
    f"{test_graph.x.shape[0]:,}"
)


# ============================================================
# CLASS LABEL CONVERSION
# ============================================================
#
# graph objects use:
#
# -1 = unknown
#  0 = licit
#  1 = illicit
#
# We train/evaluate only on labeled nodes.
# ============================================================

train_labeled = train_graph.y >= 0
val_labeled = val_graph.y >= 0
test_labeled = test_graph.y >= 0

y_train = (
    train_graph.y[
        train_labeled
    ]
    .cpu()
    .numpy()
    .astype(int)
)

y_val = (
    val_graph.y[
        val_labeled
    ]
    .cpu()
    .numpy()
    .astype(int)
)

y_test = (
    test_graph.y[
        test_labeled
    ]
    .cpu()
    .numpy()
    .astype(int)
)


print("\nLabeled nodes:")

print(
    f"Train      : {len(y_train):,}"
)

print(
    f"Validation : {len(y_val):,}"
)

print(
    f"Test       : {len(y_test):,}"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_licit = np.sum(y_train == 0)
train_illicit = np.sum(y_train == 1)

class_weights = torch.tensor(
    [
        len(y_train) /
        (2 * train_licit),

        len(y_train) /
        (2 * train_illicit)
    ],
    dtype=torch.float32,
    device=DEVICE
)

print("\nClass weights:")

print(
    class_weights.detach()
    .cpu()
    .numpy()
)


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

# Graph feature matrix contains:
# 165 transaction features
#
# Fit scaler using TRAIN graph only.

train_x_full = train_graph.x.cpu().numpy()
val_x_full = val_graph.x.cpu().numpy()
test_x_full = test_graph.x.cpu().numpy()

scaler = StandardScaler()

train_x_scaled = scaler.fit_transform(
    train_x_full
)

val_x_scaled = scaler.transform(
    val_x_full
)

test_x_scaled = scaler.transform(
    test_x_full
)

print(
    "\n[PASS] Feature scaler fitted "
    "on training graph only."
)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    y_true,
    probabilities
):

    thresholds = np.linspace(
        0.01,
        0.99,
        99
    )

    best_threshold = 0.50
    best_f1 = -1

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    return (
        best_threshold,
        best_f1
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

    return {
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities
        ),

        "pr_auc": average_precision_score(
            y_true,
            probabilities
        )
    }


# ============================================================
# MODEL DEFINITIONS
# ============================================================

class GCNModel(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=64
    ):

        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = GCNConv(
            hidden_dim,
            2
        )

        self.dropout = DROPOUT

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

        return x


class GraphSAGEModel(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=64
    ):

        super().__init__()

        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            2
        )

        self.dropout = DROPOUT

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

        return x


class GATModel(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        heads=4
    ):

        super().__init__()

        self.conv1 = GATConv(
            input_dim,
            hidden_dim,
            heads=heads,
            dropout=DROPOUT
        )

        self.conv2 = GATConv(
            hidden_dim * heads,
            2,
            heads=1,
            concat=False,
            dropout=DROPOUT
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.elu(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x


# ============================================================
# TRAIN ONE GNN
# ============================================================

def train_gnn(
    model,
    train_x,
    val_x,
    test_x,
    train_graph,
    val_graph,
    test_graph,
    model_name,
    learning_rate
):

    model = model.to(DEVICE)

    train_x = torch.tensor(
        train_x,
        dtype=torch.float32,
        device=DEVICE
    )

    val_x = torch.tensor(
        val_x,
        dtype=torch.float32,
        device=DEVICE
    )

    test_x = torch.tensor(
        test_x,
        dtype=torch.float32,
        device=DEVICE
    )

    train_edge_index = (
        train_graph.edge_index
        .to(DEVICE)
    )

    val_edge_index = (
        val_graph.edge_index
        .to(DEVICE)
    )

    test_edge_index = (
        test_graph.edge_index
        .to(DEVICE)
    )

    train_y = (
        train_graph.y
        .to(DEVICE)
    )

    val_y = (
        val_graph.y
        .to(DEVICE)
    )

    test_y = (
        test_graph.y
        .to(DEVICE)
    )

    train_mask = (
        train_y >= 0
    )

    val_mask = (
        val_y >= 0
    )

    test_mask = (
        test_y >= 0
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    best_state = None
    best_pr_auc = -1
    best_epoch = 0
    patience_counter = 0

    print(
        f"\nTraining {model_name}..."
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        optimizer.zero_grad()

        train_logits = model(
            train_x,
            train_edge_index
        )

        loss = criterion(
            train_logits[train_mask],
            train_y[train_mask]
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_logits = model(
                val_x,
                val_edge_index
            )

            val_prob = torch.softmax(
                val_logits,
                dim=1
            )[:, 1]

            val_prob_np = (
                val_prob[val_mask]
                .detach()
                .cpu()
                .numpy()
            )

        val_pr_auc = average_precision_score(
            y_val,
            val_prob_np
        )

        if val_pr_auc > best_pr_auc:

            best_pr_auc = val_pr_auc

            best_epoch = epoch

            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v in model.state_dict().items()
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
                f"Early stopping at epoch "
                f"{epoch}"
            )

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    model.eval()

    with torch.no_grad():

        val_logits = model(
            val_x,
            val_edge_index
        )

        test_logits = model(
            test_x,
            test_edge_index
        )

        val_prob = torch.softmax(
            val_logits,
            dim=1
        )[:, 1]

        test_prob = torch.softmax(
            test_logits,
            dim=1
        )[:, 1]

    val_prob = (
        val_prob[val_mask]
        .detach()
        .cpu()
        .numpy()
    )

    test_prob = (
        test_prob[test_mask]
        .detach()
        .cpu()
        .numpy()
    )

    threshold, val_f1 = (
        find_best_threshold(
            y_val,
            val_prob
        )
    )

    test_metrics = calculate_metrics(
        y_test,
        test_prob,
        threshold
    )

    print(
        f"\n{model_name}"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation PR-AUC: "
        f"{best_pr_auc:.4f}"
    )

    print(
        f"Selected threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{val_f1:.4f}"
    )

    print(
        f"Test Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Test Recall: "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1: "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"Test PR-AUC: "
        f"{test_metrics['pr_auc']:.4f}"
    )

    return {
        "threshold": threshold,
        "val_f1": val_f1,
        "val_pr_auc": best_pr_auc,
        **test_metrics
    }


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# RUN ABLATION
# ============================================================

for k in TOP_K_VALUES:

    print("\n")
    print("=" * 70)
    print(f"FEATURE SET: TOP {k}")
    print("=" * 70)

    selected_features = (
        ranked_features[:k]
    )

    selected_indices = [
        feature_to_index[f]
        for f in selected_features
    ]

    train_x = train_x_scaled[
        :,
        selected_indices
    ]

    val_x = val_x_scaled[
        :,
        selected_indices
    ]

    test_x = test_x_scaled[
        :,
        selected_indices
    ]

    print(
        f"Using {k} transaction features"
    )


    # ========================================================
    # GCN
    # ========================================================

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    gcn = GCNModel(
        input_dim=k,
        hidden_dim=HIDDEN_GCN
    )

    gcn_result = train_gnn(
        gcn,
        train_x,
        val_x,
        test_x,
        train_graph,
        val_graph,
        test_graph,
        f"GCN Top-{k}",
        LR_GCN
    )

    results.append({
        "model": "GCN",
        "top_k": k,
        **gcn_result
    })


    # ========================================================
    # GRAPHSAGE
    # ========================================================

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    sage = GraphSAGEModel(
        input_dim=k,
        hidden_dim=HIDDEN_SAGE
    )

    sage_result = train_gnn(
        sage,
        train_x,
        val_x,
        test_x,
        train_graph,
        val_graph,
        test_graph,
        f"GraphSAGE Top-{k}",
        LR_SAGE
    )

    results.append({
        "model": "GraphSAGE",
        "top_k": k,
        **sage_result
    })


    # ========================================================
    # GAT
    # ========================================================

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    gat = GATModel(
        input_dim=k,
        hidden_dim=HIDDEN_GAT,
        heads=GAT_HEADS
    )

    gat_result = train_gnn(
        gat,
        train_x,
        val_x,
        test_x,
        train_graph,
        val_graph,
        test_graph,
        f"GAT Top-{k}",
        LR_GAT
    )

    results.append({
        "model": "GAT",
        "top_k": k,
        **gat_result
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

output_path = os.path.join(
    RESULTS_DIR,
    "gnn_feature_ablation.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL GNN FEATURE ABLATION RESULTS")
print("=" * 70)

display_cols = [
    "model",
    "top_k",
    "val_pr_auc",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc"
]

print(
    results_df[
        display_cols
    ].to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# BEST PR-AUC
# ============================================================

print("\n")
print("=" * 70)
print("BEST TEST PR-AUC BY MODEL")
print("=" * 70)

for model_name in [
    "GCN",
    "GraphSAGE",
    "GAT"
]:

    subset = results_df[
        results_df["model"] == model_name
    ]

    best = subset.loc[
        subset["pr_auc"].idxmax()
    ]

    print(
        f"{model_name}: "
        f"Top-{int(best['top_k'])} | "
        f"PR-AUC="
        f"{best['pr_auc']:.4f}"
    )


# ============================================================
# FILE SAVED
# ============================================================

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(output_path)

print("\n")
print("=" * 70)
print("GNN FEATURE ABLATION COMPLETE")
print("=" * 70)