import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from torch_geometric.nn import GCNConv, SAGEConv, GATConv


# ============================================================
# CONFIGURATION
# ============================================================

SEEDS = [42, 123, 2024, 7, 99]

OUTPUT_DIR = "results/robustness"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print("=" * 70)
print("GRAPHSHIELD - MULTI-SEED ROBUSTNESS ANALYSIS")
print("=" * 70)

print(f"\nDevice: {DEVICE}")
print(f"Seeds: {SEEDS}")


# ============================================================
# LOAD TEMPORAL TABULAR DATA
# ============================================================

train = pd.read_csv(
    "data/processed/train_temporal.csv"
)

validation = pd.read_csv(
    "data/processed/validation_temporal.csv"
)

test = pd.read_csv(
    "data/processed/test_temporal.csv"
)


feature_cols = [
    c
    for c in train.columns
    if c.startswith("feature_")
]


# ============================================================
# LABEL CONVERSION
# 1 = illicit
# 2 = licit
#
# Model representation:
# 1 = illicit
# 0 = licit
# ============================================================

def convert_labels(df):

    return (
        df["class"]
        .map({
            1: 1,
            2: 0
        })
        .values
        .astype(int)
    )


X_train = train[feature_cols].values
X_val = validation[feature_cols].values
X_test = test[feature_cols].values

y_train = convert_labels(train)
y_val = convert_labels(validation)
y_test = convert_labels(test)


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# LOAD GRAPHS
# ============================================================

train_graph = torch.load(
    "data/processed/graph_train.pt",
    map_location="cpu",
    weights_only=False
)

validation_graph = torch.load(
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
# GRAPH DATA
# ============================================================

train_x = train_graph.x.float().to(DEVICE)
val_x = validation_graph.x.float().to(DEVICE)
test_x = test_graph.x.float().to(DEVICE)

train_edges = train_graph.edge_index.to(DEVICE)
val_edges = validation_graph.edge_index.to(DEVICE)
test_edges = test_graph.edge_index.to(DEVICE)

train_y = train_graph.y.to(DEVICE)
val_y = validation_graph.y.to(DEVICE)
test_y = test_graph.y.to(DEVICE)

train_mask = (
    train_graph.y >= 0
).to(DEVICE)

val_mask = (
    validation_graph.y >= 0
).to(DEVICE)

test_mask = (
    test_graph.y >= 0
).to(DEVICE)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weights = torch.tensor(
    [
        0.5654888,
        4.3174467
    ],
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def set_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_device():

    if DEVICE.type == "mps":
        torch.mps.synchronize()

    elif DEVICE.type == "cuda":
        torch.cuda.synchronize()


def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5
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
# THRESHOLD SELECTION
# ============================================================

def select_threshold(
    y_true,
    probabilities
):

    thresholds = np.linspace(
        0.01,
        0.99,
        99
    )

    best_threshold = 0.5
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
# GNN ARCHITECTURES
# ============================================================

class GCN(nn.Module):

    def __init__(
        self,
        input_dim=165,
        hidden=64
    ):

        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            hidden
        )

        self.conv2 = GCNConv(
            hidden,
            2
        )

        self.dropout = 0.3

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

        return x


class GraphSAGE(nn.Module):

    def __init__(
        self,
        input_dim=165,
        hidden=64
    ):

        super().__init__()

        self.conv1 = SAGEConv(
            input_dim,
            hidden
        )

        self.conv2 = SAGEConv(
            hidden,
            2
        )

        self.dropout = 0.3

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

        return x


class GAT(nn.Module):

    def __init__(
        self,
        input_dim=165,
        hidden=32,
        heads=4
    ):

        super().__init__()

        self.conv1 = GATConv(
            input_dim,
            hidden,
            heads=heads,
            dropout=0.3
        )

        self.conv2 = GATConv(
            hidden * heads,
            2,
            heads=1,
            concat=False,
            dropout=0.3
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

        x = torch.nn.functional.elu(
            x
        )

        x = self.conv2(
            x,
            edge_index
        )

        return x


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n")
print("=" * 70)
print("RANDOM FOREST ROBUSTNESS")
print("=" * 70)


for seed in SEEDS:

    print(
        f"\nSeed {seed}"
    )

    set_seed(seed)

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1
    )

    start = time.perf_counter()

    model.fit(
        X_train_scaled,
        y_train
    )

    training_time = (
        time.perf_counter()
        - start
    )

    val_prob = model.predict_proba(
        X_val_scaled
    )[:, 1]

    test_prob = model.predict_proba(
        X_test_scaled
    )[:, 1]

    threshold, val_f1 = select_threshold(
        y_val,
        val_prob
    )

    metrics = calculate_metrics(
        y_test,
        test_prob,
        threshold
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"PR-AUC: {metrics['pr_auc']:.4f} | "
        f"F1: {metrics['f1']:.4f}"
    )

    results.append({

        "model":
            "Random Forest",

        "seed":
            seed,

        "threshold":
            threshold,

        "precision":
            metrics["precision"],

        "recall":
            metrics["recall"],

        "f1":
            metrics["f1"],

        "roc_auc":
            metrics["roc_auc"],

        "pr_auc":
            metrics["pr_auc"],

        "training_time_sec":
            training_time
    })


# ============================================================
# GNN TRAINING FUNCTION
# ============================================================

def run_gnn(
    model,
    model_name,
    seed,
    learning_rate
):

    set_seed(seed)

    model = model.to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=5e-4
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    best_state = None

    best_val_pr = -1

    best_epoch = 0

    patience = 15

    patience_counter = 0

    max_epochs = 100


    sync_device()

    start = time.perf_counter()


    for epoch in range(
        1,
        max_epochs + 1
    ):

        model.train()

        optimizer.zero_grad()

        logits = model(
            train_x,
            train_edges
        )

        loss = criterion(
            logits[train_mask],
            train_y[train_mask]
        )

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_logits = model(
                val_x,
                val_edges
            )

            val_prob = torch.softmax(
                val_logits,
                dim=1
            )[:, 1]

            val_prob_labeled = (
                val_prob[val_mask]
                .detach()
                .cpu()
                .numpy()
            )


        val_pr = average_precision_score(
            y_val,
            val_prob_labeled
        )


        if val_pr > best_val_pr:

            best_val_pr = val_pr

            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1


        if patience_counter >= patience:

            break


    training_time = (
        time.perf_counter()
        - start
    )


    # --------------------------------------------------------
    # RESTORE BEST MODEL
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    model = model.to(
        DEVICE
    )

    model.eval()


    # --------------------------------------------------------
    # VALIDATION PREDICTIONS
    # --------------------------------------------------------

    with torch.no_grad():

        val_logits = model(
            val_x,
            val_edges
        )

        val_prob = torch.softmax(
            val_logits,
            dim=1
        )[:, 1]

        val_prob_labeled = (
            val_prob[val_mask]
            .detach()
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # TEST PREDICTIONS
        # ----------------------------------------------------

        test_logits = model(
            test_x,
            test_edges
        )

        test_prob = torch.softmax(
            test_logits,
            dim=1
        )[:, 1]

        test_prob_labeled = (
            test_prob[test_mask]
            .detach()
            .cpu()
            .numpy()
        )


    threshold, val_f1 = select_threshold(
        y_val,
        val_prob_labeled
    )


    metrics = calculate_metrics(
        y_test,
        test_prob_labeled,
        threshold
    )


    print(
        f"Seed {seed} | "
        f"Best epoch {best_epoch} | "
        f"Val PR-AUC {best_val_pr:.4f} | "
        f"Test PR-AUC {metrics['pr_auc']:.4f} | "
        f"Test F1 {metrics['f1']:.4f}"
    )


    results.append({

        "model":
            model_name,

        "seed":
            seed,

        "threshold":
            threshold,

        "precision":
            metrics["precision"],

        "recall":
            metrics["recall"],

        "f1":
            metrics["f1"],

        "roc_auc":
            metrics["roc_auc"],

        "pr_auc":
            metrics["pr_auc"],

        "training_time_sec":
            training_time,

        "best_epoch":
            best_epoch,

        "validation_pr_auc":
            best_val_pr
    })


# ============================================================
# GCN
# ============================================================

print("\n")
print("=" * 70)
print("GCN ROBUSTNESS")
print("=" * 70)


for seed in SEEDS:

    print(
        f"\nRunning GCN - seed {seed}"
    )

    model = GCN(
        input_dim=165,
        hidden=64
    )

    run_gnn(
        model,
        "GCN",
        seed,
        0.01
    )


# ============================================================
# GRAPHSAGE
# ============================================================

print("\n")
print("=" * 70)
print("GRAPHSAGE ROBUSTNESS")
print("=" * 70)


for seed in SEEDS:

    print(
        f"\nRunning GraphSAGE - seed {seed}"
    )

    model = GraphSAGE(
        input_dim=165,
        hidden=64
    )

    run_gnn(
        model,
        "GraphSAGE",
        seed,
        0.01
    )


# ============================================================
# GAT
# ============================================================

print("\n")
print("=" * 70)
print("GAT ROBUSTNESS")
print("=" * 70)


for seed in SEEDS:

    print(
        f"\nRunning GAT - seed {seed}"
    )

    model = GAT(
        input_dim=165,
        hidden=32,
        heads=4
    )

    run_gnn(
        model,
        "GAT",
        seed,
        0.005
    )


# ============================================================
# SAVE INDIVIDUAL RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


raw_path = (
    f"{OUTPUT_DIR}/"
    "robustness_per_seed.csv"
)


results_df.to_csv(
    raw_path,
    index=False
)


# ============================================================
# AGGREGATED RESULTS
# ============================================================

metric_columns = [
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc"
]


summary = (
    results_df
    .groupby("model")[metric_columns]
    .agg(["mean", "std"])
)


summary_path = (
    f"{OUTPUT_DIR}/"
    "robustness_summary.csv"
)


summary.to_csv(
    summary_path
)


# ============================================================
# CLEAN SUMMARY TABLE
# ============================================================

summary_rows = []


for model_name in (
    results_df["model"].unique()
):

    subset = results_df[
        results_df["model"]
        == model_name
    ]

    row = {
        "model":
            model_name
    }

    for metric in metric_columns:

        row[
            f"{metric}_mean"
        ] = subset[
            metric
        ].mean()

        row[
            f"{metric}_std"
        ] = subset[
            metric
        ].std()

    summary_rows.append(
        row
    )


clean_summary = pd.DataFrame(
    summary_rows
)


clean_path = (
    f"{OUTPUT_DIR}/"
    "robustness_clean_summary.csv"
)


clean_summary.to_csv(
    clean_path,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("ROBUSTNESS SUMMARY")
print("=" * 70)


for _, row in clean_summary.iterrows():

    print(
        f"\n{row['model']}"
    )

    print(
        f"Precision: "
        f"{row['precision_mean']:.4f} "
        f"± "
        f"{row['precision_std']:.4f}"
    )

    print(
        f"Recall: "
        f"{row['recall_mean']:.4f} "
        f"± "
        f"{row['recall_std']:.4f}"
    )

    print(
        f"F1: "
        f"{row['f1_mean']:.4f} "
        f"± "
        f"{row['f1_std']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{row['roc_auc_mean']:.4f} "
        f"± "
        f"{row['roc_auc_std']:.4f}"
    )

    print(
        f"PR-AUC: "
        f"{row['pr_auc_mean']:.4f} "
        f"± "
        f"{row['pr_auc_std']:.4f}"
    )


# ============================================================
# FILES
# ============================================================

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    raw_path
)

print(
    summary_path
)

print(
    clean_path
)


print("\n")
print("=" * 70)
print("MULTI-SEED ROBUSTNESS ANALYSIS COMPLETE")
print("=" * 70)