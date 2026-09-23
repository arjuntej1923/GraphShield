import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from torch_geometric.nn import GCNConv, SAGEConv, GATConv


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

OUTPUT_DIR = "results/computational_efficiency"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")

else:
    DEVICE = torch.device("cpu")


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("GRAPHSHIELD - COMPUTATIONAL EFFICIENCY ANALYSIS")
print("=" * 70)

print(f"\nDevice: {DEVICE}")


# ============================================================
# LOAD TABULAR DATA
# ============================================================

train = pd.read_csv(
    "data/processed/train_temporal.csv"
)

test = pd.read_csv(
    "data/processed/test_temporal.csv"
)


feature_cols = [
    c
    for c in train.columns
    if c.startswith("feature_")
]


X_train = train[
    feature_cols
].values

X_test = test[
    feature_cols
].values


# ============================================================
# LABEL CONVERSION
# ============================================================
#
# Dataset:
# 1 = illicit
# 2 = licit
#
# Model:
# 1 = illicit
# 0 = licit
# ============================================================

y_train = (
    train["class"]
    .map({
        1: 1,
        2: 0
    })
    .values
    .astype(int)
)

y_test = (
    test["class"]
    .map({
        1: 1,
        2: 0
    })
    .values
    .astype(int)
)


# ============================================================
# SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# LOAD GRAPH DATA
# ============================================================

train_graph = torch.load(
    "data/processed/graph_train.pt",
    map_location="cpu",
    weights_only=False
)

test_graph = torch.load(
    "data/processed/graph_test.pt",
    map_location="cpu",
    weights_only=False
)


train_mask = (
    train_graph.y >= 0
)

test_mask = (
    test_graph.y >= 0
)


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# DEVICE SYNCHRONIZATION
# ============================================================

def sync_device():

    if DEVICE.type == "mps":

        torch.mps.synchronize()

    elif DEVICE.type == "cuda":

        torch.cuda.synchronize()


def elapsed(start):

    sync_device()

    return (
        time.perf_counter()
        - start
    )


# ============================================================
# TRADITIONAL ML
# ============================================================

print("\n")
print("=" * 70)
print("TRADITIONAL ML MODELS")
print("=" * 70)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")

lr_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=SEED
)

start = time.perf_counter()

lr_model.fit(
    X_train_scaled,
    y_train
)

training_time = (
    time.perf_counter()
    - start
)


start = time.perf_counter()

_ = lr_model.predict_proba(
    X_test_scaled
)

inference_time = (
    time.perf_counter()
    - start
)


parameter_count = (
    lr_model.coef_.size
    +
    lr_model.intercept_.size
)


results.append({

    "model":
        "Logistic Regression",

    "family":
        "Traditional ML",

    "parameters":
        int(parameter_count),

    "training_time_sec":
        training_time,

    "test_inference_time_sec":
        inference_time,

    "test_inference_ms_per_sample":
        inference_time /
        len(X_test) *
        1000
})


print(
    f"Training: "
    f"{training_time:.3f}s | "
    f"Inference: "
    f"{inference_time:.3f}s | "
    f"Parameters: "
    f"{parameter_count:,}"
)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(

    n_estimators=300,

    class_weight="balanced",

    random_state=SEED,

    n_jobs=-1
)


start = time.perf_counter()

rf_model.fit(
    X_train_scaled,
    y_train
)

training_time = (
    time.perf_counter()
    - start
)


start = time.perf_counter()

_ = rf_model.predict_proba(
    X_test_scaled
)

inference_time = (
    time.perf_counter()
    - start
)


tree_nodes = sum(
    tree.tree_.node_count
    for tree in rf_model.estimators_
)


results.append({

    "model":
        "Random Forest",

    "family":
        "Traditional ML",

    "parameters":
        int(tree_nodes),

    "training_time_sec":
        training_time,

    "test_inference_time_sec":
        inference_time,

    "test_inference_ms_per_sample":
        inference_time /
        len(X_test) *
        1000
})


print(
    f"Training: "
    f"{training_time:.3f}s | "
    f"Inference: "
    f"{inference_time:.3f}s | "
    f"Tree nodes: "
    f"{tree_nodes:,}"
)


# ============================================================
# XGBOOST
# ============================================================
#
# IMPORTANT:
# XGBoost previously caused a native segmentation fault
# on this macOS environment.
#
# We therefore DO NOT execute XGBoost again here.
# ============================================================

print("\nXGBoost...")

print(
    "[SKIPPED] XGBoost has been excluded "
    "from the timing benchmark because "
    "it previously caused a native "
    "segmentation fault on this environment."
)


results.append({

    "model":
        "XGBoost",

    "family":
        "Traditional ML",

    "parameters":
        np.nan,

    "training_time_sec":
        np.nan,

    "test_inference_time_sec":
        np.nan,

    "test_inference_ms_per_sample":
        np.nan
})


# ============================================================
# GNN MODEL DEFINITIONS
# ============================================================


class GCN(nn.Module):

    def __init__(
        self,
        input_dim,
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
            p=0.3,
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
        input_dim,
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
            p=0.3,
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
        input_dim,
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
# GRAPH DATA
# ============================================================

train_x = (
    train_graph.x
    .float()
    .to(DEVICE)
)

test_x = (
    test_graph.x
    .float()
    .to(DEVICE)
)

train_edges = (
    train_graph.edge_index
    .to(DEVICE)
)

test_edges = (
    test_graph.edge_index
    .to(DEVICE)
)

train_y = (
    train_graph.y
    .to(DEVICE)
)


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


criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# GNN BENCHMARK
# ============================================================

def benchmark_gnn(
    model,
    model_name,
    learning_rate
):

    model = model.to(
        DEVICE
    )

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=learning_rate,

        weight_decay=5e-4
    )


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    sync_device()

    start = time.perf_counter()


    for epoch in range(100):

        optimizer.zero_grad()

        logits = model(
            train_x,
            train_edges
        )

        loss = criterion(
            logits[
                train_mask.to(DEVICE)
            ],
            train_y[
                train_mask.to(DEVICE)
            ]
        )

        loss.backward()

        optimizer.step()


    training_time = elapsed(
        start
    )


    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        sync_device()

        start = time.perf_counter()

        logits = model(
            test_x,
            test_edges
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=1
            )[:, 1]
        )

        _ = probabilities[
            test_mask.to(DEVICE)
        ]

        inference_time = elapsed(
            start
        )


    parameter_count = sum(

        p.numel()

        for p in model.parameters()
    )


    print(
        f"\n{model_name}"
    )

    print(
        f"Training: "
        f"{training_time:.3f}s"
    )

    print(
        f"Inference: "
        f"{inference_time:.3f}s"
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )


    results.append({

        "model":
            model_name,

        "family":
            "GNN",

        "parameters":
            int(parameter_count),

        "training_time_sec":
            training_time,

        "test_inference_time_sec":
            inference_time,

        "test_inference_ms_per_sample":
            inference_time /
            len(y_test) *
            1000
    })


# ============================================================
# GCN
# ============================================================

print("\n")
print("=" * 70)
print("GRAPH NEURAL NETWORKS")
print("=" * 70)

torch.manual_seed(SEED)

gcn = GCN(
    input_dim=165,
    hidden=64
)

benchmark_gnn(
    gcn,
    "GCN",
    0.01
)


# ============================================================
# GRAPHSAGE
# ============================================================

torch.manual_seed(SEED)

sage = GraphSAGE(
    input_dim=165,
    hidden=64
)

benchmark_gnn(
    sage,
    "GraphSAGE",
    0.01
)


# ============================================================
# GAT
# ============================================================

torch.manual_seed(SEED)

gat = GAT(
    input_dim=165,
    hidden=32,
    heads=4
)

benchmark_gnn(
    gat,
    "GAT",
    0.005
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


output_path = os.path.join(

    OUTPUT_DIR,

    "computational_efficiency.csv"
)


results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# FINAL TABLE
# ============================================================

print("\n")
print("=" * 70)
print("COMPUTATIONAL EFFICIENCY RESULTS")
print("=" * 70)


print(
    results_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# FILE
# ============================================================

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    output_path
)


print("\n")
print("=" * 70)
print("COMPUTATIONAL EFFICIENCY ANALYSIS COMPLETE")
print("=" * 70)