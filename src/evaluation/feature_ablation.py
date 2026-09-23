import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

import xgboost as xgb


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TOP_K_VALUES = [10, 20, 30, 50, 165]

RESULTS_DIR = "results/feature_ablation"
IMPORTANCE_DIR = "results/feature_importance"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("GRAPHSHIELD - TOP-K FEATURE ABLATION")
print("=" * 70)

train = pd.read_csv(
    "data/processed/train_temporal.csv"
)

val = pd.read_csv(
    "data/processed/validation_temporal.csv"
)

test = pd.read_csv(
    "data/processed/test_temporal.csv"
)


# ============================================================
# FEATURE COLUMNS
# ============================================================

feature_cols = [
    c for c in train.columns
    if c.startswith("feature_")
]


print("\nDataset sizes:")
print(f"Train      : {len(train):,}")
print(f"Validation : {len(val):,}")
print(f"Test       : {len(test):,}")
print(f"Features   : {len(feature_cols)}")


# ============================================================
# CONVERT ELLIPTIC LABELS
# ============================================================
#
# Original processed dataset encoding:
#
# 1 = illicit
# 2 = licit
#
# Convert to:
#
# 1 = illicit
# 0 = licit
#
# ============================================================

label_mapping = {
    1: 1,   # illicit
    2: 0    # licit
}


def convert_labels(df, name):

    unique_labels = sorted(
        df["class"].unique()
    )

    print(
        f"\n{name} original labels: "
        f"{unique_labels}"
    )

    invalid = [
        x for x in unique_labels
        if x not in label_mapping
    ]

    if invalid:
        raise ValueError(
            f"{name} contains unexpected class values: "
            f"{invalid}"
        )

    y = df["class"].map(label_mapping).astype(int)

    return y.values


y_train = convert_labels(train, "Train")
y_val = convert_labels(val, "Validation")
y_test = convert_labels(test, "Test")


# ============================================================
# VERIFY CLASS DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("CONVERTED CLASS DISTRIBUTION")
print("=" * 70)

print(
    f"Train      -> "
    f"Licit: {(y_train == 0).sum():,} | "
    f"Illicit: {(y_train == 1).sum():,}"
)

print(
    f"Validation -> "
    f"Licit: {(y_val == 0).sum():,} | "
    f"Illicit: {(y_val == 1).sum():,}"
)

print(
    f"Test       -> "
    f"Licit: {(y_test == 0).sum():,} | "
    f"Illicit: {(y_test == 1).sum():,}"
)


# ============================================================
# EXTRACT FEATURES
# ============================================================

X_train_full = train[feature_cols].values
X_val_full = val[feature_cols].values
X_test_full = test[feature_cols].values


# ============================================================
# LOAD FEATURE IMPORTANCE RANKING
# ============================================================

combined_path = os.path.join(
    IMPORTANCE_DIR,
    "combined_feature_importance.csv"
)

importance_df = pd.read_csv(
    combined_path
)

importance_df = importance_df.sort_values(
    "mean_importance",
    ascending=False
).reset_index(drop=True)


ranked_features = (
    importance_df["feature"]
    .tolist()
)


print("\n")
print("=" * 70)
print("FEATURE RANKING")
print("=" * 70)

print("\nTop 10 features:")

for i, feature in enumerate(
    ranked_features[:10],
    start=1
):
    print(
        f"{i:2d}. {feature}"
    )


# ============================================================
# SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_full
)

X_val_scaled = scaler.transform(
    X_val_full
)

X_test_scaled = scaler.transform(
    X_test_full
)

print("\n[PASS] Scaler fitted on training data only.")


# ============================================================
# THRESHOLD SELECTION
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

        f1 = f1_score(
            y_true,
            predictions,
            pos_label=1,
            average="binary",
            zero_division=0
        )

        if f1 > best_f1:

            best_f1 = f1
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
            pos_label=1,
            average="binary",
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            predictions,
            pos_label=1,
            average="binary",
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            predictions,
            pos_label=1,
            average="binary",
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
# RESULTS
# ============================================================

results = []


# ============================================================
# ABLATION EXPERIMENT
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
        feature_cols.index(feature)
        for feature in selected_features
    ]

    Xtr = X_train_scaled[
        :,
        selected_indices
    ]

    Xv = X_val_scaled[
        :,
        selected_indices
    ]

    Xte = X_test_scaled[
        :,
        selected_indices
    ]

    print(
        f"Using {len(selected_features)} features"
    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    print("\nTraining Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf.fit(
        Xtr,
        y_train
    )

    rf_val_prob = rf.predict_proba(
        Xv
    )[:, 1]

    rf_threshold, rf_val_f1 = (
        find_best_threshold(
            y_val,
            rf_val_prob
        )
    )

    rf_test_prob = rf.predict_proba(
        Xte
    )[:, 1]

    rf_metrics = calculate_metrics(
        y_test,
        rf_test_prob,
        rf_threshold
    )

    print(
        f"RF threshold: "
        f"{rf_threshold:.2f} | "
        f"Val F1: "
        f"{rf_val_f1:.4f}"
    )

    print(
        f"RF Test | "
        f"Precision="
        f"{rf_metrics['precision']:.4f} | "
        f"Recall="
        f"{rf_metrics['recall']:.4f} | "
        f"F1="
        f"{rf_metrics['f1']:.4f} | "
        f"ROC-AUC="
        f"{rf_metrics['roc_auc']:.4f} | "
        f"PR-AUC="
        f"{rf_metrics['pr_auc']:.4f}"
    )

    results.append({

        "model": "Random Forest",

        "top_k": k,

        "threshold": rf_threshold,

        **rf_metrics
    })


    # ========================================================
    # XGBOOST
    # ========================================================

    print("\nTraining XGBoost...")

    xgb_model = xgb.XGBClassifier(

        n_estimators=300,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="binary:logistic",

        eval_metric="logloss",

        random_state=RANDOM_STATE,

        n_jobs=1,

        tree_method="hist"
    )

    xgb_model.fit(
        Xtr,
        y_train
    )

    xgb_val_prob = (
        xgb_model
        .predict_proba(Xv)[:, 1]
    )

    xgb_threshold, xgb_val_f1 = (
        find_best_threshold(
            y_val,
            xgb_val_prob
        )
    )

    xgb_test_prob = (
        xgb_model
        .predict_proba(Xte)[:, 1]
    )

    xgb_metrics = calculate_metrics(
        y_test,
        xgb_test_prob,
        xgb_threshold
    )

    print(
        f"XGB threshold: "
        f"{xgb_threshold:.2f} | "
        f"Val F1: "
        f"{xgb_val_f1:.4f}"
    )

    print(
        f"XGB Test | "
        f"Precision="
        f"{xgb_metrics['precision']:.4f} | "
        f"Recall="
        f"{xgb_metrics['recall']:.4f} | "
        f"F1="
        f"{xgb_metrics['f1']:.4f} | "
        f"ROC-AUC="
        f"{xgb_metrics['roc_auc']:.4f} | "
        f"PR-AUC="
        f"{xgb_metrics['pr_auc']:.4f}"
    )

    results.append({

        "model": "XGBoost",

        "top_k": k,

        "threshold": xgb_threshold,

        **xgb_metrics
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

output_path = os.path.join(
    RESULTS_DIR,
    "top_k_feature_ablation.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# FINAL RESULTS TABLE
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TOP-K ABLATION RESULTS")
print("=" * 70)

display_cols = [
    "model",
    "top_k",
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
print("BEST PR-AUC BY MODEL")
print("=" * 70)

for model in [
    "Random Forest",
    "XGBoost"
]:

    subset = results_df[
        results_df["model"] == model
    ]

    if len(subset) == 0:

        print(
            f"{model}: "
            f"No successful result"
        )

        continue

    best = subset.loc[
        subset["pr_auc"].idxmax()
    ]

    print(
        f"{model}: "
        f"Top-{int(best['top_k'])} | "
        f"PR-AUC="
        f"{best['pr_auc']:.4f}"
    )


# ============================================================
# FILES
# ============================================================

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(output_path)

print("\n")
print("=" * 70)
print("TOP-K FEATURE ABLATION COMPLETE")
print("=" * 70)