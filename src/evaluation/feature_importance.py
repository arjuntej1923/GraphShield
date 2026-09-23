from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ============================================================
# GRAPHSHIELD - FEATURE IMPORTANCE ANALYSIS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "feature_importance"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_FILE = DATA_DIR / "train_temporal.csv"
VAL_FILE = DATA_DIR / "validation_temporal.csv"
TEST_FILE = DATA_DIR / "test_temporal.csv"


RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("GRAPHSHIELD - FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VAL_FILE)
test = pd.read_csv(TEST_FILE)

print("\nDataset sizes:")
print(f"Train      : {len(train):,}")
print(f"Validation : {len(validation):,}")
print(f"Test       : {len(test):,}")


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [
    f"feature_{i}"
    for i in range(1, 166)
]

print(
    f"\nTransaction features: {len(FEATURE_COLUMNS)}"
)


X_train = train[FEATURE_COLUMNS].copy()
X_val = validation[FEATURE_COLUMNS].copy()
X_test = test[FEATURE_COLUMNS].copy()


y_train = (
    train["class"].astype(str) == "1"
).astype(int).to_numpy()

y_val = (
    validation["class"].astype(str) == "1"
).astype(int).to_numpy()

y_test = (
    test["class"].astype(str) == "1"
).astype(int).to_numpy()


# ============================================================
# NORMALIZATION
#
# Same preprocessing used by the original baseline.
# RF does not require scaling, but we keep the data pipeline
# explicit and use raw features for tree importance.
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)

print(
    "[PASS] Scaler fitted on training data only."
)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

rf.fit(
    X_train,
    y_train,
)

rf_importance = rf.feature_importances_


rf_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "random_forest_importance": rf_importance,
})


rf_df = rf_df.sort_values(
    "random_forest_importance",
    ascending=False,
).reset_index(drop=True)


rf_df["random_forest_rank"] = (
    np.arange(len(rf_df)) + 1
)


rf_df.to_csv(
    RESULTS_DIR / "random_forest_feature_importance.csv",
    index=False,
)


print("\nTop 20 Random Forest features:")

print(
    rf_df.head(20).to_string(index=False)
)


# ============================================================
# XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=RANDOM_STATE,

    # Important for your Mac environment.
    n_jobs=1,

    tree_method="hist",
)

xgb.fit(
    X_train,
    y_train,
)

xgb_importance = xgb.feature_importances_


xgb_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "xgboost_importance": xgb_importance,
})


xgb_df = xgb_df.sort_values(
    "xgboost_importance",
    ascending=False,
).reset_index(drop=True)


xgb_df["xgboost_rank"] = (
    np.arange(len(xgb_df)) + 1
)


xgb_df.to_csv(
    RESULTS_DIR / "xgboost_feature_importance.csv",
    index=False,
)


print("\nTop 20 XGBoost features:")

print(
    xgb_df.head(20).to_string(index=False)
)


# ============================================================
# COMBINED FEATURE IMPORTANCE
# ============================================================

combined = pd.DataFrame({
    "feature": FEATURE_COLUMNS,

    "random_forest_importance": (
        rf_importance
    ),

    "xgboost_importance": (
        xgb_importance
    ),
})


# Normalize each importance vector so that
# the values sum to 1.

combined["rf_normalized"] = (
    combined["random_forest_importance"]
    /
    combined["random_forest_importance"].sum()
)


combined["xgb_normalized"] = (
    combined["xgboost_importance"]
    /
    combined["xgboost_importance"].sum()
)


# Average importance across the two tree models.

combined["mean_importance"] = (
    combined["rf_normalized"]
    +
    combined["xgb_normalized"]
) / 2


# Rank separately

combined["rf_rank"] = (
    combined[
        "random_forest_importance"
    ]
    .rank(
        ascending=False,
        method="min",
    )
    .astype(int)
)


combined["xgb_rank"] = (
    combined[
        "xgboost_importance"
    ]
    .rank(
        ascending=False,
        method="min",
    )
    .astype(int)
)


# Average rank

combined["mean_rank"] = (
    combined["rf_rank"]
    +
    combined["xgb_rank"]
) / 2


combined = combined.sort_values(
    "mean_importance",
    ascending=False,
).reset_index(drop=True)


combined["combined_rank"] = (
    np.arange(len(combined)) + 1
)


combined.to_csv(
    RESULTS_DIR / "combined_feature_importance.csv",
    index=False,
)


# ============================================================
# TOP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP 30 FEATURES - COMBINED")
print("=" * 70)

print(
    combined[
        [
            "combined_rank",
            "feature",
            "random_forest_importance",
            "xgboost_importance",
            "mean_importance",
            "rf_rank",
            "xgb_rank",
        ]
    ]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# RANK CORRELATION
# ============================================================

spearman_correlation = (
    combined[
        "random_forest_importance"
    ]
    .rank()
    .corr(
        combined[
            "xgboost_importance"
        ].rank(),
        method="spearman",
    )
)


print("\n" + "=" * 70)
print("FEATURE IMPORTANCE AGREEMENT")
print("=" * 70)

print(
    f"Spearman rank correlation: "
    f"{spearman_correlation:.4f}"
)


# ============================================================
# TOP-K OVERLAP
# ============================================================

overlap_rows = []


for k in [5, 10, 20, 30, 50]:

    rf_top = set(
        rf_df.head(k)["feature"]
    )

    xgb_top = set(
        xgb_df.head(k)["feature"]
    )

    overlap = rf_top.intersection(
        xgb_top
    )

    overlap_rows.append({
        "top_k": k,
        "random_forest_features": len(rf_top),
        "xgboost_features": len(xgb_top),
        "overlap": len(overlap),
        "overlap_percentage":
            100 * len(overlap) / k,
    })


overlap_df = pd.DataFrame(
    overlap_rows
)


overlap_df.to_csv(
    RESULTS_DIR / "top_k_feature_overlap.csv",
    index=False,
)


print("\nTop-K feature overlap:")

print(
    overlap_df.to_string(index=False)
)


# ============================================================
# SAVE TOP FEATURES LIST
# ============================================================

top_features = combined.head(30)[
    [
        "combined_rank",
        "feature",
        "random_forest_importance",
        "xgboost_importance",
        "mean_importance",
    ]
]


top_features.to_csv(
    RESULTS_DIR / "top_30_features.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

for path in sorted(
    RESULTS_DIR.glob("*.csv")
):
    print(path)


print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("=" * 70)
