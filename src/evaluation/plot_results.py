from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT / "results"

INPUT_FILE = RESULTS_DIR / "all_models_comparison.csv"

PLOT_DIR = RESULTS_DIR / "plots"


# ============================================================
# CONFIGURATION
# ============================================================

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD RESULTS
# ============================================================

print("=" * 70)
print("GRAPHSHIELD - RESULTS VISUALIZATION")
print("=" * 70)

print("\nLoading final six-model comparison...")

df = pd.read_csv(INPUT_FILE)

print("\nModels found:")

print(
    df["model"].to_string(
        index=False
    )
)


# ============================================================
# DISPLAY DATA
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    df[
        [
            "model",
            "test_precision",
            "test_recall",
            "test_f1",
            "roc_auc",
            "pr_auc",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 1. PR-AUC COMPARISON
# ============================================================

plt.figure(figsize=(11, 6))

plt.bar(
    df["model"],
    df["pr_auc"]
)

plt.xlabel("Model")
plt.ylabel("PR-AUC")
plt.title("GraphShield - Test PR-AUC Comparison")

plt.ylim(
    0,
    max(df["pr_auc"]) * 1.15
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

pr_auc_file = (
    PLOT_DIR /
    "test_pr_auc_comparison.png"
)

plt.savefig(
    pr_auc_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"\n[PASS] Saved: {pr_auc_file}"
)


# ============================================================
# 2. F1 COMPARISON
# ============================================================

plt.figure(figsize=(11, 6))

plt.bar(
    df["model"],
    df["test_f1"]
)

plt.xlabel("Model")
plt.ylabel("F1-score")
plt.title("GraphShield - Test F1 Comparison")

plt.ylim(
    0,
    max(df["test_f1"]) * 1.15
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

f1_file = (
    PLOT_DIR /
    "test_f1_comparison.png"
)

plt.savefig(
    f1_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"[PASS] Saved: {f1_file}"
)


# ============================================================
# 3. PRECISION AND RECALL
# ============================================================

plt.figure(figsize=(11, 6))

x = range(len(df))
width = 0.35

plt.bar(
    [i - width / 2 for i in x],
    df["test_precision"],
    width=width,
    label="Precision"
)

plt.bar(
    [i + width / 2 for i in x],
    df["test_recall"],
    width=width,
    label="Recall"
)

plt.xlabel("Model")
plt.ylabel("Score")
plt.title("GraphShield - Test Precision vs Recall")

plt.xticks(
    list(x),
    df["model"],
    rotation=25,
    ha="right"
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.tight_layout()

precision_recall_file = (
    PLOT_DIR /
    "test_precision_recall_comparison.png"
)

plt.savefig(
    precision_recall_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"[PASS] Saved: {precision_recall_file}"
)


# ============================================================
# 4. ROC-AUC COMPARISON
# ============================================================

plt.figure(figsize=(11, 6))

plt.bar(
    df["model"],
    df["roc_auc"]
)

plt.xlabel("Model")
plt.ylabel("ROC-AUC")
plt.title("GraphShield - Test ROC-AUC Comparison")

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

roc_auc_file = (
    PLOT_DIR /
    "test_roc_auc_comparison.png"
)

plt.savefig(
    roc_auc_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"[PASS] Saved: {roc_auc_file}"
)


# ============================================================
# 5. VALIDATION VS TEST PR-AUC
# ============================================================

plt.figure(figsize=(11, 6))

x = range(len(df))
width = 0.35

# Validation PR-AUC is reconstructed from the original
# model-specific experiment results.

validation_pr_auc = {
    "Logistic Regression": 0.4066,
    "Random Forest": 0.9221,
    "XGBoost": 0.9271,
    "GCN": 0.3948,
    "GraphSAGE": 0.5866,
    "GAT": 0.4578,
}

validation_values = [
    validation_pr_auc[model]
    for model in df["model"]
]

plt.bar(
    [i - width / 2 for i in x],
    validation_values,
    width=width,
    label="Validation PR-AUC"
)

plt.bar(
    [i + width / 2 for i in x],
    df["pr_auc"],
    width=width,
    label="Test PR-AUC"
)

plt.xlabel("Model")
plt.ylabel("PR-AUC")
plt.title(
    "GraphShield - Validation vs Test PR-AUC"
)

plt.xticks(
    list(x),
    df["model"],
    rotation=25,
    ha="right"
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.tight_layout()

temporal_gap_file = (
    PLOT_DIR /
    "validation_vs_test_pr_auc.png"
)

plt.savefig(
    temporal_gap_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"[PASS] Saved: {temporal_gap_file}"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print("\nPlots saved in:")

print(PLOT_DIR)

print("\nGenerated files:")

for file in sorted(PLOT_DIR.glob("*.png")):
    print(
        f"  - {file.name}"
    )

print("\n")
print("=" * 70)
