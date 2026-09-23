import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon


# ============================================================
# GRAPHSHIELD - STATISTICAL COMPARISON
# ============================================================

INPUT_FILE = (
    "results/robustness/"
    "robustness_per_seed.csv"
)

OUTPUT_DIR = (
    "results/statistical_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("=" * 70)
print("GRAPHSHIELD - STATISTICAL COMPARISON")
print("=" * 70)


# ============================================================
# LOAD ROBUSTNESS RESULTS
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"\nLoaded {len(df)} experiment results."
)

print(
    f"Models: "
    f"{', '.join(df['model'].unique())}"
)

print(
    f"Seeds: "
    f"{sorted(df['seed'].unique())}"
)


# ============================================================
# CHECK DATA
# ============================================================

required_columns = [
    "model",
    "seed",
    "pr_auc",
    "roc_auc",
    "f1",
    "precision",
    "recall"
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# MODEL ORDER
# ============================================================

model_order = [
    "Random Forest",
    "GCN",
    "GraphSAGE",
    "GAT"
]


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

metrics = [
    "pr_auc",
    "roc_auc",
    "f1",
    "precision",
    "recall"
]


summary_rows = []


for model in model_order:

    subset = df[
        df["model"] == model
    ]

    if len(subset) == 0:
        continue

    row = {
        "model": model,
        "n_runs": len(subset)
    }

    for metric in metrics:

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

        row[
            f"{metric}_median"
        ] = subset[
            metric
        ].median()

        row[
            f"{metric}_min"
        ] = subset[
            metric
        ].min()

        row[
            f"{metric}_max"
        ] = subset[
            metric
        ].max()

    summary_rows.append(row)


summary_df = pd.DataFrame(
    summary_rows
)


summary_path = (
    f"{OUTPUT_DIR}/"
    "descriptive_statistics.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)


for _, row in summary_df.iterrows():

    print(
        f"\n{row['model']}"
    )

    print(
        f"PR-AUC: "
        f"{row['pr_auc_mean']:.4f} "
        f"± "
        f"{row['pr_auc_std']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{row['roc_auc_mean']:.4f} "
        f"± "
        f"{row['roc_auc_std']:.4f}"
    )

    print(
        f"F1: "
        f"{row['f1_mean']:.4f} "
        f"± "
        f"{row['f1_std']:.4f}"
    )


# ============================================================
# PAIRED PR-AUC COMPARISONS
# ============================================================
#
# Each model was evaluated on the SAME five random seeds.
#
# Therefore, we compare PR-AUC values seed-by-seed.
#
# Wilcoxon signed-rank test is used because:
# - only five paired observations exist
# - we do not assume normality
# ============================================================

print("\n")
print("=" * 70)
print("PAIRED PR-AUC COMPARISONS")
print("=" * 70)


wide_pr = (
    df[
        ["model", "seed", "pr_auc"]
    ]
    .pivot(
        index="seed",
        columns="model",
        values="pr_auc"
    )
)


comparisons = []


for i in range(
    len(model_order)
):

    for j in range(
        i + 1,
        len(model_order)
    ):

        model_a = model_order[i]
        model_b = model_order[j]

        if (
            model_a not in wide_pr.columns
            or
            model_b not in wide_pr.columns
        ):
            continue

        paired = wide_pr[
            [model_a, model_b]
        ].dropna()

        if len(paired) < 2:

            continue

        values_a = (
            paired[model_a]
            .values
        )

        values_b = (
            paired[model_b]
            .values
        )

        differences = (
            values_a - values_b
        )

        mean_difference = (
            differences.mean()
        )

        median_difference = (
            np.median(differences)
        )


        try:

            statistic, p_value = (
                wilcoxon(
                    values_a,
                    values_b,
                    alternative="two-sided"
                )
            )

        except ValueError:

            statistic = np.nan
            p_value = np.nan


        comparisons.append({

            "model_a":
                model_a,

            "model_b":
                model_b,

            "n_pairs":
                len(paired),

            "mean_pr_auc_a":
                values_a.mean(),

            "mean_pr_auc_b":
                values_b.mean(),

            "mean_difference_a_minus_b":
                mean_difference,

            "median_difference_a_minus_b":
                median_difference,

            "wilcoxon_statistic":
                statistic,

            "p_value":
                p_value
        })


comparison_df = pd.DataFrame(
    comparisons
)


comparison_path = (
    f"{OUTPUT_DIR}/"
    "paired_pr_auc_comparisons.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# PRINT PAIRWISE RESULTS
# ============================================================

for _, row in comparison_df.iterrows():

    print(
        f"\n{row['model_a']} "
        f"vs "
        f"{row['model_b']}"
    )

    print(
        f"Mean PR-AUC difference: "
        f"{row['mean_difference_a_minus_b']:.4f}"
    )

    print(
        f"Wilcoxon statistic: "
        f"{row['wilcoxon_statistic']}"
    )

    print(
        f"p-value: "
        f"{row['p_value']:.4f}"
    )


# ============================================================
# MULTIPLE-COMPARISON CORRECTION
# ============================================================
#
# Six pairwise comparisons are performed.
#
# Bonferroni correction:
# corrected alpha = 0.05 / 6
# ============================================================

if len(comparison_df) > 0:

    comparison_df[
        "bonferroni_p_value"
    ] = np.minimum(
        comparison_df["p_value"] * len(comparison_df),
        1.0
    )

    comparison_df[
        "significant_at_0.05_after_correction"
    ] = (
        comparison_df[
            "bonferroni_p_value"
        ] < 0.05
    )


comparison_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# PR-AUC ROBUSTNESS PLOT
# ============================================================

print("\n")
print("=" * 70)
print("GENERATING PR-AUC ROBUSTNESS PLOT")
print("=" * 70)


plot_data = []


for model in model_order:

    values = df[
        df["model"] == model
    ]["pr_auc"].values

    if len(values) == 0:
        continue

    plot_data.append(
        values
    )


plt.figure(
    figsize=(10, 6)
)

plt.boxplot(
    plot_data,
    tick_labels=[
        model
        for model in model_order
        if len(
            df[
                df["model"] == model
            ]
        ) > 0
    ],
    showmeans=True
)

plt.ylabel(
    "Test PR-AUC"
)

plt.xlabel(
    "Model"
)

plt.title(
    "GraphShield: PR-AUC Across Five Random Seeds"
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()


plot_path = (
    f"{OUTPUT_DIR}/"
    "pr_auc_robustness.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# METRIC MEAN ± STD BAR PLOT
# ============================================================

metric_plot_data = []


for model in model_order:

    subset = df[
        df["model"] == model
    ]

    if len(subset) == 0:
        continue

    metric_plot_data.append({

        "model": model,

        "pr_auc_mean":
            subset["pr_auc"].mean(),

        "pr_auc_std":
            subset["pr_auc"].std(),

        "f1_mean":
            subset["f1"].mean(),

        "f1_std":
            subset["f1"].std(),

        "roc_auc_mean":
            subset["roc_auc"].mean(),

        "roc_auc_std":
            subset["roc_auc"].std()
    })


metric_plot_df = pd.DataFrame(
    metric_plot_data
)


x = np.arange(
    len(metric_plot_df)
)

width = 0.25


plt.figure(
    figsize=(11, 6)
)


plt.bar(
    x - width,
    metric_plot_df["pr_auc_mean"],
    width,
    yerr=metric_plot_df["pr_auc_std"],
    capsize=4,
    label="PR-AUC"
)


plt.bar(
    x,
    metric_plot_df["roc_auc_mean"],
    width,
    yerr=metric_plot_df["roc_auc_std"],
    capsize=4,
    label="ROC-AUC"
)


plt.bar(
    x + width,
    metric_plot_df["f1_mean"],
    width,
    yerr=metric_plot_df["f1_std"],
    capsize=4,
    label="F1"
)


plt.xticks(
    x,
    metric_plot_df["model"]
)


plt.ylabel(
    "Score"
)

plt.xlabel(
    "Model"
)

plt.title(
    "GraphShield: Mean Performance Across Five Random Seeds"
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()


metrics_plot_path = (
    f"{OUTPUT_DIR}/"
    "mean_metrics_with_std.png"
)


plt.savefig(
    metrics_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE SEED-LEVEL PR-AUC TABLE
# ============================================================

seed_table = (
    df[
        ["seed", "model", "pr_auc"]
    ]
    .sort_values(
        ["seed", "model"]
    )
)


seed_table_path = (
    f"{OUTPUT_DIR}/"
    "seed_level_pr_auc.csv"
)

seed_table.to_csv(
    seed_table_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    summary_path
)

print(
    comparison_path
)

print(
    plot_path
)

print(
    metrics_plot_path
)

print(
    seed_table_path
)


print("\n")
print("=" * 70)
print("STATISTICAL ANALYSIS COMPLETE")
print("=" * 70)