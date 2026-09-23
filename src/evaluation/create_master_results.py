import os
import pandas as pd


# ============================================================
# GRAPHSHIELD - FINAL MASTER RESULTS
# ============================================================

OUTPUT_DIR = "results/final"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("=" * 70)
print("GRAPHSHIELD - FINAL MASTER RESULTS")
print("=" * 70)


# ============================================================
# MAIN MODEL RESULTS
# ============================================================

main_models = pd.DataFrame([

    {
        "Model": "Logistic Regression",
        "Family": "Traditional ML",
        "Validation_PR_AUC": 0.4103,
        "Test_PR_AUC": 0.2026,
        "Test_ROC_AUC": 0.8503,
        "Test_Precision": 0.2450,
        "Test_Recall": 0.4885,
        "Test_F1": 0.3263,
        "Threshold": 0.93
    },

    {
        "Model": "Random Forest",
        "Family": "Traditional ML",
        "Validation_PR_AUC": 0.9223,
        "Test_PR_AUC": 0.6506,
        "Test_ROC_AUC": 0.8911,
        "Test_Precision": 0.9375,
        "Test_Recall": 0.5725,
        "Test_F1": 0.7109,
        "Threshold": 0.57
    },

    {
        "Model": "XGBoost",
        "Family": "Traditional ML",
        "Validation_PR_AUC": 0.9271,
        "Test_PR_AUC": 0.6585,
        "Test_ROC_AUC": 0.8916,
        "Test_Precision": 0.9233,
        "Test_Recall": 0.5744,
        "Test_F1": 0.7082,
        "Threshold": 0.63
    },

    {
        "Model": "GCN",
        "Family": "GNN",
        "Validation_PR_AUC": 0.3948,
        "Test_PR_AUC": 0.1811,
        "Test_ROC_AUC": 0.7448,
        "Test_Precision": 0.1929,
        "Test_Recall": 0.2385,
        "Test_F1": 0.2133,
        "Threshold": 0.53
    },

    {
        "Model": "GraphSAGE",
        "Family": "GNN",
        "Validation_PR_AUC": 0.5866,
        "Test_PR_AUC": 0.2971,
        "Test_ROC_AUC": 0.8400,
        "Test_Precision": 0.3353,
        "Test_Recall": 0.4294,
        "Test_F1": 0.3766,
        "Threshold": 0.90
    },

    {
        "Model": "GAT",
        "Family": "GNN",
        "Validation_PR_AUC": 0.4578,
        "Test_PR_AUC": 0.2265,
        "Test_ROC_AUC": 0.8138,
        "Test_Precision": 0.2469,
        "Test_Recall": 0.3034,
        "Test_F1": 0.2723,
        "Threshold": 0.88
    }
])


# ============================================================
# SAVE MAIN RESULTS
# ============================================================

main_path = (
    f"{OUTPUT_DIR}/"
    "master_model_results.csv"
)

main_models.to_csv(
    main_path,
    index=False
)


# ============================================================
# MULTI-SEED ROBUSTNESS
# ============================================================

robustness = pd.read_csv(
    "results/robustness/"
    "robustness_clean_summary.csv"
)


robustness_columns = [

    "model",

    "precision_mean",
    "precision_std",

    "recall_mean",
    "recall_std",

    "f1_mean",
    "f1_std",

    "roc_auc_mean",
    "roc_auc_std",

    "pr_auc_mean",
    "pr_auc_std"
]


robustness = robustness[
    robustness_columns
]


robustness_path = (
    f"{OUTPUT_DIR}/"
    "master_robustness_results.csv"
)

robustness.to_csv(
    robustness_path,
    index=False
)


# ============================================================
# ABLATION RESULTS
# ============================================================

ablation_path = (
    "results/feature_ablation/"
    "top_k_feature_ablation.csv"
)

if os.path.exists(ablation_path):

    feature_ablation = pd.read_csv(
        ablation_path
    )

    feature_ablation.to_csv(
        f"{OUTPUT_DIR}/"
        "master_feature_ablation.csv",
        index=False
    )

    print(
        "\nFeature ablation imported."
    )


# ============================================================
# GNN FEATURE ABLATION
# ============================================================

gnn_ablation_path = (
    "results/gnn_feature_ablation/"
    "gnn_feature_ablation.csv"
)

if os.path.exists(
    gnn_ablation_path
):

    gnn_ablation = pd.read_csv(
        gnn_ablation_path
    )

    gnn_ablation.to_csv(
        f"{OUTPUT_DIR}/"
        "master_gnn_feature_ablation.csv",
        index=False
    )

    print(
        "GNN feature ablation imported."
    )


# ============================================================
# DEGREE BASELINE
# ============================================================

degree_path = (
    "results/degree_baseline/"
    "degree_baseline_comparison.csv"
)

if os.path.exists(
    degree_path
):

    degree = pd.read_csv(
        degree_path
    )

    degree.to_csv(
        f"{OUTPUT_DIR}/"
        "master_degree_baseline.csv",
        index=False
    )

    print(
        "Degree baseline imported."
    )


# ============================================================
# NEIGHBOR BASELINE
# ============================================================

neighbor_path = (
    "results/neighbor_baseline/"
    "neighbor_baseline_comparison.csv"
)

if os.path.exists(
    neighbor_path
):

    neighbor = pd.read_csv(
        neighbor_path
    )

    neighbor.to_csv(
        f"{OUTPUT_DIR}/"
        "master_neighbor_baseline.csv",
        index=False
    )

    print(
        "Neighbor baseline imported."
    )


# ============================================================
# COMPUTATIONAL EFFICIENCY
# ============================================================

efficiency_path = (
    "results/computational_efficiency/"
    "computational_efficiency.csv"
)

if os.path.exists(
    efficiency_path
):

    efficiency = pd.read_csv(
        efficiency_path
    )

    efficiency.to_csv(
        f"{OUTPUT_DIR}/"
        "master_computational_efficiency.csv",
        index=False
    )

    print(
        "Computational efficiency imported."
    )


# ============================================================
# STATISTICAL COMPARISON
# ============================================================

stat_path = (
    "results/statistical_analysis/"
    "paired_pr_auc_comparisons.csv"
)

if os.path.exists(
    stat_path
):

    statistical = pd.read_csv(
        stat_path
    )

    statistical.to_csv(
        f"{OUTPUT_DIR}/"
        "master_statistical_comparisons.csv",
        index=False
    )

    print(
        "Statistical comparisons imported."
    )


# ============================================================
# RESEARCH DATASET SUMMARY
# ============================================================

dataset_summary = pd.DataFrame([

    {
        "Property": "Total graph nodes",
        "Value": 203769
    },

    {
        "Property": "Total graph edges",
        "Value": 234355
    },

    {
        "Property": "Total time steps",
        "Value": 49
    },

    {
        "Property": "Labeled transactions",
        "Value": 46564
    },

    {
        "Property": "Illicit transactions",
        "Value": 4545
    },

    {
        "Property": "Licit transactions",
        "Value": 42019
    },

    {
        "Property": "Illicit rate among labeled",
        "Value": "9.76%"
    },

    {
        "Property": "Train time steps",
        "Value": "1-34"
    },

    {
        "Property": "Validation time steps",
        "Value": "35-40"
    },

    {
        "Property": "Test time steps",
        "Value": "41-49"
    },

    {
        "Property": "Transaction features",
        "Value": 165
    }
])


dataset_path = (
    f"{OUTPUT_DIR}/"
    "master_dataset_summary.csv"
)

dataset_summary.to_csv(
    dataset_path,
    index=False
)


# ============================================================
# FINAL RESEARCH SUMMARY
# ============================================================

research_summary = pd.DataFrame([

    {
        "Analysis":
            "Traditional ML baseline",
        "Status":
            "Completed",
        "Primary finding":
            "Tree-based models achieved strong PR-AUC using transaction features."
    },

    {
        "Analysis":
            "Graph structure",
        "Status":
            "Completed",
        "Primary finding":
            "Illicit and licit transactions exhibit different degree and neighbor structures."
    },

    {
        "Analysis":
            "Neighbor label propagation",
        "Status":
            "Completed",
        "Primary finding":
            "Neighbor labels contain relational signal but simple propagation is limited."
    },

    {
        "Analysis":
            "GNN comparison",
        "Status":
            "Completed",
        "Primary finding":
            "GNN architectures exhibit different levels of relational predictive performance."
    },

    {
        "Analysis":
            "Structural-only GNN",
        "Status":
            "Completed",
        "Primary finding":
            "Degree-only graph information is insufficient compared with transaction features."
    },

    {
        "Analysis":
            "Feature ablation",
        "Status":
            "Completed",
        "Primary finding":
            "Performance depends substantially on transaction feature information."
    },

    {
        "Analysis":
            "Error analysis",
        "Status":
            "Completed",
        "Primary finding":
            "Model errors vary across classes, confidence levels, model agreement, and time."
    },

    {
        "Analysis":
            "Computational efficiency",
        "Status":
            "Completed",
        "Primary finding":
            "GNNs require more training computation than the tested traditional baselines."
    },

    {
        "Analysis":
            "Multi-seed robustness",
        "Status":
            "Completed",
        "Primary finding":
            "Traditional Random Forest performance was highly stable, while GNN results showed greater variability."
    },

    {
        "Analysis":
            "Statistical comparison",
        "Status":
            "Completed",
        "Primary finding":
            "Five paired seeds provide descriptive robustness evidence, but limited statistical power."
    }
])


summary_path = (
    f"{OUTPUT_DIR}/"
    "research_analysis_summary.csv"
)

research_summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# PRINT MAIN RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MASTER MODEL RESULTS")
print("=" * 70)

print(
    main_models.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


print("\n")
print("=" * 70)
print("MULTI-SEED ROBUSTNESS")
print("=" * 70)

print(
    robustness.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n")
print("=" * 70)
print("MASTER FILES CREATED")
print("=" * 70)

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(
        f"{OUTPUT_DIR}/{filename}"
    )


print("\n")
print("=" * 70)
print("FINAL MASTER RESULTS COMPLETE")
print("=" * 70)