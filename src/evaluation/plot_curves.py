from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_DIR = ROOT / "results" / "predictions"
PLOTS_DIR = ROOT / "results" / "plots"

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODELS
# ============================================================

MODELS = [
    ("Logistic Regression", "logistic_regression"),
    ("Random Forest", "random_forest"),
    ("XGBoost", "xgboost"),
    ("GCN", "gcn"),
    ("GraphSAGE", "graphsage"),
    ("GAT", "gat"),
]


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions(model_key, split):

    path = (
        PREDICTIONS_DIR
        / f"{model_key}_{split}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        "y_true",
        "predicted_probability"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing column '{column}' in {path}"
            )

    y_true = df["y_true"].to_numpy()
    probabilities = df[
        "predicted_probability"
    ].to_numpy()

    return y_true, probabilities


# ============================================================
# LOAD ALL MODELS
# ============================================================

def load_all_predictions(split):

    predictions = {}

    print("\n" + "=" * 70)
    print(f"LOADING {split.upper()} PREDICTIONS")
    print("=" * 70)

    for display_name, model_key in MODELS:

        y_true, probabilities = load_predictions(
            model_key,
            split
        )

        predictions[display_name] = {
            "y_true": y_true,
            "probabilities": probabilities
        }

        print(
            f"[PASS] {display_name:<20} "
            f"{len(y_true):,} samples"
        )

    return predictions


# ============================================================
# ROC CURVES
# ============================================================

def plot_roc_curves(predictions, split):

    plt.figure(figsize=(10, 8))

    for display_name, data in predictions.items():

        y_true = data["y_true"]
        probabilities = data["probabilities"]

        fpr, tpr, _ = roc_curve(
            y_true,
            probabilities
        )

        auc = roc_auc_score(
            y_true,
            probabilities
        )

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{display_name} (AUC = {auc:.4f})"
        )

    # Random classifier reference
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Random classifier"
    )

    plt.xlabel(
        "False Positive Rate",
        fontsize=12
    )

    plt.ylabel(
        "True Positive Rate",
        fontsize=12
    )

    plt.title(
        f"ROC Curves - {split.capitalize()} Set",
        fontsize=15,
        fontweight="bold"
    )

    plt.legend(
        loc="lower right",
        fontsize=9
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / f"roc_curves_{split}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[PASS] Saved ROC curve:\n"
        f"       {output_path}"
    )


# ============================================================
# PRECISION-RECALL CURVES
# ============================================================

def plot_precision_recall_curves(
    predictions,
    split
):

    plt.figure(figsize=(10, 8))

    for display_name, data in predictions.items():

        y_true = data["y_true"]
        probabilities = data["probabilities"]

        precision, recall, _ = (
            precision_recall_curve(
                y_true,
                probabilities
            )
        )

        pr_auc = average_precision_score(
            y_true,
            probabilities
        )

        plt.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{display_name} (AP = {pr_auc:.4f})"
        )

    # Positive-class prevalence
    first_model = next(
        iter(predictions.values())
    )

    prevalence = np.mean(
        first_model["y_true"] == 1
    )

    plt.axhline(
        y=prevalence,
        linestyle="--",
        linewidth=1.5,
        label=f"Positive prevalence = {prevalence:.4f}"
    )

    plt.xlabel(
        "Recall",
        fontsize=12
    )

    plt.ylabel(
        "Precision",
        fontsize=12
    )

    plt.title(
        f"Precision-Recall Curves - {split.capitalize()} Set",
        fontsize=15,
        fontweight="bold"
    )

    plt.legend(
        loc="upper right",
        fontsize=9
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / f"precision_recall_curves_{split}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[PASS] Saved Precision-Recall curve:\n"
        f"       {output_path}"
    )


# ============================================================
# CONFUSION MATRICES
# ============================================================

def plot_confusion_matrices(
    predictions,
    split
):

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10)
    )

    axes = axes.flatten()

    for index, (display_name, data) in enumerate(
        predictions.items()
    ):

        y_true = data["y_true"]
        probabilities = data["probabilities"]

        # Use the same default threshold for visualization.
        threshold = 0.5

        y_pred = (
            probabilities >= threshold
        ).astype(int)

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "Licit",
                "Illicit"
            ]
        )

        display.plot(
            ax=axes[index],
            values_format="d",
            colorbar=False
        )

        axes[index].set_title(
            display_name,
            fontsize=12,
            fontweight="bold"
        )

        axes[index].set_xlabel(
            "Predicted Label"
        )

        axes[index].set_ylabel(
            "True Label"
        )

    fig.suptitle(
        f"Confusion Matrices - {split.capitalize()} Set",
        fontsize=17,
        fontweight="bold"
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    output_path = (
        PLOTS_DIR
        / f"confusion_matrices_{split}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[PASS] Saved confusion matrices:\n"
        f"       {output_path}"
    )


# ============================================================
# METRICS SUMMARY
# ============================================================

def print_metrics(predictions, split):

    print("\n" + "=" * 70)
    print(f"{split.upper()} CURVE METRICS")
    print("=" * 70)

    rows = []

    for display_name, data in predictions.items():

        y_true = data["y_true"]
        probabilities = data["probabilities"]

        roc_auc = roc_auc_score(
            y_true,
            probabilities
        )

        pr_auc = average_precision_score(
            y_true,
            probabilities
        )

        rows.append(
            {
                "Model": display_name,
                "ROC-AUC": roc_auc,
                "PR-AUC": pr_auc
            }
        )

    metrics_df = pd.DataFrame(rows)

    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    output_path = (
        PLOTS_DIR
        / f"curve_metrics_{split}.csv"
    )

    metrics_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\n[PASS] Saved metrics:\n"
        f"       {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("GRAPHSHIELD - ROC / PR / CONFUSION MATRIX ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_predictions = load_all_predictions(
        "validation"
    )

    print_metrics(
        validation_predictions,
        "validation"
    )

    plot_roc_curves(
        validation_predictions,
        "validation"
    )

    plot_precision_recall_curves(
        validation_predictions,
        "validation"
    )

    plot_confusion_matrices(
        validation_predictions,
        "validation"
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_predictions = load_all_predictions(
        "test"
    )

    print_metrics(
        test_predictions,
        "test"
    )

    plot_roc_curves(
        test_predictions,
        "test"
    )

    plot_precision_recall_curves(
        test_predictions,
        "test"
    )

    plot_confusion_matrices(
        test_predictions,
        "test"
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GRAPHSHIELD CURVE ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")

    expected_files = [
        "roc_curves_validation.png",
        "precision_recall_curves_validation.png",
        "confusion_matrices_validation.png",
        "roc_curves_test.png",
        "precision_recall_curves_test.png",
        "confusion_matrices_test.png",
        "curve_metrics_validation.csv",
        "curve_metrics_test.csv",
    ]

    for filename in expected_files:

        path = PLOTS_DIR / filename

        if path.exists():
            print(f"[PASS] {filename}")
        else:
            print(f"[FAIL] {filename}")

    print("\nOutput directory:")
    print(PLOTS_DIR)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
