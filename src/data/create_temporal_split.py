from pathlib import Path
import pandas as pd


# ============================================================
# GraphShield - Temporal Dataset Split
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw" / "elliptic"
PROCESSED_DIR = ROOT / "data" / "processed"

FEATURES_FILE = DATA_DIR / "elliptic_txs_features.csv"
CLASSES_FILE = DATA_DIR / "elliptic_txs_classes.csv"


def main():

    print("=" * 70)
    print("GRAPHSHIELD - TEMPORAL DATASET SPLIT")
    print("=" * 70)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    print("\nLoading features...")

    features = pd.read_csv(
        FEATURES_FILE,
        header=None
    )

    features.columns = (
        ["txId", "time_step"]
        + [f"feature_{i}" for i in range(1, 166)]
    )

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    print("Loading classes...")

    classes = pd.read_csv(
        CLASSES_FILE
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    print("Merging features and labels...")

    data = features.merge(
        classes,
        on="txId",
        how="left"
    )

    print(
        f"Total transactions: {len(data):,}"
    )

    # --------------------------------------------------------
    # Remove unknown labels
    # --------------------------------------------------------

    labeled = data[
        data["class"] != "unknown"
    ].copy()

    print(
        f"Labeled transactions: {len(labeled):,}"
    )

    # --------------------------------------------------------
    # Define temporal ranges
    # --------------------------------------------------------

    train_start = 1
    train_end = 34

    val_start = 35
    val_end = 40

    test_start = 41
    test_end = 49

    # --------------------------------------------------------
    # Create splits
    # --------------------------------------------------------

    train = labeled[
        (labeled["time_step"] >= train_start)
        &
        (labeled["time_step"] <= train_end)
    ].copy()

    validation = labeled[
        (labeled["time_step"] >= val_start)
        &
        (labeled["time_step"] <= val_end)
    ].copy()

    test = labeled[
        (labeled["time_step"] >= test_start)
        &
        (labeled["time_step"] <= test_end)
    ].copy()

    # --------------------------------------------------------
    # Print split information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEMPORAL SPLIT")
    print("=" * 70)

    print(
        f"\nTraining   : time {train_start}-{train_end}"
    )

    print(
        f"Validation : time {val_start}-{val_end}"
    )

    print(
        f"Test       : time {test_start}-{test_end}"
    )

    print("\nNumber of samples:")

    print(
        f"Training   : {len(train):,}"
    )

    print(
        f"Validation : {len(validation):,}"
    )

    print(
        f"Test       : {len(test):,}"
    )

    # --------------------------------------------------------
    # Class distributions
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    for name, split in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test)
    ]:

        print(f"\n{name}")

        counts = split["class"].value_counts()

        print(counts)

        illicit_count = (
split["class"].astype(str) == "1"        ).sum()

        total = len(split)

        print(
            f"Illicit percentage: "
            f"{illicit_count / total * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Save splits
    # --------------------------------------------------------

    train_file = (
        PROCESSED_DIR /
        "train_temporal.csv"
    )

    val_file = (
        PROCESSED_DIR /
        "validation_temporal.csv"
    )

    test_file = (
        PROCESSED_DIR /
        "test_temporal.csv"
    )

    print("\n" + "=" * 70)
    print("SAVING SPLITS")
    print("=" * 70)

    train.to_csv(
        train_file,
        index=False
    )

    validation.to_csv(
        val_file,
        index=False
    )

    test.to_csv(
        test_file,
        index=False
    )

    print(
        f"\nSaved: {train_file}"
    )

    print(
        f"Saved: {val_file}"
    )

    print(
        f"Saved: {test_file}"
    )

    # --------------------------------------------------------
    # Verify temporal separation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEMPORAL SEPARATION CHECK")
    print("=" * 70)

    max_train_time = train["time_step"].max()
    min_val_time = validation["time_step"].min()

    max_val_time = validation["time_step"].max()
    min_test_time = test["time_step"].min()

    print(
        "Maximum training time:",
        max_train_time
    )

    print(
        "Minimum validation time:",
        min_val_time
    )

    print(
        "Maximum validation time:",
        max_val_time
    )

    print(
        "Minimum test time:",
        min_test_time
    )

    if (
        max_train_time < min_val_time
        and
        max_val_time < min_test_time
    ):
        print(
            "\n[PASS] Temporal separation is valid."
        )
    else:
        print(
            "\n[FAIL] Temporal leakage detected."
        )

    print("\n" + "=" * 70)
    print("TEMPORAL SPLIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()