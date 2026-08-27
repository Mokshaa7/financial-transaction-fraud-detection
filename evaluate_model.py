import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from utils.feature_engineering import calculate_features


DATA_PATH = (
    "data/raw/transactions.csv/"
    "PS_20174392719_1491204439457_log.csv/"
    "PS_20174392719_1491204439457_log.csv"
)
MODEL_PATH = "models/fraud_detection_xgboost.pkl"


def main():

    print("=" * 70)
    print("FINANCIAL FRAUD DETECTION - HELD-OUT TEST EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    # ---------------------------------------------------------
    # 2. Create features
    # ---------------------------------------------------------

    print("Creating features...")

    X = calculate_features(df)
    y = df["isFraud"].astype(int)

    # ---------------------------------------------------------
    # 3. Recreate the same train/test split
    # ---------------------------------------------------------

    print("Creating held-out test set...")

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Test samples: {len(X_test)}")

    # ---------------------------------------------------------
    # 4. Load trained model
    # ---------------------------------------------------------

    print("Loading trained model...")

    model = joblib.load(MODEL_PATH)

    # ---------------------------------------------------------
    # 5. Predictions
    # ---------------------------------------------------------

    print("Generating predictions...")

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # ---------------------------------------------------------
    # 6. Metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    # ---------------------------------------------------------
    # 7. Display results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("HELD-OUT TEST PERFORMANCE")
    print("=" * 70)

    print(f"\nAccuracy:          {accuracy:.6f}")
    print(f"Precision:         {precision:.6f}")
    print(f"Recall:            {recall:.6f}")
    print(f"F1 Score:          {f1:.6f}")
    print(f"ROC-AUC:           {roc_auc:.6f}")
    print(f"PR-AUC:            {pr_auc:.6f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Not Fraud",
                "Fraud"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()