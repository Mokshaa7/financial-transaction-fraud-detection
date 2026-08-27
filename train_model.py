import json
import os

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)

from xgboost import XGBClassifier

from utils.feature_engineering import calculate_features


DATA_PATH = (
    "data/raw/transactions.csv/"
    "PS_20174392719_1491204439457_log.csv/"
    "PS_20174392719_1491204439457_log.csv"
)
MODEL_PATH = "models/fraud_detection_xgboost.pkl"
FEATURE_PATH = "models/feature_columns.json"


def main():

    print("=" * 60)
    print("FINANCIAL FRAUD DETECTION - MODEL TRAINING")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

    print("\n[1/7] Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    # ---------------------------------------------------------
    # 2. Check target
    # ---------------------------------------------------------

    print("\n[2/7] Checking target distribution...")

    if "isFraud" not in df.columns:
        raise ValueError(
            "Dataset must contain an 'isFraud' column."
        )

    print(df["isFraud"].value_counts())

    # ---------------------------------------------------------
    # 3. Feature engineering
    # ---------------------------------------------------------

    print("\n[3/7] Creating features...")

    X = calculate_features(df)
    y = df["isFraud"].astype(int)

    print(f"Feature matrix shape: {X.shape}")

    print("\nFeatures:")
    for feature in X.columns:
        print(f" - {feature}")

    # ---------------------------------------------------------
    # 4. Train/test split
    # ---------------------------------------------------------

    print("\n[4/7] Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # ---------------------------------------------------------
    # 5. Handle class imbalance
    # ---------------------------------------------------------

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

    print("\nClass distribution:")
    print(f"Normal transactions: {negative_count}")
    print(f"Fraud transactions:  {positive_count}")
    print(f"scale_pos_weight:     {scale_pos_weight:.2f}")

    # ---------------------------------------------------------
    # 6. Train XGBoost
    # ---------------------------------------------------------

    print("\n[5/7] Training XGBoost model...")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    print("Training completed.")

    # ---------------------------------------------------------
    # 7. Evaluate
    # ---------------------------------------------------------

    print("\n[6/7] Evaluating model...")

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\nAccuracy:")
    print(f"{accuracy:.4f}")

    print("\nROC-AUC:")
    print(f"{auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Not Fraud",
                "Fraud"
            ]
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # ---------------------------------------------------------
    # 8. Save model
    # ---------------------------------------------------------

    print("\n[7/7] Saving model...")

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        MODEL_PATH
    )

    # Save feature names separately
    feature_columns = list(X.columns)

    with open(
        FEATURE_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            feature_columns,
            file,
            indent=4
        )

    print(f"\nModel saved to:")
    print(MODEL_PATH)

    print(f"\nFeature configuration saved to:")
    print(FEATURE_PATH)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()