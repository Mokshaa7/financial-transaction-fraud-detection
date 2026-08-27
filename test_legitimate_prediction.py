import joblib
import pandas as pd

from utils.feature_engineering import calculate_features


DATA_PATH = (
    "data/raw/transactions.csv/"
    "PS_20174392719_1491204439457_log.csv/"
    "PS_20174392719_1491204439457_log.csv"
)
MODEL_PATH = "models/fraud_detection_xgboost.pkl"


def main():

    df = pd.read_csv(DATA_PATH)

    # Select an actual legitimate transaction
    transaction = df[
        df["isFraud"] == 0
    ].iloc[[0]].copy()

    model = joblib.load(MODEL_PATH)

    features = calculate_features(
        transaction
    )

    prediction = model.predict(features)[0]

    probability = model.predict_proba(
        features
    )[0][1]

    print("=" * 60)
    print("LEGITIMATE TRANSACTION TEST")
    print("=" * 60)

    print(
        f"Actual label:       {transaction['isFraud'].iloc[0]}"
    )

    print(
        f"Predicted label:    {prediction}"
    )

    print(
        f"Fraud probability:  "
        f"{probability * 100:.4f}%"
    )

    if prediction == 1:
        print("Status: FRAUD")
    else:
        print("Status: NOT FRAUD")


if __name__ == "__main__":
    main()