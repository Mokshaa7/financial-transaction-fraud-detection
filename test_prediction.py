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

    print("=" * 60)
    print("REAL FRAUD TRANSACTION TEST")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Select an actual fraudulent transaction
    fraud_transaction = df[
        df["isFraud"] == 1
    ].iloc[[0]].copy()

    print("\nActual transaction:")
    print(
        fraud_transaction[
            [
                "type",
                "amount",
                "nameOrig",
                "oldbalanceOrg",
                "newbalanceOrig",
                "nameDest",
                "oldbalanceDest",
                "newbalanceDest",
                "isFraud"
            ]
        ].to_string(index=False)
    )

    # Load model
    model = joblib.load(MODEL_PATH)

    # Create features
    features = calculate_features(
        fraud_transaction
    )

    # Predict
    prediction = model.predict(features)[0]

    probability = model.predict_proba(
        features
    )[0][1]

    print("\n" + "=" * 60)
    print("MODEL PREDICTION")
    print("=" * 60)

    print(f"\nActual label:       {fraud_transaction['isFraud'].iloc[0]}")
    print(f"Predicted label:    {prediction}")
    print(f"Fraud probability:  {probability * 100:.4f}%")

    if prediction == 1:
        print("Status: FRAUD")
    else:
        print("Status: NOT FRAUD")


if __name__ == "__main__":
    main()