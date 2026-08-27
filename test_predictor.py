from utils.predictor import FraudPredictor


def main():

    predictor = FraudPredictor()

    transaction = {
        "type": "TRANSFER",
        "amount": 5000,
        "nameOrig": "C123456",
        "oldbalanceOrg": 10000,
        "newbalanceOrig": 5000,
        "nameDest": "C987654",
        "oldbalanceDest": 2000,
        "newbalanceDest": 7000
    }

    result = predictor.predict(
        transaction
    )

    print("=" * 60)
    print("PREDICTOR TEST")
    print("=" * 60)

    print(
        f"Fraud: {result['fraud']}"
    )

    print(
        f"Fraud probability: "
        f"{result['fraud_probability'] * 100:.4f}%"
    )


if __name__ == "__main__":
    main()