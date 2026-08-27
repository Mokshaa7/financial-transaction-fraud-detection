import json
import joblib


MODEL_PATH = "models/fraud_detection_xgboost.pkl"
FEATURE_PATH = "models/feature_columns.json"


def main():

    print("=" * 60)
    print("MODEL CHECK")
    print("=" * 60)

    print("\nLoading model...")

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")

    print("\nModel type:")
    print(type(model))

    print("\nNumber of features:")
    print(model.n_features_in_)

    print("\nModel classes:")
    print(model.classes_)

    print("\nFeature names:")

    with open(
        FEATURE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        features = json.load(file)

    for index, feature in enumerate(features):
        print(f"{index}: {feature}")

    print("\nModel check completed successfully.")


if __name__ == "__main__":
    main()