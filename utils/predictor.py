import json
import joblib
import pandas as pd

from utils.feature_engineering import calculate_features


MODEL_PATH = "models/fraud_detection_xgboost.pkl"
FEATURE_PATH = "models/feature_columns.json"
METADATA_PATH = "models/model_metadata.json"


class FraudPredictor:

    def __init__(self):

        self.model = joblib.load(
            MODEL_PATH
        )

        with open(
            FEATURE_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            self.feature_columns = json.load(file)

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(file)

        self.threshold = self.metadata[
            "fraud_threshold"
        ]

    def predict(self, transaction):

        # Convert transaction to DataFrame
        df = pd.DataFrame([transaction])

        # Generate features
        features = calculate_features(df)

        # Ensure exact feature order
        features = features[
            self.feature_columns
        ]

        # Probability of class 1 = Fraud
        fraud_probability = self.model.predict_proba(
            features
        )[0][1]

        # Apply threshold
        fraud = (
            fraud_probability >= self.threshold
        )

        return {
            "fraud": bool(fraud),
            "fraud_probability": float(
                fraud_probability
            )
        }