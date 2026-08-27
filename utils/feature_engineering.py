import pandas as pd


TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER"
]


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw PaySim transaction data into model-ready features.

    The same function is used during both training and prediction
    so that the model receives a consistent feature representation.
    """

    data = df.copy()

    # Basic numerical features
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
    data["oldbalanceOrg"] = pd.to_numeric(
        data["oldbalanceOrg"], errors="coerce"
    )
    data["newbalanceOrig"] = pd.to_numeric(
        data["newbalanceOrig"], errors="coerce"
    )
    data["oldbalanceDest"] = pd.to_numeric(
        data["oldbalanceDest"], errors="coerce"
    )
    data["newbalanceDest"] = pd.to_numeric(
        data["newbalanceDest"], errors="coerce"
    )

    # Balance consistency features
    data["origin_balance_error"] = (
        data["oldbalanceOrg"]
        - data["amount"]
        - data["newbalanceOrig"]
    )

    data["destination_balance_error"] = (
        data["oldbalanceDest"]
        + data["amount"]
        - data["newbalanceDest"]
    )

    # Account indicators
    data["origin_is_merchant"] = (
        data["nameOrig"].astype(str).str.startswith("M")
    ).astype(int)

    data["destination_is_merchant"] = (
        data["nameDest"].astype(str).str.startswith("M")
    ).astype(int)

    # Transaction type encoding
    type_dummies = pd.get_dummies(
        data["type"],
        prefix="type"
    )

    # Make sure every expected transaction type column exists
    for transaction_type in TRANSACTION_TYPES:
        column = f"type_{transaction_type}"

        if column not in type_dummies.columns:
            type_dummies[column] = 0

    type_columns = [
        f"type_{transaction_type}"
        for transaction_type in TRANSACTION_TYPES
    ]

    type_dummies = type_dummies[type_columns]

    # Combine features
    features = pd.concat(
        [
            data[
                [
                    "amount",
                    "oldbalanceOrg",
                    "newbalanceOrig",
                    "oldbalanceDest",
                    "newbalanceDest",
                    "origin_balance_error",
                    "destination_balance_error",
                    "origin_is_merchant",
                    "destination_is_merchant",
                ]
            ],
            type_dummies,
        ],
        axis=1,
    )

    return features