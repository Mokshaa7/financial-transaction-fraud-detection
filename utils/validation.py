from typing import Dict, Tuple


REQUIRED_FIELDS = [
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
]


VALID_TRANSACTION_TYPES = {
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
}


NUMERIC_FIELDS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def validate_transaction(
    transaction: Dict
) -> Tuple[bool, str]:

    # ---------------------------------------------------------
    # Check required fields
    # ---------------------------------------------------------

    for field in REQUIRED_FIELDS:

        if field not in transaction:
            return (
                False,
                f"Missing required field: {field}"
            )

    # ---------------------------------------------------------
    # Validate transaction type
    # ---------------------------------------------------------

    transaction_type = transaction["type"]

    if transaction_type not in VALID_TRANSACTION_TYPES:
        return (
            False,
            "Invalid transaction type. "
            "Supported types: "
            + ", ".join(sorted(VALID_TRANSACTION_TYPES))
        )

    # ---------------------------------------------------------
    # Validate numeric fields
    # ---------------------------------------------------------

    for field in NUMERIC_FIELDS:

        try:

            value = float(
                transaction[field]
            )

        except (TypeError, ValueError):

            return (
                False,
                f"{field} must be numeric."
            )

        if value < 0:

            return (
                False,
                f"{field} cannot be negative."
            )

    # ---------------------------------------------------------
    # Validate account identifiers
    # ---------------------------------------------------------

    if not str(
        transaction["nameOrig"]
    ).strip():

        return (
            False,
            "Origin account cannot be empty."
        )

    if not str(
        transaction["nameDest"]
    ).strip():

        return (
            False,
            "Destination account cannot be empty."
        )

    return True, ""