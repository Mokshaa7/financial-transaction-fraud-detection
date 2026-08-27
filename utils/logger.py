import json
import os
from datetime import datetime, timezone


LOG_PATH = "transaction_logs.json"


def load_logs():

    if not os.path.exists(LOG_PATH):
        return []

    try:

        with open(
            LOG_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, list):
                return data

            return []

    except (json.JSONDecodeError, OSError):

        return []


def save_prediction(
    transaction,
    fraud,
    fraud_probability
):

    logs = load_logs()

    record = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "transaction": transaction,

        "fraud": bool(fraud),

        "fraud_probability": float(
            fraud_probability
        )
    }

    logs.append(record)

    with open(
        LOG_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            logs,
            file,
            indent=4
        )

    return record