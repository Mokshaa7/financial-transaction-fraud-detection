from flask import Flask, render_template, request, jsonify

app = Flask(
    __name__,
    static_folder="static",
    static_url_path="/static"
)

from utils.predictor import FraudPredictor
from utils.validation import validate_transaction
from utils.logger import (
    load_logs,
    save_prediction
)


app = Flask(__name__)


# -------------------------------------------------------------
# Load model once when application starts
# -------------------------------------------------------------

try:

    predictor = FraudPredictor()

    print("Fraud detection model loaded successfully.")

except Exception as error:

    predictor = None

    print(
        "ERROR: Could not load fraud detection model."
    )

    print(error)


# -------------------------------------------------------------
# Home page
# -------------------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# -------------------------------------------------------------
# Prediction API
# -------------------------------------------------------------

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # ---------------------------------------------------------
    # Check model
    # ---------------------------------------------------------

    if predictor is None:

        return jsonify({
            "success": False,
            "error": "Machine Learning model is unavailable."
        }), 500

    # ---------------------------------------------------------
    # Parse JSON
    # ---------------------------------------------------------

    if not request.is_json:

        return jsonify({
            "success": False,
            "error": "Request must contain JSON data."
        }), 400

    transaction = request.get_json(
        silent=True
    )

    if transaction is None:

        return jsonify({
            "success": False,
            "error": "Invalid JSON request."
        }), 400

    # ---------------------------------------------------------
    # Validate transaction
    # ---------------------------------------------------------

    valid, error_message = validate_transaction(
        transaction
    )

    if not valid:

        return jsonify({
            "success": False,
            "error": error_message
        }), 400

    # ---------------------------------------------------------
    # Convert numeric values
    # ---------------------------------------------------------

    numeric_fields = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]

    for field in numeric_fields:

        transaction[field] = float(
            transaction[field]
        )

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    try:

        result = predictor.predict(
            transaction
        )

    except Exception as error:

        return jsonify({
            "success": False,
            "error": "Prediction failed.",
            "details": str(error)
        }), 500

    # ---------------------------------------------------------
    # Determine status
    # ---------------------------------------------------------

    if result["fraud"]:

        status = "Fraud"

    else:

        status = "Not Fraud"

    # ---------------------------------------------------------
    # Log transaction
    # ---------------------------------------------------------

    try:

        save_prediction(
            transaction=transaction,
            fraud=result["fraud"],
            fraud_probability=result[
                "fraud_probability"
            ]
        )

    except Exception as error:

        return jsonify({
            "success": False,
            "error": "Prediction succeeded but logging failed.",
            "details": str(error)
        }), 500

    # ---------------------------------------------------------
    # Response
    # ---------------------------------------------------------

    return jsonify({
        "success": True,
        "fraud": result["fraud"],
        "status": status,
        "fraud_probability": result[
            "fraud_probability"
        ],
        "fraud_percentage": round(
            result["fraud_probability"] * 100,
            4
        )
    })


# -------------------------------------------------------------
# Transaction history
# -------------------------------------------------------------

@app.route(
    "/history",
    methods=["GET"]
)
def history():

    try:

        logs = load_logs()

        return jsonify({
            "success": True,
            "count": len(logs),
            "transactions": logs
        })

    except Exception as error:

        return jsonify({
            "success": False,
            "error": "Could not load transaction history.",
            "details": str(error)
        }), 500


# -------------------------------------------------------------
# Run application
# -------------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )