document.addEventListener("DOMContentLoaded", function () {

    console.log("FraudGuard JavaScript loaded.");

    const form = document.getElementById("transaction-form");

    const resultCard = document.getElementById("result-card");
    const resultIcon = document.getElementById("result-icon");
    const resultStatus = document.getElementById("result-status");
    const resultDescription = document.getElementById("result-description");
    const riskPercentage = document.getElementById("risk-percentage");
    const riskFill = document.getElementById("risk-fill");
    const resultBadge = document.getElementById("result-badge");

    const errorMessage = document.getElementById("error-message");

    const button = document.getElementById("analyze-button");
    const buttonText = document.getElementById("button-text");
    const loader = document.getElementById("button-loader");


    form.addEventListener("submit", async function (event) {

        // Prevent browser from refreshing the page
        event.preventDefault();

        console.log("Analyze button clicked.");


        // Hide previous error
        errorMessage.classList.add("hidden");
        errorMessage.textContent = "";


        // Disable button
        button.disabled = true;
        buttonText.textContent = "Analyzing...";
        loader.classList.remove("hidden");


        // Collect transaction data
        const transaction = {

            type: document.getElementById("type").value,

            amount: Number(
                document.getElementById("amount").value
            ),

            nameOrig:
                document.getElementById("nameOrig").value.trim(),

            oldbalanceOrg: Number(
                document.getElementById("oldbalanceOrg").value
            ),

            newbalanceOrig: Number(
                document.getElementById("newbalanceOrig").value
            ),

            nameDest:
                document.getElementById("nameDest").value.trim(),

            oldbalanceDest: Number(
                document.getElementById("oldbalanceDest").value
            ),

            newbalanceDest: Number(
                document.getElementById("newbalanceDest").value
            )
        };


        console.log("Sending transaction:", transaction);


        try {

            const response = await fetch("/predict", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(transaction)

            });


            const data = await response.json();

            console.log("Server response:", data);


            if (!response.ok || !data.success) {

                throw new Error(
                    data.error || "Prediction failed."
                );

            }


            // Show prediction
            showResult(data);


            // Refresh history
            loadHistory();

        }

        catch (error) {

            console.error("Prediction error:", error);

            errorMessage.textContent =
                error.message;

            errorMessage.classList.remove("hidden");

        }

        finally {

            button.disabled = false;

            buttonText.textContent =
                "Analyze Transaction";

            loader.classList.add("hidden");

        }

    });


    function showResult(data) {

        const fraud = data.fraud === true;

        const percentage =
            Number(data.fraud_percentage);


        resultCard.classList.remove("hidden");


        if (fraud) {

            resultCard.classList.add("fraud");

            resultIcon.textContent = "!";

            resultStatus.textContent = "FRAUD";

            resultDescription.textContent =
                "This transaction has a high probability of fraudulent activity and should be reviewed.";

            resultBadge.textContent =
                "High Risk";

        }

        else {

            resultCard.classList.remove("fraud");

            resultIcon.textContent = "✓";

            resultStatus.textContent =
                "NOT FRAUD";

            resultDescription.textContent =
                "This transaction appears to have a low probability of fraudulent activity.";

            resultBadge.textContent =
                "Low Risk";

        }


        riskPercentage.textContent =
            percentage.toFixed(4) + "%";


        riskFill.style.width =
            Math.min(percentage, 100) + "%";


        resultCard.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });

    }


    async function loadHistory() {

        try {

            const response =
                await fetch("/history");

            const data =
                await response.json();


            if (!response.ok || !data.success) {
                return;
            }


            const transactions =
                data.transactions || [];


            if (transactions.length === 0) {

                document.getElementById(
                    "history-container"
                ).innerHTML = `
                    <div class="empty-history">
                        No transactions analyzed yet.
                    </div>
                `;

                return;
            }


            const rows =
                transactions
                    .slice()
                    .reverse()
                    .slice(0, 10)
                    .map(record => {

                        const transaction =
                            record.transaction;

                        const probability =
                            Number(
                                record.fraud_probability
                            ) * 100;

                        const fraud =
                            record.fraud === true;


                        return `
                            <tr>

                                <td>
                                    <span class="type-badge">
                                        ${transaction.type}
                                    </span>
                                </td>

                                <td>
                                    ₹${Number(
                                        transaction.amount
                                    ).toLocaleString("en-IN")}
                                </td>

                                <td>
                                    ${transaction.nameOrig}
                                </td>

                                <td>
                                    ${transaction.nameDest}
                                </td>

                                <td>

                                    <span class="status-badge ${
                                        fraud
                                            ? "fraud"
                                            : "safe"
                                    }">

                                        ${
                                            fraud
                                                ? "Fraud"
                                                : "Not Fraud"
                                        }

                                    </span>

                                </td>

                                <td>
                                    ${probability.toFixed(4)}%
                                </td>

                            </tr>
                        `;

                    })
                    .join("");


            document.getElementById(
                "history-container"
            ).innerHTML = `

                <table class="history-table">

                    <thead>

                        <tr>
                            <th>Type</th>
                            <th>Amount</th>
                            <th>Origin</th>
                            <th>Destination</th>
                            <th>Status</th>
                            <th>Risk</th>
                        </tr>

                    </thead>

                    <tbody>
                        ${rows}
                    </tbody>

                </table>

            `;

        }

        catch (error) {

            console.error(
                "History error:",
                error
            );

        }

    }


    // Load history when page opens
    loadHistory();

});