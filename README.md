# Financial Fraud Detection System

A machine learning-based financial transaction fraud detection system built with Python, XGBoost, Flask, Pandas, NumPy, and Scikit-learn.

The application accepts transaction details through a web interface, converts the transaction into the features expected by the trained XGBoost model, predicts the probability of fraud, and displays the result as either Fraud or Not Fraud.

The project also includes model training, evaluation, testing, validation, and transaction logging.

---

## Features

- Financial transaction fraud detection
- XGBoost binary classification
- Fraud probability prediction
- Transaction type encoding
- Balance consistency feature engineering
- Account/merchant indicators
- Flask backend and API
- Web-based transaction analysis interface
- Transaction history
- JSON transaction logging
- Model training and evaluation scripts
- Held-out test evaluation
- Model loading and prediction tests
- Input validation

---

## Project Architecture

```text
User
  |
  v
Web Interface
  |
  v
Flask Backend
  |
  v
Input Validation
  |
  v
Feature Engineering
  |
  v
14 Model Features
  |
  v
XGBoost Classifier
  |
  v
Fraud Probability
  |
  v
Fraud / Not Fraud
  |
  +-------> Transaction Log
```

---

## How the Fraud Detection Works

The system does not use a single hard-coded rule such as:

```text
"If the balance is wrong, it is fraud."
```

Instead, it uses a machine learning model trained on historical financial transactions.

The transaction goes through the following process:

```text
Transaction Details
        |
        v
Input Validation
        |
        v
Feature Engineering
        |
        v
14 Numerical Features
        |
        v
XGBoost Model
        |
        v
Fraud Probability
        |
        v
Classification
        |
        v
Result shown in UI
```

The model learns patterns associated with fraudulent and legitimate transactions from the training dataset.

For example, the model may learn relationships involving:

- Transaction amount
- Account balances
- Balance changes
- Transaction type
- Origin account characteristics
- Destination account characteristics
- Combinations of the above

A transaction can therefore be classified as fraud even when its balance arithmetic looks normal.

---

# Machine Learning Model

## Algorithm: XGBoost

The project uses XGBoost (Extreme Gradient Boosting) for binary classification.

The target variable is:

```text
0 = Not Fraud
1 = Fraud
```

XGBoost is an ensemble learning algorithm based on decision trees.

Instead of relying on one decision tree, XGBoost builds many trees sequentially. Each new tree attempts to improve the errors made by the existing ensemble.

The final prediction is produced by combining the contributions of the trees.

---

## Model Configuration

The model is configured as:

```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)
```

### Parameter Explanation

| Parameter | Value | Purpose |
|---|---:|---|
| `n_estimators` | 300 | Number of boosting trees |
| `max_depth` | 6 | Maximum depth of each tree |
| `learning_rate` | 0.08 | Controls how strongly each tree contributes |
| `subsample` | 0.8 | Uses approximately 80% of training rows for each tree |
| `colsample_bytree` | 0.8 | Uses approximately 80% of features for each tree |
| `objective` | `binary:logistic` | Binary classification with probability output |
| `eval_metric` | `auc` | Uses ROC-AUC during evaluation |
| `scale_pos_weight` | Calculated | Handles the highly imbalanced fraud class |
| `random_state` | 42 | Makes training reproducible |
| `n_jobs` | -1 | Uses available CPU threads |

---

# Class Imbalance

Financial fraud datasets are normally highly imbalanced.

There are many more legitimate transactions than fraudulent transactions.

For example:

```text
Not Fraud  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Fraud      >>
```

This creates a problem.

A model that predicts every transaction as legitimate could still achieve very high accuracy while detecting no fraud.

To address this, the training process uses:

```python
scale_pos_weight
```

The weight is calculated from the ratio of non-fraudulent transactions to fraudulent transactions.

This gives the minority fraud class greater importance during training.

This is particularly important because missing fraudulent transactions is much more significant than simply maximizing overall accuracy.

---

# Feature Engineering

The raw transaction information is converted into the exact feature representation expected by the XGBoost model.

The trained model uses 14 features.

## Model Features

```text
1.  amount
2.  oldbalanceOrg
3.  newbalanceOrig
4.  oldbalanceDest
5.  newbalanceDest
6.  origin_balance_error
7.  destination_balance_error
8.  origin_is_merchant
9.  destination_is_merchant
10. type_CASH_IN
11. type_CASH_OUT
12. type_DEBIT
13. type_PAYMENT
14. type_TRANSFER
```

The feature order is important because the trained model expects the same representation during prediction that it received during training.

---

## Balance Features

### Origin Balance Error

The system creates a feature representing the difference between the expected and actual origin account balance.

Conceptually:

```text
Origin Balance Error =
Old Origin Balance - Transaction Amount - New Origin Balance
```

For example:

```text
Old balance       = 1000
Transaction       = 200
New balance       = 800

1000 - 200 - 800 = 0
```

A value close to zero indicates that the balance change is consistent.

---

### Destination Balance Error

The destination balance is represented similarly.

Conceptually:

```text
Destination Balance Error =
Old Destination Balance + Transaction Amount - New Destination Balance
```

For example:

```text
Old balance       = 500
Transaction       = 200
New balance       = 700

500 + 200 - 700 = 0
```

These are machine learning features, not standalone fraud rules.

---

# Transaction Type Encoding

Transaction type is categorical information.

The model represents the transaction types using separate binary features:

```text
type_CASH_IN
type_CASH_OUT
type_DEBIT
type_PAYMENT
type_TRANSFER
```

For a transfer transaction:

```text
type_TRANSFER = 1
```

while the other transaction-type features are zero.

This converts categorical information into numerical data that XGBoost can use.

---

# Account / Merchant Features

The model also receives:

```text
origin_is_merchant
destination_is_merchant
```

These features provide information about the type of accounts involved in the transaction.

---

# How the Model Makes a Decision

When a transaction is submitted, the model first produces a probability for the fraud class.

For example:

```text
Fraud probability = 0.0000138
```

which corresponds to approximately:

```text
0.00138%
```

Another transaction might produce:

```text
Fraud probability = 0.999992
```

which corresponds to approximately:

```text
99.9992%
```

The prediction is then converted into a fraud/not-fraud classification.

Conceptually:

```text
             Fraud Probability
                    |
          +---------+---------+
          |                   |
       Low risk            High risk
          |                   |
          v                   v
     Not Fraud              Fraud
```

The exact classification behavior is determined by the trained XGBoost classifier.

---

# Training Pipeline

The model is trained using historical labeled transaction data.

The general training process is:

```text
Historical Dataset
       |
       v
Load Dataset
       |
       v
Prepare Data
       |
       v
Feature Engineering
       |
       v
Train/Test Split
       |
       v
Calculate Class Weight
       |
       v
Train XGBoost
       |
       v
Evaluate Model
       |
       v
Save Model
```

The target column represents:

```text
0 = Not Fraud
1 = Fraud
```

---

# Training the Model

The dataset is intentionally not included in this Git repository.

Place the dataset locally inside the `data` directory.

Example:

```text
financial-fraud-detection/
└── data/
    └── dataset.csv
```

Then run:

```bash
python train_model.py
```

The training script creates the trained model and feature configuration.

The model is saved locally as:

```text
models/fraud_detection_xgboost.pkl
```

The feature configuration is saved as:

```text
models/feature_columns.json
```

---

# Why the Dataset and PKL File Are Not in Git

The dataset and trained model are intentionally excluded from the repository.

Reasons include:

- The dataset is large.
- The trained model is a generated binary file.
- Keeping large generated files out of Git keeps the repository lightweight.
- The model can be reproduced by running the training script.
- The dataset is kept local rather than distributed through the source repository.

The repository therefore contains the source code required to reproduce the project rather than the large dataset and generated model binary.

---

# Model Evaluation

The project includes a separate evaluation script:

```bash
python evaluate_model.py
```

The evaluation measures:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Classification Report

An example held-out evaluation from development produced:

```text
Accuracy:          0.999977
Precision:         0.984976
Recall:            0.997565
F1 Score:          0.991231
ROC-AUC:           0.999651
PR-AUC:            0.998527
```

Example confusion matrix:

```text
[[1270856      25]
 [      4    1639]]
```

These results are specific to the dataset and evaluation methodology used during development. They do not guarantee the same performance on new real-world financial transactions.

---

# Understanding the Evaluation Metrics

## Accuracy

The percentage of all predictions that were correct.

```text
Correct Predictions
-------------------
All Predictions
```

Accuracy alone is not sufficient for fraud detection because the dataset is highly imbalanced.

---

## Precision

Of the transactions predicted as fraud, precision measures how many were actually fraud.

```text
True Positives
-------------------------
True Positives + False Positives
```

High precision means fewer legitimate transactions are incorrectly flagged.

---

## Recall

Recall measures how many actual fraudulent transactions the model successfully detected.

```text
True Positives
------------------------
True Positives + False Negatives
```

High recall is important in fraud detection because missed fraud can be costly.

---

## F1 Score

F1 is the harmonic mean of precision and recall.

It provides a single metric that balances the two.

```text
F1 = 2 × Precision × Recall
     ------------------------
       Precision + Recall
```

---

## ROC-AUC

ROC-AUC measures how well the model separates fraudulent transactions from legitimate transactions across different classification thresholds.

A value closer to 1 indicates stronger separation on the evaluated data.

---

## PR-AUC

PR-AUC summarizes the relationship between precision and recall across different thresholds.

It is particularly useful for imbalanced classification problems such as fraud detection.

---

## Confusion Matrix

The confusion matrix contains:

```text
                 Predicted
              Not Fraud   Fraud

Actual
Not Fraud       TN          FP

Fraud           FN          TP
```

Where:

```text
TN = True Negative
FP = False Positive
FN = False Negative
TP = True Positive
```

For fraud detection:

- False Positive = legitimate transaction incorrectly flagged as fraud
- False Negative = fraudulent transaction incorrectly classified as legitimate

---

# Flask Backend

The web application uses Flask as the backend framework.

Flask is responsible for:

1. Starting the web server
2. Loading the trained model
3. Receiving transaction data
4. Validating input
5. Creating model features
6. Calling the XGBoost model
7. Calculating fraud probability
8. Returning the prediction
9. Logging successful analyses

---

# API Response

A successful prediction returns information similar to:

```json
{
    "success": true,
    "fraud": false,
    "fraud_probability": 0.0000137,
    "fraud_percentage": 0.00137,
    "status": "Not Fraud"
}
```

A fraudulent transaction can return:

```json
{
    "success": true,
    "fraud": true,
    "fraud_probability": 0.999992,
    "fraud_percentage": 99.9992,
    "status": "Fraud"
}
```

---

# Web Interface

The frontend allows the user to enter:

- Transaction type
- Transaction amount
- Origin account
- Destination account
- Origin balance before
- Origin balance after
- Destination balance before
- Destination balance after

After clicking **Analyze Transaction**, the frontend sends the transaction to the Flask backend.

The backend returns the prediction and probability.

The interface then displays:

- Risk assessment
- Fraud / Not Fraud
- Fraud probability
- Risk level
- Transaction history

---

# Transaction Logging

The application maintains a transaction history using JSON-based logging.

Successful analyses can be stored with information such as:

```text
Timestamp
Transaction type
Transaction amount
Account information
Balance information
Fraud status
Fraud probability
```

This allows previously analyzed transactions to be displayed in the application's transaction history.

---

# Project Structure

```text
financial-fraud-detection/
│
├── data/
│   └── dataset.csv                  # Local only - not committed
│
├── models/
│   ├── fraud_detection_xgboost.pkl  # Local only - not committed
│   └── feature_columns.json         # Generated feature configuration
│
├── templates/
│   └── index.html                   # Frontend
│
├── utils/
│   └── ...                          # Utility modules
│
├── app.py                           # Flask application
├── train_model.py                   # Model training
├── evaluate_model.py                # Model evaluation
├── validation.py                    # Input/data validation
├── logger.py                        # Transaction logging
├── check_model.py                   # Model verification
├── test_prediction.py               # Prediction test
├── test_legitimate_prediction.py    # Legitimate transaction test
├── test_predictor.py                # Predictor/API test
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git exclusions
└── README.md                        # Documentation
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/financial-fraud-detection.git
cd financial-fraud-detection
```

Replace `YOUR_USERNAME` with your GitHub username.

---

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

After installing dependencies and generating the model:

```bash
python app.py
```

Open the local Flask address:

```text
http://127.0.0.1:5000/
```

The web interface will then be available in the browser.

---

# Verifying the Model

Run:

```bash
python check_model.py
```

This verifies that the saved XGBoost model can be loaded and checks its configuration.

The trained model expects 14 features.

---

# Testing Predictions

Individual prediction tests can be run using:

```bash
python test_prediction.py
```

The project also contains additional prediction and legitimate/fraud test scripts.

A legitimate transaction may produce:

```text
Fraud: False
Fraud probability: 0.0014%
```

A fraudulent test transaction may produce:

```text
Fraud: True
Fraud probability: 99.9992%
```

---

# Git Configuration

The dataset and trained model should not be committed.

Recommended `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]

# Virtual environment
venv/
.venv/

# Dataset
data/
*.csv

# Trained model
models/*.pkl

# Logs
*.log

# Environment variables
.env

# IDE
.vscode/
.idea/

# Operating system
.DS_Store
Thumbs.db
```

The feature configuration file can remain tracked:

```text
models/feature_columns.json
```

while the large trained model remains local:

```text
models/fraud_detection_xgboost.pkl
```

---

# Important: Git Tracking

If the dataset or model was already added to Git before `.gitignore` was created, `.gitignore` will not automatically remove it from Git's index.

Remove them from tracking with:

```bash
git rm --cached models/fraud_detection_xgboost.pkl
```

For the dataset:

```bash
git rm --cached data/dataset.csv
```

Then commit the change:

```bash
git add .
git commit -m "Exclude dataset and trained model"
```

---

# Technologies Used

## Python

Primary programming language.

Used for:

- Data processing
- Machine learning
- Feature engineering
- Backend development
- Testing
- Validation
- Logging

## XGBoost

Used for the fraud classification model.

## Pandas

Used for loading and processing transaction data.

## NumPy

Used for numerical operations and model input preparation.

## Scikit-learn

Used for:

- Train/test splitting
- Evaluation metrics
- Classification reports
- Confusion matrices
- ROC-AUC
- PR-AUC

## Flask

Used to create the backend server and API.

## HTML / CSS / JavaScript

Used to create the browser-based frontend.

## JSON

Used for API communication, feature configuration, and transaction logging.

---

# Testing Strategy

The project was tested at multiple levels.

### Model Loading Test

Checks that the trained XGBoost model can be loaded successfully.

### Legitimate Transaction Test

Tests the model using a transaction expected to be legitimate.

### Fraud Transaction Test

Tests the model using a known fraudulent transaction.

### Predictor Test

Checks the prediction pipeline and returned probability.

### Held-Out Evaluation

Evaluates the model on data separated from the training process.

### Flask/API Test

Checks communication between the frontend and backend.

---

# Limitations

This project is an educational and demonstration system and is not a production banking fraud prevention system.

Important limitations include:

- Predictions depend on the quality and distribution of the training data.
- Historical fraud patterns may change over time.
- A high fraud probability does not prove that a transaction is fraudulent.
- A low fraud probability does not prove that a transaction is legitimate.
- The model can make false positives and false negatives.
- Real financial systems require additional security controls.
- Production systems require continuous monitoring and model retraining.
- Real-world fraud detection normally combines machine learning with business rules, transaction history, identity information, behavioral signals, and human review.

---

# Future Improvements

Potential improvements include:

- Optimize the classification threshold using validation data
- Hyperparameter tuning
- Cross-validation
- SHAP-based model explainability
- Additional behavioral features
- Transaction velocity features
- Historical account behavior
- Database-backed transaction history
- Authentication and authorization
- API rate limiting
- Model monitoring
- Data drift detection
- Automated model retraining
- Production deployment
- Human review workflows

---

# End-to-End Workflow

The complete project demonstrates the following machine learning and software engineering workflow:

```text
Historical Data
      |
      v
Data Preparation
      |
      v
Feature Engineering
      |
      v
Train/Test Split
      |
      v
Class Imbalance Handling
      |
      v
XGBoost Training
      |
      v
Model Evaluation
      |
      v
Model Serialization
      |
      v
Flask Backend
      |
      v
Web Interface
      |
      v
Transaction Input
      |
      v
Feature Engineering
      |
      v
XGBoost Prediction
      |
      v
Fraud Probability
      |
      v
Fraud / Not Fraud
      |
      v
Transaction Logging
```

---

# Project Objective

The objective of this project is to demonstrate a complete end-to-end machine learning application for financial fraud detection.

It combines:

- Data processing
- Feature engineering
- Imbalanced classification
- XGBoost
- Model evaluation
- Model serialization
- Flask API development
- Frontend development
- Input validation
- Prediction testing
- Transaction logging

The project demonstrates how a trained machine learning model can be integrated into a usable web application rather than being limited to a standalone training notebook.

---

# Disclaimer

This project is intended for educational and demonstration purposes.

Fraud probabilities are model-generated estimates and should not be treated as definitive evidence of financial fraud.

A real-world financial fraud detection platform would require additional security, compliance, privacy controls, monitoring, explainability, model governance, and human or rule-based review.
