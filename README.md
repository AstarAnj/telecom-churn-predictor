# Telecom Customer Churn Predictor

An interactive machine learning application built with Streamlit to analyze customer churn risk in a telecom setting. The project covers the full workflow from raw data ingestion and feature creation to model training, imbalance treatment, threshold tuning, and single-customer prediction.

## Overview

Customer churn is a common business problem where the goal is to identify customers who are likely to leave a service. This project uses customer profile, service usage, billing, and contract information to train classification models that estimate churn probability.

The application is designed to demonstrate not only model building, but also how model decisions change when business priorities shift. Instead of relying only on accuracy, the project highlights recall, precision, F1-score, ROC-AUC, and threshold selection.

## Main Capabilities

- Load churn data from a local CSV file
- Clean and prepare billing fields such as `TotalCharges`
- Engineer additional churn-related features from account behavior
- Train multiple machine learning models
- Address class imbalance using SMOTE
- Compare model performance across key classification metrics
- Tune the decision threshold to study the precision-recall tradeoff
- Predict churn risk for an individual customer through an interactive form

## Project Workflow

### 1. Data Loading
The application reads a telecom churn dataset from the local `data` folder and validates key fields before modeling.

### 2. Feature Engineering
Additional features are derived to improve predictive performance and business interpretability, including:

- `AvgMonthlyCharge`
- `ChargeRatio`
- `NumServices`
- `HasInternet`
- `IsAutoPayment`
- `tenure_bin`

These features help capture spending consistency, service stickiness, and customer lifecycle stage.

### 3. Preprocessing
The dataset is prepared for modeling by:

- removing non-modeling identifiers
- encoding categorical variables
- splitting into train and test sets
- scaling features for model training

### 4. Imbalanced Learning
Because churn data is typically skewed toward non-churn customers, the project applies **SMOTE** to improve minority class learning and increase sensitivity toward churn cases.

### 5. Model Training
The application compares multiple supervised learning models:

- Logistic Regression
- Random Forest
- Gradient Boosting

### 6. Threshold Tuning
Predicted churn probabilities are converted into class labels using a configurable threshold. This allows users to inspect how changing the cut-off affects:

- precision
- recall
- F1-score
- confusion matrix behavior

This is important in retention use cases where the cost of missing a real churner may be higher than contacting an extra low-risk customer.

### 7. Live Prediction
Users can manually input customer details in the Streamlit interface and receive:

- churn probability
- predicted churn status
- general retention interpretation

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- Matplotlib
- Seaborn
- Streamlit

## Folder Structure

```text
telecom-churn-predictor/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── Churn.csv