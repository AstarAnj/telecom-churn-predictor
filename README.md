# Telecom Customer Churn Predictor

> End-to-end ML pipeline for customer churn prediction — built with Python, scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

This project tackles **telecom customer churn prediction** — identifying which customers are likely to cancel their service before they do. It covers the full ML lifecycle from raw data to a live interactive predictor.

**Dataset:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 20 features, ~26% churn rate.

---

## Pipeline

| Step | What happens |
|------|-------------|
| **EDA** | Churn breakdown by contract type, tenure segment, charges distribution |
| **Feature engineering** | 6 domain-driven features derived from raw columns |
| **Imbalance handling** | SMOTE oversampling on the minority (churn) class |
| **Model training** | Logistic Regression · Random Forest · Gradient Boosting |
| **Threshold tuning** | Interactive precision/recall trade-off with live confusion matrix |
| **Live inference** | Predict churn probability for any customer profile |

---

## Engineered Features

| Feature | Logic | Signal |
|---------|-------|--------|
| `tenure_bin` | Bucket tenure → New / Mid / Loyal / Long-term | Non-linear churn decay |
| `AvgMonthlyCharge` | TotalCharges ÷ tenure | Spending consistency |
| `ChargeRatio` | MonthlyCharges ÷ (TotalCharges + 1) | High ratio → recent joiner |
| `NumServices` | Count of opted-in add-on services | More services = stickier |
| `HasInternet` | Binary: any internet service | Internet users churn differently |
| `IsAutoPayment` | Binary: automatic payment method | Manual payers leave more often |

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.7982 | 0.6364 | 0.5452 | 0.5872 | 0.8448 |
| Random Forest | 0.7954 | 0.6383 | 0.5169 | 0.5711 | 0.8373 |
| **Gradient Boosting** | **0.8046** | **0.6547** | **0.5452** | **0.5950** | **0.8554** |

*Evaluated at threshold = 0.50 with SMOTE. Metrics vary with threshold — see the app.*

---

## Project Structure

```
telecom-churn-predictor/
├── app.py              # Streamlit application
├── data/
│   └── churn.csv       # Kaggle Telco dataset (download separately)
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AstarAnj/telecom-churn-predictor.git
cd telecom-churn-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), rename it to `churn.csv`, and place it in the `data/` folder.

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Tech Stack

- **Python 3.9+**
- **Streamlit** — interactive web app
- **scikit-learn** — model training, evaluation, preprocessing
- **imbalanced-learn** — SMOTE oversampling
- **pandas / NumPy** — data manipulation
- **matplotlib / seaborn** — visualisation

---

## Key Decisions

**Why SMOTE?** With ~26% churn, training without resampling biases the model toward the majority class. SMOTE generates synthetic minority samples in feature space rather than simply duplicating rows, improving recall on the churn class.

**Why threshold tuning?** Default 0.50 is rarely optimal for imbalanced business problems. A retention campaign costs ~$20; losing a customer costs ~$500 — the asymmetric cost justifies a lower threshold to maximise recall at the expense of some precision.

**Why Gradient Boosting over Random Forest?** Boosting sequentially corrects errors from prior trees, making it more sensitive to the minority class patterns even after SMOTE.

---

## License

MIT