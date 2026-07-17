# 📡 ChurnSight — Customer Churn Prediction

> End-to-end ML project predicting telecom customer churn using Logistic Regression, deployed via Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vaibhavi-03-customer-churn-prediction.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Deployed-22c55e?style=flat-square)

---

## 🎯 Problem Statement

A telecom company wants to identify customers likely to cancel their subscription (churn) before it happens. This project builds a binary classifier to predict churn risk from customer demographics, services, and billing data — and deploys it as an interactive web app.

---

## 🚀 Live Demo

**[→ Open ChurnSight App](https://vaibhavi-03-customer-churn-prediction.streamlit.app)**

Fill in a customer profile in the sidebar → get real-time churn probability + risk signals.

---

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | 80% |
| Precision (Churn) | 65% |
| Recall (Churn) | 57% |
| F1-Score (Churn) | 61% |

**Key finding:** Month-to-month contract customers churn at ~43% vs only ~3% for two-year contracts — contract type is the single strongest predictor.

---

## 🗂️ Project Structure

```
customer-churn-prediction/
├── app.py                        # Streamlit web app
├── train_model.py                # Model training script → saves pkl files
├── requirements.txt              # Dependencies for Streamlit Cloud
├── .streamlit/
│   └── config.toml               # Dark theme configuration
├── model/
│   ├── churn_model.pkl           # Trained Logistic Regression model
│   ├── scaler.pkl                # Fitted StandardScaler
│   └── feature_cols.pkl          # Feature column names for inference
├── notebooks/
│   ├── 01_data_inspection.ipynb  # Shape, nulls, duplicates, target distribution
│   ├── 02_eda.ipynb              # Visualizations: contract, tenure, charges, heatmap
│   └── 03_preprocessing_baseline.ipynb  # Encoding, scaling, LR baseline
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── daily_log.md                  # Day-by-day learning journal (18 days)
```

---

## 🧭 Project Plan

| Weekend | Goal | Status |
|---|---|---|
| 1 — 27 Jun | Setup + Data Inspection | ✅ Done |
| 2 — 4 Jul | EDA + Visualizations | ✅ Done |
| 3 — 11 Jul | Preprocessing + Baseline Model | ✅ Done |
| 4+5 — 17 Jul | Streamlit App + Deployment | ✅ Done |

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data | Pandas, NumPy |
| ML | scikit-learn (Logistic Regression, StandardScaler) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Deployment | Streamlit Cloud |
| Version Control | Git, GitHub |

---

## 📦 Dataset

[Telco Customer Churn — Kaggle (blastchar)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- **7,043 customers** × 21 features
- Target: `Churn` (Yes / No) — 26.5% positive class
- Features: demographics, phone/internet services, contract type, billing info

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/vaibhavi-03/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates pkl files)
python train_model.py

# 4. Run the app
python -m streamlit run app.py
```

---

## 👩‍💻 Author

**Vaibhavi Prajapati** — B.Tech CSE @ BBD University, Lucknow (2023–2027)

[![GitHub](https://img.shields.io/badge/GitHub-vaibhavi--03-181717?style=flat-square&logo=github)](https://github.com/vaibhavi-03)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/vaibhavi-prajapati-66239127b)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-FF5722?style=flat-square&logo=google-chrome&logoColor=white)](https://vaibhavi-03.github.io)