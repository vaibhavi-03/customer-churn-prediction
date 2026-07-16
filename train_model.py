"""
train_model.py
Run this once locally to generate model artifacts:
    python train_model.py
Outputs: model/churn_model.pkl, model/scaler.pkl, model/feature_cols.pkl
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ── Load & clean ────────────────────────────────────────────────
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop(columns=['customerID'], inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# ── Encode ──────────────────────────────────────────────────────
cat_cols = df.select_dtypes(include='object').columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ── Split ───────────────────────────────────────────────────────
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale ───────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Train ───────────────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# ── Evaluate ────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
print("\n── Classification Report ──────────────────")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# ── Save artifacts ──────────────────────────────────────────────
Path("model").mkdir(exist_ok=True)
joblib.dump(model,        "model/churn_model.pkl")
joblib.dump(scaler,       "model/scaler.pkl")
joblib.dump(X.columns.tolist(), "model/feature_cols.pkl")

print("✅ Saved: model/churn_model.pkl")
print("✅ Saved: model/scaler.pkl")
print("✅ Saved: model/feature_cols.pkl")
