import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight | Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Root & background */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
.stApp {
    background: #0a0e1a;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1424 !important;
    border-right: 1px solid #1e2a45;
}
[data-testid="stSidebar"] * {
    color: #c8d6f0 !important;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0f1e3d 0%, #1a0b2e 50%, #0d1f3c 100%);
    border: 1px solid #1e3a6e;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #e8f0ff;
    letter-spacing: -0.5px;
    margin: 0;
}
.hero-title span {
    background: linear-gradient(90deg, #6366f1, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: #7a90b8;
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a78bfa;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 1rem;
}

/* Metric cards */
.metric-card {
    background: #0f1729;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #6366f1; }
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label {
    color: #5a7099;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.25rem;
}

/* Result card */
.result-churn {
    background: linear-gradient(135deg, #1f0a0a, #2d0f0f);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-safe {
    background: linear-gradient(135deg, #0a1f0f, #0f2d1a);
    border: 2px solid #22c55e;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-title {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.result-subtitle {
    color: #8899aa;
    font-size: 0.9rem;
}

/* Feature pills */
.feature-pill {
    display: inline-block;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a78bfa;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 0.15rem;
}

/* Section headers */
.section-header {
    color: #e8f0ff;
    font-size: 1.1rem;
    font-weight: 600;
    border-left: 3px solid #6366f1;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}

/* Streamlit overrides */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    color: #8899bb !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
.stSelectbox > div > div {
    background: #0f1729 !important;
    border-color: #1e2d4a !important;
    color: #c8d6f0 !important;
}
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("model/churn_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    feature_cols = joblib.load("model/feature_cols.pkl")
    return model, scaler, feature_cols

try:
    model, scaler, feature_cols = load_model()
    model_loaded = True
except:
    model_loaded = False


# ── Hero ───────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">📡 ML-Powered · Logistic Regression · Telco Dataset</div>
    <h1 class="hero-title">Churn<span>Sight</span></h1>
    <p class="hero-sub">Predict customer churn risk in real-time — before it's too late.</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar inputs ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Customer Profile")
    st.markdown("---")

    st.markdown('<p class="section-header">Demographics</p>', unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    st.markdown('<p class="section-header">Services</p>', unsafe_allow_html=True)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.markdown('<p class="section-header">Account</p>', unsafe_allow_html=True)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0,
                                     value=float(tenure * monthly_charges), step=10.0)

    predict_btn = st.button("🔍 Predict Churn Risk")


# ── Main content ───────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
stats = [
    ("7,043", "Customers Analysed", "#6366f1"),
    ("26.5%", "Baseline Churn Rate", "#ef4444"),
    ("80%", "Model Accuracy", "#22c55e"),
    ("57%", "Churn Recall", "#f59e0b"),
]
for col, (val, label, color) in zip([col1, col2, col3, col4], stats):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Prediction ─────────────────────────────────────────────────
if predict_btn:
    if not model_loaded:
        st.error("⚠️ Model files not found. Run `train_model.py` first to generate them.")
    else:
        # Build input row
        input_dict = {
            'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0,
            'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
            'PhoneService': phone_service, 'MultipleLines': multiple_lines,
            'InternetService': internet_service, 'OnlineSecurity': online_security,
            'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
            'TechSupport': tech_support, 'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies, 'Contract': contract,
            'PaperlessBilling': paperless, 'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
        }
        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)
        input_scaled = scaler.transform(input_encoded)

        prob = model.predict_proba(input_scaled)[0][1]
        pred = model.predict(input_scaled)[0]

        # Result
        res_col, gauge_col = st.columns([1, 1])

        with res_col:
            if pred == 1:
                st.markdown(f"""
                <div class="result-churn">
                    <div style="font-size:3rem">⚠️</div>
                    <div class="result-title" style="color:#ef4444">High Churn Risk</div>
                    <div style="font-size:2.5rem;font-weight:700;color:#ef4444;font-family:'JetBrains Mono',monospace">{prob*100:.1f}%</div>
                    <div class="result-subtitle">probability of churning</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                    <div style="font-size:3rem">✅</div>
                    <div class="result-title" style="color:#22c55e">Low Churn Risk</div>
                    <div style="font-size:2.5rem;font-weight:700;color:#22c55e;font-family:'JetBrains Mono',monospace">{prob*100:.1f}%</div>
                    <div class="result-subtitle">probability of churning</div>
                </div>""", unsafe_allow_html=True)

        with gauge_col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': '%', 'font': {'size': 36, 'color': '#e8f0ff', 'family': 'JetBrains Mono'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#3a4a6a', 'tickfont': {'color': '#5a7099'}},
                    'bar': {'color': '#ef4444' if pred == 1 else '#22c55e', 'thickness': 0.25},
                    'bgcolor': '#0f1729',
                    'bordercolor': '#1e2d4a',
                    'steps': [
                        {'range': [0, 40], 'color': '#0a1f0f'},
                        {'range': [40, 70], 'color': '#1f1a0a'},
                        {'range': [70, 100], 'color': '#1f0a0a'},
                    ],
                    'threshold': {'line': {'color': '#a78bfa', 'width': 2}, 'value': 50}
                }
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#e8f0ff'},
                height=250,
                margin=dict(t=20, b=10, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Risk factors
        st.markdown('<p class="section-header">🔍 Key Risk Signals Detected</p>', unsafe_allow_html=True)
        risk_factors = []
        if contract == "Month-to-month": risk_factors.append("📋 Month-to-month contract (highest churn risk)")
        if tenure < 12: risk_factors.append(f"⏱️ Low tenure ({tenure} months) — new customers churn more")
        if monthly_charges > 70: risk_factors.append(f"💸 High monthly charges (${monthly_charges:.0f})")
        if internet_service == "Fiber optic": risk_factors.append("🌐 Fiber optic users show higher churn rates")
        if online_security == "No": risk_factors.append("🔒 No online security add-on")
        if tech_support == "No": risk_factors.append("🛠️ No tech support add-on")
        if payment == "Electronic check": risk_factors.append("💳 Electronic check payment method correlates with churn")

        if risk_factors:
            for rf in risk_factors:
                st.markdown(f"- {rf}")
        else:
            st.success("✅ No major risk signals detected for this customer profile.")

else:
    # Default view — EDA charts
    st.markdown('<p class="section-header">📊 Dataset Insights</p>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig1 = go.Figure(go.Bar(
            x=['No Churn', 'Churn'],
            y=[5174, 1869],
            marker_color=['#6366f1', '#ef4444'],
            text=['73.5%', '26.5%'],
            textposition='outside',
            textfont={'color': '#c8d6f0', 'size': 14}
        ))
        fig1.update_layout(
            title={'text': 'Churn Distribution', 'font': {'color': '#e8f0ff', 'size': 16}},
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#8899bb'}, height=300,
            xaxis={'gridcolor': '#1e2d4a'}, yaxis={'gridcolor': '#1e2d4a'},
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig1, use_container_width=True)

    with chart_col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='No Churn', x=['Month-to-month', 'One year', 'Two year'],
                               y=[0.57, 0.89, 0.97], marker_color='#6366f1'))
        fig2.add_trace(go.Bar(name='Churn', x=['Month-to-month', 'One year', 'Two year'],
                               y=[0.43, 0.11, 0.03], marker_color='#ef4444'))
        fig2.update_layout(
            barmode='stack',
            title={'text': 'Churn Rate by Contract Type', 'font': {'color': '#e8f0ff', 'size': 16}},
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#8899bb'}, height=300,
            xaxis={'gridcolor': '#1e2d4a'}, yaxis={'gridcolor': '#1e2d4a', 'tickformat': ',.0%'},
            legend={'font': {'color': '#c8d6f0'}},
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div style="background:#0f1729;border:1px solid #1e2d4a;border-radius:12px;padding:1.25rem 1.5rem;margin-top:0.5rem">
        <p style="color:#7a90b8;font-size:0.85rem;margin:0">
        👈 <strong style="color:#a78bfa">Fill in the customer profile</strong> in the sidebar and click 
        <strong style="color:#a78bfa">Predict Churn Risk</strong> to get a real-time prediction with risk signal breakdown.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#2a3a5a;font-size:0.8rem;border-top:1px solid #1e2d4a;padding-top:1rem">
    Built by <strong style="color:#6366f1">Vaibhavi Prajapati</strong> · 
    B.Tech CSE '27 · BBD University · 
    <a href="https://github.com/vaibhavi-03" style="color:#6366f1;text-decoration:none">GitHub</a> · 
    <a href="https://linkedin.com/in/vaibhavi-prajapati-66239127b" style="color:#6366f1;text-decoration:none">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
