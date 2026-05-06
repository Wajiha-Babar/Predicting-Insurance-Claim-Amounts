# app.py

import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pathlib import Path


# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Insurance Claim Prediction Dashboard",
    page_icon="💎",
    layout="wide"
)


# -------------------------------
# Luxury CSS Styling
# -------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #070A13 0%, #111827 50%, #1B1B2F 100%);
        color: #F8F4E3;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #F5D76E;
        text-align: center;
        margin-bottom: 8px;
    }

    .sub-title {
        font-size: 17px;
        color: #E5E7EB;
        text-align: center;
        margin-bottom: 30px;
    }

    .metric-card {
        background: linear-gradient(135deg, #111827, #1F2937);
        border: 1px solid #D4AF37;
        border-radius: 20px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(212, 175, 55, 0.15);
    }

    .metric-label {
        color: #C9C9C9;
        font-size: 15px;
    }

    .metric-value {
        color: #F5D76E;
        font-size: 30px;
        font-weight: 800;
    }

    section[data-testid="stSidebar"] {
        background-color: #0B1020;
        border-right: 1px solid #D4AF37;
    }

    .stButton>button {
        background: linear-gradient(90deg, #D4AF37, #F5D76E);
        color: #0B1020;
        font-weight: 700;
        border-radius: 12px;
        padding: 0.7rem 1.2rem;
        border: none;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #F5D76E, #D4AF37);
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------
# Paths
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "medical_insurance.csv"
MODEL_PATH = BASE_DIR / "models" / "insurance_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
OPTIONS_PATH = BASE_DIR / "models" / "feature_options.json"

TARGET = "annual_medical_cost"


# -------------------------------
# Load Files
# -------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


@st.cache_data
def load_options():
    with open(OPTIONS_PATH, "r") as f:
        return json.load(f)


df = load_data()
model = load_model()
metrics = load_metrics()
options = load_options()


# -------------------------------
# Header
# -------------------------------

st.markdown('<div class="main-title">💎 Insurance Claim Amount Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A premium regression dashboard to estimate annual medical insurance claim amounts using personal and health data.</div>',
    unsafe_allow_html=True
)


# -------------------------------
# Sidebar Inputs
# -------------------------------

st.sidebar.title("🔮 Enter Customer Details")

age = st.sidebar.slider(
    "Age",
    int(options["age"]["min"]),
    int(options["age"]["max"]),
    int(options["age"]["median"])
)

bmi = st.sidebar.slider(
    "BMI",
    float(options["bmi"]["min"]),
    float(options["bmi"]["max"]),
    float(options["bmi"]["median"])
)

income = st.sidebar.number_input(
    "Income",
    min_value=float(options["income"]["min"]),
    max_value=float(options["income"]["max"]),
    value=float(options["income"]["median"])
)

visits_last_year = st.sidebar.slider(
    "Doctor Visits Last Year",
    int(options["visits_last_year"]["min"]),
    int(options["visits_last_year"]["max"]),
    int(options["visits_last_year"]["median"])
)

hospitalizations = st.sidebar.slider(
    "Hospitalizations Last 3 Years",
    int(options["hospitalizations_last_3yrs"]["min"]),
    int(options["hospitalizations_last_3yrs"]["max"]),
    int(options["hospitalizations_last_3yrs"]["median"])
)

medication_count = st.sidebar.slider(
    "Medication Count",
    int(options["medication_count"]["min"]),
    int(options["medication_count"]["max"]),
    int(options["medication_count"]["median"])
)

deductible = st.sidebar.selectbox(
    "Deductible",
    sorted(df["deductible"].dropna().unique())
)

copay = st.sidebar.selectbox(
    "Copay",
    sorted(df["copay"].dropna().unique())
)

risk_score = st.sidebar.slider(
    "Risk Score",
    float(options["risk_score"]["min"]),
    float(options["risk_score"]["max"]),
    float(options["risk_score"]["median"])
)

chronic_count = st.sidebar.slider(
    "Chronic Disease Count",
    int(options["chronic_count"]["min"]),
    int(options["chronic_count"]["max"]),
    int(options["chronic_count"]["median"])
)

hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
had_major_procedure = st.sidebar.selectbox("Had Major Procedure", [0, 1])

sex = st.sidebar.selectbox("Sex", options["sex"])
region = st.sidebar.selectbox("Region", options["region"])
urban_rural = st.sidebar.selectbox("Area Type", options["urban_rural"])
education = st.sidebar.selectbox("Education", options["education"])
smoker = st.sidebar.selectbox("Smoking Status", options["smoker"])
plan_type = st.sidebar.selectbox("Plan Type", options["plan_type"])
network_tier = st.sidebar.selectbox("Network Tier", options["network_tier"])


# -------------------------------
# Prepare Input
# -------------------------------

input_data = pd.DataFrame({
    "age": [age],
    "income": [income],
    "bmi": [bmi],
    "visits_last_year": [visits_last_year],
    "hospitalizations_last_3yrs": [hospitalizations],
    "medication_count": [medication_count],
    "deductible": [deductible],
    "copay": [copay],
    "risk_score": [risk_score],
    "chronic_count": [chronic_count],
    "hypertension": [hypertension],
    "diabetes": [diabetes],
    "had_major_procedure": [had_major_procedure],
    "sex": [sex],
    "region": [region],
    "urban_rural": [urban_rural],
    "education": [education],
    "smoker": [smoker],
    "plan_type": [plan_type],
    "network_tier": [network_tier]
})


prediction = model.predict(input_data)[0]


# -------------------------------
# Metrics Cards
# -------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Claim Amount</div>
            <div class="metric-value">${prediction:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">MAE</div>
            <div class="metric-value">${metrics["MAE"]:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">${metrics["RMSE"]:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">{metrics["R2_Score"]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")
st.write("")


# -------------------------------
# Tabs
# -------------------------------

tab1, tab2, tab3 = st.tabs(["📊 Prediction Insights", "🔥 EDA Visuals", "📌 Input Summary"])


# -------------------------------
# Tab 1: Prediction Insights
# -------------------------------

with tab1:
    st.subheader("Prediction Compared with Dataset Trends")

    sample_df = df.sample(min(6000, len(df)), random_state=42)

    fig_age = px.scatter(
        sample_df,
        x="age",
        y=TARGET,
        color="smoker",
        title="Age vs Insurance Claim Amount",
        labels={
            "age": "Age",
            TARGET: "Annual Medical Cost",
            "smoker": "Smoker"
        },
        template="plotly_dark"
    )

    fig_age.add_trace(
        go.Scatter(
            x=[age],
            y=[prediction],
            mode="markers",
            marker=dict(size=18, color="#F5D76E", symbol="star"),
            name="Your Prediction"
        )
    )

    fig_age.update_layout(
        paper_bgcolor="#0B1020",
        plot_bgcolor="#111827",
        font=dict(color="#F8F4E3"),
        title_font=dict(size=22, color="#F5D76E")
    )

    st.plotly_chart(fig_age, use_container_width=True)


# -------------------------------
# Tab 2: EDA Visuals
# -------------------------------

with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        fig_bmi = px.scatter(
            df.sample(min(6000, len(df)), random_state=22),
            x="bmi",
            y=TARGET,
            color="smoker",
            title="BMI vs Insurance Charges",
            template="plotly_dark"
        )

        fig_bmi.add_trace(
            go.Scatter(
                x=[bmi],
                y=[prediction],
                mode="markers",
                marker=dict(size=18, color="#F5D76E", symbol="star"),
                name="Your Prediction"
            )
        )

        fig_bmi.update_layout(
            paper_bgcolor="#0B1020",
            plot_bgcolor="#111827",
            font=dict(color="#F8F4E3"),
            title_font=dict(color="#F5D76E")
        )

        st.plotly_chart(fig_bmi, use_container_width=True)

    with col_b:
        smoker_avg = df.groupby("smoker")[TARGET].mean().reset_index()

        fig_smoker = px.bar(
            smoker_avg,
            x="smoker",
            y=TARGET,
            title="Average Claim Amount by Smoking Status",
            template="plotly_dark",
            text_auto=".2s"
        )

        fig_smoker.update_traces(marker_color="#D4AF37")

        fig_smoker.update_layout(
            paper_bgcolor="#0B1020",
            plot_bgcolor="#111827",
            font=dict(color="#F8F4E3"),
            title_font=dict(color="#F5D76E")
        )

        st.plotly_chart(fig_smoker, use_container_width=True)

    region_avg = df.groupby("region")[TARGET].mean().reset_index()

    fig_region = px.bar(
        region_avg,
        x="region",
        y=TARGET,
        title="Average Claim Amount by Region",
        template="plotly_dark",
        text_auto=".2s"
    )

    fig_region.update_traces(marker_color="#C9A227")

    fig_region.update_layout(
        paper_bgcolor="#0B1020",
        plot_bgcolor="#111827",
        font=dict(color="#F8F4E3"),
        title_font=dict(color="#F5D76E")
    )

    st.plotly_chart(fig_region, use_container_width=True)


# -------------------------------
# Tab 3: Input Summary
# -------------------------------

with tab3:
    st.subheader("Customer Input Summary")

    st.dataframe(input_data, use_container_width=True)

    st.success(
        f"Based on the provided inputs, the estimated annual insurance claim amount is: ${prediction:,.2f}"
    )