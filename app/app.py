import os
import sys
import streamlit as st
import pandas as pd

# Make sure Python can find the src/ package
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.data_cleaning import load_data, basic_cleaning
from src.insights_engine import (
    superstore_kpis,
    sales_trend,
    profit_trend,
    forecast_sales,
    churn_kpis,
    auto_insights_superstore,
    auto_insights_churn,
)

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Business Insights Platform",
    layout="wide",
)

st.title("📊 AI-Augmented Business Insights Platform")
st.caption("Superstore Sales + Telco Churn | Built by Rendell Rocky Padu")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_processed_data():
    # Adjust file names if needed
    superstore_raw = load_data("data/raw/superstore.csv")
    churn_raw = load_data("data/raw/Telco_customer_churn.csv")

    super_clean = basic_cleaning(superstore_raw)
    churn_clean = basic_cleaning(churn_raw)

    return super_clean, churn_clean

df_super, df_churn = load_processed_data()

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a view",
    ["Superstore Overview", "Churn Overview", "AI Insights"],
)

st.sidebar.markdown("---")
st.sidebar.write(f"Superstore rows: **{len(df_super)}**")
st.sidebar.write(f"Churn rows: **{len(df_churn)}**")

# ------------------------------------------------------------
# SUPERSTORE OVERVIEW
# ------------------------------------------------------------
if page == "Superstore Overview":
    st.subheader("🛒 Superstore Sales & Profit Overview")

    kpis = superstore_kpis(df_super)
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("Total Sales", f"${kpis['Total Sales']:,.0f}")
    col2.metric("Total Profit", f"${kpis['Total Profit']:,.0f}")
    col3.metric("Avg Discount", f"{kpis['Avg Discount (%)']:.2f}%")

    col4.metric("Total Orders", f"{kpis['Total Orders']:,}")
    col5.metric("Top Category", kpis["Top Category"])
    col6.metric("Most Profitable Region", kpis["Most Profitable Region"])

    st.markdown("---")
    st.markdown("### 📈 Monthly Sales & Profit Trend")

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Monthly Sales")
        st.line_chart(sales_trend(df_super), x="month", y="Sales")

    with col_b:
        st.write("Monthly Profit")
        st.line_chart(profit_trend(df_super), x="month", y="Profit")

    st.markdown("---")

    st.subheader("📈 Sales Forecast (Next 6 Months)")

    forecast = forecast_sales(df_super)

    forecast_plot = (
        forecast.rename(columns={"ds": "Date", "yhat": "Forecasted Sales"})
        [["Date", "Forecasted Sales"]]
        .copy()
    )

    # Ensure correct dtypes
    forecast_plot["Date"] = pd.to_datetime(forecast_plot["Date"], errors="coerce")
    forecast_plot["Forecasted Sales"] = pd.to_numeric(
        forecast_plot["Forecasted Sales"], errors="coerce"
    )

    # Drop any rows where we couldn't parse properly
    forecast_plot = forecast_plot.dropna(subset=["Date", "Forecasted Sales"])

    # ✅ Explicitly tell Streamlit which column is y
    st.line_chart(forecast_plot, x="Date", y="Forecasted Sales")



# ------------------------------------------------------------
# CHURN OVERVIEW
# ------------------------------------------------------------
elif page == "Churn Overview":
    st.subheader("📞 Telco Customer Churn Overview")

    kpis = churn_kpis(df_churn)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Churn Rate", f"{kpis['Churn Rate (%)']:.2f}%")
    col2.metric("Avg Monthly Charge", f"${kpis['Avg Monthly Charge']:,.2f}")
    col3.metric("Avg Tenure", f"{kpis['Avg Tenure (months)']:.1f} months")
    col4.metric("Top Churn Reason", kpis["Top Churn Reason"])

    st.markdown("---")
    st.markdown("### 🔍 Churn Breakdown by Contract & Payment Method")

    col_a, col_b = st.columns(2)
    with col_a:
        if "Contract" in df_churn.columns:
            contract_churn = (
                df_churn.groupby("Contract")["Churn Value"].mean().reset_index()
            )
            contract_churn["Churn Rate (%)"] = contract_churn["Churn Value"] * 100
            st.bar_chart(contract_churn, x="Contract", y="Churn Rate (%)")
        else:
            st.info("No 'Contract' column found.")

    with col_b:
        if "Payment Method" in df_churn.columns:
            pay_churn = (
                df_churn.groupby("Payment Method")["Churn Value"].mean().reset_index()
            )
            pay_churn["Churn Rate (%)"] = pay_churn["Churn Value"] * 100
            st.bar_chart(pay_churn, x="Payment Method", y="Churn Rate (%)")
        else:
            st.info("No 'Payment Method' column found.")


# ------------------------------------------------------------
# AI INSIGHTS PAGE
# ------------------------------------------------------------
elif page == "AI Insights":
    st.subheader("🤖 AI-Augmented Insights")

    st.markdown("#### 🛒 Superstore Insights")
    insights_super = auto_insights_superstore(df_super)
    if insights_super:
        for text in insights_super:
            st.write(text)
    else:
        st.info("No specific insights generated for Superstore dataset.")

    st.markdown("---")
    st.markdown("#### 📞 Churn Insights")

    insights_churn = auto_insights_churn(df_churn)
    if insights_churn:
        for text in insights_churn:
            st.write(text)
    else:
        st.info("No specific insights generated for Churn dataset.")
