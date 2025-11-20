import os
import sys
import streamlit as st
import pandas as pd
import altair as alt
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------
# PATH SETUP SO WE CAN IMPORT FROM src/
# ------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.ai_agent import build_ai_agent
from src.data_cleaning import load_data, basic_cleaning
from src.insights_engine import (
    superstore_kpis,
    sales_trend,
    profit_trend,
    forecast_sales_prophet,
    prepare_forecast_data,
    detect_anomalies,
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
# LOAD & CACHE DATA
# ------------------------------------------------------------
@st.cache_data
def load_processed_data():
    superstore_raw = load_data("data/raw/superstore.csv")
    churn_raw = load_data("data/raw/Telco_customer_churn.csv")

    super_clean = basic_cleaning(superstore_raw)
    churn_clean = basic_cleaning(churn_raw)

    return super_clean, churn_clean


with st.spinner("Loading and cleaning data..."):
    df_super, df_churn = load_processed_data()

# ------------------------------------------------------------
# CACHE AI AGENT
# ------------------------------------------------------------
@st.cache_resource
def get_ai_agent():
    return build_ai_agent(df_super, df_churn)

# ------------------------------------------------------------
# SIDEBAR – META INFO / FILTERS
# ------------------------------------------------------------
st.sidebar.header("Dataset Summary")
st.sidebar.write(f"🛒 Superstore rows: **{len(df_super)}**")
st.sidebar.write(f"📞 Churn rows: **{len(df_churn)}**")
st.sidebar.markdown("---")
st.sidebar.caption("Use the top tabs to explore Sales, Churn, Forecast, and the AI Agent.")

# ------------------------------------------------------------
# TOP-LEVEL NAVIGATION TABS
# ------------------------------------------------------------
tab_sales, tab_churn, tab_forecast, tab_ai = st.tabs(
    ["📊 Sales Overview", "📞 Churn Overview", "📈 Forecast", "🤖 AI Chat Agent"]
)

# ============================================================
# TAB 1: SALES OVERVIEW
# ============================================================
with tab_sales:
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
    st.markdown("### 💡 Auto-Generated Business Insights")
    insights = auto_insights_superstore(df_super)
    if insights:
        for i in insights:
            st.write(f"- {i}")
    else:
        st.info("No specific insights generated for this dataset yet.")

# ============================================================
# TAB 2: CHURN OVERVIEW
# ============================================================
with tab_churn:
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

    st.markdown("---")
    st.markdown("### 💡 Auto-Generated Churn Insights")
    churn_insights = auto_insights_churn(df_churn)
    if churn_insights:
        for i in churn_insights:
            st.write(f"- {i}")
    else:
        st.info("No specific churn insights generated yet.")

# ============================================================
# TAB 3: FORECAST (PROPHET + ANOMALIES)
# ============================================================
with tab_forecast:
    st.subheader("📈 Sales Forecast & Anomaly Detection")

    st.markdown(
        """
We use **Prophet** (a time-series forecasting model) to predict monthly sales
and detect **anomalies** in the Superstore data.

**What counts as an anomaly?**

- The **actual sales** for a month are **outside** the model's expected range  
  (between `yhat_lower` and `yhat_upper`).
- These months may indicate:
  - 🔻 A **problem** (sudden drop in sales, operational issue, lost customers)  
  - 🔺 An unexpected **win** (successful campaign, seasonal spike, large deal)
        """
    )

    with st.spinner("Training Prophet model and generating forecast..."):
        forecast = forecast_sales_prophet(df_super, periods=6)

    # Prepare data for simple forecast line chart
    forecast_plot = (
        forecast.rename(columns={"ds": "Date", "yhat": "Forecasted Sales"})
        [["Date", "Forecasted Sales"]]
        .copy()
    )

    forecast_plot["Date"] = pd.to_datetime(forecast_plot["Date"], errors="coerce")
    forecast_plot["Forecasted Sales"] = pd.to_numeric(
        forecast_plot["Forecasted Sales"], errors="coerce"
    )
    forecast_plot = forecast_plot.dropna(subset=["Date", "Forecasted Sales"])

    st.markdown("### 🔮 Prophet Forecast (Next 6 Months)")
    st.line_chart(forecast_plot, x="Date", y="Forecasted Sales")

    # Anomaly detection
    monthly_actual = prepare_forecast_data(df_super)
    anomalies, merged = detect_anomalies(monthly_actual, forecast)

    # Prepare data for Altair plot (actual vs forecast)
    plot_df = merged.copy().sort_values("ds")
    plot_df["ds"] = pd.to_datetime(plot_df["ds"])

    chart_df = plot_df.melt(
        id_vars=["ds"],
        value_vars=["y", "yhat"],
        var_name="Type",
        value_name="Sales",
    )

    color_scale = alt.Scale(
        domain=["y", "yhat"],
        range=["#1f77b4", "#ff7f0e"],  # blue = actual, orange = forecast
    )

    st.markdown("### 📉 Actual vs Forecasted Sales")
    chart = (
        alt.Chart(chart_df)
        .mark_line()
        .encode(
            x=alt.X("ds:T", title="Month"),
            y=alt.Y("Sales:Q", title="Sales"),
            color=alt.Color("Type:N", scale=color_scale, title="Legend"),
            tooltip=["ds", "Type", "Sales"],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Lines: **blue = actual sales (y)**, **orange = model forecast (yhat)**")

    st.markdown("### 🔍 Detected Anomalies")
    if anomalies.empty:
        st.success(
            "No anomalies found in the historical period – actual sales stayed within "
            "the model's expected range."
        )
    else:
        st.warning(
            "We found some months where actual sales were unusually high or low "
            "compared to the forecast."
        )
        st.dataframe(
            anomalies[["ds", "y", "yhat", "yhat_lower", "yhat_upper"]],
            use_container_width=True,
        )
        st.caption(
            "Rows where **actual sales (y)** are outside the forecast interval "
            "(yhat_lower – yhat_upper)."
        )

# ============================================================
# TAB 4: AI CHAT AGENT
# ============================================================
with tab_ai:
    st.subheader("🤖 AI Chat Agent – Ask Questions About Your Data")

    st.markdown(
        """
This assistant can answer **natural-language questions** about:

- 🛒 The **Superstore** sales dataset  
- 📞 The **Telco churn** dataset  

Under the hood, it uses a **Pandas-aware LLM agent** and your dataframes.  
It can:
- Summarize KPIs
- Compare segments / regions / plans
- Explain patterns in churn or profit
        """
    )

    st.info(
        "Tip: You can mention dataset names directly, e.g. "
        "`superstore` or `churn`, inside your question."
    )

    st.markdown("**Try one of these example questions:**")
    example_queries = [
        "In the superstore data, which region has the highest total profit and why might that be?",
        "Using the churn dataset, what are the top 3 factors that seem most related to churn?",
        "Compare average sales and profit between the Technology and Furniture categories in superstore.",
        "For churn, how does churn rate differ by contract type and payment method?",
    ]

    cols = st.columns(len(example_queries))
    chosen_example = None
    for i, q in enumerate(example_queries):
        if cols[i].button(f"Example {i+1}"):
            chosen_example = q

    default_text = chosen_example if chosen_example else ""
    user_question = st.text_area(
        "Ask a question about your data:",
        value=default_text,
        placeholder="e.g. What are the main drivers of churn among high-paying customers?",
        height=120,
    )

    if st.button("Run AI Analysis"):
        if not user_question.strip():
            st.warning("Please enter a question first.")
        else:
            try:
                agent = get_ai_agent()
                with st.spinner("Thinking with your data..."):
                    result = agent.invoke({"input": user_question})
                    answer = result.get("output", result)

                st.markdown("### 🧠 AI Answer")
                st.write(answer)

                st.caption(
                    "Note: This answer is generated by an LLM grounded in the "
                    "Superstore and Telco churn dataframes."
                )

            except Exception as e:
                st.error(f"❌ Something went wrong while running the AI agent: {e}")
                st.info(
                    "Check your API key, installed langchain / LLM client packages, "
                    "and that `build_ai_agent` is configured correctly."
                )
