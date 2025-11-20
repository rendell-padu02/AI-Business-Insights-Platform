import pandas as pd
from prophet import Prophet


# ============================================================
# SUPERSTORE KPIs
# ============================================================

def superstore_kpis(df: pd.DataFrame) -> dict:
    df = df.copy()
    return {
        "Total Sales": round(df["Sales"].sum(), 2),
        "Total Profit": round(df["Profit"].sum(), 2),
        "Avg Discount (%)": round(df["Discount"].mean() * 100, 2),
        "Total Orders": df["Order ID"].nunique(),
        "Top Category": df.groupby("Category")["Sales"].sum().idxmax(),
        "Most Profitable Region": df.groupby("Region")["Profit"].sum().idxmax()
    }


# ============================================================
# SUPERSTORE TRENDS
# ============================================================

def _parse_order_date(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Order Date with day-first format."""
    df = df.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Order Date"])
    return df

def sales_trend(df: pd.DataFrame) -> pd.DataFrame:
    df = _parse_order_date(df)
    df["month"] = df["Order Date"].dt.to_period("M")
    trend = df.groupby("month")["Sales"].sum().reset_index()
    trend["month"] = trend["month"].astype(str)
    return trend

def profit_trend(df: pd.DataFrame) -> pd.DataFrame:
    df = _parse_order_date(df)
    df["month"] = df["Order Date"].dt.to_period("M")
    trend = df.groupby("month")["Profit"].sum().reset_index()
    trend["month"] = trend["month"].astype(str)
    return trend


# ============================================================
# PROPHET FORECASTING
# ============================================================

def forecast_sales(df, periods=6):
    df = df.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Order Date"])

    ts = (
        df.groupby(df["Order Date"].dt.to_period("M"))["Sales"]
        .sum()
        .reset_index()
    )
    ts["Order Date"] = ts["Order Date"].astype(str)

    prophet_df = ts.rename(columns={"Order Date": "ds", "Sales": "y"})
    prophet_df["y"] = pd.to_numeric(prophet_df["y"], errors="coerce")

    model = Prophet()
    model.fit(prophet_df)

    # Use 'MS' (month start) instead of deprecated 'M'
    future = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


# ============================================================
# SUPERSTORE FORECASTING (PROPHET)
# ============================================================

from prophet import Prophet

def prepare_forecast_data(df):
    df = df.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Order Date"])

    # Aggregate monthly sales
    df["month"] = df["Order Date"].dt.to_period("M")
    monthly = df.groupby("month")["Sales"].sum().reset_index()

    # Convert monthly period to a real timestamp (month start)
    monthly["ds"] = monthly["month"].dt.to_timestamp()

    monthly = monthly.drop(columns=["month"])
    monthly = monthly.rename(columns={"Sales": "y"})

    return monthly



def forecast_sales_prophet(df, periods=6):
    prophet_df = prepare_forecast_data(df)

    model = Prophet()
    model.fit(prophet_df)

    # Use month start frequency to align with our ds timestamps
    future = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)

    # Keep only relevant columns
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


# ============================================================
# Anomly Detection
# ============================================================

def detect_anomalies(df, forecast):
    # Make copies so we don't mutate inputs
    df = df.copy()
    forecast = forecast.copy()

    # Ensure 'ds' is datetime on BOTH dataframes
    df["ds"] = pd.to_datetime(df["ds"])
    forecast["ds"] = pd.to_datetime(forecast["ds"])

    # Merge on aligned datetime key
    merged = df.merge(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds",
        how="left"
    )

    # Flag anomalies where actual is outside prediction interval
    merged["anomaly"] = (
        (merged["y"] < merged["yhat_lower"]) |
        (merged["y"] > merged["yhat_upper"])
    )

    anomalies = merged[merged["anomaly"]]

    return anomalies, merged


# ============================================================
# CHURN KPIs
# ============================================================

def churn_kpis(df: pd.DataFrame) -> dict:
    df = df.copy()

    churn_rate = df["Churn Value"].mean() * 100
    month_charge = df["Monthly Charges"].mean()
    tenure_avg = df["Tenure Months"].mean()

    if "Churn Reason" in df.columns and df["Churn Reason"].dropna().shape[0]:
        top_churn_reason = df["Churn Reason"].mode()[0]
    else:
        top_churn_reason = "N/A"

    return {
        "Churn Rate (%)": round(churn_rate, 2),
        "Avg Monthly Charge": round(month_charge, 2),
        "Avg Tenure (months)": round(tenure_avg, 2),
        "Top Churn Reason": top_churn_reason
    }


# ============================================================
# AUTO INSIGHTS (RULE-BASED)
# ============================================================

def auto_insights_superstore(df: pd.DataFrame) -> list[str]:
    insights = []
    df = df.copy()

    # Discount hurting profit
    if "Discount" in df.columns and "Profit" in df.columns:
        high_discount_loss = df[(df["Discount"] > 0.4) & (df["Profit"] < 0)]
        if len(high_discount_loss) > 0:
            insights.append(
                "⚠️ High discounts (>40%) often lead to negative profit. "
                "Consider reviewing your discounting strategy."
            )

    # Top category profitability
    if "Category" in df.columns and "Profit" in df.columns:
        top_category = df.groupby("Category")["Profit"].mean().idxmax()
        worst_category = df.groupby("Category")["Profit"].mean().idxmin()
        insights.append(f"💡 The most profitable category is **{top_category}**.")
        insights.append(f"🔻 The least profitable category is **{worst_category}**.")

    return insights


def auto_insights_churn(df: pd.DataFrame) -> list[str]:
    insights = []
    df = df.copy()

    # Overall churn level
    rate = df["Churn Value"].mean() * 100
    if rate > 25:
        insights.append(
            f"⚠️ High churn rate detected (**{rate:.1f}%**). "
            "You likely need a stronger retention strategy."
        )
    else:
        insights.append(
            f"✅ Churn rate (**{rate:.1f}%**) is within acceptable limits compared to typical telco benchmarks."
        )

    # Contract type
    if "Contract" in df.columns:
        contract_churn = df.groupby("Contract")["Churn Value"].mean().sort_values(ascending=False)
        top = contract_churn.index[0]
        insights.append(
            f"💡 Customers on **{top}** contracts churn the most. "
            "Consider incentives to move them to longer-term contracts."
        )

    # Payment method
    if "Payment Method" in df.columns:
        method = df.groupby("Payment Method")["Churn Value"].mean().idxmax()
        insights.append(
            f"🔻 Customers paying via **{method}** have higher churn. "
            "Investigate friction or dissatisfaction with this payment method."
        )

    return insights
