# src/data_cleaning.py

import pandas as pd

def load_data(path: str, encoding: str = "latin1") -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(path, encoding=encoding)


# ============================================================
# GENERIC CLEANING
# ============================================================

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Perform general cleaning applicable to most business datasets."""
    df = df.copy()

    # Strip column names
    df.columns = df.columns.str.strip()

    # Convert object columns with numbers to numeric
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda x: x.replace(",", "") if isinstance(x, str) else x
            )

    # Handle missing values safely
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


# ============================================================
# TELCO CHURN CLEANING (SPECIAL HANDLING)
# ============================================================

def clean_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Telco Customer Churn dataset."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Remove useless combined lat-long string column
    if "Lat Long" in df.columns:
        df.drop(columns=["Lat Long"], inplace=True)

    # Convert total charges to numeric
    if "Total Charges" in df.columns:
        df["Total Charges"] = (
            df["Total Charges"]
            .replace(" ", "0")  # some blank values
            .astype(float)
        )

    # Convert Yes/No columns to 1/0
    yes_no_cols = [
        col for col in df.columns
        if df[col].dtype == "object" and df[col].isin(["Yes", "No"]).any()
    ]

    for col in yes_no_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Handle Churn Reason missing values
    if "Churn Reason" in df.columns:
        df["Churn Reason"] = df["Churn Reason"].fillna("Unknown")

    # Final type fixes
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass  # leave non-numeric columns as is

    return df


# ============================================================
# SAVE CLEANED OUTPUT
# ============================================================

def save_processed(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned file to: {output_path}")


# ============================================================
# MAIN SCRIPT (RUN MANUALLY)
# ============================================================

if __name__ == "__main__":
    # SUPERSTORE CLEANING
    superstore = load_data("data/raw/superstore.csv")
    superstore_clean = basic_cleaning(superstore)
    save_processed(superstore_clean, "data/processed/superstore_clean.csv")

    # CHURN CLEANING
    churn = load_data("data/raw/Telco_customer_churn.csv")
    churn_clean = clean_churn(churn)
    save_processed(churn_clean, "data/processed/churn_clean.csv")
