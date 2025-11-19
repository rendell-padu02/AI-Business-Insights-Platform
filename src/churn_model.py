# src/churn_model.py

import pandas as pd
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


TARGET_COL = "Churn Value"


def prepare_churn_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare Telco churn dataset for modeling.
    - Drops obvious ID / text-only cols
    - Converts numeric-like text columns
    - One-hot encodes categoricals
    """

    df = df.copy()

    # Drop columns that shouldn't be features
    drop_cols = [
        "CustomerID",
        "Churn Label",
        "Churn Reason",
        "Lat Long",      # redundant with Lat/Long
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe.")

    y = df[TARGET_COL]
    X = df.drop(columns=drop_cols + [TARGET_COL])

    # Fix numeric-like column "Total Charges" if it is object
    if "Total Charges" in X.columns and X["Total Charges"].dtype == "object":
        X["Total Charges"] = pd.to_numeric(X["Total Charges"], errors="coerce")

    # Basic missing value handling before dummies
    X = X.fillna(0)

    # One-hot encode categoricals
    X = pd.get_dummies(X, drop_first=True)

    return X, y


def train_test_split_churn(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """Split churn data into train/test sets."""
    X, y = prepare_churn_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Train Logistic Regression model."""
    log_reg = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced"  # helpful if churn is imbalanced
    )
    log_reg.fit(X_train, y_train)
    return log_reg


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Train Random Forest model."""
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_model(
    model, X_test, y_test, model_name: str = "Model"
) -> Dict[str, float]:
    """Compute and print standard classification metrics."""
    y_pred = model.predict(X_test)

    # Probabilities for ROC-AUC (if available)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = float("nan")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== {model_name} =====")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def train_and_evaluate_churn_models(df: pd.DataFrame):
    """
    Full pipeline:
    - prepare data
    - train/test split
    - train Logistic Regression + Random Forest
    - evaluate both
    Returns:
        models, metrics
    """
    X_train, X_test, y_train, y_test = train_test_split_churn(df)

    # Baseline linear model
    log_reg = train_logistic_regression(X_train, y_train)
    log_reg_metrics = evaluate_model(log_reg, X_test, y_test, "Logistic Regression")

    # Non-linear ensemble model
    rf = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test, "Random Forest")

    models = {
        "logistic_regression": log_reg,
        "random_forest": rf,
    }

    metrics = {
        "logistic_regression": log_reg_metrics,
        "random_forest": rf_metrics,
    }

    return models, metrics
