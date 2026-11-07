"""Reusable data prep, feature engineering and model definitions for the Telco churn project."""
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Telco-Customer-Churn.csv"
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RANDOM_STATE = 42

SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def load_raw() -> pd.DataFrame:
    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA)
    return pd.read_csv(DATA)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """TotalCharges is stored as text; 11 brand-new customers (tenure 0) have a blank value -> 0."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce").fillna(0.0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["num_addon_services"] = (df[SERVICES] == "Yes").sum(axis=1)
    df["tenure_group"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72],
                                labels=["0-12m", "13-24m", "25-48m", "49-72m"]).astype(str)
    df["avg_monthly_spend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"].clip(lower=1),
                                       df["MonthlyCharges"])
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["auto_pay"] = df["PaymentMethod"].str.contains("automatic").astype(int)
    return df


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=["customerID", "Churn"])
    return X, df["Churn"]


def feature_lists(X: pd.DataFrame):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return num, cat


def build_models(X: pd.DataFrame) -> dict:
    num, cat = feature_lists(X)
    pre_lr = ColumnTransformer([("num", StandardScaler(), num),
                                ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat)])
    pre_gb = ColumnTransformer([("num", "passthrough", num),
                                ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                                      sparse_output=False), cat)])
    return {
        "Logistic Regression": Pipeline([("prep", pre_lr),
                                         ("model", LogisticRegression(max_iter=2000, C=1.0))]),
        "Gradient Boosting": Pipeline([("prep", pre_gb),
                                       ("model", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                                                            max_depth=3, subsample=0.8,
                                                                            random_state=RANDOM_STATE))]),
    }
