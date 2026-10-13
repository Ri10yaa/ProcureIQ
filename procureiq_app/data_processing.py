from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Purchase_Orders.xlsx"


def load_procurement_data(path: Path | str | None = None) -> pd.DataFrame:
    if path is None:
        path = DATA_PATH
    df = pd.read_excel(path)
    df.columns = [str(col).strip() for col in df.columns]

    for col in ["Order_Date", "Delivery_Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["lead_time_days"] = (df["Delivery_Date"] - df["Order_Date"]).dt.days.clip(lower=0)
    df["spend_amount"] = df["Quantity"] * df["Unit_Price"]
    df["negotiated_amount"] = df["Quantity"] * df["Negotiated_Price"]
    df["savings_opportunity"] = (df["Negotiated_Price"] - df["Unit_Price"]) * df["Quantity"]
    df["price_variance"] = df["Unit_Price"] - df["Negotiated_Price"]
    df["pct_price_variance"] = (df["price_variance"].abs() / df["Negotiated_Price"].replace(0, 1)) * 100
    df["defect_rate"] = df["Defective_Units"] / df["Quantity"].replace(0, np.nan)
    df["is_compliant"] = df["Compliance"].astype(str).str.lower().eq("yes")
    df["is_delivered"] = df["Order_Status"].astype(str).str.lower().eq("delivered")
    df["is_on_time"] = df["Delivery_Status"].astype(str).str.lower().eq("delivered")

    grouped_median = df.groupby("Item_Category")["Unit_Price"].transform("median")
    grouped_std = df.groupby("Item_Category")["Unit_Price"].transform("std").fillna(0)
    df["price_zscore"] = (df["Unit_Price"] - grouped_median) / grouped_std.replace(0, np.nan)
    df["is_anomaly"] = df["pct_price_variance"] > 15

    df["month"] = df["Order_Date"].dt.to_period("M").dt.to_timestamp()
    return df
