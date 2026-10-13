from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_kpis(df: pd.DataFrame) -> dict:
    total_spend = float(df["spend_amount"].sum())
    total_orders = int(df["PO_ID"].nunique())
    total_quantity = int(df["Quantity"].sum())
    on_time_rate = float(df["is_on_time"].mean() * 100) if not df.empty else 0.0
    defect_rate = float(df["Defective_Units"].sum() / df["Quantity"].sum() * 100) if df["Quantity"].sum() else 0.0
    compliance_rate = float(df["is_compliant"].mean() * 100) if not df.empty else 0.0
    savings = float(df["savings_opportunity"].sum())
    avg_lead_time = float(df["lead_time_days"].mean()) if not df.empty else 0.0
    anomaly_count = int(df["is_anomaly"].sum())
    avg_unit_price = float(df["Unit_Price"].mean()) if not df.empty else 0.0
    avg_negotiated_price = float(df["Negotiated_Price"].mean()) if not df.empty else 0.0
    return {
        "total_spend": total_spend,
        "total_orders": total_orders,
        "total_quantity": total_quantity,
        "supplier_count": int(df["Supplier"].nunique()),
        "category_count": int(df["Item_Category"].nunique()),
        "product_count": int(df["Product_Id"].nunique()) if "Product_Id" in df.columns else 0,
        "on_time_rate": on_time_rate,
        "defect_rate": defect_rate,
        "compliance_rate": compliance_rate,
        "savings": savings,
        "avg_lead_time": avg_lead_time,
        "anomaly_count": anomaly_count,
        "avg_unit_price": avg_unit_price,
        "avg_negotiated_price": avg_negotiated_price,
    }


def get_supplier_summary(df: pd.DataFrame) -> pd.DataFrame:
    supplier_summary = (
        df.groupby("Supplier", as_index=False)
        .agg(
            total_spend=("spend_amount", "sum"),
            order_count=("PO_ID", "count"),
            on_time_rate=("is_on_time", "mean"),
            defect_rate=("Defective_Units", lambda s: (s.sum() / df.loc[s.index, "Quantity"].sum()) * 100 if df.loc[s.index, "Quantity"].sum() else 0),
            compliance_rate=("is_compliant", "mean"),
            avg_lead_time=("lead_time_days", "mean"),
            savings_opportunity=("savings_opportunity", "sum"),
            anomaly_count=("is_anomaly", "sum"),
        )
        .sort_values("total_spend", ascending=False)
        .reset_index(drop=True)
    )

    supplier_summary["risk_score"] = (
        (1 - supplier_summary["on_time_rate"]) * 0.35
        + (supplier_summary["defect_rate"] / 100) * 0.30
        + (1 - supplier_summary["compliance_rate"]) * 0.20
        + (supplier_summary["avg_lead_time"] / supplier_summary["avg_lead_time"].max().max()) * 0.15
    )
    supplier_summary["risk_level"] = np.where(
        supplier_summary["risk_score"] >= 0.45,
        "High Risk",
        np.where(supplier_summary["risk_score"] >= 0.25, "Moderate Risk", "Healthy"),
    )
    return supplier_summary


def get_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("Item_Category", as_index=False)
        .agg(
            total_spend=("spend_amount", "sum"),
            order_count=("PO_ID", "count"),
            average_price=("Unit_Price", "mean"),
            savings_opportunity=("savings_opportunity", "sum"),
            anomaly_count=("is_anomaly", "sum"),
        )
        .sort_values("total_spend", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def get_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Product_Id", as_index=False)
        .agg(
            total_spend=("spend_amount", "sum"),
            total_quantity=("Quantity", "sum"),
            order_count=("PO_ID", "count"),
            supplier_count=("Supplier", "nunique"),
            average_price=("Unit_Price", "mean"),
            average_negotiated_price=("Negotiated_Price", "mean"),
            anomaly_count=("is_anomaly", "sum"),
            defect_rate=("Defective_Units", lambda s: (s.sum() / df.loc[s.index, "Quantity"].sum()) * 100 if df.loc[s.index, "Quantity"].sum() else 0),
        )
        .sort_values("total_spend", ascending=False)
        .reset_index(drop=True)
    )


def get_price_trend(df: pd.DataFrame, group_col: str = "Item_Category") -> pd.DataFrame:
    return (
        df.groupby(["month", group_col], as_index=False)
        .agg(
            avg_unit_price=("Unit_Price", "mean"),
            avg_negotiated_price=("Negotiated_Price", "mean"),
            order_count=("PO_ID", "count"),
            total_quantity=("Quantity", "sum"),
        )
        .sort_values(["month", group_col])
        .reset_index(drop=True)
    )


def get_supplier_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Supplier", "Item_Category"], as_index=False)
        .agg(
            total_spend=("spend_amount", "sum"),
            total_quantity=("Quantity", "sum"),
            order_count=("PO_ID", "count"),
            avg_unit_price=("Unit_Price", "mean"),
            avg_negotiated_price=("Negotiated_Price", "mean"),
            anomaly_count=("is_anomaly", "sum"),
        )
        .sort_values("total_spend", ascending=False)
        .reset_index(drop=True)
    )


def get_savings_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    supplier_category = get_supplier_category_summary(df)
    if supplier_category.empty:
        return pd.DataFrame()

    benchmarks = (
        supplier_category.sort_values("avg_unit_price")
        .groupby("Item_Category", as_index=False)
        .first()[["Item_Category", "Supplier", "avg_unit_price"]]
        .rename(columns={"Supplier": "benchmark_supplier", "avg_unit_price": "benchmark_price"})
    )
    opportunities = supplier_category.merge(benchmarks, on="Item_Category", how="left")
    opportunities["estimated_savings"] = (
        (opportunities["avg_unit_price"] - opportunities["benchmark_price"])
        * opportunities["total_quantity"]
    ).clip(lower=0)
    return opportunities.sort_values("estimated_savings", ascending=False).reset_index(drop=True)


def get_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer grouping by the precomputed `month` column if present (stable across pandas versions)
    if "month" in df.columns:
        trend = (
            df.groupby("month", as_index=False)
            .agg(
                spend=("spend_amount", "sum"),
                orders=("PO_ID", "count"),
                avg_unit_price=("Unit_Price", "mean"),
                savings=("savings_opportunity", "sum"),
            )
        )
        trend["month"] = pd.to_datetime(trend["month"])
    else:
        # Fallback: group by Order_Date using a Grouper (may behave differently across pandas versions)
        trend = (
            df.groupby(pd.Grouper(key="Order_Date", freq="MS"), as_index=False)
            .agg(
                spend=("spend_amount", "sum"),
                orders=("PO_ID", "count"),
                avg_unit_price=("Unit_Price", "mean"),
                savings=("savings_opportunity", "sum"),
            )
        )
        # For older/newer pandas, the grouped column name may vary; attempt to find and normalize it
        if "Order_Date" in trend.columns:
            trend = trend.rename(columns={"Order_Date": "month"})
        elif "month" not in trend.columns and trend.columns.size > 0:
            # try to infer the first datetime-like column as month
            for c in trend.columns:
                if pd.api.types.is_datetime64_any_dtype(trend[c]):
                    trend = trend.rename(columns={c: "month"})
                    break
        if "month" in trend.columns:
            trend["month"] = pd.to_datetime(trend["month"])

    return trend.sort_values("month").reset_index(drop=True)


def get_anomaly_records(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.loc[df["is_anomaly"], [
            "PO_ID",
            "Supplier",
            "Item_Category",
            "Order_Date",
            "Unit_Price",
            "Negotiated_Price",
            "price_variance",
            "Quantity",
            "savings_opportunity",
            "Order_Status",
        ]]
        .sort_values("price_variance", ascending=False)
        .reset_index(drop=True)
    )


def forecast_monthly_spend(df: pd.DataFrame, horizon_months: int = 6) -> pd.DataFrame:
    trend = get_monthly_trend(df)
    if trend.empty:
        return pd.DataFrame(columns=["month", "forecast_spend"])

    x = np.arange(len(trend)).reshape(-1, 1)
    y = trend["spend"].values.astype(float)

    model = LinearRegression()
    model.fit(x, y)

    future = np.arange(len(trend), len(trend) + horizon_months).reshape(-1, 1)
    forecast_values = model.predict(future)
    forecast_index = pd.date_range(trend["month"].max() + pd.offsets.MonthBegin(1), periods=horizon_months, freq="MS")

    forecast = pd.DataFrame({"month": forecast_index, "forecast_spend": forecast_values})
    return forecast


def forecast_monthly_quantity(df: pd.DataFrame, horizon_months: int = 6) -> pd.DataFrame:
    trend = get_monthly_trend(df)
    if trend.empty:
        return pd.DataFrame(columns=["month", "forecast_quantity"])

    monthly_quantity = df.groupby("month", as_index=False)["Quantity"].sum().sort_values("month")
    x = np.arange(len(monthly_quantity)).reshape(-1, 1)
    y = monthly_quantity["Quantity"].values.astype(float)

    model = LinearRegression()
    model.fit(x, y)

    future = np.arange(len(monthly_quantity), len(monthly_quantity) + horizon_months).reshape(-1, 1)
    forecast_values = model.predict(future).clip(min=0)
    forecast_index = pd.date_range(monthly_quantity["month"].max() + pd.offsets.MonthBegin(1), periods=horizon_months, freq="MS")
    return pd.DataFrame({"month": forecast_index, "forecast_quantity": forecast_values})
