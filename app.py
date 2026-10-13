from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from procureiq_app.agent import ProcurementAgent
from procureiq_app.analytics import (
    compute_kpis,
    forecast_monthly_quantity,
    forecast_monthly_spend,
    get_anomaly_records,
    get_category_summary,
    get_monthly_trend,
    get_price_trend,
    get_product_summary,
    get_savings_opportunities,
    get_supplier_category_summary,
    get_supplier_summary,
)
from procureiq_app.data_processing import load_procurement_data

st.set_page_config(
    page_title="ProcureIQ Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_data():
    return load_procurement_data()


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def main() -> None:
    df = get_data()

    st.title("ProcureIQ: Intelligent Procurement Analytics Dashboard")
    st.caption("Python-based procurement intelligence dashboard for spend analysis, supplier monitoring, anomaly detection, and forecasting.")

    with st.sidebar:
        st.header("Filters")
        suppliers = ["All"] + sorted(df["Supplier"].unique().tolist())
        categories = ["All"] + sorted(df["Item_Category"].unique().tolist())

        selected_supplier = st.selectbox("Supplier", suppliers)
        selected_category = st.selectbox("Category", categories)

        min_date = df["Order_Date"].min().date()
        max_date = df["Order_Date"].max().date()
        start_date, end_date = st.date_input(
            "Date range",
            (min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(start_date, tuple):
            start_date, end_date = start_date

        filtered = df.copy()
        if selected_supplier != "All":
            filtered = filtered[filtered["Supplier"] == selected_supplier]
        if selected_category != "All":
            filtered = filtered[filtered["Item_Category"] == selected_category]
        filtered = filtered[
            (filtered["Order_Date"] >= pd.Timestamp(start_date))
            & (filtered["Order_Date"] <= pd.Timestamp(end_date))
        ]

        st.markdown("---")
        st.subheader("Insight mode")
        st.write("Use the procurement query panel to investigate spend, supplier risk, anomalies, and sourcing alternatives.")

    if filtered.empty:
        st.warning("No records match the selected filters. Please widen the selection.")
        return

    kpis = compute_kpis(filtered)
    trend = get_monthly_trend(filtered)
    supplier_summary = get_supplier_summary(filtered)
    category_summary = get_category_summary(filtered)
    product_summary = get_product_summary(filtered)
    supplier_category = get_supplier_category_summary(filtered)
    price_trend = get_price_trend(filtered)
    anomaly_records = get_anomaly_records(filtered)
    savings_opportunities = get_savings_opportunities(filtered)
    forecast = forecast_monthly_spend(filtered, horizon_months=6)
    quantity_forecast = forecast_monthly_quantity(filtered, horizon_months=6)

    st.markdown("---")
    st.subheader("Procurement KPI Dashboard")
    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Total Spend", format_currency(kpis["total_spend"]))
    kpi_cols[1].metric("Purchase Volume", f"{kpis['total_quantity']:,}")
    kpi_cols[2].metric("Total Orders", f"{kpis['total_orders']:,}")
    kpi_cols[3].metric("Suppliers", f"{kpis['supplier_count']:,}")
    kpi_cols[4].metric("Categories", f"{kpis['category_count']:,}")
    kpi_cols[5].metric("Products", f"{kpis['product_count']:,}")

    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Avg Unit Price", format_currency(kpis["avg_unit_price"]))
    kpi_cols[1].metric("Avg Negotiated Price", format_currency(kpis["avg_negotiated_price"]))
    kpi_cols[2].metric("On-Time Delivery", f"{kpis['on_time_rate']:.1f}%")
    kpi_cols[3].metric("Compliance Rate", f"{kpis['compliance_rate']:.1f}%")
    kpi_cols[4].metric("Defect Rate", f"{kpis['defect_rate']:.2f}%")
    kpi_cols[5].metric("Price Anomalies", f"{kpis['anomaly_count']:,}")

    tabs = st.tabs([
        "Spend & Volume",
        "Supplier Analytics",
        "Price & Anomalies",
        "Forecasting",
        "Savings",
        "AI Agent",
    ])

    with tabs[0]:
        left_col, right_col = st.columns(2)
        with left_col:
            spend_fig = px.line(
                trend,
                x="month",
                y="spend",
                markers=True,
                title="Monthly Procurement Spend",
                template="plotly_white",
            )
            spend_fig.update_layout(height=390, xaxis_title="Month", yaxis_title="Spend")
            st.plotly_chart(spend_fig, width="stretch")
        with right_col:
            volume_fig = px.bar(
                filtered.groupby("month", as_index=False)["Quantity"].sum(),
                x="month",
                y="Quantity",
                title="Monthly Purchase Volume",
                template="plotly_white",
            )
            volume_fig.update_layout(height=390, xaxis_title="Month", yaxis_title="Units Purchased")
            st.plotly_chart(volume_fig, width="stretch")

        left_col, right_col = st.columns(2)
        with left_col:
            category_spend_fig = px.bar(
                category_summary.sort_values("total_spend", ascending=True),
                x="total_spend",
                y="Item_Category",
                orientation="h",
                color="total_spend",
                title="Spend by Category",
                template="plotly_white",
            )
            category_spend_fig.update_layout(height=390, xaxis_title="Spend", yaxis_title="Category")
            st.plotly_chart(category_spend_fig, width="stretch")
        with right_col:
            product_volume_fig = px.bar(
                product_summary.sort_values("total_quantity", ascending=False).head(10),
                x="Product_Id",
                y="total_quantity",
                color="total_spend",
                title="Top Products by Purchase Volume",
                template="plotly_white",
            )
            product_volume_fig.update_layout(height=390, xaxis_title="Product", yaxis_title="Units Purchased")
            st.plotly_chart(product_volume_fig, width="stretch")

        spend_mix = filtered.groupby(["Supplier", "Item_Category"], as_index=False)["spend_amount"].sum()
        mix_fig = px.bar(
            spend_mix,
            x="Supplier",
            y="spend_amount",
            color="Item_Category",
            title="Supplier Spend Mix by Category",
            template="plotly_white",
        )
        mix_fig.update_layout(height=430, xaxis_title="Supplier", yaxis_title="Spend")
        st.plotly_chart(mix_fig, width="stretch")

    with tabs[1]:
        left_col, right_col = st.columns(2)
        with left_col:
            supplier_risk_fig = px.scatter(
                supplier_summary,
                x="on_time_rate",
                y="defect_rate",
                size="total_spend",
                color="risk_level",
                hover_name="Supplier",
                title="Supplier Risk Map",
                template="plotly_white",
            )
            supplier_risk_fig.update_layout(height=410, xaxis_title="On-Time Rate", yaxis_title="Defect Rate")
            st.plotly_chart(supplier_risk_fig, width="stretch")
        with right_col:
            supplier_spend_fig = px.bar(
                supplier_summary.sort_values("total_spend", ascending=False),
                x="Supplier",
                y="total_spend",
                color="risk_level",
                title="Supplier Spend and Risk Level",
                template="plotly_white",
            )
            supplier_spend_fig.update_layout(height=410, xaxis_title="Supplier", yaxis_title="Spend")
            st.plotly_chart(supplier_spend_fig, width="stretch")

        supplier_trend = filtered.groupby(["month", "Supplier"], as_index=False)["spend_amount"].sum()
        supplier_trend_fig = px.line(
            supplier_trend,
            x="month",
            y="spend_amount",
            color="Supplier",
            markers=True,
            title="Supplier Spend Trend",
            template="plotly_white",
        )
        supplier_trend_fig.update_layout(height=430, xaxis_title="Month", yaxis_title="Spend")
        st.plotly_chart(supplier_trend_fig, width="stretch")

        st.dataframe(
            supplier_summary[[
                "Supplier",
                "total_spend",
                "order_count",
                "on_time_rate",
                "defect_rate",
                "compliance_rate",
                "avg_lead_time",
                "savings_opportunity",
                "anomaly_count",
                "risk_score",
                "risk_level",
            ]].sort_values("risk_score", ascending=False),
            width="stretch",
        )

    with tabs[2]:
        left_col, right_col = st.columns(2)
        with left_col:
            price_fig = px.line(
                price_trend,
                x="month",
                y="avg_unit_price",
                color="Item_Category",
                markers=True,
                title="Average Price Trend by Category",
                template="plotly_white",
            )
            price_fig.update_layout(height=410, xaxis_title="Month", yaxis_title="Average Unit Price")
            st.plotly_chart(price_fig, width="stretch")
        with right_col:
            variance_fig = px.box(
                filtered,
                x="Item_Category",
                y="Unit_Price",
                color="Item_Category",
                points="outliers",
                title="Price Distribution and Outliers",
                template="plotly_white",
            )
            variance_fig.update_layout(height=410, xaxis_title="Category", yaxis_title="Unit Price", showlegend=False)
            st.plotly_chart(variance_fig, width="stretch")

        anomaly_fig = px.scatter(
            filtered,
            x="Negotiated_Price",
            y="Unit_Price",
            color="is_anomaly",
            size="Quantity",
            hover_data=["PO_ID", "Supplier", "Item_Category", "price_variance", "pct_price_variance"],
            title="Price Anomaly Detection",
            template="plotly_white",
        )
        anomaly_fig.update_layout(height=430, xaxis_title="Negotiated Price", yaxis_title="Unit Price")
        st.plotly_chart(anomaly_fig, width="stretch")

        st.dataframe(
            anomaly_records[[
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
            ]].head(25),
            width="stretch",
        )

    with tabs[3]:
        left_col, right_col = st.columns(2)
        with left_col:
            forecast_fig = px.line(
                pd.concat([
                    trend[["month", "spend"]].rename(columns={"spend": "forecast_spend"}),
                    forecast,
                ], ignore_index=True),
                x="month",
                y="forecast_spend",
                markers=True,
                title="Spend Forecast",
                template="plotly_white",
            )
            forecast_fig.update_layout(height=410, xaxis_title="Month", yaxis_title="Spend")
            st.plotly_chart(forecast_fig, width="stretch")
        with right_col:
            quantity_history = filtered.groupby("month", as_index=False)["Quantity"].sum().rename(columns={"Quantity": "forecast_quantity"})
            quantity_forecast_fig = px.line(
                pd.concat([quantity_history, quantity_forecast], ignore_index=True),
                x="month",
                y="forecast_quantity",
                markers=True,
                title="Demand Forecast by Purchase Volume",
                template="plotly_white",
            )
            quantity_forecast_fig.update_layout(height=410, xaxis_title="Month", yaxis_title="Units")
            st.plotly_chart(quantity_forecast_fig, width="stretch")

        demand_by_category = filtered.groupby(["month", "Item_Category"], as_index=False)["Quantity"].sum()
        demand_fig = px.area(
            demand_by_category,
            x="month",
            y="Quantity",
            color="Item_Category",
            title="Demand Pattern by Category",
            template="plotly_white",
        )
        demand_fig.update_layout(height=430, xaxis_title="Month", yaxis_title="Units Purchased")
        st.plotly_chart(demand_fig, width="stretch")

    with tabs[4]:
        top_savings = savings_opportunities.head(15)
        left_col, right_col = st.columns(2)
        with left_col:
            savings_fig = px.bar(
                top_savings,
                x="estimated_savings",
                y="Supplier",
                color="Item_Category",
                orientation="h",
                title="Potential Savings vs Lowest Category Supplier Price",
                template="plotly_white",
            )
            savings_fig.update_layout(height=430, xaxis_title="Estimated Savings", yaxis_title="Supplier")
            st.plotly_chart(savings_fig, width="stretch")
        with right_col:
            alt_fig = px.scatter(
                supplier_category,
                x="avg_unit_price",
                y="total_quantity",
                color="Supplier",
                facet_col="Item_Category",
                facet_col_wrap=2,
                size="total_spend",
                title="Alternative Supplier Price and Volume Comparison",
                template="plotly_white",
            )
            alt_fig.update_layout(height=430, xaxis_title="Average Unit Price", yaxis_title="Quantity")
            st.plotly_chart(alt_fig, width="stretch")

        st.dataframe(
            savings_opportunities[[
                "Supplier",
                "Item_Category",
                "avg_unit_price",
                "benchmark_supplier",
                "benchmark_price",
                "total_quantity",
                "estimated_savings",
                "order_count",
                "anomaly_count",
            ]],
            width="stretch",
        )

    with tabs[5]:
        st.subheader("AI Procurement Investigation Agent")
        query = st.text_input("Ask a procurement question", key="procure_query", value="")
        agent = ProcurementAgent(filtered)
        if st.button("Run Query"):
            q = st.session_state.get("procure_query", "").strip()
            if not q:
                st.warning("Please enter a question and click Run Query.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        answer = agent.answer(q)
                    except Exception as err:
                        st.error(f"Agent error: {err}")
                        answer = None
                if answer:
                    st.info(answer)
                else:
                    st.error("No LLM answer was returned. Check the LLM configuration and try again.")

        st.markdown("#### Investigation prompts")
        st.write("- Why is this supplier risky?")
        st.write("- Which supplier can replace the current supplier for this category?")
        st.write("- Which purchase orders have abnormal prices?")
        st.write("- What savings can we get from alternative supplier pricing?")
        st.write("- What procurement trend changed recently?")

    st.markdown("---")
    st.subheader("Analysis Results")
    if not supplier_summary.empty and not category_summary.empty:
        top_risk = supplier_summary.sort_values("risk_score", ascending=False).iloc[0]
        top_spend_supplier = supplier_summary.sort_values("total_spend", ascending=False).iloc[0]
        top_category = category_summary.sort_values("total_spend", ascending=False).iloc[0]
        top_saving = savings_opportunities.iloc[0] if not savings_opportunities.empty else None
        st.write(f"- Highest spend supplier: {top_spend_supplier['Supplier']} with {format_currency(top_spend_supplier['total_spend'])}.")
        st.write(f"- Highest risk supplier: {top_risk['Supplier']} ({top_risk['risk_level']}) with {top_risk['on_time_rate'] * 100:.1f}% on-time delivery and {top_risk['defect_rate']:.2f}% defect rate.")
        st.write(f"- Largest spend category: {top_category['Item_Category']} with {format_currency(top_category['total_spend'])}.")
        st.write(f"- Price anomalies flagged: {kpis['anomaly_count']:,}.")
        if top_saving is not None:
            st.write(
                f"- Largest alternative-supplier savings estimate: {format_currency(top_saving['estimated_savings'])} "
                f"for {top_saving['Supplier']} in {top_saving['Item_Category']} compared with {top_saving['benchmark_supplier']}."
            )

    # Database schema / columns description section
    st.markdown("---")
    st.subheader("Database Columns & Description")
    st.caption("List of dataset columns and a short description to help understand the source table.")

    # human-friendly descriptions for known columns
    column_descriptions = {
        "PO_ID": "Purchase order identifier (unique per order)",
        "Order_Date": "Date the purchase order was issued",
        "Delivery_Date": "Date the goods/services were delivered",
        "Supplier": "Name of the supplier/vendor",
        "Item_Category": "High-level category of the purchased item",
        "Quantity": "Number of units ordered",
        "Unit_Price": "Price per unit listed on the PO",
        "Negotiated_Price": "Agreed/negotiated unit price (if available)",
        "spend_amount": "Computed: Quantity * Unit_Price (total spend for the row)",
        "negotiated_amount": "Computed: Quantity * Negotiated_Price",
        "savings_opportunity": "Computed: (Negotiated_Price - Unit_Price) * Quantity",
        "price_variance": "Difference between listed and negotiated unit price",
        "pct_price_variance": "Percent variance of price vs negotiated price",
        "price_zscore": "Within-category price z-score (outlier detection)",
        "Defective_Units": "Number of defective units reported for the order",
        "defect_rate": "Computed: Defective_Units / Quantity",
        "Compliance": "Procurement compliance flag (Yes/No)",
        "is_compliant": "Boolean derived from `Compliance`",
        "Order_Status": "Status of the order (e.g., Open, Closed)",
        "Delivery_Status": "Delivery state (e.g., Delivered, Pending)",
        "is_on_time": "Boolean indicating on-time delivery",
        "lead_time_days": "Computed: (Delivery_Date - Order_Date) in days",
        "month": "Normalized month timestamp derived from `Order_Date`",
        "is_anomaly": "Boolean flag for price anomaly (pct variance > 15%)",
    }

    rows = []
    for col in filtered.columns:
        desc = column_descriptions.get(col, "Column from source dataset (no description available)")
        rows.append({"column": col, "description": desc})

    st.table(rows)


if __name__ == "__main__":
    main()
