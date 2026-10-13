from __future__ import annotations

import json
import os

import pandas as pd
import requests
import re
import difflib
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from procureiq_app.analytics import (
    compute_kpis,
    forecast_monthly_quantity,
    forecast_monthly_spend,
    get_anomaly_records,
    get_category_summary,
    get_product_summary,
    get_savings_opportunities,
    get_supplier_category_summary,
    get_supplier_summary,
)
from dotenv import load_dotenv

load_dotenv()


class ProcurementAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.knowledge_base = self._build_knowledge_base()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.document_matrix = self.vectorizer.fit_transform(
            [doc["text"] for doc in self.knowledge_base]
        )

    def _dataset_context(self) -> dict:
        ordering_columns = [
            "PO_ID",
            "Supplier",
            "Item_Category",
            "Order_Date",
            "Delivery_Date",
            "Quantity",
            "Unit_Price",
            "Negotiated_Price",
            "Defective_Units",
            "Compliance",
            "Delivery_Status",
            "Order_Status",
        ]

        top_suppliers = (
            self.df.groupby("Supplier", as_index=False)["spend_amount"]
            .sum()
            .sort_values("spend_amount", ascending=False)
            .head(5)
            .to_dict(orient="records")
        )

        top_categories = (
            self.df.groupby("Item_Category", as_index=False)["spend_amount"]
            .sum()
            .sort_values("spend_amount", ascending=False)
            .head(5)
            .to_dict(orient="records")
        )

        anomaly_rows = (
            self.df.loc[self.df["is_anomaly"], ["Supplier", "Item_Category", "Unit_Price", "Negotiated_Price", "price_variance", "Quantity"]]
            .sort_values("price_variance", ascending=False)
            .head(10)
            .to_dict(orient="records")
        )

        context = {
            "dataset_name": "Procurement Purchase Orders",
            "description": "This dataset contains purchase order records for supplier spend, item categories, quantities, negotiated prices, quality, delivery compliance, and defect details.",
            "schema": ordering_columns,
            "key_metrics": {
                "total_orders": int(self.df["PO_ID"].nunique()),
                "total_suppliers": int(self.df["Supplier"].nunique()),
                "total_categories": int(self.df["Item_Category"].nunique()),
                "total_spend": float(self.df["spend_amount"].sum()),
                "average_order_value": float(self.df["spend_amount"].mean()),
                "on_time_delivery_rate": float(self.df["is_on_time"].mean() * 100),
                "compliance_rate": float(self.df["is_compliant"].mean() * 100),
                "defect_rate": float((self.df["Defective_Units"].sum() / self.df["Quantity"].sum()) * 100) if self.df["Quantity"].sum() else 0.0,
                "average_lead_time_days": float(self.df["lead_time_days"].mean()),
                "savings_opportunity": float(self.df["savings_opportunity"].sum()),
                "anomaly_count": int(self.df["is_anomaly"].sum()),
            },
            "top_suppliers_by_spend": top_suppliers,
            "top_categories_by_spend": top_categories,
            "anomaly_examples": anomaly_rows,
            "sample_rows": self.df[ordering_columns].head(12).to_dict(orient="records"),
        }
        return context

    def _build_prompt(self, question: str) -> str:
        context = json.dumps(self._dataset_context(), default=str, ensure_ascii=False)
        return (
            "You are ProcureIQ, a procurement analytics assistant built specifically for this dataset. "
            "Answer procurement questions using only the dataset provided below. "
            "Use clear business language and provide practical operational insight.\n\n"
            f"Question: {question}\n\n"
            f"Dataset context:\n{context}\n\n"
            "Instructions:\n"
            "- Use the data from the dataset only.\n"
            "- Mention concrete metrics when relevant.\n"
            "- If asked about a supplier or category, compare performance using spend, compliance, on-time delivery, defects, and pricing.\n"
            "- If the question asks for monitoring or action, recommend a practical procurement action.\n"
            "- Keep the answer concise but analytical."
        )

    def _build_knowledge_base(self) -> list[dict]:
        docs: list[dict] = []
        kpis = compute_kpis(self.df)
        docs.append({
            "type": "kpi",
            "title": "Overall procurement KPI summary",
            "text": (
                f"Overall procurement KPI summary. Total spend ${kpis['total_spend']:,.0f}. "
                f"Total orders {kpis['total_orders']}. On-time delivery rate {kpis['on_time_rate']:.1f}%. "
                f"Compliance rate {kpis['compliance_rate']:.1f}%. Defect rate {kpis['defect_rate']:.2f}%. "
                f"Average lead time {kpis['avg_lead_time']:.1f} days. "
                f"Savings opportunity {kpis['savings']:,.0f}. Pricing anomaly count {kpis['anomaly_count']}."
            ),
        })

        for _, r in get_supplier_summary(self.df).iterrows():
            docs.append({
                "type": "supplier",
                "title": f"Supplier {r['Supplier']}",
                "text": (
                    f"Supplier {r['Supplier']} procurement performance. Total spend ${r['total_spend']:,.0f}. "
                    f"Order count {int(r['order_count'])}. On-time delivery {r['on_time_rate'] * 100:.1f}%. "
                    f"Defect rate {r['defect_rate']:.2f}%. Compliance {r['compliance_rate'] * 100:.1f}%. "
                    f"Average lead time {r['avg_lead_time']:.1f} days. "
                    f"Savings opportunity ${r['savings_opportunity']:,.0f}. "
                    f"Pricing anomalies {int(r['anomaly_count'])}. Risk level {r['risk_level']}."
                ),
            })

        for _, r in get_product_summary(self.df).head(50).iterrows():
            docs.append({
                "type": "product",
                "title": f"Product {r['Product_Id']}",
                "text": (
                    f"Product {r['Product_Id']} procurement summary. Total spend ${r['total_spend']:,.0f}. "
                    f"Total quantity {int(r['total_quantity'])}. Order count {int(r['order_count'])}. "
                    f"Supplier count {int(r['supplier_count'])}. Average unit price ${r['average_price']:,.2f}. "
                    f"Average negotiated price ${r['average_negotiated_price']:,.2f}. "
                    f"Anomaly count {int(r['anomaly_count'])}. Defect rate {r['defect_rate']:.2f}%."
                ),
            })

        category_defects = (
            self.df.groupby("Item_Category")
            .apply(lambda g: pd.Series({
                "defect_rate": (g["Defective_Units"].sum() / g["Quantity"].sum() * 100) if g["Quantity"].sum() else 0,
                "defective_units": g["Defective_Units"].sum(),
            }))
            .reset_index()
        )
        category_summary = get_category_summary(self.df).merge(category_defects, on="Item_Category", how="left")
        for _, r in category_summary.iterrows():
            docs.append({
                "type": "category",
                "title": f"Category {r['Item_Category']}",
                "text": (
                    f"Category {r['Item_Category']} procurement analysis. Total spend ${r['total_spend']:,.0f}. "
                    f"Order count {int(r['order_count'])}. Average unit price ${r['average_price']:,.2f}. "
                    f"Savings opportunity ${r['savings_opportunity']:,.0f}. "
                    f"Pricing anomalies {int(r['anomaly_count'])}. "
                    f"Defect rate {r['defect_rate']:.2f}% with {int(r['defective_units']):,} defective units."
                ),
            })

        for _, r in get_supplier_category_summary(self.df).iterrows():
            docs.append({
                "type": "supplier_category",
                "title": f"{r['Supplier']} {r['Item_Category']} supplier category performance",
                "text": (
                    f"Supplier {r['Supplier']} in category {r['Item_Category']}. "
                    f"Total spend ${r['total_spend']:,.0f}. Total quantity {int(r['total_quantity'])}. "
                    f"Order count {int(r['order_count'])}. Average unit price ${r['avg_unit_price']:,.2f}. "
                    f"Average negotiated price ${r['avg_negotiated_price']:,.2f}. "
                    f"Pricing anomalies {int(r['anomaly_count'])}."
                ),
            })

        for _, r in get_savings_opportunities(self.df).head(25).iterrows():
            docs.append({
                "type": "savings_opportunity",
                "title": f"Savings opportunity {r['Supplier']} {r['Item_Category']}",
                "text": (
                    f"Potential savings opportunity for supplier {r['Supplier']} in category {r['Item_Category']}. "
                    f"Current average unit price ${r['avg_unit_price']:,.2f}. "
                    f"Alternative benchmark supplier {r['benchmark_supplier']} average price ${r['benchmark_price']:,.2f}. "
                    f"Total quantity {int(r['total_quantity'])}. Estimated savings ${r['estimated_savings']:,.0f}. "
                    f"Order count {int(r['order_count'])}. Anomaly count {int(r['anomaly_count'])}."
                ),
            })

        monthly = self.df.groupby("month", as_index=False).agg(
            spend=("spend_amount", "sum"),
            orders=("PO_ID", "count"),
            avg_unit_price=("Unit_Price", "mean"),
            savings=("savings_opportunity", "sum"),
            defects=("Defective_Units", "sum"),
        )
        for _, r in monthly.sort_values("month").iterrows():
            docs.append({
                "type": "month",
                "title": f"Monthly procurement {pd.to_datetime(r['month']).strftime('%Y-%m')}",
                "text": (
                    f"Monthly procurement trend for {pd.to_datetime(r['month']).strftime('%Y-%m')}. "
                    f"Spend ${r['spend']:,.0f}. Orders {int(r['orders'])}. "
                    f"Average unit price ${r['avg_unit_price']:,.2f}. "
                    f"Savings opportunity ${r['savings']:,.0f}. Defective units {int(r['defects'])}."
                ),
            })

        for _, r in forecast_monthly_spend(self.df, horizon_months=6).iterrows():
            docs.append({
                "type": "forecast",
                "title": f"Spend forecast {pd.to_datetime(r['month']).strftime('%Y-%m')}",
                "text": (
                    f"Spend forecast for {pd.to_datetime(r['month']).strftime('%Y-%m')}. "
                    f"Forecast spend ${r['forecast_spend']:,.0f}."
                ),
            })

        for _, r in forecast_monthly_quantity(self.df, horizon_months=6).iterrows():
            docs.append({
                "type": "forecast",
                "title": f"Demand forecast {pd.to_datetime(r['month']).strftime('%Y-%m')}",
                "text": (
                    f"Demand forecast for {pd.to_datetime(r['month']).strftime('%Y-%m')}. "
                    f"Forecast purchase volume {r['forecast_quantity']:,.0f} units."
                ),
            })

        for _, r in get_anomaly_records(self.df).head(50).iterrows():
            docs.append({
                "type": "anomaly",
                "title": f"Pricing anomaly {r['PO_ID']}",
                "text": (
                    f"Pricing anomaly purchase order {r['PO_ID']}. Supplier {r['Supplier']}. "
                    f"Category {r['Item_Category']}. Order date {pd.to_datetime(r['Order_Date']).date()}. "
                    f"Unit price ${r['Unit_Price']:,.2f}. Negotiated price ${r['Negotiated_Price']:,.2f}. "
                    f"Price variance ${r['price_variance']:,.2f}. Quantity {int(r['Quantity'])}. "
                    f"Savings opportunity ${r['savings_opportunity']:,.0f}. Order status {r['Order_Status']}."
                ),
            })

        row_columns = [
            "PO_ID", "Supplier", "Item_Category", "Order_Date", "Delivery_Date",
            "Quantity", "Unit_Price", "Negotiated_Price", "Defective_Units",
            "Compliance", "Delivery_Status", "Order_Status", "lead_time_days",
            "spend_amount", "price_variance", "savings_opportunity",
        ]
        for _, r in self.df[row_columns].iterrows():
            docs.append({
                "type": "purchase_order",
                "title": f"Purchase order {r['PO_ID']}",
                "text": (
                    f"Purchase order {r['PO_ID']}. Supplier {r['Supplier']}. Category {r['Item_Category']}. "
                    f"Order date {pd.to_datetime(r['Order_Date']).date()}. Delivery date {pd.to_datetime(r['Delivery_Date']).date()}. "
                    f"Quantity {int(r['Quantity'])}. Unit price ${r['Unit_Price']:,.2f}. "
                    f"Negotiated price ${r['Negotiated_Price']:,.2f}. Spend ${r['spend_amount']:,.0f}. "
                    f"Price variance ${r['price_variance']:,.2f}. Savings opportunity ${r['savings_opportunity']:,.0f}. "
                    f"Defective units {int(r['Defective_Units'])}. Compliance {r['Compliance']}. "
                    f"Delivery status {r['Delivery_Status']}. Order status {r['Order_Status']}. "
                    f"Lead time {int(r['lead_time_days'])} days."
                ),
            })
        return docs

    def _retrieve_context(self, question: str, k: int = 12) -> list[dict]:
        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.document_matrix).ravel()
        q_lower = question.lower()

        for i, doc in enumerate(self.knowledge_base):
            text_lower = doc["text"].lower()
            if doc["type"] in {"supplier", "category"} and doc["title"].lower() in q_lower:
                scores[i] += 0.35
            if any(term in q_lower and term in text_lower for term in ["risk", "defect", "compliance", "delivery", "price", "forecast", "spend", "saving"]):
                scores[i] += 0.03

        top_indices = scores.argsort()[::-1][:k]
        return [
            {**self.knowledge_base[i], "score": float(scores[i])}
            for i in top_indices
            if scores[i] > 0
        ]

    def _build_rag_prompt(self, question: str, retrieved: list[dict]) -> str:
        evidence = "\n".join(
            f"[{i + 1}] {doc['title']} ({doc['type']}): {doc['text']}"
            for i, doc in enumerate(retrieved)
        )
        return (
            "You are ProcureIQ, a retrieval-augmented procurement analysis agent. "
            "Answer the user's question using only the retrieved procurement evidence below. "
            "If the exact answer is not present, say what the closest evidence indicates and avoid inventing facts. "
            "Use concise business language, include numbers, and recommend an action when useful.\n\n"
            f"Question: {question}\n\n"
            f"Retrieved evidence:\n{evidence}"
        )

    def _call_openai_llm(self, prompt: str) -> str | None:
        if os.getenv("USE_LLM", "true").lower() in {"0", "false", "no"}:
            return None

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a specialized procurement analysis assistant. "
                                "Use the data provided to answer questions about spend, supplier performance, defects, compliance, delivery, trends, forecasting, and pricing anomalies."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=45,
            )
            if response.status_code != 200:
                return None
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

    def _answer_top_suppliers(self, n: int = 5) -> str:
        top = (
            self.df.groupby("Supplier", as_index=False)["spend_amount"]
            .sum()
            .sort_values("spend_amount", ascending=False)
            .head(n)
        )
        lines = [f"Top {n} suppliers by spend:"]
        for _, r in top.iterrows():
            lines.append(f"- {r['Supplier']}: ${r['spend_amount']:,.0f}")
        return "\n".join(lines)

    def _match_supplier(self, name: str) -> Optional[str]:
        if not name:
            return None
        candidates = sorted(self.df["Supplier"].dropna().unique())
        # exact match first
        for c in candidates:
            if c.lower() == name.lower():
                return c
        # fuzzy match
        matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.5)
        return matches[0] if matches else None

    def _answer_supplier_details(self, name: str) -> str:
        match = self._match_supplier(name)
        if not match:
            return f"No supplier matched '{name}'. Try a different name or check spelling."
        s = self.df[self.df["Supplier"] == match]
        total_spend = float(s["spend_amount"].sum())
        orders = int(s["PO_ID"].nunique())
        on_time = float(s["is_on_time"].mean() * 100)
        defect = float((s["Defective_Units"].sum() / s["Quantity"].sum() * 100) if s["Quantity"].sum() else 0)
        compliance = float(s["is_compliant"].mean() * 100)
        return (
            f"Supplier '{match}' — total spend: ${total_spend:,.0f}; orders: {orders}; "
            f"on-time: {on_time:.1f}%; defects: {defect:.2f}%; compliance: {compliance:.1f}%"
        )

    def _answer_top_categories(self, n: int = 5) -> str:
        top = (
            self.df.groupby("Item_Category", as_index=False)["spend_amount"]
            .sum()
            .sort_values("spend_amount", ascending=False)
            .head(n)
        )
        lines = [f"Top {n} categories by spend:"]
        for _, r in top.iterrows():
            lines.append(f"- {r['Item_Category']}: ${r['spend_amount']:,.0f}")
        return "\n".join(lines)

    def _match_category(self, name: str) -> Optional[str]:
        if not name:
            return None
        candidates = sorted(self.df["Item_Category"].dropna().unique())
        for c in candidates:
            if c.lower() == name.lower():
                return c
        matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.5)
        return matches[0] if matches else None

    def _answer_category_details(self, name: str) -> str:
        match = self._match_category(name)
        if not match:
            return f"No category matched '{name}'. Try a different category name or check spelling."
        c = self.df[self.df["Item_Category"] == match]
        total_spend = float(c["spend_amount"].sum())
        orders = int(c["PO_ID"].nunique())
        avg_price = float(c["Unit_Price"].mean())
        defect = float((c["Defective_Units"].sum() / c["Quantity"].sum() * 100) if c["Quantity"].sum() else 0)
        anomalies = int(c["is_anomaly"].sum())
        return (
            f"Category '{match}' — spend: ${total_spend:,.0f}; orders: {orders}; "
            f"average unit price: ${avg_price:,.2f}; defects: {defect:.2f}%; pricing anomalies: {anomalies}."
        )

    def _answer_kpi(self, metric: str) -> str:
        kpis = compute_kpis(self.df)
        if metric == "spend":
            return f"Total procurement spend is ${kpis['total_spend']:,.0f} across {kpis['total_orders']:,} orders."
        if metric == "orders":
            return f"Total orders: {kpis['total_orders']:,}."
        if metric == "delivery":
            late_rate = 100 - kpis["on_time_rate"]
            return f"On-time delivery rate is {kpis['on_time_rate']:.1f}%, so about {late_rate:.1f}% of orders are late or not marked delivered."
        if metric == "compliance":
            return f"Compliance rate is {kpis['compliance_rate']:.1f}% across the selected purchase orders."
        if metric == "defects":
            return f"Overall defect rate is {kpis['defect_rate']:.2f}% based on {int(self.df['Defective_Units'].sum()):,} defective units out of {int(self.df['Quantity'].sum()):,} units ordered."
        if metric == "savings":
            direction = "opportunity" if kpis["savings"] >= 0 else "favorable negotiated-price impact"
            return f"Savings {direction} is ${abs(kpis['savings']):,.0f}. Negative savings in this dataset means negotiated prices are below the listed unit prices."
        if metric == "lead_time":
            return f"Average lead time is {kpis['avg_lead_time']:.1f} days."
        return ""

    def _answer_monthly_trend(self, n: int = 6) -> str:
        monthly = (
            self.df.groupby("month", as_index=False)["spend_amount"]
            .sum()
            .sort_values("month")
        )
        if monthly.empty:
            return "No monthly spend trend is available for the selected data."
        latest = monthly.tail(n)
        peak = monthly.sort_values("spend_amount", ascending=False).iloc[0]
        lines = [f"Recent monthly spend trend, last {len(latest)} months:"]
        for _, r in latest.iterrows():
            lines.append(f"- {pd.to_datetime(r['month']).strftime('%Y-%m')}: ${r['spend_amount']:,.0f}")
        lines.append(f"Peak spend month is {pd.to_datetime(peak['month']).strftime('%Y-%m')} at ${peak['spend_amount']:,.0f}.")
        return "\n".join(lines)

    def _answer_supplier_performance(self) -> str:
        summary = get_supplier_summary(self.df)
        best = summary.sort_values(["risk_score", "total_spend"], ascending=[True, False]).iloc[0]
        worst = summary.sort_values("risk_score", ascending=False).iloc[0]
        return (
            "Supplier performance snapshot:\n"
            f"- Best overall profile: {best['Supplier']} ({best['risk_level']}), on-time {best['on_time_rate'] * 100:.1f}%, defects {best['defect_rate']:.2f}%, compliance {best['compliance_rate'] * 100:.1f}%.\n"
            f"- Highest risk profile: {worst['Supplier']} ({worst['risk_level']}), on-time {worst['on_time_rate'] * 100:.1f}%, defects {worst['defect_rate']:.2f}%, compliance {worst['compliance_rate'] * 100:.1f}%."
        )

    def _answer_risk_suppliers(self, n: int = 3) -> str:
        summary = get_supplier_summary(self.df).sort_values("risk_score", ascending=False).head(n)
        lines = [f"Top {len(summary)} supplier risks:"]
        for _, r in summary.iterrows():
            lines.append(
                f"- {r['Supplier']}: {r['risk_level']} "
                f"(on-time {r['on_time_rate'] * 100:.1f}%, defects {r['defect_rate']:.2f}%, compliance {r['compliance_rate'] * 100:.1f}%)"
            )
        lines.append("Action: prioritize performance reviews for suppliers with weak delivery, quality, or compliance scores.")
        return "\n".join(lines)

    def _answer_category_defects(self, n: int = 5) -> str:
        category_defects = (
            self.df.groupby("Item_Category")
            .apply(lambda g: pd.Series({
                "defect_rate": (g["Defective_Units"].sum() / g["Quantity"].sum() * 100) if g["Quantity"].sum() else 0,
                "defective_units": g["Defective_Units"].sum(),
            }))
            .reset_index()
            .sort_values("defect_rate", ascending=False)
            .head(n)
        )
        lines = [f"Top {len(category_defects)} categories by defect rate:"]
        for _, r in category_defects.iterrows():
            lines.append(f"- {r['Item_Category']}: {r['defect_rate']:.2f}% ({int(r['defective_units']):,} defective units)")
        return "\n".join(lines)

    def _answer_anomalies(self, n: int = 5) -> str:
        anomalies = get_anomaly_records(self.df).head(n)
        if anomalies.empty:
            return "No pricing anomalies are flagged in the selected data."
        lines = [f"{int(self.df['is_anomaly'].sum())} pricing anomalies are flagged. Largest examples:"]
        for _, r in anomalies.iterrows():
            lines.append(
                f"- {r['Supplier']} / {r['Item_Category']}: unit ${r['Unit_Price']:,.2f} vs negotiated ${r['Negotiated_Price']:,.2f} "
                f"(variance ${r['price_variance']:,.2f})"
            )
        return "\n".join(lines)

    def _answer_recommended_actions(self) -> str:
        suppliers = get_supplier_summary(self.df)
        categories = get_category_summary(self.df)
        risky = suppliers.sort_values("risk_score", ascending=False).iloc[0]
        savings = categories.assign(savings_abs=categories["savings_opportunity"].abs()).sort_values("savings_abs", ascending=False).iloc[0]
        anomalies = int(self.df["is_anomaly"].sum())
        return (
            "Recommended actions:\n"
            f"- Review {risky['Supplier']} first; it has the highest risk score with on-time delivery at {risky['on_time_rate'] * 100:.1f}% and defect rate at {risky['defect_rate']:.2f}%.\n"
            f"- Recheck pricing in {savings['Item_Category']}; it has the largest pricing impact at ${abs(savings['savings_opportunity']):,.0f}.\n"
            f"- Audit the {anomalies} flagged pricing anomalies before renewing supplier terms."
        )

    def _answer_compare_suppliers(self, a: str, b: str) -> str:
        def supplier_metrics(name: str) -> dict:
            matched = self._match_supplier(name)
            if not matched:
                return {}
            s = self.df[self.df["Supplier"] == matched]
            if s.empty:
                return {}
            return {
                "supplier": matched,
                "total_spend": float(s["spend_amount"].sum()),
                "orders": int(s["PO_ID"].nunique()),
                "on_time": float(s["is_on_time"].mean() * 100),
                "defect_rate": float((s["Defective_Units"].sum() / s["Quantity"].sum() * 100) if s["Quantity"].sum() else 0),
                "compliance": float(s["is_compliant"].mean() * 100),
            }

        ma = supplier_metrics(a)
        mb = supplier_metrics(b)
        if not ma:
            return f"No records found for supplier '{a}'."
        if not mb:
            return f"No records found for supplier '{b}'."

        lines = [f"Comparison: {ma['supplier']} vs {mb['supplier']}"]
        lines.append(f"- {ma['supplier']} spend: ${ma['total_spend']:,.0f} ({ma['orders']} orders), on-time {ma['on_time']:.1f}%, defects {ma['defect_rate']:.2f}%, compliance {ma['compliance']:.1f}%")
        lines.append(f"- {mb['supplier']} spend: ${mb['total_spend']:,.0f} ({mb['orders']} orders), on-time {mb['on_time']:.1f}%, defects {mb['defect_rate']:.2f}%, compliance {mb['compliance']:.1f}%")
        # simple suggestion
        if ma["total_spend"] > mb["total_spend"]:
            lines.append(f"Suggestion: {ma['supplier']} has higher spend; review pricing and performance for potential savings.")
        else:
            lines.append(f"Suggestion: {mb['supplier']} has higher spend; review pricing and performance for potential savings.")
        return "\n".join(lines)

    def _answer_forecast(self, months: int = 6) -> str:
        fc = forecast_monthly_spend(self.df, horizon_months=months)
        if fc.empty:
            return "Not enough historical data to produce a forecast."
        lines = [f"Forecast for next {months} months:"]
        for _, r in fc.iterrows():
            m = pd.to_datetime(r["month"]).strftime("%Y-%m")
            val = float(r.get("forecast_spend", r.get("predicted_spend", 0)))
            lines.append(f"- {m}: ${val:,.0f}")
        return "\n".join(lines)

    def answer(self, question: str) -> str:
        q = (question or "").strip()
        if not q:
            return "Please ask a procurement question about spend, supplier performance, delivery, pricing, compliance, defects, or demand."

        retrieved = self._retrieve_context(q)
        llm_answer = self._call_openai_llm(self._build_rag_prompt(q, retrieved))
        if llm_answer:
            return llm_answer

        return (
            "LLM answering is not configured. Set USE_LLM=true and provide OPENAI_API_KEY "
            "so ProcureIQ can answer using the retrieved procurement dataset context."
        )
