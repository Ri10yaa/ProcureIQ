# ProcureIQ: Intelligent Procurement Analytics Dashboard

A Python-based procurement analytics dashboard and AI investigation assistant designed to analyze supplier performance, spending, pricing variance, and demand forecasting using a procurement dataset.

## Overview

This project combines:
- procurement KPI dashboards
- supplier performance analysis
- spend and category analysis
- price and anomaly tracking
- demand forecasting
- an AI-powered procurement query agent

The dashboard is implemented in Python using `Streamlit` and `Plotly`.

## Project Structure

```text
latestProcureIQ/
├── app.py
├── .env
├── .env.example
├── requirements.txt
├── README.md
├── data/
│   └── Purchase_Orders.xlsx
├── docs/
│   └── ProcureIQ_abstract.pdf
├── repo_src/
│   └── README.md
├── procureiq_app/
│   ├── __init__.py
│   ├── agent.py
│   ├── analytics.py
│   ├── data_processing.py
│   └── __pycache__/
```

## Dataset

Source dataset used:
- `data/Purchase_Orders.xlsx`

This dataset contains procurement order records including supplier, category, order dates, quantity, unit price, negotiated price, compliance, delivery status, and defect counts.

## Features

- Executive KPI cards
- Monthly procurement spend trends
- Supplier performance scorecard
- Cost and category analysis
- Forecasted spend projection
- Price anomaly view
- AI procurement investigation agent
- Natural-language procurement Q&A

## Tech Stack

- Python
- Streamlit
- Plotly
- Pandas
- NumPy
- scikit-learn
- OpenAI API (optional LLM support)

## Setup

1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Configure environment variables

Copy `.env.example` to `.env` and set your values:

```bash
cp .env.example .env
```

Then update `.env`:

```env
USE_LLM=true
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

## Run the dashboard

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Example procurement questions

- Which supplier has the highest spend?
- What is the total procurement spend?
- Who is the highest risk supplier?
- What is the on-time delivery rate?
- Which category has the highest spend?
- Compare Alpha_Inc vs Delta_Logistics
- What is the savings opportunity?
- What is the forecast?
- Tell me about anomalies
- How is Delta_Logistics performing?

## Understanding Procurement Analytics

If you are new to procurement, this project is about using purchase order data to help organizations understand:

- how much money they are spending on suppliers
- which suppliers are reliable or risky
- which categories consume the most budget
- whether prices are unusually high or low
- whether deliveries are on time
- whether orders are compliant with policy
- whether quality issues or defects are frequent
- how future demand and spend may change

Procurement is the process of buying goods and services needed by an organization. In practice, procurement teams need to monitor purchase activity carefully because poor supplier performance, late deliveries, high prices, and quality defects can increase costs and reduce business efficiency.

This project turns raw procurement records into decision-making insight by converting purchase orders into measurable KPIs and actionable intelligence.

## Dataset Metrics

The selected dataset contains 777 purchase orders across 5 suppliers and 5 product categories.

Key metrics in the project:
- Total spend: $49,304,823
- On-time delivery rate: 72.1%
- Compliance rate: 82.4%
- Defect rate: 5.64%
- Average lead time: 12.1 days
- Average order value: $63,455
- Savings opportunity: -$3,931,126 (negative value indicates the negotiated price is lower than benchmark or current unit price in this analytical perspective)
- Price anomaly count: 117 flagged anomalies

These figures help explain the overall health of procurement operations in the dataset and identify where management action is needed.

## Why this project matters

Organizations benefit from procurement analytics because it helps them:

- reduce unnecessary spend
- identify high-risk suppliers
- negotiate better pricing
- avoid late or non-compliant deliveries
- control defect and quality issues
- plan future procurement demand more accurately

In short, this dashboard turns large purchasing data into practical decision support for buyers, procurement managers, and leadership teams.

## Notes

- The procurement investigation agent requires the LLM configuration in `.env`.
- The agent uses retrieval-augmented context from the procurement dataset before calling the LLM.
- The app is designed for procurement analytics and sourcing investigations using procurement order data.

## License

This project is intended for academic and portfolio use.
