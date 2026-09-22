# 🛒 E-Commerce Sales Data Analytics

> **IBM SkillsBuild – Data Analytics with AI Academic Internship**
> BharatCares & AICTE | September 2026

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikitlearn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview

This project delivers a complete **end-to-end data analytics solution** for an e-commerce business using Python and Business Intelligence (BI) tooling. Starting from raw transactional CSV data, it progresses through preprocessing, EDA, customer segmentation (RFM), predictive revenue modelling, and an interactive **Streamlit dashboard** — all within a single, reproducible codebase.

**Internship Domain:** Data Analytics with AI
**Program:** IBM SkillsBuild Academic Internship (BharatCares & AICTE)

---

## 📂 Project Structure

```
ecommerce-analytics/
│
├── ecommerce_sales_34500.csv   # Raw dataset (34,500 transactions)
├── ecommerce_analysis.py       # Code analysis script                    # 
├── requirements.txt            # Python dependencies
├── report.md                   # Full project report (.docx-style)
└── README.md                   # This file
```

---

## 📊 Dataset Details

| Property | Value |
|---|---|
| File | `ecommerce_sales_34500.csv` |
| Rows | 34,500 transactions |
| Time Range | 2023 – 2025 |
| Columns | 17 (order ID, customer, product, price, discount, quantity, payment, date, region, return status, revenue, shipping cost, profit margin, age, gender) |

**Key Columns:**
- `total_amount` — Final order revenue (₹)
- `profit_margin` — Profit % per order
- `order_date` — Transaction date (parsed to year/month/quarter)
- `category` — Electronics, Fashion, Grocery, Beauty, Home, Sports, Toys
- `region` — North, South, East, West
- `customer_id` — Used for RFM segmentation

---

## 🛠 Technologies Used

| Library | Version | Purpose |
|---|---|---|
| `pandas` | 2.2.2 | Data loading, wrangling, aggregation |
| `numpy` | 1.26.4 | Numerical operations |
| `matplotlib` | 3.9.0 | Static plotting |
| `seaborn` | 0.13.2 | Statistical visualisations |
| `scikit-learn` | 1.5.0 | Linear Regression, preprocessing, metrics |
| `streamlit` | 1.35.0 | Interactive BI dashboard |
| `openpyxl` | 3.1.4 | Excel export support |

---

## ⚙️ Setup Instructions

### 1. Clone / Download the project
```bash
git clone https://github.com/name/ecommerce-analytics.git
cd ecommerce-analytics
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the standalone analysis script
```bash
python ecommerce_analysis.py
```
This will print EDA stats, model results, and save all charts as `.png` files.

### 5. Launch the Streamlit dashboard
```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**

---

## 🖥 Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Executive Overview | 8 live KPI cards, monthly trend, revenue by region & category |
| 📊 EDA & Sales Trends | Quarterly heatmap, payment methods, profit margins, delivery times |
| 👥 Customer Analysis | Age/gender splits, RFM segmentation, top customers by LTV |
| 🤖 Predictive Model | Linear Regression metrics, Actual vs Predicted plot, 6-month forecast |

---

## 📈 Key Business Insights

1. **Electronics & Fashion** are the top two revenue-generating categories — prioritise inventory investment here before Q4.

2. **Q4 seasonal spike** (Oct–Dec) is consistent across all years — launch promotional campaigns 4–6 weeks in advance.

3. **South region** records the highest order volume, while **North** has the longest delivery times — negotiate faster logistics SLAs for North.

4. **26–45 age group** contributes the most revenue — target this segment with mobile-first, personalised marketing.

5. **Credit Card** is the dominant payment method — introduce cashback/rewards for UPI to shift mix and reduce processing costs.

6. **RFM Champions (~top 25% customers)** drive disproportionate revenue — invest in a VIP loyalty programme to retain them.

7. **Grocery category** has the lowest (often negative) profit margins — audit pricing strategy or shift to premium / bundled SKUs.

8. **Linear Regression forecast** projects a continued upward revenue trend over the next 6 months, supporting investment in scaling operations.

---

## 🤖 Machine Learning Model

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ~₹45 | ~₹120 | ~0.87 |

- **Algorithm:** `sklearn.linear_model.LinearRegression`
- **Target:** `total_amount` (order-level revenue)
- **Features:** 14 (price, discount, quantity, shipping cost, delivery time, profit margin, month, quarter, year, customer age, category, region, payment method, gender)
- **Split:** 80/20 train-test

---

## 📋 BI Framework Summary

### KPIs Tracked
- Total Revenue · AOV · Avg Profit Margin · Repeat Customer Rate · CLV · Return Rate

### Segmentation
- **RFM Segmentation:** Champions → Loyal → Potential Loyalists → At Risk → Lost Customers

### Strategic Levers
- Inventory optimisation aligned to seasonal demand patterns
- Targeted lifecycle marketing by RFM segment and age group
- Operational improvements (delivery SLA, return reduction)
- Revenue forecasting for budgeting and capacity planning

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
