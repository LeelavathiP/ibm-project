# =============================================================================
#  E-Commerce Sales Data Analytics
#  IBM SkillsBuild – Data Analytics with AI Academic Internship
#  BharatCares & AICTEpyth
# =============================================================================
#  Dataset  : ecommerce_sales_34500.csv
#  Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit
#
#  ─── HOW TO RUN ───────────────────────────────────────────────────────────
#  Script mode   →  python PuchakayalaLeelavathi_EcommerceSalesAnalysis.py
#  Dashboard mode→  streamlit run PuchakayalaLeelavathi_EcommerceSalesAnalysis.py
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Detect whether we are running inside Streamlit
_STREAMLIT_MODE = "streamlit" in sys.modules or any("streamlit" in a for a in sys.argv)

if _STREAMLIT_MODE:
    import streamlit as st

# ── Global plot style ─────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 13, "axes.labelsize": 11})

DATASET = "ecommerce_sales_34500.csv"

# =============================================================================
# ░░  SHARED UTILITIES  ░░
# =============================================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, parse dates, and engineer all derived columns."""

    # ── Missing values ────────────────────────────────────────────────────────
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # ── Date parsing & temporal features ─────────────────────────────────────
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["year"]       = df["order_date"].dt.year
    df["month"]      = df["order_date"].dt.month
    df["quarter"]    = df["order_date"].dt.quarter
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)

    # ── Derived columns ───────────────────────────────────────────────────────
    df["gross_profit"] = df["total_amount"] * (df["profit_margin"] / 100)
    df["net_revenue"]  = df["total_amount"] - df["shipping_cost"]
    df["is_returned"]  = (df["returned"].str.lower() == "yes").astype(int)
    df["age_group"]    = pd.cut(
        df["customer_age"],
        bins=[0, 25, 35, 45, 60, 120],
        labels=["18-25", "26-35", "36-45", "46-60", "60+"]
    )
    return df


def rfm_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """Build RFM scores and assign segment labels for every customer."""
    snapshot = df["order_date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_id").agg(
        recency  =("order_date",   lambda x: (snapshot - x.max()).days),
        frequency=("order_id",     "count"),
        monetary =("total_amount", "sum"),
    ).reset_index()

    rfm["R"] = pd.qcut(rfm["recency"],                        q=4, labels=[4,3,2,1]).astype(int)
    rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), q=4, labels=[1,2,3,4]).astype(int)
    rfm["M"] = pd.qcut(rfm["monetary"],                       q=4, labels=[1,2,3,4]).astype(int)
    rfm["RFM_Score"] = rfm["R"] + rfm["F"] + rfm["M"]

    def _label(s):
        if s >= 10: return "Champions"
        elif s >= 8: return "Loyal Customers"
        elif s >= 6: return "Potential Loyalists"
        elif s >= 4: return "At Risk"
        else:        return "Lost Customers"

    rfm["Segment"] = rfm["RFM_Score"].apply(_label)
    return rfm


def build_ml_model(df: pd.DataFrame):
    """
    Train a Linear Regression model to predict order-level revenue.
    Returns (model, X_test, y_test, y_pred, feature_names).
    """
    feat_cols = [
        "price", "discount", "quantity", "shipping_cost",
        "delivery_time_days", "profit_margin",
        "month", "quarter", "year", "customer_age",
        "category", "region", "payment_method", "customer_gender",
    ]
    mdf = df[feat_cols + ["total_amount"]].copy()
    le  = LabelEncoder()
    for col in ["category", "region", "payment_method", "customer_gender"]:
        mdf[col] = le.fit_transform(mdf[col].astype(str))

    X, y = mdf.drop("total_amount", axis=1), mdf["total_amount"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred, feat_cols


def revenue_forecast(df: pd.DataFrame, months: int = 6):
    """Return (monthly_df_with_t, future_t_array, future_rev_array)."""
    monthly = (
        df.groupby("year_month")["total_amount"]
        .sum().reset_index().sort_values("year_month")
    )
    monthly["t"] = np.arange(len(monthly))
    trend = LinearRegression()
    trend.fit(monthly[["t"]], monthly["total_amount"])
    future_t   = np.arange(len(monthly), len(monthly) + months).reshape(-1, 1)
    future_rev = trend.predict(future_t)
    return monthly, future_t.flatten(), future_rev


# =============================================================================
# ░░  SCRIPT MODE  ░░
# =============================================================================

def run_script():
    """Full standalone pipeline: load → EDA → ML → forecast (no Streamlit)."""

    # ── Load & preprocess ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  E-COMMERCE SALES DATA ANALYTICS")
    print("  IBM SkillsBuild | BharatCares & AICTE")
    print("="*60)

    df = preprocess(pd.read_csv(DATASET))
    print(f"✅  Loaded  →  {len(df):,} rows × {df.shape[1]} columns")

    print(f"\n  Date range  : {df['order_date'].min().date()} → {df['order_date'].max().date()}")
    print(f"  Customers   : {df['customer_id'].nunique():,}")
    print(f"  Products    : {df['product_id'].nunique():,}")
    print(f"  Categories  : {sorted(df['category'].unique())}")
    print(f"  Regions     : {sorted(df['region'].unique())}")
    print(f"  Return rate : {df['is_returned'].mean()*100:.2f}%")

    # ── RFM & KPIs ────────────────────────────────────────────────────────────
    rfm = rfm_segmentation(df)

    total_rev    = df["total_amount"].sum()
    total_orders = df["order_id"].nunique()
    aov          = total_rev / total_orders
    avg_pm       = df["profit_margin"].mean()
    repeat_rate  = rfm[rfm["frequency"] > 1].shape[0] / rfm.shape[0] * 100
    return_rate  = df["is_returned"].mean() * 100
    clv          = rfm["monetary"].mean()

    print("\n" + "="*60)
    print("  KEY PERFORMANCE INDICATORS (KPIs)")
    print("="*60)
    print(f"  Total Revenue         : ₹{total_rev:>15,.2f}")
    print(f"  Total Orders          : {total_orders:>16,}")
    print(f"  Avg Order Value (AOV) : ₹{aov:>15,.2f}")
    print(f"  Avg Profit Margin     : {avg_pm:>15.2f}%")
    print(f"  Repeat Customer Rate  : {repeat_rate:>15.2f}%")
    print(f"  Return Rate           : {return_rate:>15.2f}%")
    print(f"  Avg Customer LTV      : ₹{clv:>15,.2f}")

    # ── EDA PLOTS ─────────────────────────────────────────────────────────────

    # Plot 1 – Top 10 categories by revenue
    cat_rev = (df.groupby("category")["total_amount"]
               .sum().sort_values(ascending=False).head(10))
    fig, ax = plt.subplots(figsize=(10, 5))
    cat_rev[::-1].plot(kind="barh", ax=ax, color=sns.color_palette("Blues_r", 10))
    ax.set_title("Top 10 Product Categories by Total Revenue")
    ax.set_xlabel("Total Revenue (₹)")
    for i, v in enumerate(cat_rev[::-1]):
        ax.text(v + 200, i, f"₹{v:,.0f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("plot_01_top10_categories.png")
    plt.show()

    # Plot 2 – Monthly revenue trend
    monthly_df = (df.groupby("year_month")["total_amount"]
                  .sum().reset_index().sort_values("year_month"))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly_df["year_month"], monthly_df["total_amount"],
            marker="o", linewidth=2, color="#3b82d4")
    ax.fill_between(monthly_df["year_month"], monthly_df["total_amount"],
                    alpha=0.12, color="#3b82d4")
    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (₹)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig("plot_02_monthly_trend.png")
    plt.show()

    # Plot 3 – Revenue by region
    region_rev = df.groupby("region")["total_amount"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    region_rev.plot(kind="bar", ax=ax,
                    color=sns.color_palette("Set2", len(region_rev)), edgecolor="white")
    ax.set_title("Total Revenue by Region")
    ax.set_ylabel("Revenue (₹)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("plot_03_region_revenue.png")
    plt.show()

    # Plot 4 – Payment method pie
    payment = df["payment_method"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(payment.values, labels=payment.index, autopct="%1.1f%%",
           startangle=140, colors=sns.color_palette("pastel"),
           wedgeprops={"edgecolor": "white"})
    ax.set_title("Payment Method Distribution")
    plt.tight_layout()
    plt.savefig("plot_04_payment_methods.png")
    plt.show()

    # Plot 5 – Customer demographics
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(df["customer_age"], bins=20, color="#7c5cd8", edgecolor="white")
    axes[0].set_title("Customer Age Distribution")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")
    gender_rev = df.groupby("customer_gender")["total_amount"].sum()
    axes[1].bar(gender_rev.index, gender_rev.values,
                color=sns.color_palette("Set1", len(gender_rev)))
    axes[1].set_title("Revenue by Gender")
    axes[1].set_ylabel("Revenue (₹)")
    plt.tight_layout()
    plt.savefig("plot_05_customer_demographics.png")
    plt.show()

    # Plot 6 – Profit margin by category
    fig, ax = plt.subplots(figsize=(12, 5))
    cat_order = (df.groupby("category")["profit_margin"]
                 .median().sort_values(ascending=False).index)
    sns.boxplot(data=df, x="category", y="profit_margin",
                order=cat_order, palette="Set3", ax=ax)
    ax.axhline(0, color="red", linestyle="--", linewidth=1, label="Break-even")
    ax.set_title("Profit Margin Distribution by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Profit Margin (%)")
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("plot_06_profit_margins.png")
    plt.show()

    # Plot 7 – RFM segment distribution
    seg_counts = rfm["Segment"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    seg_counts.plot(kind="bar", ax=ax,
                    color=sns.color_palette("husl", len(seg_counts)), edgecolor="white")
    ax.set_title("RFM Customer Segments")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Number of Customers")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("plot_07_rfm_segments.png")
    plt.show()

    print("\n  RFM Segment Summary (averages):")
    summary = (rfm.groupby("Segment")[["recency", "frequency", "monetary"]]
               .mean().round(1))
    print(summary.to_string())

    # ── ML MODEL ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  LINEAR REGRESSION – REVENUE PREDICTION")
    print("="*60)

    model, X_test, y_test, y_pred, feat_cols = build_ml_model(df)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"  MAE    : ₹{mae:,.2f}")
    print(f"  RMSE   : ₹{rmse:,.2f}")
    print(f"  R²     : {r2:.4f}")

    # Plot 8 – Actual vs Predicted
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.3, color="#3b82d4", s=10)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Fit")
    ax.set_xlabel("Actual Revenue (₹)")
    ax.set_ylabel("Predicted Revenue (₹)")
    ax.set_title("Linear Regression – Actual vs Predicted Revenue")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plot_08_actual_vs_predicted.png")
    plt.show()

    # Plot 9 – Residuals
    residuals = y_test.values - y_pred
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=40, color="#7c5cd8", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual (₹)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("plot_09_residuals.png")
    plt.show()

    # ── FORECAST ──────────────────────────────────────────────────────────────
    monthly, future_t, future_rev = revenue_forecast(df, months=6)

    # Plot 10 – Revenue forecast
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly["t"], monthly["total_amount"],
            marker="o", label="Historical", color="#3b82d4")
    ax.plot(future_t, future_rev,
            marker="s", linestyle="--", color="#e05c2a",
            label="Forecast (+6 months)")
    ax.set_title("Monthly Revenue – Historical + 6-Month Forecast")
    ax.set_xlabel("Month Index")
    ax.set_ylabel("Revenue (₹)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plot_10_revenue_forecast.png")
    plt.show()

    print("\n  6-Month Revenue Forecast:")
    for i, r in enumerate(future_rev, 1):
        print(f"    Month +{i} : ₹{r:,.2f}")

    print("\n✅  All 10 plots saved. Analysis complete.")


# =============================================================================
# ░░  STREAMLIT DASHBOARD MODE  ░░
# =============================================================================

def run_dashboard():
    """Four-page interactive Streamlit BI dashboard."""

    # ── Page config ───────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="E-Commerce Sales Analytics",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
        .main { background-color: #f7f8fa; }

        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,.06);
        }
        div[data-testid="metric-container"] label {
            color: #57606a !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        div[data-testid="metric-container"] div[data-testid="metric-value"] {
            color: #1f2328 !important;
            font-size: 24px !important;
            font-weight: 700 !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #1f2328;
        }
        .section-header {
            font-size: 17px;
            font-weight: 700;
            color: #1f2328;
            border-left: 4px solid #3b82d4;
            padding-left: 10px;
            margin: 24px 0 12px 0;
        }
        .footer {
            text-align: center;
            font-size: 11px;
            color: #57606a;
            margin-top: 48px;
            padding-top: 12px;
            border-top: 1px solid #e5e7eb;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Cached helpers ────────────────────────────────────────────────────────
    @st.cache_data(show_spinner="Loading & preprocessing data…")
    def _load(path):
        return preprocess(pd.read_csv(path))

    @st.cache_data(show_spinner="Running RFM segmentation…")
    def _rfm(_df):
        return rfm_segmentation(_df)

    @st.cache_data(show_spinner="Training ML model…")
    def _model(_df):
        return build_ml_model(_df)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🛒 E-Commerce Analytics")
        st.markdown("**IBM SkillsBuild Internship**")
        st.markdown("BharatCares & AICTE")
        st.divider()

        uploaded  = st.file_uploader("Upload CSV (optional)", type="csv")
        data_path = uploaded if uploaded else DATASET

        df_raw = _load(data_path)
        st.divider()
        st.markdown("### 🔍 Filters")

        sel_years  = st.multiselect(
            "Year", sorted(df_raw["year"].dropna().unique().astype(int).tolist()),
            default=sorted(df_raw["year"].dropna().unique().astype(int).tolist())
        )
        sel_regions    = st.multiselect("Region",   sorted(df_raw["region"].unique()),   default=sorted(df_raw["region"].unique()))
        sel_categories = st.multiselect("Category", sorted(df_raw["category"].unique()), default=sorted(df_raw["category"].unique()))

        st.divider()
        page = st.radio("📋 Dashboard", [
            "🏠 Executive Overview",
            "📊 EDA & Sales Trends",
            "👥 Customer Analysis",
            "🤖 Predictive Model",
        ])

    # ── Apply filters ─────────────────────────────────────────────────────────
    df = df_raw[
        df_raw["year"].isin(sel_years) &
        df_raw["region"].isin(sel_regions) &
        df_raw["category"].isin(sel_categories)
    ].copy()

    rfm = _rfm(df)

    # =========================================================================
    # PAGE 1 – EXECUTIVE OVERVIEW
    # =========================================================================
    if page == "🏠 Executive Overview":
        st.title("🛒 E-Commerce Sales Analytics Dashboard")
        st.caption("IBM SkillsBuild · Data Analytics with AI · BharatCares & AICTE")
        st.divider()

        total_rev    = df["total_amount"].sum()
        total_orders = df["order_id"].nunique()
        aov          = total_rev / total_orders
        avg_pm       = df["profit_margin"].mean()
        repeat_rate  = rfm[rfm["frequency"] > 1].shape[0] / rfm.shape[0] * 100
        return_rate  = df["is_returned"].mean() * 100
        clv          = rfm["monetary"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Total Revenue",        f"₹{total_rev:,.0f}")
        c2.metric("🛍️ Total Orders",          f"{total_orders:,}")
        c3.metric("📦 Avg Order Value (AOV)", f"₹{aov:,.2f}")
        c4.metric("📈 Avg Profit Margin",     f"{avg_pm:.1f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("🔁 Repeat Customer Rate",  f"{repeat_rate:.1f}%")
        c6.metric("↩️ Return Rate",            f"{return_rate:.1f}%")
        c7.metric("👤 Unique Customers",       f"{df['customer_id'].nunique():,}")
        c8.metric("💎 Avg Customer LTV",       f"₹{clv:,.0f}")

        st.markdown('<div class="section-header">Monthly Revenue Trend</div>', unsafe_allow_html=True)
        monthly = df.groupby("year_month")["total_amount"].sum().reset_index().sort_values("year_month")
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(monthly["year_month"], monthly["total_amount"],
                marker="o", color="#3b82d4", linewidth=2)
        ax.fill_between(monthly["year_month"], monthly["total_amount"],
                        alpha=0.1, color="#3b82d4")
        ax.set_ylabel("Revenue (₹)")
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<div class="section-header">Revenue by Region</div>', unsafe_allow_html=True)
            reg = df.groupby("region")["total_amount"].sum().sort_values(ascending=False)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            reg.plot(kind="bar", ax=ax2,
                     color=sns.color_palette("Set2", len(reg)), edgecolor="white")
            ax2.set_ylabel("Revenue (₹)")
            ax2.set_xlabel("")
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        with col_r:
            st.markdown('<div class="section-header">Revenue by Category</div>', unsafe_allow_html=True)
            cat = df.groupby("category")["total_amount"].sum().sort_values(ascending=False).head(8)
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            cat[::-1].plot(kind="barh", ax=ax3, color=sns.color_palette("Blues_r", 8))
            ax3.set_xlabel("Revenue (₹)")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

    # =========================================================================
    # PAGE 2 – EDA & SALES TRENDS
    # =========================================================================
    elif page == "📊 EDA & Sales Trends":
        st.title("📊 Exploratory Data Analysis")
        st.divider()

        st.markdown('<div class="section-header">Quarterly Revenue Heatmap (Year × Quarter)</div>', unsafe_allow_html=True)
        pivot = df.pivot_table(
            values="total_amount", index="year",
            columns="quarter", aggfunc="sum", fill_value=0
        )
        pivot.columns = [f"Q{c}" for c in pivot.columns]
        fig_h, ax_h = plt.subplots(figsize=(8, 3))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                    linewidths=0.5, ax=ax_h)
        ax_h.set_title("Revenue Heatmap – Year × Quarter")
        plt.tight_layout()
        st.pyplot(fig_h)
        plt.close()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Payment Method Split</div>', unsafe_allow_html=True)
            pay = df["payment_method"].value_counts()
            fig_p, ax_p = plt.subplots(figsize=(5, 5))
            ax_p.pie(pay.values, labels=pay.index, autopct="%1.1f%%",
                     colors=sns.color_palette("pastel"), startangle=140,
                     wedgeprops={"edgecolor": "white"})
            plt.tight_layout()
            st.pyplot(fig_p)
            plt.close()

        with col2:
            st.markdown('<div class="section-header">Profit Margin by Category</div>', unsafe_allow_html=True)
            cat_order = (df.groupby("category")["profit_margin"]
                         .median().sort_values(ascending=False).index)
            fig_b, ax_b = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x="category", y="profit_margin",
                        order=cat_order, palette="Set3", ax=ax_b)
            ax_b.axhline(0, color="red", linestyle="--", linewidth=1)
            plt.xticks(rotation=30, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_b)
            plt.close()

        st.markdown('<div class="section-header">Average Delivery Time by Region</div>', unsafe_allow_html=True)
        del_time = df.groupby("region")["delivery_time_days"].mean().sort_values()
        fig_d, ax_d = plt.subplots(figsize=(8, 3))
        del_time.plot(kind="barh", ax=ax_d,
                      color=sns.color_palette("rocket", len(del_time)))
        ax_d.set_xlabel("Avg Delivery Days")
        plt.tight_layout()
        st.pyplot(fig_d)
        plt.close()

        st.markdown('<div class="section-header">Top Categories – Orders vs Revenue</div>', unsafe_allow_html=True)
        cat_stats = df.groupby("category").agg(
            Orders  =("order_id",     "count"),
            Revenue =("total_amount", "sum")
        ).sort_values("Revenue", ascending=False).head(10)
        st.dataframe(
            cat_stats.style.background_gradient(cmap="Blues"),
            use_container_width=True
        )

    # =========================================================================
    # PAGE 3 – CUSTOMER ANALYSIS
    # =========================================================================
    elif page == "👥 Customer Analysis":
        st.title("👥 Customer Analysis & RFM Segmentation")
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Revenue by Age Group</div>', unsafe_allow_html=True)
            age_rev = df.groupby("age_group", observed=True)["total_amount"].sum()
            fig_a, ax_a = plt.subplots(figsize=(6, 4))
            age_rev.plot(kind="bar", ax=ax_a,
                         color=sns.color_palette("viridis", len(age_rev)), edgecolor="white")
            ax_a.set_ylabel("Revenue (₹)")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig_a)
            plt.close()

        with col2:
            st.markdown('<div class="section-header">Revenue by Gender</div>', unsafe_allow_html=True)
            gen_rev = df.groupby("customer_gender")["total_amount"].sum()
            fig_g, ax_g = plt.subplots(figsize=(6, 4))
            ax_g.pie(gen_rev.values, labels=gen_rev.index, autopct="%1.1f%%",
                     colors=["#3b82d4", "#7c5cd8", "#e05c2a"],
                     wedgeprops={"edgecolor": "white"}, startangle=140)
            plt.tight_layout()
            st.pyplot(fig_g)
            plt.close()

        st.markdown('<div class="section-header">RFM Customer Segments</div>', unsafe_allow_html=True)
        seg_counts = rfm["Segment"].value_counts()
        col3, col4 = st.columns([1, 2])
        with col3:
            fig_r, ax_r = plt.subplots(figsize=(5, 4))
            ax_r.pie(seg_counts.values, labels=seg_counts.index, autopct="%1.0f%%",
                     colors=sns.color_palette("husl", len(seg_counts)),
                     wedgeprops={"edgecolor": "white"})
            ax_r.set_title("Segment Share")
            plt.tight_layout()
            st.pyplot(fig_r)
            plt.close()

        with col4:
            seg_summary = (rfm.groupby("Segment")[["recency", "frequency", "monetary"]]
                           .mean().round(1))
            seg_summary.columns = ["Avg Recency (days)", "Avg Frequency", "Avg Monetary (₹)"]
            st.dataframe(
                seg_summary.style.background_gradient(
                    subset=["Avg Monetary (₹)"], cmap="Greens"
                ),
                use_container_width=True
            )

        st.markdown('<div class="section-header">Top 10 Customers by Lifetime Value</div>', unsafe_allow_html=True)
        top_cust = (rfm.sort_values("monetary", ascending=False)
                    .head(10)[["customer_id", "recency", "frequency", "monetary", "Segment"]]
                    .reset_index(drop=True))
        top_cust.columns = ["Customer ID", "Recency (days)", "Orders", "Total Spend (₹)", "Segment"]
        st.dataframe(
            top_cust.style.background_gradient(subset=["Total Spend (₹)"], cmap="Blues"),
            use_container_width=True
        )

    # =========================================================================
    # PAGE 4 – PREDICTIVE MODEL
    # =========================================================================
    elif page == "🤖 Predictive Model":
        st.title("🤖 Predictive Modelling – Revenue Forecast")
        st.divider()

        model, X_test, y_test, y_pred, _ = _model(df)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        c1, c2, c3 = st.columns(3)
        c1.metric("📉 MAE",      f"₹{mae:,.2f}",  help="Mean Absolute Error")
        c2.metric("📉 RMSE",     f"₹{rmse:,.2f}", help="Root Mean Squared Error")
        c3.metric("📈 R² Score", f"{r2:.4f}",      help="Coefficient of Determination")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Actual vs Predicted Revenue</div>', unsafe_allow_html=True)
            fig_m, ax_m = plt.subplots(figsize=(6, 6))
            ax_m.scatter(y_test, y_pred, alpha=0.3, color="#3b82d4", s=8)
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax_m.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Fit")
            ax_m.set_xlabel("Actual Revenue (₹)")
            ax_m.set_ylabel("Predicted Revenue (₹)")
            ax_m.legend()
            plt.tight_layout()
            st.pyplot(fig_m)
            plt.close()

        with col2:
            st.markdown('<div class="section-header">Residual Distribution</div>', unsafe_allow_html=True)
            residuals = y_test.values - y_pred
            fig_res, ax_res = plt.subplots(figsize=(6, 6))
            ax_res.hist(residuals, bins=40, color="#7c5cd8", edgecolor="white")
            ax_res.axvline(0, color="red", linestyle="--")
            ax_res.set_xlabel("Residual (₹)")
            ax_res.set_ylabel("Frequency")
            ax_res.set_title("Residual Distribution")
            plt.tight_layout()
            st.pyplot(fig_res)
            plt.close()

        st.markdown('<div class="section-header">6-Month Revenue Forecast (Linear Trend)</div>', unsafe_allow_html=True)
        monthly, future_t, future_rev = revenue_forecast(df, months=6)

        fig_f, ax_f = plt.subplots(figsize=(13, 4))
        ax_f.plot(monthly["t"], monthly["total_amount"],
                  marker="o", label="Historical", color="#3b82d4")
        ax_f.plot(future_t, future_rev,
                  marker="s", linestyle="--", color="#e05c2a",
                  label="Forecast (+6 months)")
        ax_f.set_xlabel("Month Index")
        ax_f.set_ylabel("Revenue (₹)")
        ax_f.legend()
        plt.tight_layout()
        st.pyplot(fig_f)
        plt.close()

        forecast_df = pd.DataFrame({
            "Month Ahead":           [f"+{i}" for i in range(1, 7)],
            "Forecasted Revenue (₹)":[f"₹{r:,.2f}" for r in future_rev],
        })
        st.dataframe(forecast_df, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="footer">IBM SkillsBuild · Data Analytics with AI Academic Internship · BharatCares & AICTE</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# ░░  ENTRY POINT  ░░
# =============================================================================

if _STREAMLIT_MODE:
    run_dashboard()
else:
    run_script()
