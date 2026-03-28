import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Insights Dashboard", layout="wide")

# ---------------- SESSION ----------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ---------------- THEME ----------------
def apply_theme():
    dark = st.session_state.theme == "dark"
    bg = "#0b1220" if dark else "#f9fafb"
    text = "#f1f5f9" if dark else "#111827"

    st.markdown(f"""
    <style>
    .stApp {{ background: {bg}; color: {text}; }}
    .header {{
        background: linear-gradient(135deg,#7c3aed,#ec4899);
        padding: 30px;
        border-radius: 20px;
        text-align:center;
        color:white;
        font-weight:800;
        font-size:30px;
    }}
    .card {{
        background: linear-gradient(135deg,#7c3aed,#ec4899);
        padding:20px;
        border-radius:15px;
        text-align:center;
        color:white;
    }}
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# ---------------- HEADER ----------------
st.markdown("""
<div class="header">
📊 Data-Driven Insight Generation using Python<br>
<small>AI • Analytics • Machine Learning</small>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data(file):
    if file:
        df = pd.read_csv(file, encoding="latin1")
    else:
        df = pd.read_csv("superstore.csv", encoding="latin1")

    df.columns = df.columns.str.strip()

    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

    return df

# ---------------- CLEAN DATA ----------------
def clean_data(df):
    df = df.copy()
    df = df.drop_duplicates()
    df = df.ffill()   # ✅ fixed
    return df

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ Controls")

    file = st.file_uploader("Upload CSV")

    col1, col2 = st.columns(2)
    if col1.button("☀️"):
        st.session_state.theme = "light"
        st.rerun()
    if col2.button("🌙"):
        st.session_state.theme = "dark"
        st.rerun()

    view = st.radio("Navigation", [
        "Raw Data",
        "Cleaned Data",
        "Dashboard",
        "AI Insights",
        "ML Forecast"
    ])

# ---------------- LOAD ----------------
df_raw = load_data(file)

if df_raw.empty:
    st.warning("Upload dataset")
    st.stop()

df = clean_data(df_raw)

# ---------------- RAW DATA ----------------
if view == "Raw Data":
    st.subheader("📄 Raw Dataset")
    st.dataframe(df_raw)
    st.info(f"Rows: {df_raw.shape[0]}, Columns: {df_raw.shape[1]}")

# ---------------- CLEANED DATA ----------------
elif view == "Cleaned Data":
    st.subheader("🧹 Cleaned Dataset")
    st.success("✔ Removed duplicates & handled missing values")
    st.dataframe(df)
    st.write("Missing Values:")
    st.dataframe(df.isnull().sum())

# ---------------- DASHBOARD ----------------
elif view == "Dashboard":

    st.subheader("📊 Business Dashboard")

    sales = df["Sales"].sum() if "Sales" in df.columns else 0
    profit = df["Profit"].sum() if "Profit" in df.columns else 0
    orders = len(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sales", f"${sales:,.0f}")
    c2.metric("Total Profit", f"${profit:,.0f}")
    c3.metric("Orders", orders)

    st.divider()

    col1, col2 = st.columns(2)

    # BAR
    with col1:
        if "Category" in df.columns:
            fig = px.bar(df, x="Category", y="Sales", color="Category")
            st.plotly_chart(fig, width="stretch")  # ✅ fixed

    # PIE SAFE
    with col2:
        col_choice = None
        for col in ["Segment", "Region", "Category"]:
            if col in df.columns:
                col_choice = col
                break

        if col_choice:
            fig = px.pie(df, names=col_choice, values="Sales", hole=0.4)
            st.plotly_chart(fig, width="stretch")  # ✅ fixed

    # LINE
    if "Order Date" in df.columns:
        trend = df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"].sum().reset_index()  # ✅ fixed
        fig = px.line(trend, x="Order Date", y="Sales", title="Sales Trend")
        st.plotly_chart(fig, width="stretch")

    # HEATMAP
    if "Region" in df.columns and "Category" in df.columns:
        pivot = df.pivot_table(values="Sales", index="Region", columns="Category", aggfunc="sum")
        fig = px.imshow(pivot, text_auto=True)
        st.plotly_chart(fig, width="stretch")

# ---------------- AI INSIGHTS ----------------
elif view == "AI Insights":

    st.subheader("🤖 AI Generated Insights")

    if "Sales" in df.columns:
        st.success(f"💰 Total Sales: ${df['Sales'].sum():,.0f}")

    if "Profit" in df.columns:
        if df["Profit"].sum() > 0:
            st.success("📈 Business is profitable")
        else:
            st.warning("⚠️ Loss detected")

    if "Category" in df.columns:
        best = df.groupby("Category")["Sales"].sum().idxmax()
        st.success(f"🏆 Best Category: {best}")

    if "Region" in df.columns:
        best = df.groupby("Region")["Sales"].sum().idxmax()
        st.success(f"🌍 Top Region: {best}")

# ---------------- ML FORECAST ----------------
elif view == "ML Forecast":

    st.subheader("🤖 Sales Forecast (Random Forest)")

    if "Order Date" in df.columns:

        trend = df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"].sum().reset_index()  # ✅ fixed
        trend = trend.sort_values("Order Date")

        if len(trend) < 6:
            st.warning("Not enough data")
            st.stop()

        trend["Month"] = trend["Order Date"].dt.month
        trend["Year"] = trend["Order Date"].dt.year
        trend["Index"] = np.arange(len(trend))

        X = trend[["Month", "Year", "Index"]]
        y = trend["Sales"]

        model = RandomForestRegressor(n_estimators=200)
        model.fit(X, y)

        future_dates = pd.date_range(trend["Order Date"].max(), periods=7, freq="ME")[1:]  # ✅ fixed

        future_df = pd.DataFrame({
            "Order Date": future_dates
        })

        future_df["Month"] = future_df["Order Date"].dt.month
        future_df["Year"] = future_df["Order Date"].dt.year
        future_df["Index"] = np.arange(len(trend), len(trend)+6)

        preds = model.predict(future_df[["Month", "Year", "Index"]])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend["Order Date"], y=y, name="Actual"))
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=preds,
            name="Forecast",
            line=dict(dash="dash")
        ))

        st.plotly_chart(fig, width="stretch")  
