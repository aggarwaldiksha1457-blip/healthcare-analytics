"""
03_financial.py — Financial Analysis Dashboard Page.
"""
import streamlit as st
import sys
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import src.visualizations as vis

st.set_page_config(page_title="Financial | HealthPro", page_icon="💰", layout="wide")

if "filtered_df" not in st.session_state:
    st.warning("Please navigate to the main app page first to load data.")
    st.stop()

df = st.session_state["filtered_df"]

if len(df) == 0:
    st.warning("No data available for the selected filters.")
    st.stop()

st.title("💰 Financial Analysis")
st.markdown("---")

# Row 1: Metrics
m1, m2, m3, m4 = st.columns(4)
total_rev = df["Billing Amount"].sum()
max_bill = df["Billing Amount"].max()
min_bill = df["Billing Amount"].min()

# Calc "This Month" revenue (last month in data)
if "Date of Admission" in df.columns:
    last_month = df["Date of Admission"].max().replace(day=1)
    month_rev = df[df["Date of Admission"] >= last_month]["Billing Amount"].sum()
else:
    month_rev = 0

with m1:
    st.markdown('<div data-testid="metric-container">', unsafe_allow_html=True)
    st.metric("Total Revenue", f"${total_rev:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div data-testid="metric-container">', unsafe_allow_html=True)
    st.metric("Max Bill", f"${max_bill:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div data-testid="metric-container">', unsafe_allow_html=True)
    st.metric("Min Bill", f"${min_bill:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div data-testid="metric-container">', unsafe_allow_html=True)
    st.metric("Revenue This Month", f"${month_rev:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Row 2: Billing dist & admission
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(vis.billing_histogram(df), use_container_width=True)
with c2:
    if "Admission Type" in df.columns:
        st.plotly_chart(vis.billing_by_admission_box(df), use_container_width=True)

# Row 3: Top Hospitals by Rev & Monthly
c3, c4 = st.columns([1, 2])
with c3:
    if "Hospital" in df.columns:
        st.plotly_chart(vis.top_hospitals_bar(df), use_container_width=True)
with c4:
    if "Date of Admission" in df.columns:
        st.plotly_chart(vis.monthly_revenue_line(df), use_container_width=True)
