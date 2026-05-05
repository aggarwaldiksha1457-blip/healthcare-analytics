"""
01_overview.py — Executive Overview Dashboard Page.
"""
import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import src.visualizations as vis

st.set_page_config(page_title="Overview | HealthPro", page_icon="📊", layout="wide")

if "filtered_df" not in st.session_state:
    st.warning("Please navigate to the main app page first to load data.")
    st.stop()

df = st.session_state["filtered_df"]

if len(df) == 0:
    st.warning("No data available for the selected filters.")
    st.stop()

st.title("📊 Executive Overview")
st.markdown("---")

# Row 1: Metrics
m1, m2, m3, m4, m5 = st.columns(5)
total_pts = len(df)
total_rev = df["Billing Amount"].sum()
avg_bill = df["Billing Amount"].mean()
avg_stay = df.get("Days_in_Hospital", df["Age"] * 0).mean() # Fallback if missing
em_rate = df.get("Is_Emergency", df["Age"] * 0).mean() * 100

with m1:
    st.markdown('<div data-testid="metric-container">', unsafe_allow_html=True)
    st.metric("Total Patients", f"{total_pts:,}")
    st.markdown('</div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div data-testid="metric-container">', unsafe_allow_html=True)
    st.metric("Total Revenue", f"${total_rev:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div data-testid="metric-container">', unsafe_allow_html=True)
    st.metric("Avg Billing", f"${avg_bill:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div data-testid="metric-container">', unsafe_allow_html=True)
    st.metric("Avg Stay (Days)", f"{avg_stay:.1f}")
    st.markdown('</div>', unsafe_allow_html=True)
with m5:
    st.markdown('<div data-testid="metric-container">', unsafe_allow_html=True)
    st.metric("Emergency Rate", f"{em_rate:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Row 2: Monthly Admissions & Admission Type
c1, c2 = st.columns([2, 1])
with c1:
    if "Date of Admission" in df.columns:
        st.plotly_chart(vis.monthly_admissions_line(df), use_container_width=True)
    else:
        st.info("Date of Admission data missing.")
with c2:
    if "Admission Type" in df.columns:
        st.plotly_chart(vis.admission_type_donut(df), use_container_width=True)

# Row 3: Top Conditions & Revenue by Insurer
c3, c4 = st.columns(2)
with c3:
    if "Medical Condition" in df.columns:
        st.plotly_chart(vis.top_conditions_bar(df), use_container_width=True)
with c4:
    if "Insurance Provider" in df.columns:
        st.plotly_chart(vis.revenue_by_insurer(df), use_container_width=True)

# Row 4: Monthly Revenue Trend
st.markdown("<br>", unsafe_allow_html=True)
if "Date of Admission" in df.columns and "Billing Amount" in df.columns:
    st.plotly_chart(vis.monthly_revenue_line(df), use_container_width=True)
