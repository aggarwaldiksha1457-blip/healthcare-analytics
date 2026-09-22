"""
01_overview.py — Executive Overview Dashboard Page.
"""
import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import src.visualizations as vis
import src.ui as ui

st.set_page_config(page_title="Overview | HealthPro", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# Load CSS
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

ui.render_top_nav()

if "filtered_df" not in st.session_state:
    st.warning("Please navigate to the main app page first to load data.")
    st.stop()

df = st.session_state["filtered_df"]

if len(df) == 0:
    st.warning("No data available for the selected filters.")
    st.stop()

ui.render_kpis(df)

st.title("📊 Executive Overview")
st.markdown("---")

# Row 1: Admissions & Admission Type
c1, c2 = st.columns([2, 1])
with c1:
    if "Date of Admission" in df.columns:
        st.plotly_chart(vis.monthly_admissions_line(df), use_container_width=True)
    else:
        st.info("Date of Admission data missing.")
with c2:
    if "Admission Type" in df.columns:
        st.plotly_chart(vis.admission_type_donut(df), use_container_width=True)

# Row 2: Heatmap
if "Date of Admission" in df.columns:
    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(vis.admissions_heatmap(df), use_container_width=True)

# Row 3: Top Conditions & Revenue by Insurer
st.markdown("<br>", unsafe_allow_html=True)
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

