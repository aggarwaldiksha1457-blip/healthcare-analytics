"""
04_hospital_performance.py — Hospital Performance Dashboard Page.
"""
import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import src.visualizations as vis
import src.ui as ui

st.set_page_config(page_title="Hospital Performance | HealthPro", page_icon="🏥", layout="wide", initial_sidebar_state="collapsed")

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

st.title("🏥 Hospital & Provider Performance")
st.markdown("---")

ui.render_kpis(df)

c1, c2 = st.columns(2)
with c1:
    if "Hospital" in df.columns:
        import plotly.express as px
        top_h = df["Hospital"].value_counts().head(10).reset_index()
        top_h.columns = ["Hospital", "Patients"]
        fig1 = px.bar(top_h, x="Patients", y="Hospital", orientation="h",
                      color="Patients", color_continuous_scale=["#E2E8F0", vis.PALETTE[1]])
        fig1 = vis._apply(fig1, "Top 10 Hospitals by Volume")
        st.plotly_chart(fig1, use_container_width=True)
with c2:
    if "Doctor" in df.columns:
        st.plotly_chart(vis.top_doctors_bar(df), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(vis.avg_days_by_condition(df), use_container_width=True)
with c4:
    if "Hospital" in df.columns and "Medical Condition" in df.columns:
        st.plotly_chart(vis.hospital_condition_heatmap(df), use_container_width=True)
