"""
02_demographics.py — Patient Demographics Dashboard Page.
"""
import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import src.visualizations as vis
import src.ui as ui

st.set_page_config(page_title="Demographics | HealthPro", page_icon="👥", layout="wide", initial_sidebar_state="collapsed")

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

st.title("👥 Patient Demographics")
st.markdown("---")

ui.render_kpis(df)

# Row 1: Metrics
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown('<div data-testid="metric-container">', unsafe_allow_html=True)
    st.metric("Average Age", f"{df['Age'].mean():.1f} years")
    st.markdown('</div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div data-testid="metric-container">', unsafe_allow_html=True)
    top_cond = df["Medical Condition"].mode()[0] if not df["Medical Condition"].empty else "N/A"
    st.metric("Most Common Condition", top_cond)
    st.markdown('</div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div data-testid="metric-container">', unsafe_allow_html=True)
    top_blood = df["Blood Type"].mode()[0] if not df["Blood Type"].empty else "N/A"
    st.metric("Most Common Blood Type", top_blood)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Row 2: Age Hist & Gender
import plotly.express as px
c1, c2 = st.columns(2)
with c1:
    fig_age = px.histogram(df, x="Age", nbins=30, color_discrete_sequence=[vis.PALETTE[0]])
    fig_age = vis._apply(fig_age, "Age Distribution")
    st.plotly_chart(fig_age, use_container_width=True)
with c2:
    if "Age_Group" in df.columns:
        fig_ga = px.histogram(df, x="Age_Group", color="Gender", barmode="group",
                              color_discrete_sequence=vis.PALETTE,
                              category_orders={"Age_Group": ["Child", "Young Adult", "Adult", "Senior", "Elderly"]})
        fig_ga = vis._apply(fig_ga, "Gender by Age Group")
        st.plotly_chart(fig_ga, use_container_width=True)
    else:
        st.plotly_chart(vis.gender_donut(df), use_container_width=True)

# Row 3: Blood Type & Heatmap
c3, c4 = st.columns([1, 2])
with c3:
    if "Blood Type" in df.columns:
        st.plotly_chart(vis.blood_type_bar(df), use_container_width=True)
with c4:
    if "Medical Condition" in df.columns and "Gender" in df.columns:
        st.plotly_chart(vis.condition_gender_heatmap(df), use_container_width=True)
