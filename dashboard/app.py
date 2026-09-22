"""
app.py — Main Streamlit entrypoint.
Run: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import CLEANED_CSV
import src.ui as ui

# Page configuration
st.set_page_config(
    page_title="Healthcare Analytics Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Data Loading ──
@st.cache_data
def load_data():
    if not CLEANED_CSV.exists():
        st.error(f"Cleaned data not found at {CLEANED_CSV}. Please run preprocessing first.")
        st.stop()
    df = pd.read_csv(CLEANED_CSV)
    if "Date of Admission" in df.columns:
        df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
    if "Discharge Date" in df.columns:
        df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])
    return df

try:
    raw_df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ── Global Navigation & Filters ──
ui.render_top_nav()

st.title("🏥 Healthcare Analytics Dashboard")
st.markdown("Explore comprehensive healthcare data with advanced analytics and predictive modeling.")

filtered_df = ui.render_filters(raw_df)

if len(filtered_df) == 0:
    st.warning("No data matches the current filters. Please adjust the settings.")
else:
    # ── KPIs ──
    ui.render_kpis(filtered_df)
    
    st.markdown("### Welcome to HealthPro Analytics")
    st.markdown("""
    Use the navigation menu at the top to explore different analytical views:
    - **Overview**: High-level executive KPIs and trends
    - **Patients**: Deep dive into patient demographics
    - **Financial**: Revenue, billing analysis, and insurance
    - **AI Risk**: Machine learning models for risk and billing
    """)
    
    # Professional Extras: Export Buttons
    st.markdown("---")
    st.markdown("### 📥 Data Export & Summary")
    c1, c2 = st.columns([1, 5])
    with c1:
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='healthpro_data.csv',
            mime='text/csv',
        )
    with c2:
        pass # Reserved for PDF export in future
        
    st.dataframe(filtered_df.head(100), use_container_width=True)
    st.caption("Showing preview of up to 100 rows.")

# Footer
st.markdown('<div class="custom-footer">© 2026 Healthcare Analytics Pro. Designed for excellence.</div>', unsafe_allow_html=True)
