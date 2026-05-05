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

# Page configuration
st.set_page_config(
    page_title="Healthcare Analytics Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
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

# ── Sidebar Filters ──
st.sidebar.markdown("<h2 style='text-align: center; color: #00D4FF;'>🏥 HealthPro</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")

# Date range
if "Date of Admission" in raw_df.columns:
    min_date = raw_df["Date of Admission"].min().date()
    max_date = raw_df["Date of Admission"].max().date()
    date_range = st.sidebar.slider("Admission Date Range", 
                                   min_value=min_date, max_value=max_date,
                                   value=(min_date, max_date))
else:
    date_range = None

# Multiselects
hospitals = sorted(raw_df["Hospital"].dropna().unique())
sel_hospitals = st.sidebar.multiselect("Select Hospitals", hospitals, default=[])

conditions = sorted(raw_df["Medical Condition"].dropna().unique())
sel_conditions = st.sidebar.multiselect("Medical Conditions", conditions, default=[])

# Gender
genders = ["All"] + sorted(raw_df["Gender"].dropna().unique().tolist())
sel_gender = st.sidebar.radio("Gender", genders)

# Age
min_age = int(raw_df["Age"].min())
max_age = int(raw_df["Age"].max())
age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))

# ── Apply Filters ──
filtered_df = raw_df.copy()

if date_range and "Date of Admission" in filtered_df.columns:
    mask = (filtered_df["Date of Admission"].dt.date >= date_range[0]) & \
           (filtered_df["Date of Admission"].dt.date <= date_range[1])
    filtered_df = filtered_df[mask]

if sel_hospitals:
    filtered_df = filtered_df[filtered_df["Hospital"].isin(sel_hospitals)]

if sel_conditions:
    filtered_df = filtered_df[filtered_df["Medical Condition"].isin(sel_conditions)]

if sel_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == sel_gender]

filtered_df = filtered_df[(filtered_df["Age"] >= age_range[0]) & (filtered_df["Age"] <= age_range[1])]

# ── Save to session state ──
st.session_state["filtered_df"] = filtered_df
st.session_state["raw_df"] = raw_df

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.metric("Filtered Patients", f"{len(filtered_df):,}")
st.sidebar.markdown(f"<p style='color: #888; font-size: 12px;'>{(len(filtered_df)/len(raw_df))*100:.1f}% of total data</p>", unsafe_allow_html=True)

st.sidebar.markdown("<br>"*5, unsafe_allow_html=True)

# Main page wrapper
st.title("🏥 Healthcare Analytics Dashboard")
st.markdown("Use the navigation sidebar to explore different analytical views.")
if len(filtered_df) == 0:
    st.warning("No data matches the current filters. Please adjust the sidebar settings.")

st.markdown("---")
st.markdown("""
### Welcome to HealthPro Analytics
Select a page from the sidebar to begin exploring the data:
- **01 Overview**: High-level executive KPIs and trends
- **02 Demographics**: Deep dive into patient populations
- **03 Financial**: Revenue, billing analysis, and insurance
- **04 Hospital Performance**: Provider comparisons and efficiency
- **05 Predictions**: Machine learning models for risk and billing
""")
