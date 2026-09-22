"""
ui.py - Shared Streamlit UI components (Navigation, Filters, KPIs).
"""
import streamlit as st
import pandas as pd
import src.visualizations as vis

def render_top_nav():
    c1, c2, c3, c4, c5, c6 = st.columns([2, 1, 1, 1, 1, 1])
    with c1:
        st.markdown("<h3 style='margin-top: 0; padding-top: 0; color: #1B4F72;'>🏥 HealthPro</h3>", unsafe_allow_html=True)
    with c2:
        st.page_link("app.py", label="Home", icon="🏠")
    with c3:
        st.page_link("pages/01_overview.py", label="Overview", icon="📊")
    with c4:
        st.page_link("pages/02_demographics.py", label="Patients", icon="👥")
    with c5:
        st.page_link("pages/03_financial.py", label="Financial", icon="💵")
    with c6:
        st.page_link("pages/05_predictions.py", label="AI Risk", icon="🔮")
    st.markdown("---")

def render_filters(df):
    with st.expander("🔍 Global Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            if "Date of Admission" in df.columns:
                min_date = df["Date of Admission"].min().date()
                max_date = df["Date of Admission"].max().date()
                date_range = st.date_input("Admission Date Range", (min_date, max_date), min_value=min_date, max_value=max_date)
            else:
                date_range = None
                
        with c2:
            hospitals = sorted(df["Hospital"].dropna().unique())
            sel_hospitals = st.multiselect("Select Hospitals", hospitals, default=[])
            
        with c3:
            conditions = sorted(df["Medical Condition"].dropna().unique())
            sel_conditions = st.multiselect("Medical Conditions", conditions, default=[])
            
        with c4:
            genders = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
            sel_gender = st.selectbox("Gender", genders)
            
        # Apply Filters
        filtered = df.copy()
        if date_range and len(date_range) == 2 and "Date of Admission" in filtered.columns:
            mask = (filtered["Date of Admission"].dt.date >= date_range[0]) & \
                   (filtered["Date of Admission"].dt.date <= date_range[1])
            filtered = filtered[mask]
            
        if sel_hospitals:
            filtered = filtered[filtered["Hospital"].isin(sel_hospitals)]
        if sel_conditions:
            filtered = filtered[filtered["Medical Condition"].isin(sel_conditions)]
        if sel_gender != "All":
            filtered = filtered[filtered["Gender"] == sel_gender]
            
        st.session_state["filtered_df"] = filtered
        st.session_state["raw_df"] = df
        
        st.caption(f"Showing **{len(filtered):,}** records ({(len(filtered)/len(df))*100:.1f}% of total).")
        return filtered

def render_kpis(df):
    if len(df) == 0:
        return
    c1, c2, c3 = st.columns(3)
    
    # 1. Total Patients
    with c1:
        st.markdown('<div data-testid="metric-container">', unsafe_allow_html=True)
        st.metric("Total Patients", f"{len(df):,}")
        st.plotly_chart(vis.sparkline(df, "Date of Admission", color="#1B4F72"), use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
        
    # 2. Avg Billing
    with c2:
        st.markdown('<div data-testid="metric-container">', unsafe_allow_html=True)
        avg_bill = df["Billing Amount"].mean()
        st.metric("Avg Billing", f"${avg_bill:,.0f}")
        st.plotly_chart(vis.sparkline(df, "Date of Admission", "Billing Amount", color="#148F77"), use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
        
    # 3. Avg Stay
    with c3:
        st.markdown('<div data-testid="metric-container">', unsafe_allow_html=True)
        if "Days_in_Hospital" in df.columns:
            avg_stay = df["Days_in_Hospital"].mean()
            st.metric("Avg Hospital Stay", f"{avg_stay:.1f} Days")
            st.plotly_chart(vis.sparkline(df, "Date of Admission", "Days_in_Hospital", color="#D35400"), use_container_width=True, config={'displayModeBar': False})
        else:
            st.metric("Avg Hospital Stay", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
