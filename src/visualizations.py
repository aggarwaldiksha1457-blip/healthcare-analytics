"""
visualizations.py — Reusable Plotly chart helpers for the Streamlit dashboard.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = ['#1B4F72', '#148F77', '#2E86C1', '#117A65', '#28B463', '#2471A3', '#D35400', '#F39C12']

LAYOUT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(family="Inter, sans-serif", color="#333333"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0"),
)

def _apply(fig, title=""):
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=18, color="#1B4F72")))
    fig.update_layout(**LAYOUT)
    fig.update_xaxes(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0")
    return fig

# ── KPIs ─────────────────────────────────────────────────────────────────────
def kpi_card_metric(label, value, delta=None):
    """Returns dict for st.metric."""
    return {"label": label, "value": value, "delta": delta}

# ── Line Charts ──────────────────────────────────────────────────────────────
def monthly_admissions_line(df):
    monthly = df.groupby(df["Date of Admission"].dt.to_period("M")).size().reset_index()
    monthly["Date of Admission"] = monthly["Date of Admission"].astype(str)
    monthly.columns = ["Month", "Admissions"]
    fig = px.line(monthly, x="Month", y="Admissions", markers=True,
                  color_discrete_sequence=[PALETTE[0]])
    return _apply(fig, "Monthly Admissions Trend")

def monthly_revenue_line(df):
    monthly = df.groupby(df["Date of Admission"].dt.to_period("M"))["Billing Amount"].sum().reset_index()
    monthly["Date of Admission"] = monthly["Date of Admission"].astype(str)
    monthly.columns = ["Month", "Revenue"]
    monthly["MoM%"] = monthly["Revenue"].pct_change() * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Revenue"], name="Revenue",
                             line=dict(color=PALETTE[0], width=3), mode="lines+markers"), secondary_y=False)
    fig.add_trace(go.Bar(x=monthly["Month"], y=monthly["MoM%"], name="MoM %",
                         marker_color=PALETTE[2], opacity=0.5), secondary_y=True)
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
    fig.update_yaxes(title_text="MoM Change (%)", secondary_y=True)
    return _apply(fig, "Monthly Revenue + MoM % Change")

# ── Bar Charts ────────────────────────────────────────────────────────────────
def top_conditions_bar(df, n=10):
    top = df["Medical Condition"].value_counts().head(n).reset_index()
    top.columns = ["Condition", "Count"]
    fig = px.bar(top, x="Count", y="Condition", orientation="h",
                 color="Count", color_continuous_scale=["#E2E8F0", PALETTE[0]])
    return _apply(fig, "Top Medical Conditions")

def revenue_by_insurer(df):
    rev = df.groupby("Insurance Provider")["Billing Amount"].sum().reset_index()
    rev.columns = ["Insurer", "Revenue"]
    rev = rev.sort_values("Revenue", ascending=True)
    fig = px.bar(rev, x="Revenue", y="Insurer", orientation="h",
                 color="Revenue", color_continuous_scale=["#E2E8F0", PALETTE[1]])
    return _apply(fig, "Revenue by Insurance Provider")

def top_hospitals_bar(df, n=10):
    top = df.groupby("Hospital")["Billing Amount"].sum().nlargest(n).reset_index()
    top.columns = ["Hospital", "Revenue"]
    fig = px.bar(top, x="Revenue", y="Hospital", orientation="h",
                 color="Revenue", color_continuous_scale=["#E2E8F0", PALETTE[2]])
    return _apply(fig, "Top 10 Hospitals by Revenue")

def top_doctors_bar(df, n=10):
    top = df["Doctor"].value_counts().head(n).reset_index()
    top.columns = ["Doctor", "Patients"]
    fig = px.bar(top, x="Patients", y="Doctor", orientation="h",
                 color="Patients", color_continuous_scale=["#E2E8F0", PALETTE[4]])
    return _apply(fig, "Top 10 Doctors by Patients Treated")

def avg_days_by_condition(df):
    if "Days_in_Hospital" not in df.columns:
        return go.Figure()
    avg = df.groupby("Medical Condition")["Days_in_Hospital"].mean().reset_index()
    avg.columns = ["Condition", "Avg Days"]
    avg = avg.sort_values("Avg Days", ascending=True)
    fig = px.bar(avg, x="Avg Days", y="Condition", orientation="h",
                 color="Avg Days", color_continuous_scale=["#E2E8F0", PALETTE[3]])
    return _apply(fig, "Avg Days in Hospital by Condition")

# ── Donut / Pie ───────────────────────────────────────────────────────────────
def admission_type_donut(df):
    counts = df["Admission Type"].value_counts().reset_index()
    counts.columns = ["Type", "Count"]
    fig = px.pie(counts, names="Type", values="Count", hole=0.55,
                 color_discrete_sequence=PALETTE)
    return _apply(fig, "Admission Type Distribution")

def gender_donut(df):
    counts = df["Gender"].value_counts().reset_index()
    counts.columns = ["Gender", "Count"]
    fig = px.pie(counts, names="Gender", values="Count", hole=0.55,
                 color_discrete_sequence=PALETTE)
    return _apply(fig, "Gender Distribution")

def blood_type_bar(df):
    bt = df["Blood Type"].value_counts().reset_index()
    bt.columns = ["Blood Type", "Count"]
    fig = px.bar(bt, x="Blood Type", y="Count",
                 color="Blood Type", color_discrete_sequence=PALETTE)
    return _apply(fig, "Blood Type Distribution")

# ── Heatmaps ─────────────────────────────────────────────────────────────────
def condition_gender_heatmap(df):
    pivot = pd.crosstab(df["Medical Condition"], df["Gender"])
    fig = px.imshow(pivot, color_continuous_scale="Teal", text_auto=True, aspect="auto")
    return _apply(fig, "Medical Condition vs Gender")

def hospital_condition_heatmap(df):
    top_h = df["Hospital"].value_counts().head(10).index
    subset = df[df["Hospital"].isin(top_h)]
    pivot = pd.crosstab(subset["Hospital"], subset["Medical Condition"])
    fig = px.imshow(pivot, color_continuous_scale="Teal", text_auto=True, aspect="auto")
    return _apply(fig, "Hospital vs Medical Condition (Top 10 Hospitals)")

def admissions_heatmap(df):
    if "Date of Admission" not in df.columns:
        return go.Figure()
    
    df_temp = df.copy()
    df_temp['Month'] = df_temp['Date of Admission'].dt.month_name()
    df_temp['DayOfWeek'] = df_temp['Date of Admission'].dt.day_name()
    
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    pivot = pd.crosstab(df_temp['Month'], df_temp['DayOfWeek'])
    # Filter only available months/days
    months_present = [m for m in months_order if m in pivot.index]
    days_present = [d for d in days_order if d in pivot.columns]
    pivot = pivot.reindex(index=months_present, columns=days_present, fill_value=0)
    
    fig = px.imshow(pivot, color_continuous_scale="Teal", text_auto=True, aspect="auto")
    return _apply(fig, "Admissions by Month and Day of Week")

# ── Scatter ───────────────────────────────────────────────────────────────────
def age_billing_scatter(df):
    fig = px.scatter(df, x="Age", y="Billing Amount", color="Medical Condition",
                     opacity=0.6, size_max=8, color_discrete_sequence=PALETTE)
    return _apply(fig, "Age vs Billing Amount")

def cluster_scatter(df):
    if "Cluster_Label" not in df.columns:
        return go.Figure()
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    feat_cols = [c for c in ["Age","Billing Amount","Days_in_Hospital","Risk_Score"] if c in df.columns]
    X = StandardScaler().fit_transform(df[feat_cols].fillna(0))
    X2d = PCA(n_components=2, random_state=42).fit_transform(X)
    plot_df = pd.DataFrame({"PC1": X2d[:,0], "PC2": X2d[:,1], "Cluster": df["Cluster_Label"].values})
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster",
                     opacity=0.7, color_discrete_sequence=PALETTE)
    return _apply(fig, "Patient Clusters (PCA)")

# ── Financial ─────────────────────────────────────────────────────────────────
def billing_histogram(df):
    fig = px.histogram(df, x="Billing Amount", nbins=50,
                       color_discrete_sequence=[PALETTE[0]])
    fig.update_traces(marker_line_color="#FFFFFF", marker_line_width=0.5)
    return _apply(fig, "Billing Amount Distribution")

def billing_by_admission_box(df):
    fig = px.box(df, x="Admission Type", y="Billing Amount",
                 color="Admission Type", color_discrete_sequence=PALETTE)
    return _apply(fig, "Billing Amount by Admission Type")

# ── Gauge (risk probability) ──────────────────────────────────────────────────
def risk_gauge(probability: float, risk_label: str = ""):
    color = "#148F77" if probability < 0.33 else "#F39C12" if probability < 0.66 else "#D35400"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 28, "color": "#1B4F72"}},
        title={"text": f"Emergency Risk: {risk_label}", "font": {"size": 16, "color": "#333333"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1"},
            "bar": {"color": color},
            "bgcolor": "#F8F9FA",
            "borderwidth": 2,
            "bordercolor": "#E2E8F0",
            "steps": [
                {"range": [0,  33], "color": "#E8F8F5"},
                {"range": [33, 66], "color": "#FEF5E7"},
                {"range": [66,100], "color": "#FDF2E9"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.75, "value": probability*100},
        }
    ))
    fig.update_layout(paper_bgcolor="#FFFFFF", font=dict(color="#333333"),
                      margin=dict(l=30, r=30, t=60, b=20))
    return fig

def sparkline(df, date_col, value_col=None, color="#148F77"):
    if date_col not in df.columns:
        return go.Figure()
    
    if value_col:
        trend = df.groupby(df[date_col].dt.to_period("M"))[value_col].sum().reset_index()
    else:
        trend = df.groupby(df[date_col].dt.to_period("M")).size().reset_index()
    
    trend[date_col] = trend[date_col].astype(str)
    y_col = value_col if value_col else 0
    
    fig = go.Figure(go.Scatter(x=trend[date_col], y=trend[y_col], mode='lines', line=dict(color=color, width=3)))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=50, showlegend=False
    )
    return fig
