"""
eda.py — Exploratory Data Analysis: generates and saves 23 charts.
Run standalone: python src/eda.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import CLEANED_CSV, CHARTS_EDA, ensure_dirs, get_logger, save_fig, print_section, PALETTE

logger = get_logger("eda")
plt.style.use("dark_background")
C = PALETTE

def _fig(figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0D1117")
    ax.set_facecolor("#0D1117")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.tick_params(colors="#AAA"); ax.xaxis.label.set_color("#AAA"); ax.yaxis.label.set_color("#AAA"); ax.title.set_color("#FFF")
    return fig, ax

def _save(fig, name):
    p = CHARTS_EDA / f"{name}.png"
    save_fig(fig, p, dpi=150); plt.close(fig); logger.info(f"  Saved → {p.name}")

# ── 1 Age distribution ──────────────────────────────────────────────────────
def chart01(df):
    fig, ax = _fig((12, 6))
    sns.histplot(df["Age"], bins=30, kde=True, color=C[0], edgecolor="#333", ax=ax, line_kws={"lw": 2})
    ax.set_title("Patient Age Distribution", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Age (years)"); ax.set_ylabel("Count")
    ax.axvline(df["Age"].mean(), color=C[2], ls="--", lw=2, label=f"Mean: {df['Age'].mean():.1f}")
    ax.legend(facecolor="#1A1A2E", labelcolor="white")
    fig.tight_layout(); _save(fig, "01_age_distribution")

# ── 2 Gender pie ─────────────────────────────────────────────────────────────
def chart02(df):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0D1117")
    counts = df["Gender"].value_counts()
    wedges, texts, autotexts = ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
        colors=C[:len(counts)], startangle=90,
        wedgeprops={"edgecolor": "#0D1117", "lw": 2}, textprops={"color": "white", "fontsize": 13})
    for at in autotexts: at.set_color("white"); at.set_fontsize(13)
    ax.set_title("Gender Distribution", fontsize=16, fontweight="bold", color="white", pad=20)
    fig.tight_layout(); _save(fig, "02_gender_pie")

# ── 3 Blood type ─────────────────────────────────────────────────────────────
def chart03(df):
    fig, ax = _fig((10, 6))
    bt = df["Blood Type"].value_counts()
    bars = ax.bar(bt.index, bt.values, color=C[:len(bt)], edgecolor="#333")
    ax.set_title("Blood Type Distribution", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Blood Type"); ax.set_ylabel("Count")
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+20, f"{int(b.get_height()):,}", ha="center", color="white", fontsize=10)
    fig.tight_layout(); _save(fig, "03_blood_type")

# ── 4 Age group bar ──────────────────────────────────────────────────────────
def chart04(df):
    if "Age_Group" not in df.columns: return
    fig, ax = _fig((10, 6))
    order = ["Child", "Young Adult", "Adult", "Senior", "Elderly"]
    ag = df["Age_Group"].astype(str).value_counts().reindex(order, fill_value=0)
    bars = ax.bar(ag.index, ag.values, color=C[:5], edgecolor="#333")
    ax.set_title("Patient Age Group Distribution", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Age Group"); ax.set_ylabel("Count")
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+20, f"{int(b.get_height()):,}", ha="center", color="white", fontsize=10)
    fig.tight_layout(); _save(fig, "04_age_group")

# ── 5 Gender vs Age group ────────────────────────────────────────────────────
def chart05(df):
    if "Age_Group" not in df.columns: return
    fig, ax = _fig((12, 6))
    pivot = df.groupby(["Age_Group", "Gender"]).size().unstack(fill_value=0)
    order = ["Child", "Young Adult", "Adult", "Senior", "Elderly"]
    pivot = pivot.reindex([o for o in order if o in pivot.index])
    x = np.arange(len(pivot)); w = 0.35
    for i, (g, col) in enumerate(zip(pivot.columns, C)): ax.bar(x+i*w, pivot[g], w, label=g, color=col, edgecolor="#333")
    ax.set_xticks(x+w/2); ax.set_xticklabels(pivot.index, rotation=15)
    ax.set_title("Gender vs Age Group", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Age Group"); ax.set_ylabel("Count")
    ax.legend(facecolor="#1A1A2E", labelcolor="white")
    fig.tight_layout(); _save(fig, "05_gender_age_group")

# ── 6 Top conditions ─────────────────────────────────────────────────────────
def chart06(df):
    fig, ax = _fig((12, 7))
    top = df["Medical Condition"].value_counts().head(10)
    bars = ax.barh(top.index[::-1], top.values[::-1], color=C[:10], edgecolor="#333")
    ax.set_title("Top 10 Medical Conditions", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Patient Count")
    for b in bars: ax.text(b.get_width()+5, b.get_y()+b.get_height()/2, f"{int(b.get_width()):,}", va="center", color="white", fontsize=9)
    fig.tight_layout(); _save(fig, "06_top_conditions")

# ── 7 Condition vs admission stacked ─────────────────────────────────────────
def chart07(df):
    fig, ax = _fig((14, 7))
    pivot = df.groupby(["Medical Condition", "Admission Type"]).size().unstack(fill_value=0)
    pivot.plot(kind="bar", stacked=True, ax=ax, color=C[:len(pivot.columns)], edgecolor="#333", width=0.7)
    ax.set_title("Medical Condition vs Admission Type", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Medical Condition"); ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(facecolor="#1A1A2E", labelcolor="white")
    fig.tight_layout(); _save(fig, "07_condition_admission_stacked")

# ── 8 Test results donut ─────────────────────────────────────────────────────
def chart08(df):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0D1117")
    counts = df["Test Results"].value_counts()
    wedges, texts, autotexts = ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
        colors=C[:len(counts)], startangle=90,
        wedgeprops={"edgecolor": "#0D1117", "lw": 3, "width": 0.6},
        textprops={"color": "white", "fontsize": 13})
    for at in autotexts: at.set_color("white")
    ax.set_title("Test Results Distribution", fontsize=16, fontweight="bold", color="white", pad=20)
    fig.tight_layout(); _save(fig, "08_test_results_donut")

# ── 9 Medication ─────────────────────────────────────────────────────────────
def chart09(df):
    fig, ax = _fig((12, 6))
    med = df["Medication"].value_counts()
    bars = ax.bar(med.index, med.values, color=C[:len(med)], edgecolor="#333")
    ax.set_title("Medication Usage", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Medication"); ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    for b in bars: ax.text(b.get_x()+b.get_width()/2, b.get_height()+10, f"{int(b.get_height()):,}", ha="center", color="white", fontsize=9)
    fig.tight_layout(); _save(fig, "09_medication")

# ── 10 Condition vs test results heatmap ─────────────────────────────────────
def chart10(df):
    fig, ax = _fig((13, 7))
    pivot = df.groupby(["Medical Condition", "Test Results"]).size().unstack(fill_value=0)
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.5, linecolor="#333", annot_kws={"size": 10, "color": "black"})
    ax.set_title("Medical Condition vs Test Results", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Test Result"); ax.set_ylabel("Medical Condition")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); _save(fig, "10_condition_test_heatmap")

# ── 11 Top hospitals ─────────────────────────────────────────────────────────
def chart11(df):
    fig, ax = _fig((13, 7))
    top = df["Hospital"].value_counts().head(10)
    bars = ax.barh(top.index[::-1], top.values[::-1], color=C[1], edgecolor="#333")
    ax.set_title("Top 10 Hospitals by Patient Count", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Patient Count")
    for b in bars: ax.text(b.get_width()+1, b.get_y()+b.get_height()/2, f"{int(b.get_width()):,}", va="center", color="white", fontsize=9)
    fig.tight_layout(); _save(fig, "11_top_hospitals")

# ── 12 Top doctors ───────────────────────────────────────────────────────────
def chart12(df):
    fig, ax = _fig((13, 7))
    top = df["Doctor"].value_counts().head(10)
    bars = ax.barh(top.index[::-1], top.values[::-1], color=C[2], edgecolor="#333")
    ax.set_title("Top 10 Doctors by Patient Count", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Patient Count")
    for b in bars: ax.text(b.get_width()+0.5, b.get_y()+b.get_height()/2, f"{int(b.get_width()):,}", va="center", color="white", fontsize=9)
    fig.tight_layout(); _save(fig, "12_top_doctors")

# ── 13 Avg billing by hospital ───────────────────────────────────────────────
def chart13(df):
    fig, ax = _fig((14, 7))
    top_h = df["Hospital"].value_counts().head(10).index
    avg = df[df["Hospital"].isin(top_h)].groupby("Hospital")["Billing Amount"].mean().sort_values()
    bars = ax.barh(avg.index, avg.values, color=C[3], edgecolor="#333")
    ax.set_title("Avg Billing Amount — Top 10 Hospitals", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Average Billing ($)")
    for b in bars: ax.text(b.get_width()+100, b.get_y()+b.get_height()/2, f"${b.get_width():,.0f}", va="center", color="white", fontsize=9)
    fig.tight_layout(); _save(fig, "13_avg_billing_hospital")

# ── 14 Days in hospital boxplot ──────────────────────────────────────────────
def chart14(df):
    if "Days_in_Hospital" not in df.columns: return
    fig, ax = _fig((10, 6))
    bp = ax.boxplot(df["Days_in_Hospital"].dropna(), patch_artist=True,
        boxprops=dict(facecolor=C[0], color=C[0]), whiskerprops=dict(color=C[1]),
        capprops=dict(color=C[2]), flierprops=dict(markerfacecolor=C[3], marker="o", markersize=4),
        medianprops=dict(color=C[4], linewidth=2))
    ax.set_title("Days in Hospital Distribution", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Days"); ax.set_xticks([1]); ax.set_xticklabels(["Days"])
    fig.tight_layout(); _save(fig, "14_days_hospital_boxplot")

# ── 15 Billing histogram ─────────────────────────────────────────────────────
def chart15(df):
    fig, ax = _fig((12, 6))
    sns.histplot(df["Billing Amount"], bins=40, kde=True, color=C[4], edgecolor="#333", ax=ax, line_kws={"lw": 2})
    ax.set_title("Billing Amount Distribution", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Billing Amount ($)"); ax.set_ylabel("Count")
    ax.axvline(df["Billing Amount"].mean(), color=C[2], ls="--", lw=2, label=f"Mean: ${df['Billing Amount'].mean():,.0f}")
    ax.legend(facecolor="#1A1A2E", labelcolor="white")
    fig.tight_layout(); _save(fig, "15_billing_histogram")

# ── 16 Billing by insurance boxplot ─────────────────────────────────────────
def chart16(df):
    fig, ax = _fig((14, 7))
    ins_list = df["Insurance Provider"].unique()
    data = [df[df["Insurance Provider"]==ins]["Billing Amount"].dropna().values for ins in ins_list]
    bp = ax.boxplot(data, patch_artist=True, labels=ins_list,
        medianprops=dict(color=C[0], lw=2), flierprops=dict(markerfacecolor=C[3], marker="o", ms=3),
        whiskerprops=dict(color="#AAA"), capprops=dict(color="#AAA"))
    for i, patch in enumerate(bp["boxes"]): patch.set_facecolor(C[i % len(C)])
    ax.set_title("Billing Amount by Insurance Provider", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Insurance Provider"); ax.set_ylabel("Billing Amount ($)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); _save(fig, "16_billing_insurance_boxplot")

# ── 17 Billing by admission violin ───────────────────────────────────────────
def chart17(df):
    fig, ax = _fig((10, 7))
    adm_types = df["Admission Type"].unique()
    data = [df[df["Admission Type"]==a]["Billing Amount"].dropna().values for a in adm_types]
    parts = ax.violinplot(data, positions=range(len(adm_types)), showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]): pc.set_facecolor(C[i % len(C)]); pc.set_alpha(0.8)
    ax.set_xticks(range(len(adm_types))); ax.set_xticklabels(adm_types, rotation=10)
    ax.set_title("Billing Amount by Admission Type", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Admission Type"); ax.set_ylabel("Billing Amount ($)")
    fig.tight_layout(); _save(fig, "17_billing_admission_violin")

# ── 18 Monthly revenue line ──────────────────────────────────────────────────
def chart18(df):
    if "Date of Admission" not in df.columns: return
    fig, ax = _fig((14, 6))
    monthly = df.groupby(df["Date of Admission"].dt.to_period("M"))["Billing Amount"].sum().reset_index()
    monthly["Date of Admission"] = monthly["Date of Admission"].astype(str)
    ax.plot(monthly["Date of Admission"], monthly["Billing Amount"], color=C[0], lw=2.5, marker="o", ms=4)
    ax.fill_between(monthly["Date of Admission"], monthly["Billing Amount"], alpha=0.2, color=C[0])
    ax.set_title("Monthly Revenue Trend", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Month"); ax.set_ylabel("Total Revenue ($)")
    step = max(1, len(monthly)//12)
    ax.set_xticks(range(0, len(monthly), step))
    ax.set_xticklabels(monthly["Date of Admission"].iloc[::step], rotation=45, ha="right")
    fig.tight_layout(); _save(fig, "18_monthly_revenue")

# ── 19 Revenue by condition ──────────────────────────────────────────────────
def chart19(df):
    fig, ax = _fig((13, 7))
    rev = df.groupby("Medical Condition")["Billing Amount"].sum().sort_values()
    bars = ax.barh(rev.index, rev.values, color=C[:len(rev)], edgecolor="#333")
    ax.set_title("Total Revenue by Medical Condition", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Total Revenue ($)")
    for b in bars: ax.text(b.get_width()+1000, b.get_y()+b.get_height()/2, f"${b.get_width():,.0f}", va="center", color="white", fontsize=9)
    fig.tight_layout(); _save(fig, "19_revenue_by_condition")

# ── 20 Billing category pie ──────────────────────────────────────────────────
def chart20(df):
    if "Billing_Category" not in df.columns: return
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0D1117")
    counts = df["Billing_Category"].astype(str).value_counts()
    wedges, texts, autotexts = ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
        colors=[C[2], C[4], C[3]][:len(counts)], startangle=90,
        wedgeprops={"edgecolor": "#0D1117", "lw": 2}, textprops={"color": "white", "fontsize": 13})
    for at in autotexts: at.set_color("white")
    ax.set_title("Billing Category Distribution", fontsize=16, fontweight="bold", color="white", pad=20)
    fig.tight_layout(); _save(fig, "20_billing_category_pie")

# ── 21 Correlation heatmap ───────────────────────────────────────────────────
def chart21(df):
    fig, ax = _fig((13, 10))
    num_df = df.select_dtypes(include=[np.number]).drop(columns=[c for c in df.columns if "_Enc" in c], errors="ignore")
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
        linewidths=0.5, linecolor="#333", annot_kws={"size": 9}, vmin=-1, vmax=1, center=0)
    ax.set_title("Correlation Heatmap — Numeric Features", fontsize=16, fontweight="bold", pad=15)
    fig.tight_layout(); _save(fig, "21_correlation_heatmap")

# ── 22 Age vs billing scatter ────────────────────────────────────────────────
def chart22(df):
    fig, ax = _fig((13, 7))
    for i, cond in enumerate(df["Medical Condition"].unique()):
        sub = df[df["Medical Condition"]==cond]
        ax.scatter(sub["Age"], sub["Billing Amount"], alpha=0.5, s=15, color=C[i % len(C)], label=cond)
    ax.set_title("Age vs Billing Amount by Medical Condition", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Age (years)"); ax.set_ylabel("Billing Amount ($)")
    ax.legend(facecolor="#1A1A2E", labelcolor="white", markerscale=2, fontsize=8)
    fig.tight_layout(); _save(fig, "22_age_vs_billing_scatter")

# ── 23 Days vs billing scatter ───────────────────────────────────────────────
def chart23(df):
    if "Days_in_Hospital" not in df.columns: return
    fig, ax = _fig((12, 6))
    ax.scatter(df["Days_in_Hospital"], df["Billing Amount"], alpha=0.4, s=15, color=C[0], edgecolors="none")
    z = np.polyfit(df["Days_in_Hospital"].fillna(0), df["Billing Amount"], 1)
    xline = np.linspace(df["Days_in_Hospital"].min(), df["Days_in_Hospital"].max(), 200)
    ax.plot(xline, np.poly1d(z)(xline), color=C[2], lw=2, label="Trend")
    ax.set_title("Days in Hospital vs Billing Amount", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Days in Hospital"); ax.set_ylabel("Billing Amount ($)")
    ax.legend(facecolor="#1A1A2E", labelcolor="white")
    fig.tight_layout(); _save(fig, "23_days_vs_billing_scatter")

# ─────────────────────────────────────────────────────────────────────────────
def run_eda(df=None):
    ensure_dirs()
    print_section("EDA — GENERATING ALL 23 CHARTS")
    if df is None:
        if not CLEANED_CSV.exists():
            logger.error(f"Cleaned CSV not found. Run preprocessing first."); return
        df = pd.read_csv(CLEANED_CSV, parse_dates=["Date of Admission", "Discharge Date"])
    fns = [chart01,chart02,chart03,chart04,chart05,chart06,chart07,chart08,chart09,chart10,
           chart11,chart12,chart13,chart14,chart15,chart16,chart17,chart18,chart19,chart20,
           chart21,chart22,chart23]
    for i, fn in enumerate(fns, 1):
        try:
            logger.info(f"Chart {i:02d}/{len(fns)} — {fn.__name__}")
            fn(df)
        except Exception as e:
            logger.error(f"  Chart {i} failed: {e}")
    print(f"\n✅ EDA complete! {len(fns)} charts saved to {CHARTS_EDA}\n")

if __name__ == "__main__":
    run_eda()
