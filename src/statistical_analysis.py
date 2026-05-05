"""
statistical_analysis.py — Descriptive stats, hypothesis tests, and correlation analysis.
Run standalone: python src/statistical_analysis.py
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
from scipy import stats
from src.utils import CLEANED_CSV, CHARTS_STAT, ensure_dirs, get_logger, save_fig, print_section, PALETTE

logger = get_logger("stats")
plt.style.use("dark_background")
C = PALETTE

def _fig(figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0D1117")
    ax.set_facecolor("#0D1117")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.tick_params(colors="#AAA"); ax.xaxis.label.set_color("#AAA"); ax.yaxis.label.set_color("#AAA"); ax.title.set_color("#FFF")
    return fig, ax

def _save(fig, name):
    p = CHARTS_STAT / f"{name}.png"
    save_fig(fig, p, dpi=150); plt.close(fig); logger.info(f"  Saved → {p.name}")

def _conclusion(p, alpha=0.05, reject_msg="", fail_msg=""):
    if p < alpha:
        return f"✅ SIGNIFICANT (p={p:.4f} < {alpha}) — {reject_msg}"
    return f"❌ NOT significant (p={p:.4f} ≥ {alpha}) — {fail_msg}"

# ─────────────────────────────────────────────────────────────────────────────
# DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def descriptive_stats(df):
    print_section("DESCRIPTIVE STATISTICS")
    print("\n── Full describe() ──")
    print(df.describe(include="all").to_string())

    print("\n── Skewness & Kurtosis ──")
    for col in ["Billing Amount", "Age", "Days_in_Hospital"]:
        if col in df.columns:
            s = df[col].skew(); k = df[col].kurtosis()
            print(f"  {col:25s}  skew={s:+.3f}  kurtosis={k:+.3f}")

    print("\n── IQR Outlier Detection ──")
    for col in ["Billing Amount", "Age", "Days_in_Hospital"]:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            n_out = ((df[col] < lo) | (df[col] > hi)).sum()
            print(f"  {col:25s}  IQR={IQR:,.1f}  [{lo:,.1f}, {hi:,.1f}]  outliers={n_out:,}")

    print("\n── Top 5 Patients by Billing ──")
    cols = ["Name","Age","Medical Condition","Billing Amount"] if "Name" in df.columns else ["Age","Medical Condition","Billing Amount"]
    print(df.nlargest(5, "Billing Amount")[cols].to_string())


# ─────────────────────────────────────────────────────────────────────────────
# HYPOTHESIS TESTS
# ─────────────────────────────────────────────────────────────────────────────
def hypothesis_tests(df):
    print_section("HYPOTHESIS TESTS")

    # T-test: Emergency vs Non-Emergency billing
    if "Is_Emergency" in df.columns:
        emerg  = df[df["Is_Emergency"]==1]["Billing Amount"].dropna()
        normal = df[df["Is_Emergency"]==0]["Billing Amount"].dropna()
        t, p = stats.ttest_ind(emerg, normal, equal_var=False)
        print(f"\n[T-TEST] Emergency vs Non-Emergency Billing")
        print(f"  t-stat={t:.4f}  p={p:.4f}")
        print(f"  {_conclusion(p, reject_msg='Emergency patients are billed significantly differently.', fail_msg='No significant billing difference.')}")

    # ANOVA: Billing across Medical Conditions
    groups = [df[df["Medical Condition"]==c]["Billing Amount"].dropna().values
              for c in df["Medical Condition"].unique()]
    f, p = stats.f_oneway(*groups)
    print(f"\n[ANOVA] Billing Amount across Medical Conditions")
    print(f"  F-stat={f:.4f}  p={p:.4f}")
    print(f"  {_conclusion(p, reject_msg='Billing differs significantly by condition.', fail_msg='No significant billing difference by condition.')}")

    # Chi-square: Gender vs Medical Condition
    ct = pd.crosstab(df["Gender"], df["Medical Condition"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    print(f"\n[CHI-SQUARE] Gender vs Medical Condition")
    print(f"  chi2={chi2:.4f}  p={p:.4f}  dof={dof}")
    print(f"  {_conclusion(p, reject_msg='Significant relationship between gender and condition.', fail_msg='No significant relationship.')}")

    # Chi-square: Blood Type vs Medical Condition
    ct2 = pd.crosstab(df["Blood Type"], df["Medical Condition"])
    chi2, p, dof, _ = stats.chi2_contingency(ct2)
    print(f"\n[CHI-SQUARE] Blood Type vs Medical Condition")
    print(f"  chi2={chi2:.4f}  p={p:.4f}  dof={dof}")
    print(f"  {_conclusion(p, reject_msg='Significant relationship between blood type and condition.', fail_msg='No significant relationship.')}")

    # Mann-Whitney U: Billing Male vs Female
    male   = df[df["Gender"].str.lower()=="male"]["Billing Amount"].dropna()
    female = df[df["Gender"].str.lower()=="female"]["Billing Amount"].dropna()
    u, p = stats.mannwhitneyu(male, female, alternative="two-sided")
    print(f"\n[MANN-WHITNEY U] Billing — Male vs Female")
    print(f"  U={u:.1f}  p={p:.4f}")
    print(f"  {_conclusion(p, reject_msg='Significant billing difference between genders.', fail_msg='No significant billing difference by gender.')}")


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def correlation_analysis(df):
    print_section("CORRELATION ANALYSIS")
    cols = [c for c in ["Age", "Billing Amount", "Days_in_Hospital"] if c in df.columns]
    subset = df[cols].dropna()

    print("\n── Pearson Correlations ──")
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            r, p = stats.pearsonr(subset[cols[i]], subset[cols[j]])
            sig = "✅ Significant" if p < 0.05 else "❌ Not significant"
            print(f"  {cols[i]:25s} vs {cols[j]:25s}  r={r:+.4f}  p={p:.4f}  {sig}")

    print("\n── Spearman Correlations ──")
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            r, p = stats.spearmanr(subset[cols[i]], subset[cols[j]])
            sig = "✅ Significant" if p < 0.05 else "❌ Not significant"
            print(f"  {cols[i]:25s} vs {cols[j]:25s}  rho={r:+.4f}  p={p:.4f}  {sig}")

    # Save correlation heatmap (numeric only, no encoded cols)
    fig, ax = _fig((10, 8))
    num_df = df[cols]
    sns.heatmap(num_df.corr(), annot=True, fmt=".3f", cmap="coolwarm", ax=ax,
        linewidths=0.5, annot_kws={"size": 12}, vmin=-1, vmax=1, center=0)
    ax.set_title("Pearson Correlation — Key Numeric Features", fontsize=15, fontweight="bold", pad=15)
    fig.tight_layout(); _save(fig, "correlation_key_features")

    # Outlier visualisation: billing boxplot
    fig2, ax2 = _fig((12, 5))
    bp = ax2.boxplot(df["Billing Amount"].dropna(), vert=False, patch_artist=True,
        boxprops=dict(facecolor=C[0]), medianprops=dict(color=C[2], lw=2),
        flierprops=dict(marker="o", markerfacecolor=C[3], ms=3),
        whiskerprops=dict(color="#AAA"), capprops=dict(color="#AAA"))
    ax2.set_title("Billing Amount — Outlier Detection (IQR)", fontsize=15, fontweight="bold", pad=15)
    ax2.set_xlabel("Billing Amount ($)")
    fig2.tight_layout(); _save(fig2, "billing_outlier_boxplot")

    # Distributions for key columns
    for col in cols:
        fig3, ax3 = _fig((10, 5))
        sns.histplot(df[col].dropna(), kde=True, color=C[cols.index(col) % len(C)],
            edgecolor="#333", ax=ax3, line_kws={"lw": 2})
        ax3.set_title(f"{col} Distribution", fontsize=15, fontweight="bold", pad=15)
        ax3.set_xlabel(col); ax3.set_ylabel("Count")
        fig3.tight_layout(); _save(fig3, f"dist_{col.lower().replace(' ','_')}")


# ─────────────────────────────────────────────────────────────────────────────
def run_statistical_analysis(df=None):
    ensure_dirs()
    if df is None:
        if not CLEANED_CSV.exists():
            logger.error("Cleaned CSV not found. Run preprocessing first."); return
        df = pd.read_csv(CLEANED_CSV, parse_dates=["Date of Admission", "Discharge Date"])
    descriptive_stats(df)
    hypothesis_tests(df)
    correlation_analysis(df)
    print("\n✅ Statistical analysis complete!\n")

if __name__ == "__main__":
    run_statistical_analysis()
