"""
ml_models.py — Three ML models: Billing Predictor, Risk Classifier, Patient Clustering.
Run standalone: python src/ml_models.py
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
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix, roc_curve)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from src.utils import CLEANED_CSV, CHARTS_ML, MODELS_DIR, ensure_dirs, get_logger, save_fig, print_section, PALETTE

logger = get_logger("ml")
plt.style.use("dark_background")
C = PALETTE

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not installed — using GradientBoosting as substitute")

def _fig(figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0D1117")
    ax.set_facecolor("#0D1117")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.tick_params(colors="#AAA"); ax.xaxis.label.set_color("#AAA"); ax.yaxis.label.set_color("#AAA"); ax.title.set_color("#FFF")
    return fig, ax

def _save(fig, name):
    p = CHARTS_ML / f"{name}.png"
    save_fig(fig, p, dpi=150); plt.close(fig); logger.info(f"  Saved → {p.name}")

def _encode_features(df, feature_cols):
    """Label-encode any string columns in feature_cols, return array."""
    X = df[feature_cols].copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    return X.fillna(0).values

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — BILLING PREDICTOR (Regression)
# ─────────────────────────────────────────────────────────────────────────────
def model1_billing_predictor(df):
    print_section("MODEL 1 — BILLING AMOUNT PREDICTOR (Regression)")
    FEATURE_COLS = ["Age", "Gender", "Medical Condition", "Admission Type",
                    "Insurance Provider", "Days_in_Hospital", "Blood Type"]
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = _encode_features(df, feat_cols)
    y = df["Billing Amount"].fillna(df["Billing Amount"].median()).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    results = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "model": mdl, "preds": preds}
        logger.info(f"  {name:25s}  MAE=${mae:,.0f}  RMSE=${rmse:,.0f}  R²={r2:.4f}")

    print("\n── Model Comparison Table ──")
    print(f"{'Model':25s}  {'MAE':>10}  {'RMSE':>10}  {'R²':>8}")
    print("─" * 60)
    for name, r in results.items():
        print(f"  {name:23s}  ${r['MAE']:>9,.0f}  ${r['RMSE']:>9,.0f}  {r['R2']:>8.4f}")

    best_name = max(results, key=lambda k: results[k]["R2"])
    best_mdl  = results[best_name]["model"]
    best_pred = results[best_name]["preds"]
    print(f"\n  🏆 Best model: {best_name} (R²={results[best_name]['R2']:.4f})")

    # Save best model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_mdl, MODELS_DIR / "billing_predictor.pkl")
    logger.info(f"  Saved → models/billing_predictor.pkl")

    # Feature importance
    if hasattr(best_mdl, "feature_importances_"):
        fig, ax = _fig((12, 6))
        fi = pd.Series(best_mdl.feature_importances_, index=feat_cols).sort_values(ascending=True)
        fi.plot(kind="barh", ax=ax, color=C[0], edgecolor="#333")
        ax.set_title(f"Feature Importance — {best_name}", fontsize=15, fontweight="bold", pad=15)
        ax.set_xlabel("Importance")
        fig.tight_layout(); _save(fig, "m1_feature_importance")

    # Actual vs Predicted
    fig2, ax2 = _fig((10, 7))
    ax2.scatter(y_test, best_pred, alpha=0.4, s=15, color=C[0], edgecolors="none")
    lim = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
    ax2.plot(lim, lim, color=C[2], lw=2, label="Perfect prediction")
    ax2.set_title("Actual vs Predicted — Billing Amount", fontsize=15, fontweight="bold", pad=15)
    ax2.set_xlabel("Actual ($)"); ax2.set_ylabel("Predicted ($)")
    ax2.legend(facecolor="#1A1A2E", labelcolor="white")
    fig2.tight_layout(); _save(fig2, "m1_actual_vs_predicted")
    return best_mdl, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — READMISSION RISK CLASSIFIER (Classification)
# ─────────────────────────────────────────────────────────────────────────────
def model2_risk_classifier(df):
    print_section("MODEL 2 — READMISSION RISK CLASSIFIER (Classification)")
    FEATURE_COLS = ["Age", "Gender", "Medical Condition", "Billing Amount",
                    "Days_in_Hospital", "Test Results", "Medication"]
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = _encode_features(df, feat_cols)
    y = df["Is_Emergency"].fillna(0).astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    if XGB_AVAILABLE:
        clf_models["XGBoost"] = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0)
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        clf_models["GradientBoosting"] = GradientBoostingClassifier(n_estimators=100, random_state=42)

    results = {}
    for name, mdl in clf_models.items():
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        proba = mdl.predict_proba(X_test)[:, 1] if hasattr(mdl, "predict_proba") else preds
        acc   = accuracy_score(y_test, preds)
        prec  = precision_score(y_test, preds, zero_division=0)
        rec   = recall_score(y_test, preds, zero_division=0)
        f1    = f1_score(y_test, preds, zero_division=0)
        auc   = roc_auc_score(y_test, proba)
        results[name] = {"Acc": acc, "Prec": prec, "Rec": rec, "F1": f1, "AUC": auc,
                         "model": mdl, "preds": preds, "proba": proba}
        logger.info(f"  {name:22s}  Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

    print("\n── Classifier Comparison ──")
    print(f"{'Model':22s}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'AUC':>6}")
    print("─" * 62)
    for name, r in results.items():
        print(f"  {name:20s}  {r['Acc']:.3f}  {r['Prec']:.3f}  {r['Rec']:.3f}  {r['F1']:.3f}  {r['AUC']:.3f}")

    best_name = max(results, key=lambda k: results[k]["F1"])
    best_clf  = results[best_name]["model"]
    best_pred = results[best_name]["preds"]
    print(f"\n  🏆 Best model: {best_name} (F1={results[best_name]['F1']:.4f})")
    print("\n── Classification Report (best model) ──")
    print(classification_report(y_test, best_pred, target_names=["Non-Emergency","Emergency"]))

    joblib.dump(best_clf, MODELS_DIR / "risk_classifier.pkl")
    logger.info("  Saved → models/risk_classifier.pkl")

    # Confusion Matrix
    cm = confusion_matrix(y_test, best_pred)
    fig, ax = _fig((7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Non-Emerg","Emerg"], yticklabels=["Non-Emerg","Emerg"],
        linewidths=1, annot_kws={"size": 14})
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); _save(fig, "m2_confusion_matrix")

    # ROC Curves
    fig2, ax2 = _fig((10, 7))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        ax2.plot(fpr, tpr, lw=2, label=f"{name} (AUC={r['AUC']:.3f})")
    ax2.plot([0,1],[0,1], "w--", lw=1)
    ax2.set_title("ROC Curve Comparison", fontsize=15, fontweight="bold", pad=15)
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
    ax2.legend(facecolor="#1A1A2E", labelcolor="white")
    fig2.tight_layout(); _save(fig2, "m2_roc_curves")
    return best_clf, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — PATIENT CLUSTERING (KMeans)
# ─────────────────────────────────────────────────────────────────────────────
def model3_clustering(df):
    print_section("MODEL 3 — PATIENT CLUSTERING (KMeans k=4)")
    FEATURE_COLS = ["Age", "Billing Amount", "Days_in_Hospital", "Risk_Score"]
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X_raw = df[feat_cols].fillna(0).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Elbow Method
    inertias = []
    K_range  = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    fig, ax = _fig((10, 5))
    ax.plot(K_range, inertias, marker="o", color=C[0], lw=2.5)
    ax.fill_between(K_range, inertias, alpha=0.2, color=C[0])
    ax.axvline(4, color=C[2], ls="--", lw=2, label="k=4 chosen")
    ax.set_title("Elbow Method — Optimal k for KMeans", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Clusters (k)"); ax.set_ylabel("Inertia")
    ax.legend(facecolor="#1A1A2E", labelcolor="white")
    fig.tight_layout(); _save(fig, "m3_elbow_method")

    # Final KMeans
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    df = df.copy()
    df["Cluster"] = labels

    # Label clusters by avg risk score
    cluster_risk = df.groupby("Cluster")["Risk_Score"].mean().sort_values()
    label_map = {
        cluster_risk.index[0]: "Low Risk",
        cluster_risk.index[1]: "Medium Risk",
        cluster_risk.index[2]: "High Risk",
        cluster_risk.index[3]: "Critical",
    }
    df["Cluster_Label"] = df["Cluster"].map(label_map)

    # Cluster summary
    print("\n── Cluster Summary ──")
    summary = df.groupby("Cluster_Label")[feat_cols].mean().round(2)
    summary["Count"] = df.groupby("Cluster_Label").size()
    print(summary.to_string())

    # PCA scatter
    pca = PCA(n_components=2, random_state=42)
    X2d = pca.fit_transform(X)
    cluster_colors = {"Low Risk": C[2], "Medium Risk": C[4], "High Risk": C[3], "Critical": C[3]}
    fig2, ax2 = _fig((12, 8))
    for lab, col in zip(["Low Risk","Medium Risk","High Risk","Critical"], C[:4]):
        mask = df["Cluster_Label"] == lab
        ax2.scatter(X2d[mask, 0], X2d[mask, 1], alpha=0.6, s=25, color=col, label=lab, edgecolors="none")
    ax2.set_title("Patient Clusters (PCA 2D Projection)", fontsize=15, fontweight="bold", pad=15)
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax2.legend(facecolor="#1A1A2E", labelcolor="white")
    fig2.tight_layout(); _save(fig2, "m3_pca_scatter")

    # Cluster profile bar charts
    avg = df.groupby("Cluster_Label")[feat_cols].mean()
    for col in feat_cols:
        fig3, ax3 = _fig((8, 5))
        bars = ax3.bar(avg.index, avg[col], color=C[:4], edgecolor="#333")
        ax3.set_title(f"Avg {col} by Cluster", fontsize=14, fontweight="bold", pad=12)
        ax3.set_xlabel("Cluster"); ax3.set_ylabel(col)
        ax3.tick_params(axis="x", rotation=15)
        for b in bars:
            ax3.text(b.get_x()+b.get_width()/2, b.get_height()*1.01, f"{b.get_height():.1f}", ha="center", color="white", fontsize=10)
        fig3.tight_layout(); _save(fig3, f"m3_cluster_{col.lower().replace(' ','_')}")

    joblib.dump({"model": km, "scaler": scaler, "label_map": label_map, "features": feat_cols},
                MODELS_DIR / "cluster_model.pkl")
    logger.info("  Saved → models/cluster_model.pkl")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def run_ml_models(df=None):
    ensure_dirs()
    if df is None:
        if not CLEANED_CSV.exists():
            logger.error("Cleaned CSV not found. Run preprocessing first."); return
        df = pd.read_csv(CLEANED_CSV, parse_dates=["Date of Admission", "Discharge Date"])
    model1_billing_predictor(df)
    model2_risk_classifier(df)
    model3_clustering(df)
    print("\n✅ All ML models complete!\n")

if __name__ == "__main__":
    run_ml_models()
