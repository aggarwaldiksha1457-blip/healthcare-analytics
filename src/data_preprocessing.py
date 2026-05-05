"""
data_preprocessing.py — Load, clean, and engineer features for the healthcare dataset.

Run standalone:
    python src/data_preprocessing.py
"""

import sys
from pathlib import Path

# Allow running from project root OR from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.utils import (
    RAW_CSV, CLEANED_CSV, ensure_dirs, get_logger, print_section
)

logger = get_logger("preprocessing")


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV and print a quick profile."""
    print_section("STEP 1 — LOADING RAW DATA")
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded  : {path}")
        logger.info(f"Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
        print("\n── Column dtypes ──")
        print(df.dtypes.to_string())
        print("\n── Missing values ──")
        missing = df.isnull().sum()
        print(missing[missing > 0].to_string() if missing.any() else "  None ✅")
        print(f"\n── Duplicate rows: {df.duplicated().sum():,} ──")
        return df
    except FileNotFoundError:
        logger.error(f"CSV not found at {path}. Place it in data/raw/ first.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLEAN
# ─────────────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix missing values, duplicates, and parse dates."""
    print_section("STEP 2 — CLEANING")

    original_shape = df.shape

    # Remove duplicates
    df = df.drop_duplicates()
    logger.info(f"Removed {original_shape[0] - len(df):,} duplicate rows")

    # Fill numeric missing values with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        n_miss = df[col].isnull().sum()
        if n_miss:
            df[col].fillna(df[col].median(), inplace=True)
            logger.info(f"  Filled {n_miss} NaN in '{col}' with median")

    # Fill categorical missing values with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        n_miss = df[col].isnull().sum()
        if n_miss:
            df[col].fillna(df[col].mode()[0], inplace=True)
            logger.info(f"  Filled {n_miss} NaN in '{col}' with mode")

    # Strip whitespace from string columns
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()

    # Parse dates (must be AFTER strip)
    for col in ["Date of Admission", "Discharge Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            logger.info(f"  Parsed '{col}' as datetime")

    logger.info(f"Clean shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived columns used by EDA, statistics, and ML."""
    print_section("STEP 3 — FEATURE ENGINEERING")

    # Days in hospital
    if "Date of Admission" in df.columns and "Discharge Date" in df.columns:
        admit = pd.to_datetime(df["Date of Admission"], errors="coerce")
        discharge = pd.to_datetime(df["Discharge Date"], errors="coerce")
        df["Days_in_Hospital"] = (discharge - admit).dt.days.clip(lower=0)
        logger.info("Created: Days_in_Hospital")

    # Age group
    if "Age" in df.columns:
        bins   = [0, 17, 35, 55, 75, 200]
        labels = ["Child", "Young Adult", "Adult", "Senior", "Elderly"]
        df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)
        logger.info("Created: Age_Group")

    # Billing category
    if "Billing Amount" in df.columns:
        df["Billing_Category"] = pd.cut(
            df["Billing Amount"],
            bins=[-np.inf, 10_000, 30_000, np.inf],
            labels=["Low", "Medium", "High"]
        )
        logger.info("Created: Billing_Category")

    # Admission temporal features
    if "Date of Admission" in df.columns:
        df["Month_of_Admission"] = df["Date of Admission"].dt.month_name()
        df["Year_of_Admission"]  = df["Date of Admission"].dt.year
        logger.info("Created: Month_of_Admission, Year_of_Admission")

    # Binary flags
    if "Admission Type" in df.columns:
        df["Is_Emergency"] = (df["Admission Type"].str.strip().str.lower() == "emergency").astype(int)
        logger.info("Created: Is_Emergency")

    if "Test Results" in df.columns:
        df["Is_Abnormal"] = (df["Test Results"].str.strip().str.lower() == "abnormal").astype(int)
        logger.info("Created: Is_Abnormal")

    # Risk score (0–6)
    high_bill = (df["Billing Amount"] > 30_000).astype(int) if "Billing Amount" in df.columns else 0
    df["Risk_Score"] = df.get("Is_Emergency", 0) * 2 + df.get("Is_Abnormal", 0) * 3 + high_bill
    logger.info("Created: Risk_Score")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENCODE FOR ML
# ─────────────────────────────────────────────────────────────────────────────
def encode_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """Add label-encoded versions of key categorical columns for ML use."""
    print_section("STEP 4 — LABEL ENCODING (ML columns)")

    from sklearn.preprocessing import LabelEncoder

    encode_cols = [
        "Gender", "Blood Type", "Medical Condition",
        "Admission Type", "Medication", "Test Results",
        "Insurance Provider", "Hospital", "Doctor",
        "Age_Group", "Billing_Category", "Month_of_Admission"
    ]

    le = LabelEncoder()
    for col in encode_cols:
        if col in df.columns:
            df[f"{col}_Enc"] = le.fit_transform(df[col].astype(str))
            logger.info(f"  Encoded '{col}' → '{col}_Enc'")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────────────────────────────────────
def save_cleaned(df: pd.DataFrame, path: Path = CLEANED_CSV) -> None:
    """Persist the cleaned DataFrame to CSV."""
    print_section("STEP 5 — SAVING CLEANED DATA")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved to: {path}")
    logger.info(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("\n── Sample (5 rows) ──")
    print(df.head().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_preprocessing() -> pd.DataFrame:
    ensure_dirs()
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_for_ml(df)
    save_cleaned(df)
    print("\n✅ Preprocessing complete!\n")
    return df


if __name__ == "__main__":
    run_preprocessing()