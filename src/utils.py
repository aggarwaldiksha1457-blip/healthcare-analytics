"""
utils.py — Shared utility functions for the Healthcare Analytics Project
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# ── Project root (one level above src/) ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Standard paths ────────────────────────────────────────────────────────────
DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXPORTS   = PROJECT_ROOT / "data" / "exports"
CHARTS_EDA     = PROJECT_ROOT / "charts" / "eda"
CHARTS_STAT    = PROJECT_ROOT / "charts" / "statistical"
CHARTS_ML      = PROJECT_ROOT / "charts" / "ml"
MODELS_DIR     = PROJECT_ROOT / "models"
REPORTS_DIR    = PROJECT_ROOT / "reports"

RAW_CSV        = DATA_RAW      / "healthcare_dataset.csv"
CLEANED_CSV    = DATA_PROCESSED / "cleaned_healthcare_data.csv"
SUMMARY_XLSX   = DATA_EXPORTS  / "summary_report.xlsx"

PALETTE = ['#00D4FF', '#7B2FBE', '#00C896', '#FF4757', '#FFB347',
           '#FF6B9D', '#C77DFF', '#4CC9F0', '#F72585', '#480CA8']


def ensure_dirs() -> None:
    """Create all required output directories if they don't exist."""
    for d in [DATA_RAW, DATA_PROCESSED, DATA_EXPORTS,
              CHARTS_EDA, CHARTS_STAT, CHARTS_ML,
              MODELS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s — %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def save_fig(fig, path: Path, dpi: int = 150) -> None:
    """Save a Matplotlib figure to *path* with the given DPI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())


def print_section(title: str, width: int = 70) -> None:
    """Pretty-print a section header."""
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def progress_bar(current: int, total: int, prefix: str = "", width: int = 40) -> None:
    """Print an inline ASCII progress bar."""
    pct   = current / total
    filled = int(width * pct)
    bar   = "█" * filled + "░" * (width - filled)
    print(f"\r{prefix} [{bar}] {pct*100:.1f}%", end="", flush=True)
    if current == total:
        print()


def step_banner(step: int, total: int, description: str) -> None:
    """Print a numbered step banner for run_project.py."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] ─── STEP {step}/{total}: {description} ───")
