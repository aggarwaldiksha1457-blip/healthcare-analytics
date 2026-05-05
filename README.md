# 🏥 Healthcare Data Analytics Pro

```text
  _   _            _ _   _    ____                 ____  
 | | | | ___  __ _| | |_| |__|  _ \ _ __ ___  ___ |  _ \ 
 | |_| |/ _ \/ _` | | __| '_ \ |_) | '__/ _ \/ _ \| |_) |
 |  _  |  __/ (_| | | |_| | | |  __/| | | (_) | (_) |  __/ 
 |_| |_|\___|\__,_|_|\__|_| |_|_|   |_|  \___/ \___/|_|    
```

A complete, production-ready full-stack Python data science project analyzing healthcare records, generating statistical insights, training machine learning models, and serving interactive visualizations via a Streamlit dashboard.

## 🌟 Features
- 🧹 **Automated Data Pipeline**: Cleaning, imputation, and feature engineering.
- 📊 **Comprehensive EDA**: 23 high-quality charts spanning demographics, financials, and medical insights.
- 📈 **Statistical Rigor**: Hypothesis testing (ANOVA, T-Tests, Chi-Square) and correlation analysis.
- 🤖 **Machine Learning**: 
  - Regression (Billing Amount Prediction)
  - Classification (Readmission/Emergency Risk)
  - Clustering (Patient Cohort Identification)
- 🖥️ **Interactive Dashboard**: A beautiful, multi-page Streamlit app with dynamic filtering.
- 📓 **Jupyter Integration**: Auto-generated notebooks for presentation and exploratory work.
- 📑 **Automated Reporting**: Generates markdown reports and Excel summaries.

## 📂 Folder Structure
```
HEALTHCARE_PROJECT/
├── data/
│   ├── raw/               # Original CSV data
│   ├── processed/         # Cleaned data ready for ML/EDA
│   └── exports/           # Excel summaries
├── src/                   # Core Python modules
├── notebooks/             # Auto-generated Jupyter Notebooks
├── dashboard/             # Streamlit Application
│   ├── pages/             # Multi-page routing
│   └── assets/            # CSS styling
├── charts/                # Saved PNG outputs (EDA, Stat, ML)
├── models/                # Serialized `.pkl` ML models
└── reports/               # Markdown and text reports
```

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.9+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
Execute the master script to run the entire pipeline from end-to-end (preprocessing, EDA, stats, ML, reports):
```bash
python run_project.py
```

### 3. Launch the Dashboard
Once the pipeline is complete, launch the interactive Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```

## 🛠️ Technologies Used
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Machine Learning**: `scikit-learn`, `xgboost`, `scipy`
- **Web Dashboard**: `streamlit`
- **Notebooks**: `jupyter`, `nbformat`

## 💡 Key Insights Example
- Identified significant correlation between patient age, medical condition, and hospital length of stay.
- Gradient Boosting algorithms successfully predicted billing amounts with high R² scores.
- Unsupervised clustering revealed distinct cohorts of high-risk patients needing specialized care pathways.

---
*Built with ❤️ by Antigravity*
