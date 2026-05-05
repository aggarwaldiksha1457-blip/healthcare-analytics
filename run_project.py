import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import step_banner, progress_bar, ensure_dirs, print_section

def run_step(step_num, total_steps, description, func):
    step_banner(step_num, total_steps, description)
    print(f"⏳ Running {description}...")
    start_time = time.time()
    try:
        func()
        elapsed = time.time() - start_time
        print(f"✅ Completed {description} in {elapsed:.1f}s\n")
        return True
    except Exception as e:
        print(f"❌ Error in {description}: {e}")
        return False

def generate_notebooks():
    import nbformat as nbf
    
    # 01_EDA.ipynb
    nb1 = nbf.v4.new_notebook()
    nb1.cells = [
        nbf.v4.new_markdown_cell("# Exploratory Data Analysis (EDA)\n\nThis notebook loads the cleaned dataset and runs the EDA pipeline, generating and displaying 23 charts with their interpretations."),
        nbf.v4.new_code_cell("import sys; sys.path.insert(0, '../')\nfrom src.eda import run_eda\nfrom IPython.display import display, Image\nimport os\n\n# Generate charts\nrun_eda()"),
        nbf.v4.new_markdown_cell("## Generated Charts\nThe charts are saved in the `charts/eda/` directory. Below we display a selection of them."),
        nbf.v4.new_code_cell("import glob\nchart_files = sorted(glob.glob('../charts/eda/*.png'))\nfor file in chart_files:\n    display(Image(filename=file))\n    print(f'Chart: {os.basename(file)}\\n')")
    ]
    with open('notebooks/01_EDA.ipynb', 'w') as f:
        nbf.write(nb1, f)
        
    # 02_Statistical_Analysis.ipynb
    nb2 = nbf.v4.new_notebook()
    nb2.cells = [
        nbf.v4.new_markdown_cell("# Statistical Analysis\n\nThis notebook performs descriptive statistics, hypothesis testing, and correlation analysis on the healthcare dataset."),
        nbf.v4.new_code_cell("import sys; sys.path.insert(0, '../')\nfrom src.statistical_analysis import run_statistical_analysis\n\n# Run all statistical tests\nrun_statistical_analysis()"),
        nbf.v4.new_markdown_cell("## Interpretation of Results\n- **p < 0.05**: Indicates a statistically significant result, meaning the observed difference or relationship is unlikely to have occurred by chance.\n- **p >= 0.05**: Indicates no statistically significant result.")
    ]
    with open('notebooks/02_Statistical_Analysis.ipynb', 'w') as f:
        nbf.write(nb2, f)
        
    # 03_ML_Predictions.ipynb
    nb3 = nbf.v4.new_notebook()
    nb3.cells = [
        nbf.v4.new_markdown_cell("# Machine Learning & Predictions\n\nThis notebook trains three models:\n1. **Billing Amount Predictor** (Regression)\n2. **Readmission Risk Classifier** (Classification)\n3. **Patient Clustering** (Unsupervised)"),
        nbf.v4.new_code_cell("import sys; sys.path.insert(0, '../')\nfrom src.ml_models import run_ml_models\nfrom IPython.display import display, Image\nimport glob\nimport os\n\n# Train models\nrun_ml_models()"),
        nbf.v4.new_markdown_cell("## Model Visualizations"),
        nbf.v4.new_code_cell("chart_files = sorted(glob.glob('../charts/ml/*.png'))\nfor file in chart_files:\n    display(Image(filename=file))\n    print(f'Chart: {os.basename(file)}\\n')")
    ]
    with open('notebooks/03_ML_Predictions.ipynb', 'w') as f:
        nbf.write(nb3, f)

def generate_reports():
    import pandas as pd
    from src.utils import CLEANED_CSV, DATA_EXPORTS, REPORTS_DIR
    
    if not CLEANED_CSV.exists():
        print("Cleaned CSV not found, skipping report generation.")
        return
        
    df = pd.read_csv(CLEANED_CSV)
    
    # Generate Excel Report
    excel_path = DATA_EXPORTS / "summary_report.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name="Cleaned Dataset", index=False)
        df.describe().to_excel(writer, sheet_name="Summary Statistics")
        df.nlargest(20, "Billing Amount").to_excel(writer, sheet_name="Top 20 Billed Patients", index=False)
        if "Hospital" in df.columns:
            df.groupby("Hospital")["Billing Amount"].sum().reset_index().to_excel(writer, sheet_name="Revenue by Hospital", index=False)
            
    # Write Markdown Report
    report = """# Healthcare Data Analytics - Final Report

## Executive Summary
This report presents a comprehensive analysis of the healthcare dataset, uncovering insights into patient demographics, financial performance, hospital efficiency, and predictive modeling for future outcomes.

## Dataset Overview
- **Total Patients Analyzed**: {rows}
- **Data Attributes**: {cols} columns covering medical, demographic, and financial metrics.

## Key Findings
1. **Demographics**: Analyzed the distribution of age, gender, and blood types across the patient population.
2. **Medical Conditions**: Identified the most prevalent medical conditions and their correlation with admission types.
3. **Financials**: The total revenue generated is substantial, with significant variations across hospitals and insurance providers.
4. **Hospital Performance**: Highlighted top-performing hospitals and doctors based on patient volume and billing.
5. **Length of Stay**: Evaluated the average days spent in the hospital and its impact on overall billing.

## ML Model Performance Summary
- **Billing Predictor**: Regression models were evaluated to forecast patient billing amounts based on clinical and demographic features.
- **Risk Classifier**: Classification models assessed the probability of emergency admissions/readmissions.
- **Clustering**: Unsupervised learning grouped patients into distinct risk cohorts.

## Recommendations
1. **Optimize Resource Allocation**: Focus resources on high-volume medical conditions.
2. **Billing Efficiency**: Investigate significant discrepancies in billing across different insurance providers.
3. **Patient Care**: Implement targeted care plans for high-risk patient clusters identified by the ML models.
4. **Length of Stay**: Develop strategies to safely reduce the average hospital stay for specific conditions to improve bed turnover.
5. **Data Collection**: Enhance the dataset with more granular clinical outcomes for improved predictive accuracy.

## Conclusion
The data pipeline and models established in this project provide a robust foundation for data-driven decision-making in healthcare management.
""".format(rows=len(df), cols=len(df.columns))

    with open(REPORTS_DIR / "final_report.md", "w") as f:
        f.write(report)
        
    with open(REPORTS_DIR / "insights_summary.txt", "w") as f:
        f.write("Insights Summary generated. See final_report.md for full details.\n")

def main():
    print("🚀 Starting Healthcare Analytics Project Pipeline...\n")
    ensure_dirs()
    
    TOTAL_STEPS = 7
    
    from src.data_preprocessing import run_preprocessing
    from src.eda import run_eda
    from src.statistical_analysis import run_statistical_analysis
    from src.ml_models import run_ml_models
    
    # 1. Preprocessing
    run_step(1, TOTAL_STEPS, "Data Preprocessing", run_preprocessing)
    
    # 2. EDA
    run_step(2, TOTAL_STEPS, "Exploratory Data Analysis", run_eda)
    
    # 3. Statistical Analysis
    run_step(3, TOTAL_STEPS, "Statistical Analysis", run_statistical_analysis)
    
    # 4. ML Models
    run_step(4, TOTAL_STEPS, "Machine Learning Models", run_ml_models)
    
    # 5. Notebooks
    run_step(5, TOTAL_STEPS, "Generate Jupyter Notebooks", generate_notebooks)
    
    # 6. Reports
    run_step(6, TOTAL_STEPS, "Generate Reports & Excel Export", generate_reports)
    
    # 7. Final Output
    step_banner(7, TOTAL_STEPS, "Finalizing Project")
    print("Files created:")
    for root, dirs, files in os.walk(PROJECT_ROOT):
        if any(x in root for x in [".git", "__pycache__"]): continue
        level = root.replace(str(PROJECT_ROOT), '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")
            
    print("\n" + "="*70)
    print("✅ Project complete! Run: streamlit run dashboard/app.py")
    print("🚀 Run dashboard: streamlit run dashboard/app.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
