# =============================================================================
# Project: Smart Glucose Forecasting (MSc Thesis)
# Author: Asha Deepthi Yarabati
# Description: This script orchestrates the entire data processing, modeling, 
#              and evaluation pipeline for the Smart Glucose Forecasting project. 
#              It sequentially executes all necessary steps in workflow from 
#              data cleaning to model training and results compilation.
# Version: 1.0
# Last Updated: 06 Apr 2026
# Usage:
# - Run in project root directory "pwsh ./run_all.ps1"
# =============================================================================

CD "C:\AshaYarabati\Code\Python\SmartGlucoseForecast"

# Set up virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

 
python 02_data_cleaning.py `
  --input "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\GlucoBench_benchmark_dataset.csv" `
  --output-dir "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\cleaned"

python 03_resample_align.py `
  --input "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\cleaned\cleaned_glucobench.csv" `
  --output-dir "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\resampled"

python 04_make_windows.py `
  --input "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\resampled\resampled_glucobench_5min.csv" `
  --output-dir "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\windows" `
  --modes "cgm_only,cgm_context" `
  --horizons "30,60" `
  --history-minutes 90

python 05_run_baselines_v2.py `
  --windows-dir "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\windows" `
  --output-dir "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\baselines" `
  --modes "cgm_only,cgm_context" `
  --horizons "30,60"

python 06_plot_baselines_comparison.py `
  --predictions-dir "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\baselines\predictions" `
  --output-dir "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\baselines\plots"

python 07_train_deep_models_v2.py `
  --windows-dir "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\windows" `
  --output-dir "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\deep" `
  --models "lstm,tcn" `
  --modes "cgm_only,cgm_context" `
  --horizons "30,60" `
  --epochs 20 `
  --batch-size 128 `
  --patience 5 `
  --lr 0.001 `
  --device cpu

python 08_merge_paper_tables.py `
  --baseline-table "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\baselines\baseline_results_paper_table.csv" `
  --deep-tables "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\deep\deep_results_paper_table_lstm_h30-60.csv,C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\deep\deep_results_paper_table_tcn_h30-60.csv" `
  --output-dir "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\merged"

python 09_export_table_to_word.py `
  --input "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\merged\merged_results_paper_table.csv" `
  --output-dir "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\word"

python 10_plot_all_models_comparison.py `
  --baseline-predictions-dir "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\baselines\predictions" `
  --deep-predictions-dir "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\deep\predictions" `
  --output-dir "C:\AshaYarabati\Code\Python\SmartGlucoseForecast\outputs\comparison_plots" `
  --modes "cgm_only,cgm_context" `
  --horizons "30,60" `
  --max-points 300