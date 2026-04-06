## Overview
This repository contains the Python implementation for an MSc thesis on personalised short-term glucose forecasting using Continuous Glucose Monitoring (CGM) data and contextual variables.

The project evaluates forecasting performance at *+30 minute* and *+60 minute* prediction horizons using both *CGM-only* and *CGM + context* input configurations.

## Author and Project Context:
- Asha Deepthi Yarabati
- MSc Data Science and Artificial Intelligence
- Liverpool John Moores University
- Thesis Title: Smart Glucose Forecasting: A Machine Learning Study Integrating Lifestyle and Sensor Data

## Project Aim
To build and evaluate a leakage-safe machine learning pipeline for personalised short-term glucose forecasting.

# GlucoBench end-to-end forecasting pipeline
This bundle regenerates the full thesis pipeline for the uploaded `GlucoBench_benchmark_dataset.csv` and runs:

## What the Code Does
- data cleaning
- 5-minute resampling and alignment
- supervised window generation
- baseline models: persistence, ridge regression, SARIMAX
- deep models: LSTM, TCN
- merged paper-ready result tables
- Word export of the merged table 

## Models Included
Baseline models:
- Persistence  
- Ridge Regression  
- SARIMAX  
Deep learning models:
- LSTM  
- Temporal Convolutional Network (TCN)  

## Forecasting Setup
- Prediction horizons: +30 minutes, +60 minutes  
- Input settings: CGM-only, CGM + context  
- Split strategy: chronological train / validation / test split  
- Evaluation metrics: MAE, RMSE  

## Setup
Python version: Python 3.10 or above  
Install packages: pip install -r requirements.txt

Required packages:
- matplotlib
- numpy
- pandas
- python-docx
- scikit-learn
- statsmodels
- tensorflow
- torch

Dataset Required:
- Uses the GlucoBench benchmark dataset
- Dataset is not included in this repository
- Obtain access from the official source and place locally before running
- https://www.kaggle.com/datasets/omenkj/glucobench-glucose-monitoring-and-lifestyle-data


## How to Run: 
The entire pipeline end-to-end
pwsh ./run_all.ps1

Individual Components  (Please change input and output folders to your working directory)
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

## Output locations
- `outputs/cleaned/cleaned_glucobench.csv`
- `outputs/resampled/resampled_glucobench_5min.csv`
- `outputs/windows/...`
- `outputs/baselines/baseline_results_paper_table.csv`
- `outputs/deep/deep_results_paper_table_lstm_h30-60.csv`
- `outputs/deep/deep_results_paper_table_tcn_h30-60.csv`
- `outputs/merged/merged_results_paper_table.csv`
- `outputs/word/merged_results_table.docx`