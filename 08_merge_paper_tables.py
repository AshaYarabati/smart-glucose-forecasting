# =============================================================================
# Project: Smart Glucose Forecasting (MSc Thesis)
# Author: Asha Deepthi Yarabati
# Description: Merge baseline and deep-model paper tables into a single CSV 
#              for easier comparison and presentation in the thesis.  
# Version: 1.0
# Last Updated: 06 Apr 2026
# Usage:
# - Run in project root directory "pwsh ./run_all.ps1"
# =============================================================================


from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import ensure_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge baseline and deep-model paper tables.')
    parser.add_argument('--baseline-table', required=True)
    parser.add_argument('--deep-tables', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--output-name', default='merged_results_paper_table.csv')
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    tables = [pd.read_csv(args.baseline_table)]
    for item in args.deep_tables.split(','):
        item = item.strip()
        if item:
            tables.append(pd.read_csv(item))
    merged = pd.concat(tables, ignore_index=True)
    order = {'persistence': 0, 'ridge': 1, 'sarimax': 2, 'lstm': 3, 'tcn': 4}
    merged['_order'] = merged['Model'].str.lower().map(order).fillna(999)
    merged = merged.sort_values(['Input setting', 'Horizon (min)', '_order', 'Model']).drop(columns=['_order'])
    for col in ['MAE', 'RMSE', 'TIR_true', 'TIR_pred']:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce').round(2)
    out_path = Path(out_dir) / args.output_name
    merged.to_csv(out_path, index=False)
    print(f'Wrote merged table: {out_path}')
