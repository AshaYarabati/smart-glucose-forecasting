# =============================================================================
# Project: Smart Glucose Forecasting (MSc Thesis)
# Author: Asha Deepthi Yarabati
# Description: Data cleaning for GlucoBench dataset. This includes:
#            - Parsing timestamps, Coercing numeric columns
#            - Removing implausible values based on domain knowledge
#            - Filtering out users with insufficient data  
# Version: 1.0
# Last Updated: 06 Apr 2026
# Usage:
# - Run in project root directory "pwsh ./run_all.ps1"
# =============================================================================

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import RAW_NUMERIC_COLUMNS, coerce_numeric, ensure_dir, parse_timestamps

IMPLAUSIBLE_RULES = {
    'glucose': (40, 400),
    'insulin_bolus': (0, 50),
    'insulin_basal': (0, 10),
    'carbs': (0, 250),
    'exercise_steps': (0, 50000),
    'heart_rate': (30, 220),
    'skin_temp': (20, 45),
    'gsr': (0, 20),
    'stress_level': (0, 10),
    'alcohol': (0, 20),
    'hbA1c': (3, 20),
    'age': (1, 110),
    'weight': (20, 300),
    'carb_ratio': (1, 100),
    'insulin_sensitivity': (1, 200),
    'glucose_lag_1': (40, 400),
    'glucose_lag_3': (40, 400),
    'glucose_lag_6': (40, 400),
    'glucose_roll_mean_1h': (40, 400),
}


def clean_dataframe(df: pd.DataFrame, filter_bad_cgm: bool = True) -> pd.DataFrame:
    out = parse_timestamps(df)
    out = coerce_numeric(out, [c for c in RAW_NUMERIC_COLUMNS if c in out.columns])
    out = out.sort_values(['user_id', 'timestamp']).drop_duplicates(subset=['user_id', 'timestamp'], keep='last')
    for col, (low, high) in IMPLAUSIBLE_RULES.items():
        if col in out.columns:
            out.loc[(out[col] < low) | (out[col] > high), col] = np.nan
    if filter_bad_cgm and 'cgm_quality_flag' in out.columns and 'glucose' in out.columns:
        out.loc[out['cgm_quality_flag'].fillna(1) <= 0, 'glucose'] = np.nan
    counts = out.groupby('user_id')['glucose'].apply(lambda s: s.notna().sum())
    keep_users = counts[counts >= 100].index
    out = out[out['user_id'].isin(keep_users)].copy()
    return out.reset_index(drop=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean raw GlucoBench CSV.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--output-name', default='cleaned_glucobench.csv')
    parser.add_argument('--keep-bad-cgm', action='store_true')
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    df = pd.read_csv(args.input)
    cleaned = clean_dataframe(df, filter_bad_cgm=not args.keep_bad_cgm)
    out_path = Path(out_dir) / args.output_name
    cleaned.to_csv(out_path, index=False)
    print(f'Wrote cleaned data: {out_path} | rows={len(cleaned)} | users={cleaned.user_id.nunique()}')
