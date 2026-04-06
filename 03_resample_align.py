# =============================================================================
# Project: Smart Glucose Forecasting (MSc Thesis)
# Author: Asha Deepthi Yarabati
# Description: This script resamples the cleaned GlucoBench dataset to a regular 5-minute grid. It handles:
#            - Continuous variables: Interpolation for glucose, forward/backward fill for others
#            - Event variables: Summation within each interval
#            - Categorical variables: Last observation carried forward, with 'unknown' for all-NaN intervals
#            - Static variables: Forward fill the first non-null value for each user    
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

from common import ensure_dir, add_time_features

CONTINUOUS_COLS = [
    'glucose', 'insulin_basal', 'heart_rate', 'skin_temp', 'gsr', 'stress_level',
    'glucose_lag_1', 'glucose_lag_3', 'glucose_lag_6', 'glucose_roll_mean_1h'
]
EVENT_COLS = ['insulin_bolus', 'carbs', 'exercise_steps', 'alcohol']
CATEGORY_COLS = ['meal_type', 'exercise_intensity', 'sleep_stage']
STATIC_COLS = ['hbA1c', 'age', 'sex', 'weight', 'carb_ratio', 'insulin_sensitivity', 'timezone', 'region', 'device_id']


def resample_user(user_df: pd.DataFrame, freq: str = '5min') -> pd.DataFrame:
    user_df = user_df.sort_values('timestamp').copy()
    user_id = user_df['user_id'].iloc[0]
    user_df = user_df.set_index('timestamp')
    grid = pd.date_range(user_df.index.min().floor(freq), user_df.index.max().ceil(freq), freq=freq)
    out = pd.DataFrame(index=grid)
    out.index.name = 'timestamp'

    for col in CONTINUOUS_COLS:
        if col in user_df.columns:
            series = user_df[col].resample(freq).mean()
            if col == 'glucose':
                out[col] = series.reindex(grid).interpolate(method='time', limit_direction='both')
            else:
                out[col] = series.reindex(grid).ffill().bfill()

    for col in EVENT_COLS:
        if col in user_df.columns:
            out[col] = user_df[col].resample(freq).sum(min_count=1).reindex(grid).fillna(0.0)

    for col in CATEGORY_COLS:
        if col in user_df.columns:
            series = user_df[col].astype('string').resample(freq).last()
            out[col] = series.reindex(grid).ffill().bfill().fillna('unknown')

    for col in STATIC_COLS:
        if col in user_df.columns:
            non_null = user_df[col].dropna()
            out[col] = non_null.iloc[0] if len(non_null) else np.nan

    if 'cgm_quality_flag' in user_df.columns:
        out['cgm_quality_flag'] = user_df['cgm_quality_flag'].resample(freq).max().reindex(grid).ffill().bfill()

    out['user_id'] = user_id
    out = out.reset_index()
    out = add_time_features(out)
    if 'glucose' in out.columns:
        out['glucose_lag_1'] = out['glucose'].shift(1)
        out['glucose_lag_3'] = out['glucose'].shift(3)
        out['glucose_lag_6'] = out['glucose'].shift(6)
        out['glucose_roll_mean_1h'] = out['glucose'].rolling(window=12, min_periods=1).mean()
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resample cleaned data to a 5-minute grid.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--output-name', default='resampled_glucobench_5min.csv')
    parser.add_argument('--freq', default='5min')
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    df = pd.read_csv(args.input, parse_dates=['timestamp'])
    users = [resample_user(group, freq=args.freq) for _, group in df.groupby('user_id', sort=True)]
    out = pd.concat(users, ignore_index=True).sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    out_path = Path(out_dir) / args.output_name
    out.to_csv(out_path, index=False)
    print(f'Wrote resampled data: {out_path} | rows={len(out)} | users={out.user_id.nunique()}')
