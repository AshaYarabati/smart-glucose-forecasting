# =============================================================================
# Project: Smart Glucose Forecasting (MSc Thesis)
# Author: Asha Deepthi Yarabati
# Description: This script creates comparison plots for all available models 
#              (baseline and deep) against the ground truth for each user, 
#              mode, and horizon. It ensures all models are plotted on the 
#              same y-axis scale for fair visual comparison. 
#              It also generates overview sheets that tile individual 
#              user plots for each mode+horizon combination.  
# Version: 1.0
# Last Updated: 06 Apr 2026
# Usage:
# - Run in project root directory "pwsh ./run_all.ps1"
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

RAW_NUMERIC_COLUMNS = [
    'glucose', 'cgm_quality_flag', 'insulin_bolus', 'insulin_basal', 'carbs',
    'exercise_steps', 'heart_rate', 'skin_temp', 'gsr', 'stress_level', 'alcohol',
    'hbA1c', 'age', 'weight', 'carb_ratio', 'insulin_sensitivity',
    'glucose_lag_1', 'glucose_lag_3', 'glucose_lag_6', 'glucose_roll_mean_1h'
]

CGM_CONTEXT_NUMERIC = [
    'glucose', 'insulin_bolus', 'insulin_basal', 'carbs', 'exercise_steps',
    'heart_rate', 'skin_temp', 'gsr', 'stress_level', 'alcohol',
    'glucose_lag_1', 'glucose_lag_3', 'glucose_lag_6', 'glucose_roll_mean_1h'
]
CGM_CONTEXT_CATEGORICAL = ['meal_type', 'exercise_intensity', 'sleep_stage']
DEFAULT_SPLIT = (0.70, 0.15, 0.15)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_timestamps(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors='coerce')
    return out.dropna(subset=[timestamp_col])


def coerce_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')
    return out


def add_time_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col])
    minutes = ts.dt.hour * 60 + ts.dt.minute
    out['time_sin'] = np.sin(2 * np.pi * minutes / 1440.0)
    out['time_cos'] = np.cos(2 * np.pi * minutes / 1440.0)
    weekday = ts.dt.dayofweek
    out['dow_sin'] = np.sin(2 * np.pi * weekday / 7.0)
    out['dow_cos'] = np.cos(2 * np.pi * weekday / 7.0)
    return out


def chronological_split(n: int, split: Sequence[float] = DEFAULT_SPLIT) -> tuple[slice, slice, slice]:
    train_frac, val_frac, test_frac = split
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError('Split fractions must sum to 1.0')
    if n < 10:
        raise ValueError(f'Not enough samples to split chronologically: {n}')
    train_end = max(int(n * train_frac), 1)
    val_end = min(max(train_end + int(n * val_frac), train_end + 1), n - 1)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {'MAE': mae, 'RMSE': rmse}


def compute_tir(values: np.ndarray, low: float = 70.0, high: float = 180.0) -> float:
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return float('nan')
    return float(np.mean((arr >= low) & (arr <= high)) * 100.0)


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if pd.notna(v)]
    return float(np.mean(vals)) if vals else float('nan')


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
