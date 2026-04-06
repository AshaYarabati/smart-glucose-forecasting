# =============================================================================
# Project: Smart Glucose Forecasting (MSc Thesis)
# Author: Asha Deepthi Yarabati
# Description: This script creates supervised learning windows from the resampled GlucoBench dataset. It supports two modes:
#            - 'cgm_only': Uses only glucose values as features
#            - 'cgm_context': Uses glucose plus contextual numeric and categorical features. For each user, it generates input sequences of a specified history length and targets at specified future horizons
# Version: 1.0
# Last Updated: 06 Apr 2026
# Usage:
# - Run in project root directory "pwsh ./run_all.ps1"
# =============================================================================
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import CGM_CONTEXT_CATEGORICAL, CGM_CONTEXT_NUMERIC, chronological_split, ensure_dir

MODE_TO_FEATURES = {
    'cgm_only': {'numeric': ['glucose'], 'categorical': []},
    'cgm_context': {
        'numeric': CGM_CONTEXT_NUMERIC + ['time_sin', 'time_cos', 'dow_sin', 'dow_cos'],
        'categorical': CGM_CONTEXT_CATEGORICAL,
    },
}
REQUIRED_COLUMNS = ['user_id', 'timestamp', 'glucose']


def prepare_feature_frame(user_df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, list[str]]:
    config = MODE_TO_FEATURES[mode]
    num_cols = [c for c in config['numeric'] if c in user_df.columns]
    cat_cols = [c for c in config['categorical'] if c in user_df.columns]
    selected_cols = REQUIRED_COLUMNS + [c for c in num_cols if c not in REQUIRED_COLUMNS] + cat_cols
    frame = user_df[selected_cols].copy()
    for col in num_cols:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame[num_cols] = frame[num_cols].ffill().bfill()
    if cat_cols:
        dummies = pd.get_dummies(frame[cat_cols].astype(str), prefix=cat_cols, dummy_na=False)
        feature_df = pd.concat([frame[num_cols], dummies], axis=1)
    else:
        feature_df = frame[num_cols].copy()
    feature_df = feature_df.ffill().bfill().fillna(0.0)
    return feature_df, list(feature_df.columns)


def build_windows(user_df: pd.DataFrame, mode: str, history_steps: int, horizon_minutes: int, freq_minutes: int = 5):
    horizon_steps = horizon_minutes // freq_minutes
    features, feature_names = prepare_feature_frame(user_df, mode)
    glucose = pd.to_numeric(user_df['glucose'], errors='coerce')
    timestamps = pd.to_datetime(user_df['timestamp'])
    X_list, y_list, ts_list = [], [], []
    total = len(user_df)
    start_idx = history_steps - 1
    end_idx = total - horizon_steps - 1
    for t in range(start_idx, end_idx + 1):
        window = features.iloc[t - history_steps + 1 : t + 1].to_numpy(dtype=np.float32)
        target = glucose.iloc[t + horizon_steps]
        if np.isnan(window).any() or pd.isna(target):
            continue
        X_list.append(window)
        y_list.append(np.float32(target))
        ts_list.append(str(timestamps.iloc[t]))
    X = np.stack(X_list) if X_list else np.empty((0, history_steps, len(feature_names)), dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    ts = np.asarray(ts_list)
    return X, y, ts, feature_names


def save_npz(X: np.ndarray, y: np.ndarray, ts: np.ndarray, feature_names: list[str], path: Path):
    train_sl, val_sl, test_sl = chronological_split(len(X))
    np.savez_compressed(
        path,
        X_train=X[train_sl], y_train=y[train_sl], ts_train=ts[train_sl],
        X_val=X[val_sl], y_val=y[val_sl], ts_val=ts[val_sl],
        X_test=X[test_sl], y_test=y[test_sl], ts_test=ts[test_sl],
        feature_names=np.asarray(feature_names, dtype=object),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create per-user supervised windows.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--modes', default='cgm_only,cgm_context')
    parser.add_argument('--horizons', default='30,60')
    parser.add_argument('--history-minutes', type=int, default=90)
    parser.add_argument('--freq-minutes', type=int, default=5)
    parser.add_argument('--min-samples', type=int, default=50)
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    df = pd.read_csv(args.input, parse_dates=['timestamp'])
    history_steps = args.history_minutes // args.freq_minutes
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(',') if h.strip()]

    manifest = []
    for user_id, user_df in df.groupby('user_id', sort=True):
        user_df = user_df.sort_values('timestamp').reset_index(drop=True)
        for mode in modes:
            for horizon in horizons:
                X, y, ts, feature_names = build_windows(user_df, mode, history_steps, horizon, args.freq_minutes)
                if len(X) < args.min_samples:
                    print(f'Skipping {user_id} {mode} h{horizon}: only {len(X)} samples')
                    continue
                user_dir = ensure_dir(out_dir / mode / f'h{horizon}')
                out_path = user_dir / f'{user_id}.npz'
                save_npz(X, y, ts, feature_names, out_path)
                manifest.append({'user_id': user_id, 'mode': mode, 'horizon': horizon, 'samples': int(len(X)), 'features': int(X.shape[-1]), 'path': str(out_path)})
                print(f'Wrote windows: {out_path} | samples={len(X)} | features={X.shape[-1]}')

    manifest_path = out_dir / 'windows_manifest.json'
    with manifest_path.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f'Wrote manifest: {manifest_path} | entries={len(manifest)}')
