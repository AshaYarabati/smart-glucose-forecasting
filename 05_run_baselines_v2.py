# =============================================================================
# Project: Smart Glucose Forecasting (MSc Thesis)
# Author: Asha Deepthi Yarabati
# Description: This script runs baseline models (persistence, Ridge regression, SARIMAX)
#  on the per-user windowed datasets. It evaluates predictions using MAE, RMSE, 
#  and TIR metrics, and saves both per-user results and overall summaries to CSV files 
#  for analysis and paper tables.    
# Version: 1.0
# Last Updated: 06 Apr 2026
# Usage:
# - Run in project root directory "pwsh ./run_all.ps1"
# =============================================================================
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import compute_tir, ensure_dir, metrics_dict


def load_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    return {
        'X_train': data['X_train'], 'y_train': data['y_train'], 'ts_train': data['ts_train'],
        'X_val': data['X_val'], 'y_val': data['y_val'], 'ts_val': data['ts_val'],
        'X_test': data['X_test'], 'y_test': data['y_test'], 'ts_test': data['ts_test'],
        'feature_names': list(data['feature_names']),
    }


def persistence_predict(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    glucose_idx = feature_names.index('glucose')
    return X[:, -1, glucose_idx]


def ridge_predict(train_X, train_y, test_X):
    model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0, random_state=42))])
    model.fit(train_X.reshape(len(train_X), -1), train_y)
    return model.predict(test_X.reshape(len(test_X), -1)), model


def sarimax_predict(train_X, train_y, test_X, feature_names):
    glucose_idx = feature_names.index('glucose')
    try:
        train_y = np.asarray(train_y, dtype=float)[-120:]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = SARIMAX(endog=train_y, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), trend='c', enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=1)
            pred = np.asarray(fit.forecast(steps=len(test_X)))
        return pred, fit
    except Exception:
        pred = test_X[:, -1, glucose_idx]
        return pred, None


def evaluate_model(name, y_true, y_pred):
    metrics = metrics_dict(y_true, y_pred)
    metrics.update({'model': name, 'TIR_true': compute_tir(y_true), 'TIR_pred': compute_tir(y_pred)})
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run baseline models over per-user NPZ windows.')
    parser.add_argument('--windows-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--modes', default='cgm_only,cgm_context')
    parser.add_argument('--horizons', default='30,60')
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(',') if h.strip()]
    all_rows = []
    per_user_dir = ensure_dir(output_dir / 'per_user')
    pred_dir = ensure_dir(output_dir / 'predictions')

    for mode in modes:
        for horizon in horizons:
            mode_rows = []
            window_dir = Path(args.windows_dir) / mode / f'h{horizon}'
            for npz_path in sorted(window_dir.glob('*.npz')):
                user_id = npz_path.stem
                print(f'Running baselines for {mode} h{horizon} {user_id}', flush=True)
                data = load_npz(npz_path)
                X_train = np.concatenate([data['X_train'], data['X_val']], axis=0)
                y_train = np.concatenate([data['y_train'], data['y_val']], axis=0)
                X_test = data['X_test']
                y_test = data['y_test']
                feature_names = data['feature_names']
                timestamps = data['ts_test']

                preds = {
                    'persistence': persistence_predict(X_test, feature_names),
                }
                preds['ridge'], _ = ridge_predict(X_train, y_train, X_test)
                preds['sarimax'], _ = sarimax_predict(X_train, y_train, X_test, feature_names)

                pred_frame = pd.DataFrame({'timestamp': timestamps, 'y_true': y_test})
                for model_name, y_pred in preds.items():
                    pred_frame[f'y_pred_{model_name}'] = y_pred
                    row = evaluate_model(model_name, y_test, y_pred)
                    row.update({'user_id': user_id, 'mode': mode, 'horizon_minutes': horizon, 'n_test': len(y_test)})
                    mode_rows.append(row)
                    all_rows.append(row)
                pred_frame.to_csv(pred_dir / f'baseline_predictions_{mode}_h{horizon}_{user_id}.csv', index=False)

            if mode_rows:
                pd.DataFrame(mode_rows).sort_values(['user_id', 'model']).to_csv(per_user_dir / f'baseline_results_{mode}_h{horizon}.csv', index=False)

    all_df = pd.DataFrame(all_rows).sort_values(['mode', 'horizon_minutes', 'model', 'user_id'])
    all_df.to_csv(output_dir / 'baseline_results_all.csv', index=False)
    summary = (
        all_df.groupby(['mode', 'horizon_minutes', 'model'], as_index=False)
        .agg(MAE=('MAE', 'mean'), RMSE=('RMSE', 'mean'), TIR_true=('TIR_true', 'mean'), TIR_pred=('TIR_pred', 'mean'))
        .sort_values(['mode', 'horizon_minutes', 'MAE'])
    )
    summary.to_csv(output_dir / 'baseline_results_summary.csv', index=False)
    paper = summary.rename(columns={'horizon_minutes': 'Horizon (min)', 'mode': 'Input setting', 'model': 'Model'})
    paper['Input setting'] = paper['Input setting'].map({'cgm_only': 'CGM-only', 'cgm_context': 'CGM + context'})
    paper['MAE'] = paper['MAE'].round(2)
    paper['RMSE'] = paper['RMSE'].round(2)
    paper.to_csv(output_dir / 'baseline_results_paper_table.csv', index=False)
    print(f"Wrote {output_dir / 'baseline_results_paper_table.csv'}", flush=True)
