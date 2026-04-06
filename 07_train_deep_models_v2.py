
# =============================================================================
# Project: Smart Glucose Forecasting (MSc Thesis)
# Author: Asha Deepthi Yarabati
# Description: Train deep models on per-user NPZ windows with train-only scaling 
#              and inverse-transformed outputs.
# Version: 1.0
# Last Updated: 06 Apr 2026
# Usage:
# - Run in project root directory "pwsh ./run_all.ps1"
# =============================================================================

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import compute_tir, ensure_dir, metrics_dict, save_json
from deep_models import LSTMRegressor, TCNRegressor


EPS = 1e-8


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.mkldnn.enabled = False


def _safe_std(arr: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    std = np.std(arr, axis=axis, keepdims=keepdims)
    return np.where(std < EPS, 1.0, std)


def fit_scalers(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, np.ndarray | float]:
    """
    Fit train-only scaling parameters.

    X is scaled per feature using statistics computed across the training samples
    and history dimension. y is scaled as a single scalar target.
    """
    x_mean = np.mean(X_train, axis=(0, 1), keepdims=True).astype(np.float32)
    x_std = _safe_std(X_train, axis=(0, 1), keepdims=True).astype(np.float32)
    y_mean = float(np.mean(y_train))
    y_std = float(_safe_std(y_train))
    return {
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std,
    }


def transform_X(X: np.ndarray, scalers: Dict[str, np.ndarray | float]) -> np.ndarray:
    return ((X - scalers['x_mean']) / scalers['x_std']).astype(np.float32)


def transform_y(y: np.ndarray, scalers: Dict[str, np.ndarray | float]) -> np.ndarray:
    return ((y - scalers['y_mean']) / scalers['y_std']).astype(np.float32)


def inverse_transform_y(y_scaled: np.ndarray, scalers: Dict[str, np.ndarray | float]) -> np.ndarray:
    return (np.asarray(y_scaled, dtype=np.float32) * scalers['y_std'] + scalers['y_mean']).astype(np.float32)


def build_loaders_from_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


def load_and_prepare_data(npz_path: Path, batch_size: int = 32, scale_X: bool = True, scale_y: bool = True):
    data = np.load(npz_path, allow_pickle=True)

    X_train_raw = np.asarray(data['X_train'], dtype=np.float32)
    y_train_raw = np.asarray(data['y_train'], dtype=np.float32)
    X_val_raw = np.asarray(data['X_val'], dtype=np.float32)
    y_val_raw = np.asarray(data['y_val'], dtype=np.float32)
    X_test_raw = np.asarray(data['X_test'], dtype=np.float32)
    y_test_raw = np.asarray(data['y_test'], dtype=np.float32)

    scalers = fit_scalers(X_train_raw, y_train_raw)

    X_train = transform_X(X_train_raw, scalers) if scale_X else X_train_raw
    X_val = transform_X(X_val_raw, scalers) if scale_X else X_val_raw
    X_test = transform_X(X_test_raw, scalers) if scale_X else X_test_raw

    y_train = transform_y(y_train_raw, scalers) if scale_y else y_train_raw
    y_val = transform_y(y_val_raw, scalers) if scale_y else y_val_raw
    y_test = transform_y(y_test_raw, scalers) if scale_y else y_test_raw

    train_loader, val_loader, test_loader = build_loaders_from_arrays(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        batch_size=batch_size,
    )

    payload = {
        'npz': data,
        'scalers': scalers,
        'X_train_raw': X_train_raw,
        'y_train_raw': y_train_raw,
        'X_val_raw': X_val_raw,
        'y_val_raw': y_val_raw,
        'X_test_raw': X_test_raw,
        'y_test_raw': y_test_raw,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
    }
    return payload, train_loader, val_loader, test_loader


def train_one_model(model, train_loader, val_loader, epochs: int, lr: float, device: str, patience: int):
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float('inf')
    best_epoch = 0
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimiser.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float('inf')
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, {'best_val_loss': best_val, 'best_epoch': best_epoch}


def predict(model, loader, device: str):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
            trues.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def build_model(model_name: str, input_size: int):
    if model_name == 'lstm':
        return LSTMRegressor(input_size=input_size, hidden_size=32, num_layers=1, dropout=0.1)
    if model_name == 'tcn':
        return TCNRegressor(input_size=input_size, channels=(32, 32), kernel_size=3, dropout=0.1)
    raise ValueError(f'Unknown model: {model_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train deep models on per-user NPZ windows with train-only scaling and inverse-transformed outputs.')
    parser.add_argument('--windows-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--models', default='lstm,tcn')
    parser.add_argument('--modes', default='cgm_only,cgm_context')
    parser.add_argument('--horizons', default='30,60')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--disable-x-scaling', action='store_true')
    parser.add_argument('--disable-y-scaling', action='store_true')
    args = parser.parse_args()

    set_seed(42)
    out_dir = ensure_dir(args.output_dir)
    per_user_dir = ensure_dir(out_dir / 'per_user')
    pred_dir = ensure_dir(out_dir / 'predictions')
    scaler_dir = ensure_dir(out_dir / 'scalers')
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(',') if h.strip()]

    rows = []
    for model_name in models:
        for mode in modes:
            for horizon in horizons:
                mode_rows = []
                window_dir = Path(args.windows_dir) / mode / f'h{horizon}'
                for npz_path in sorted(window_dir.glob('*.npz')):
                    user_id = npz_path.stem
                    print(f'Training {model_name} for {mode} h{horizon} {user_id}')
                    payload, train_loader, val_loader, test_loader = load_and_prepare_data(
                        npz_path=npz_path,
                        batch_size=args.batch_size,
                        scale_X=not args.disable_x_scaling,
                        scale_y=not args.disable_y_scaling,
                    )
                    npz_data = payload['npz']
                    scalers = payload['scalers']
                    input_size = payload['X_train'].shape[-1]

                    model = build_model(model_name, input_size=input_size)
                    model, history = train_one_model(
                        model,
                        train_loader,
                        val_loader,
                        args.epochs,
                        args.lr,
                        args.device,
                        args.patience,
                    )

                    y_pred_scaled, y_true_scaled = predict(model, test_loader, args.device)

                    if args.disable_y_scaling:
                        y_pred = y_pred_scaled.astype(np.float32)
                        y_true = y_true_scaled.astype(np.float32)
                    else:
                        y_pred = inverse_transform_y(y_pred_scaled, scalers)
                        y_true = inverse_transform_y(y_true_scaled, scalers)

                    timestamps = npz_data['ts_test']
                    metrics = metrics_dict(y_true, y_pred)
                    metrics.update({
                        'model': model_name,
                        'user_id': user_id,
                        'mode': mode,
                        'horizon_minutes': horizon,
                        'n_test': len(y_true),
                        'TIR_true': compute_tir(y_true),
                        'TIR_pred': compute_tir(y_pred),
                        'best_epoch': history['best_epoch'],
                        'best_val_loss': history['best_val_loss'],
                        'x_scaled': not args.disable_x_scaling,
                        'y_scaled': not args.disable_y_scaling,
                        'x_feature_count': int(input_size),
                        'y_train_mean_mgdl': float(scalers['y_mean']),
                        'y_train_std_mgdl': float(scalers['y_std']),
                    })
                    rows.append(metrics)
                    mode_rows.append(metrics)

                    pred_df = pd.DataFrame({
                        'timestamp': timestamps,
                        'y_true': y_true,
                        'y_pred': y_pred,
                        'y_true_scaled': y_true_scaled,
                        'y_pred_scaled': y_pred_scaled,
                    })
                    pred_path = pred_dir / f'{model_name}_predictions_{mode}_h{horizon}_{user_id}.csv'
                    pred_df.to_csv(pred_path, index=False)

                    scaler_meta = {
                        'model': model_name,
                        'user_id': user_id,
                        'mode': mode,
                        'horizon_minutes': horizon,
                        'x_scaled': not args.disable_x_scaling,
                        'y_scaled': not args.disable_y_scaling,
                        'x_mean': np.asarray(scalers['x_mean']).reshape(-1).tolist(),
                        'x_std': np.asarray(scalers['x_std']).reshape(-1).tolist(),
                        'y_mean': float(scalers['y_mean']),
                        'y_std': float(scalers['y_std']),
                    }
                    save_json(scaler_meta, scaler_dir / f'{model_name}_scaler_{mode}_h{horizon}_{user_id}.json')
                    print(f'Wrote {pred_path}')

                if mode_rows:
                    pd.DataFrame(mode_rows).sort_values(['user_id']).to_csv(
                        per_user_dir / f'{model_name}_results_{mode}_h{horizon}.csv',
                        index=False,
                    )

    if not rows:
        raise ValueError('No .npz window files were found for the requested modes and horizons.')

    all_df = pd.DataFrame(rows).sort_values(['model', 'mode', 'horizon_minutes', 'user_id'])
    all_df.to_csv(out_dir / 'deep_results_all.csv', index=False)
    summary = (
        all_df.groupby(['model', 'mode', 'horizon_minutes'], as_index=False)
        .agg(
            MAE=('MAE', 'mean'),
            RMSE=('RMSE', 'mean'),
            TIR_true=('TIR_true', 'mean'),
            TIR_pred=('TIR_pred', 'mean'),
        )
        .sort_values(['model', 'mode', 'horizon_minutes'])
    )
    summary.to_csv(out_dir / 'deep_results_summary.csv', index=False)

    for model_name in models:
        paper = summary[summary['model'] == model_name].copy()
        paper = paper.rename(columns={'horizon_minutes': 'Horizon (min)', 'mode': 'Input setting', 'model': 'Model'})
        paper['Input setting'] = paper['Input setting'].map({'cgm_only': 'CGM-only', 'cgm_context': 'CGM + context'})
        paper['MAE'] = paper['MAE'].round(2)
        paper['RMSE'] = paper['RMSE'].round(2)
        path = out_dir / f'deep_results_paper_table_{model_name}_h30-60.csv'
        paper.to_csv(path, index=False)
        print(f'Wrote {path}')
