# =============================================================================
# Project: Smart Glucose Forecasting (MSc Thesis)
# Author: Asha Deepthi Yarabati
# Description: Plots the baseline predictions against the ground truth 
#              for a visual comparison.  
# Version: 1.0
# Last Updated: 06 Apr 2026
# Usage:
# - Run in project root directory "pwsh ./run_all.ps1"
# =============================================================================

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import ensure_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot baseline predictions against ground truth.')
    parser.add_argument('--predictions-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--limit-files', type=int, default=8)
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    pred_paths = sorted(Path(args.predictions_dir).glob('baseline_predictions_*.csv'))[:args.limit_files]
    for pred_path in pred_paths:
        df = pd.read_csv(pred_path)
        plt.figure(figsize=(10, 4.5))
        plt.plot(df['y_true'].values, label='Ground truth')
        for col in [c for c in df.columns if c.startswith('y_pred_')]:
            plt.plot(df[col].values, label=col.replace('y_pred_', ''))
        plt.title(pred_path.stem)
        plt.xlabel('Test sample')
        plt.ylabel('Glucose (mg/dL)')
        plt.legend()
        plt.tight_layout()
        out_path = out_dir / f'{pred_path.stem}.png'
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f'Wrote plot: {out_path}')
