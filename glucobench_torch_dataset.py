# =============================================================================
# Project: Smart Glucose Forecasting (MSc Thesis)
# Author: Asha Deepthi Yarabati
# Description: This script defines a PyTorch Dataset class for loading the 
#              preprocessed glucose forecasting data. 
#              The dataset is expected to be in .npz format, containing the features,
#              targets, timestamps, and feature names for each split 
#              (train, validation, test).
# Version: 1.0
# Last Updated: 06 Apr 2026
# Usage:
# - Run in project root directory "pwsh ./run_all.ps1"
# =============================================================================


from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GlucoBenchTorchDataset(Dataset):
    def __init__(self, npz_path: str | Path, split: str = 'train'):
        data = np.load(npz_path, allow_pickle=True)
        split = split.lower()
        self.X = data[f'X_{split}'].astype('float32')
        self.y = data[f'y_{split}'].astype('float32')
        self.timestamps = data[f'ts_{split}']
        self.feature_names = data['feature_names']

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
