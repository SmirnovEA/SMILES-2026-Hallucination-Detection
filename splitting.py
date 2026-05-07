"""
splitting.py — 5-fold stratified CV with larger test set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.20,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """5-fold stratified CV with 20% held-out test set."""
    idx = np.arange(len(y))

    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=y
    )
    y_trainval = y[idx_trainval]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    splits = []
    for idx_tr, idx_va in skf.split(idx_trainval, y_trainval):
        splits.append((
            idx_trainval[idx_tr],
            idx_trainval[idx_va],
            idx_test,
        ))

    return splits