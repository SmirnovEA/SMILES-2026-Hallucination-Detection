"""
splitting.py — 5-fold stratified cross-validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """5-fold stratified CV with held-out test set."""
    idx = np.arange(len(y))

    # Hold out stratified test set
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=y
    )
    y_trainval = y[idx_trainval]

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    splits = []
    for idx_tr, idx_va in skf.split(idx_trainval, y_trainval):
        splits.append((
            idx_trainval[idx_tr],
            idx_trainval[idx_va],
            idx_test,
        ))

    return splits

