"""Leakage-free preprocessing: statistics are fitted on the training batch only, then frozen."""
from __future__ import annotations

import numpy as np


class FrozenStandardizer:
    """z = (x - mean_train) / std_train. Never refit on test batches."""

    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> "FrozenStandardizer":
        self.mean_ = X_train.mean(axis=0)
        self.std_ = X_train.std(axis=0)
        self.std_[self.std_ < self.eps] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("fit() must be called on training data first")
        return (X - self.mean_) / self.std_
