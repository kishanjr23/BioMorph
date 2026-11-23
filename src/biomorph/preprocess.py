"""Preprocessing fitted on Batch-1 FIT rows only.

P1 (primary):   signed log  sign(x) * log(1 + |x|),  then z-score per feature
                using mean / std of the FIT rows.
P2 (secondary): P1, then scale each sample to unit L2 length.

Why signed log: dR reaches ~670,000 while other features are O(1), and the
EMA-decay features are negative. log1p of |x| compresses the heavy tails and
the sign keeps the direction of the decay features.

Why P2 exists: per-sample L2 normalisation cancels any drift that scales all
features of a sample by the same factor. It loosely resembles antennal-lobe
gain control. Every model runs under both P1 and P2, so that a gain from
normalisation is not mistaken for a gain from the fly circuit.

Leakage rule: `fit` sees FIT rows only. `transform` uses stored statistics and
never looks at the rows it transforms beyond the per-sample L2 norm (P2), which
uses only that sample's own values.
"""

from __future__ import annotations

import numpy as np

PREPROCESSINGS = ("P1", "P2")


def signed_log(X: np.ndarray) -> np.ndarray:
    return np.sign(X) * np.log1p(np.abs(X))


class Preprocessor:
    def __init__(self, mode: str = "P1"):
        if mode not in PREPROCESSINGS:
            raise ValueError(f"unknown preprocessing {mode!r}")
        self.mode = mode
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Preprocessor":
        Z = signed_log(np.asarray(X, dtype=float))
        self.mean_ = Z.mean(axis=0)
        std = Z.std(axis=0)
        # A constant feature in FIT would divide by zero; leave it unscaled.
        self.std_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Preprocessor.transform called before fit")
        Z = (signed_log(np.asarray(X, dtype=float)) - self.mean_) / self.std_
        if self.mode == "P2":
            norm = np.linalg.norm(Z, axis=1, keepdims=True)
            Z = Z / np.where(norm > 0, norm, 1.0)
        return Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
