"""Complete fly-inspired novelty detector and its biological ablations.

All variants follow the Detector interface used by protocol.run_detector:
    fit(X_standardized_familiar, y_ignored) ; score(X) -> higher = more novel
None of them uses gas labels.

Variants (each removes ONE component; everything else is held fixed):
  full          sparse binary random wiring (conn inputs/KC), m KCs, top-k, plastic memory
  no_expansion  same, but m = number of inputs (128) instead of 2,000: no dimensional expansion
  dense         every KC receives every input with Gaussian random weights (m, k, memory same)
  no_wta        no per-sample top-k; a single fixed threshold, set on the training data so the
                AVERAGE fraction of active KCs matches k/m (sparsity is matched, competition removed)
  kc_nn         same KC code, but the plastic synaptic memory is replaced by storing every
                training code and scoring 1 - max overlap with any stored code
                (random expansion + sparse code alone, no familiarity-memory rule)
"""
from __future__ import annotations

import numpy as np

from .encoder import SparseKCEncoder, overlap
from .memory import FamiliarityMemory

VARIANTS = ("full", "no_expansion", "dense", "no_wta", "kc_nn")


class FlyFamiliarityDetector:
    def __init__(self, seed: int = 0, n_kc: int = 2000, conn: int = 7,
                 active_frac: float = 0.05, beta: float = 0.0, variant: str = "full"):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.seed, self.n_kc, self.conn = seed, n_kc, conn
        self.active_frac, self.beta, self.variant = active_frac, beta, variant

    # -------------------------------------------------------------------------------
    def _build_encoder(self, n_inputs: int) -> SparseKCEncoder:
        m = n_inputs if self.variant == "no_expansion" else self.n_kc
        enc = SparseKCEncoder(n_inputs, n_kc=m, conn_per_kc=min(self.conn, n_inputs),
                              active_frac=self.active_frac, seed=self.seed)
        if self.variant == "dense":
            rng = np.random.default_rng(self.seed + 1_000_003)
            enc.W = rng.standard_normal((m, n_inputs)).astype(np.float32)
        return enc

    def codes(self, X: np.ndarray, _fit: bool = False) -> np.ndarray:
        if self.variant == "no_wta":
            a = self.encoder_.drive(X)
            if _fit:
                self.theta_ = float(np.quantile(a, 1.0 - self.active_frac))
            return a > self.theta_
        return self.encoder_.encode(X)

    # -------------------------------------------------------------------------------
    def fit(self, X: np.ndarray, y=None) -> "FlyFamiliarityDetector":
        self.encoder_ = self._build_encoder(X.shape[1])
        train_codes = self.codes(X, _fit=True)
        if self.variant == "kc_nn":
            self.stored_ = train_codes
        else:
            self.memory_ = FamiliarityMemory(self.beta).fit(train_codes)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        c = self.codes(X)
        if self.variant == "kc_nn":
            return 1.0 - overlap(c, self.stored_).max(axis=1)
        return self.memory_.score(c)
