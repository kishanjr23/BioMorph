"""Mushroom-body-inspired sparse encoder: PN -> KC -> top-k.

Biology (simplified):
    Projection neurons (PNs) carry odor information from the antennal lobe.
    Each Kenyon cell (KC) receives a few PN inputs chosen largely at random
    (Caron et al. 2013). A global inhibitory feedback neuron (APL) keeps only
    a small fraction of KCs active per odor (Lin et al. 2014).

Engineering:
    1. PN stage    x (128 preprocessed features, signed) -> x_PN (256, >= 0)
                   ON/OFF channels: x_PN = [max(x, 0), max(-x, 0)].
                   Firing rates cannot be negative, so a signed feature is
                   carried by two non-negative channels.
    2. Projection  a = M @ x_PN,  M in {0,1}^{m x 256}, every row has exactly
                   s ones at random columns (fixed by `seed`, never trained).
    3. Top-k       h_i = 1 for the k most active KCs, else 0. Stand-in for APL
                   inhibition. Ties broken by lower KC index (deterministic).

Nothing in the encoder is fitted to data. M is random and frozen; that is the
biological point and also means the encoder cannot leak information.

Defaults: m = 2000 KCs (about the number in one fly mushroom body), s = 6
inputs per KC (Caron et al. report a handful per KC; verify the exact mean in
the paper before citing), k = 5% of m.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def on_off(X: np.ndarray) -> np.ndarray:
    """(n, d) signed -> (n, 2d) non-negative: [ON channels | OFF channels]."""
    return np.hstack([np.maximum(X, 0.0), np.maximum(-X, 0.0)])


def top_k_binary(A: np.ndarray, k: int) -> np.ndarray:
    """Binary code with exactly k ones per row: the k largest entries.

    Stable sort on -A, so equal activities are won by the lower KC index.
    """
    order = np.argsort(-A, axis=1, kind="stable")[:, :k]
    H = np.zeros(A.shape, dtype=np.uint8)
    np.put_along_axis(H, order, 1, axis=1)
    return H


class FlyEncoder:
    def __init__(self, n_features: int = 128, m: int = 2000, s: int = 6,
                 k_frac: float = 0.05, seed: int = 0):
        self.n_features = n_features
        self.n_pn = 2 * n_features
        self.m = m
        self.s = s
        self.k = max(1, int(round(k_frac * m)))
        self.seed = seed
        if not 1 <= s <= self.n_pn:
            raise ValueError("s must be between 1 and the number of PN inputs")
        rng = np.random.default_rng(seed)
        # each KC picks s distinct PN inputs uniformly at random
        self.inputs_ = np.vstack([rng.choice(self.n_pn, size=s, replace=False)
                                  for _ in range(m)])
        rows = np.repeat(np.arange(m), s)
        self.M_ = sparse.csr_matrix(
            (np.ones(m * s), (rows, self.inputs_.ravel())), shape=(m, self.n_pn)
        )

    def activity(self, X: np.ndarray) -> np.ndarray:
        """Graded KC input a = M @ x_PN, shape (n, m). Before top-k."""
        return np.asarray((self.M_ @ on_off(X).T).T)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Binary KC code h, shape (n, m), exactly k ones per row."""
        return top_k_binary(self.activity(X), self.k)

    def effective_fan_in(self) -> np.ndarray:
        """Distinct *features* per KC. The ON and OFF channel of one feature
        are never both non-zero, so a KC wired to both has one live input
        fewer from that feature."""
        feat = self.inputs_ % self.n_features
        return np.array([len(set(r)) for r in feat])
