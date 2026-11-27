"""Mushroom-body-inspired sparse encoder (PN -> KC expansion + winner-take-all).

Biology -> engineering mapping
------------------------------
* Projection neurons (PNs)   : the 128 standardized sensor features (input vector z).
* PN -> KC wiring            : fixed binary matrix W (n_kc x n_inputs). Each Kenyon cell
                               receives `conn_per_kc` inputs chosen uniformly at random
                               (Caron et al. 2013 report largely random PN->KC convergence
                               with only a handful of inputs per KC).
* KC drive                   : a = W z  (each KC sums its few inputs).
* APL feedback inhibition    : idealised as top-k winner-take-all: only the k most strongly
                               driven KCs stay active (Lin et al. 2014: sparse KC responses).
* KC output                  : binary code c in {0,1}^n_kc with exactly k ones ("FlyHash",
                               Dasgupta et al. 2017/2018).

Simplifications we state openly: no spiking dynamics, no PN-PN interactions in the antennal
lobe, inputs are z-scores (can be negative) rather than firing rates, and inhibition is an
exact top-k rule rather than a recurrent circuit. The fly has ~50 glomeruli -> ~2,000 KCs
(about 40x expansion); with 128 inputs and 2,000 KCs our expansion is about 16x.
"""
from __future__ import annotations

import numpy as np


class SparseKCEncoder:
    def __init__(self, n_inputs: int, n_kc: int = 2000, conn_per_kc: int = 7,
                 active_frac: float = 0.05, seed: int = 0):
        if not 1 <= conn_per_kc <= n_inputs:
            raise ValueError("conn_per_kc must be between 1 and n_inputs")
        if not 0 < active_frac < 1:
            raise ValueError("active_frac must be in (0, 1)")
        self.n_inputs, self.n_kc = n_inputs, n_kc
        self.conn_per_kc, self.active_frac, self.seed = conn_per_kc, active_frac, seed
        self.k = max(1, int(round(active_frac * n_kc)))

        rng = np.random.default_rng(seed)
        # For every KC, pick conn_per_kc distinct inputs uniformly at random.
        chosen = np.argsort(rng.random((n_kc, n_inputs)), axis=1)[:, :conn_per_kc]
        W = np.zeros((n_kc, n_inputs), dtype=np.float32)
        np.put_along_axis(W, chosen, 1.0, axis=1)
        self.W = W

    def drive(self, Z: np.ndarray) -> np.ndarray:
        """KC input current a = W z, shape (n_samples, n_kc)."""
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float32))
        if Z.shape[1] != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} features, got {Z.shape[1]}")
        return Z @ self.W.T

    def encode(self, Z: np.ndarray) -> np.ndarray:
        """Binary KC code: exactly k active cells per sample (bool array)."""
        a = self.drive(Z)
        top = np.argpartition(-a, self.k - 1, axis=1)[:, : self.k]
        codes = np.zeros(a.shape, dtype=bool)
        np.put_along_axis(codes, top, True, axis=1)
        return codes


def overlap(codes_a: np.ndarray, codes_b: np.ndarray) -> np.ndarray:
    """Fraction of shared active KCs between every pair of codes: |A ∩ B| / k."""
    k = codes_a.sum(axis=1, keepdims=True)
    return (codes_a.astype(np.float32) @ codes_b.T.astype(np.float32)) / k
