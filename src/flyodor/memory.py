"""Experience-dependent familiarity memory (KC -> MBON synaptic depression).

Biology
-------
In the mushroom body, Kenyon cells (KCs) synapse onto mushroom-body output neurons (MBONs).
Repeated exposure to an odor is associated with depression of the KC->MBON synapses of the
KCs that odor activates, so a familiar odor drives the output neuron weakly while a new odor
(which activates different KCs whose synapses are still strong) drives it strongly.

Engineering approximation (after Dasgupta et al., 2018, PNAS; check the paper for the exact
formulation before citing):
  * one output neuron with one weight w_j per KC, initialised to 1
  * for each familiar training sample with active KC set A:  w_j <- beta * w_j  for j in A
  * novelty score of a sample with active set A(x):  s(x) = mean_{j in A(x)} w_j  in [0, 1]

Because multiplication commutes, after training w_j = beta ** n_j, where n_j is the number of
training samples that activated KC j. beta = 0 gives a binary "seen at least once" memory
(a Bloom-filter-like structure); beta close to 1 gives graded familiarity.

Capacity limit: with beta = 0 (or small beta) every KC that fired even once is silenced. If
the training data are diverse enough to activate nearly every KC, the memory saturates and
everything scores as familiar. Sparse coding, a large KC population and graded beta all delay
this; it is tested in tests/test_flymodel.py.

Simplifications: a single MBON, no dopamine/reinforcement signal, no forgetting or recovery,
label-free (the memory never sees which gas a sample is).
"""
from __future__ import annotations

import numpy as np


class FamiliarityMemory:
    def __init__(self, beta: float = 0.0):
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        self.beta = beta
        self.w_: np.ndarray | None = None
        self.counts_: np.ndarray | None = None

    def fit(self, codes: np.ndarray) -> "FamiliarityMemory":
        """codes: bool array (n_samples, n_kc) of familiar training samples."""
        self.counts_ = codes.sum(axis=0)
        self.w_ = np.power(self.beta, self.counts_, dtype=float)   # beta**0 == 1
        return self

    def score(self, codes: np.ndarray) -> np.ndarray:
        """Mean synaptic weight over each sample's active KCs (1 = completely novel).

        A sample with no active KCs (possible only without top-k) cannot be recognised as
        familiar, so it gets the maximum novelty score of 1.
        """
        if self.w_ is None:
            raise RuntimeError("fit() first")
        n_active = codes.sum(axis=1)
        s = (codes.astype(float) @ self.w_) / np.maximum(n_active, 1)
        s[n_active == 0] = 1.0
        return s
