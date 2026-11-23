"""Familiarity memory on KC codes (simple version used for the Phase-3 MVP).

Follows the structure of the fly novelty model of Dasgupta et al. (2018), as
described in its abstract and our notes; check the full paper before citing
details (docs/literature/dasgupta2018_novelty.md).

    start:  w_i = 1 for every KC i              (KC -> output-neuron synapses)
    learn:  for every familiar sample, for every active KC i:  w_i <- eps * w_i
    score:  s(x) = (1/k) * sum_i h_i(x) * w_i   (1 = all cells fresh = novel,
                                                 0 = all cells fully depressed)

Equivalent closed form: w_i = eps ** c_i, where c_i is how many training
samples activated KC i. So the memory does not depend on sample order.

Biological reading (hedged): repeated exposure depresses KC -> MBON synapses
(dopamine-dependent plasticity), so an odor that activates mostly "used" KCs
drives the output neuron weakly = familiar. Approximations: one output neuron,
a fixed eps, no explicit dopamine signal, no forgetting over time.
"""

from __future__ import annotations

import numpy as np


class FamiliarityMemory:
    def __init__(self, eps: float):
        if not 0.0 <= eps < 1.0:
            raise ValueError("eps must be in [0, 1)")
        self.eps = eps

    def fit(self, H: np.ndarray) -> "FamiliarityMemory":
        self.counts_ = H.sum(axis=0).astype(np.int64)
        self.w_ = self.eps ** self.counts_
        self.k_ = int(H[0].sum())
        return self

    def score(self, H: np.ndarray) -> np.ndarray:
        return (H @ self.w_) / self.k_
