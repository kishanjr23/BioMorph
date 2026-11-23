"""Fly-inspired model: encoder (PN -> KC -> top-k) + familiarity memory."""

from __future__ import annotations

from ..baselines import MethodSpec
from .encoder import FlyEncoder
from .memory import FamiliarityMemory


class FlyDetector:
    """Detector interface used by experiment.py: fit(X, gas) / score(X).

    Gas labels are accepted for interface compatibility but never used.
    The seed fixes the random projection M.
    """

    def __init__(self, eps: float, seed: int, m: int = 2000, s: int = 6,
                 k_frac: float = 0.05):
        self.eps, self.seed, self.m, self.s, self.k_frac = eps, seed, m, s, k_frac

    def fit(self, X, gas=None):
        self.encoder_ = FlyEncoder(X.shape[1], self.m, self.s, self.k_frac, self.seed)
        self.memory_ = FamiliarityMemory(self.eps).fit(self.encoder_.encode(X))
        return self

    def score(self, X):
        return self.memory_.score(self.encoder_.encode(X))


EPS_GRID = [0.5, 0.8, 0.9, 0.95, 0.99]

FLY_METHODS = {
    "fly": MethodSpec("fly", "label-free", [{"eps": e} for e in EPS_GRID],
                      lambda c, s: FlyDetector(**c, seed=s)),
}

__all__ = ["FlyEncoder", "FamiliarityMemory", "FlyDetector", "FLY_METHODS"]
