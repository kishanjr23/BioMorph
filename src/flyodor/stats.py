"""Statistics helpers for Week 5: fast weighted AUROC for bootstrapping, Holm correction.

Bootstrap design (see scripts/run_week5.py)
-------------------------------------------
Two sources of variability are resampled together:
  1. the random seed (calibration split + random KC wiring)   -> resample seeds
  2. the finite set of test measurements in each batch        -> resample samples with
     replacement *within each batch and class* (familiar / held-out), so every bootstrap
     replicate keeps the original batch sizes and class counts.
A bootstrap replicate is represented by integer multinomial weights (how many times each
test sample was drawn), so AUROC can be computed for many replicates at once.
The same weights are used for every detector -> differences between detectors are *paired*.
"""
from __future__ import annotations

import numpy as np


def weighted_auc(scores: np.ndarray, labels: np.ndarray, W: np.ndarray) -> np.ndarray:
    """AUROC for R weighted replicates at once (ties count 1/2, as in Mann-Whitney U).

    scores, labels : (n,)       labels 1 = positive (novel), 0 = negative (familiar)
    W              : (R, n)     non-negative weights (bootstrap counts) per replicate
    returns        : (R,)
    """
    order = np.argsort(scores, kind="stable")
    s, y, Wo = scores[order], labels[order].astype(bool), np.atleast_2d(W)[:, order].astype(float)
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])          # groups of tied scores
    pos = np.add.reduceat(Wo * y, starts, axis=1)
    neg = np.add.reduceat(Wo * ~y, starts, axis=1)
    neg_below = np.cumsum(neg, axis=1) - neg
    num = (pos * (neg_below + 0.5 * neg)).sum(axis=1)
    den = pos.sum(axis=1) * neg.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def stratified_weights(rng: np.random.Generator, groups: list[np.ndarray], n: int, R: int) -> np.ndarray:
    """(R, n) bootstrap counts; each index group (e.g. one class in one batch) is resampled
    with replacement to its own size, independently."""
    W = np.zeros((R, n))
    for g in groups:
        if len(g):
            W[:, g] = rng.multinomial(len(g), np.full(len(g), 1.0 / len(g)), size=R)
    return W


def bootstrap_p(diff: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: difference = 0 (percentile method)."""
    diff = diff[np.isfinite(diff)]
    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    return float(min(1.0, max(p, 1.0 / len(diff))))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls family-wise error rate)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    a = (1 - level) / 2
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))
