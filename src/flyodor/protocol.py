"""Chronological leave-one-gas-out novelty-detection protocol.

For one held-out gas and one seed:
  * fit set   : Batch-1 samples of the familiar gases (used to fit scaler + detector)
  * calib set : a disjoint, stratified slice of Batch-1 familiar samples (used ONLY to set
                the decision threshold)
  * test set  : every sample of Batches 2-10 (familiar gases -> label 0, held-out gas -> 1)

Nothing from Batches 2-10 is used to fit preprocessing, the detector, or the threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from . import config as C
from .preprocess import FrozenStandardizer


class Detector(Protocol):
    """Any novelty detector: fit on familiar data, higher score = more novel."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Detector": ...
    def score(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Split:
    heldout_gas: int
    seed: int
    fit_idx: np.ndarray
    calib_idx: np.ndarray
    test_idx: np.ndarray


def make_split(df: pd.DataFrame, heldout_gas: int, seed: int,
               calib_frac: float = C.CALIB_FRACTION) -> Split:
    if heldout_gas not in C.GAS_NAMES:
        raise ValueError(f"unknown gas id {heldout_gas}")
    rng = np.random.default_rng(seed)
    batch = df["batch"].to_numpy()
    gas = df["gas_id"].to_numpy()

    fit_parts, calib_parts = [], []
    for g in C.GAS_NAMES:
        if g == heldout_gas:
            continue
        idx = np.flatnonzero((batch == C.TRAIN_BATCH) & (gas == g))
        idx = rng.permutation(idx)
        n_cal = int(round(calib_frac * len(idx)))
        calib_parts.append(idx[:n_cal])
        fit_parts.append(idx[n_cal:])

    test_idx = np.flatnonzero(np.isin(batch, C.TEST_BATCHES))
    return Split(heldout_gas, seed, np.sort(np.concatenate(fit_parts)),
                 np.sort(np.concatenate(calib_parts)), test_idx)


def check_no_leakage(df: pd.DataFrame, s: Split) -> None:
    """Raise AssertionError if the split violates the chronological protocol."""
    batch = df["batch"].to_numpy()
    gas = df["gas_id"].to_numpy()
    train = np.concatenate([s.fit_idx, s.calib_idx])
    assert np.all(batch[train] == C.TRAIN_BATCH), "training data outside Batch 1"
    assert not np.any(gas[train] == s.heldout_gas), "held-out gas leaked into training"
    assert np.intersect1d(s.fit_idx, s.calib_idx).size == 0, "fit and calib overlap"
    assert np.all(np.isin(batch[s.test_idx], C.TEST_BATCHES)), "test data from Batch 1"
    assert np.intersect1d(train, s.test_idx).size == 0, "train/test overlap"


@dataclass
class RunResult:
    heldout_gas: int
    seed: int
    detector: str
    threshold: float
    calib_fpr: float
    test_scores: np.ndarray   # aligned with split.test_idx
    test_novel: np.ndarray    # 1 = held-out gas
    test_batch: np.ndarray
    # No-drift reference (Batch 1, never used for fitting): calibration familiar scores
    # vs. scores of the held-out gas's own Batch-1 samples.
    calib_scores: np.ndarray
    b1_novel_scores: np.ndarray


def run_detector(name: str, factory: Callable[[int], Detector], df: pd.DataFrame,
                 X: np.ndarray, s: Split,
                 target_fpr: float = C.TARGET_FPR) -> RunResult:
    check_no_leakage(df, s)
    gas = df["gas_id"].to_numpy()

    scaler = FrozenStandardizer().fit(X[s.fit_idx])
    det = factory(s.seed).fit(scaler.transform(X[s.fit_idx]), gas[s.fit_idx])

    calib_scores = det.score(scaler.transform(X[s.calib_idx]))
    threshold = float(np.quantile(calib_scores, 1.0 - target_fpr))
    calib_fpr = float(np.mean(calib_scores > threshold))

    test_scores = det.score(scaler.transform(X[s.test_idx]))
    b1_novel_idx = np.flatnonzero((df["batch"].to_numpy() == C.TRAIN_BATCH) & (gas == s.heldout_gas))
    b1_novel_scores = det.score(scaler.transform(X[b1_novel_idx]))
    return RunResult(
        calib_scores=calib_scores, b1_novel_scores=b1_novel_scores,
        heldout_gas=s.heldout_gas, seed=s.seed, detector=name,
        threshold=threshold, calib_fpr=calib_fpr, test_scores=test_scores,
        test_novel=(gas[s.test_idx] == s.heldout_gas).astype(int),
        test_batch=df["batch"].to_numpy()[s.test_idx],
    )
