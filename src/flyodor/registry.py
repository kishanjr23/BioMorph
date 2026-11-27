"""All detectors compared in Week 4, their hyperparameter grids, and Batch-1-only tuning.

Tuning protocol (identical for every tunable detector)
----------------------------------------------------
For a given held-out gas, use ONLY Batch-1 samples of the remaining (familiar) gases.
Inner leave-one-familiar-gas-out: each familiar gas in turn plays a *pseudo-novel* gas; the
detector is fitted on 70% of the other familiar gases (stratified), and we measure AUROC
separating the other 30% (familiar) from the pseudo-novel gas. The setting with the best mean
inner AUROC is chosen (ties -> first in grid, i.e. the default) and then used for all seeds.

The real held-out gas and Batches 2-10 are never touched. Limitation: Batch 1 contains no
drift, so tuning optimises no-drift performance; that is unavoidable without future data.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from .baselines import (IsolationForestDetector, KNNDetector, MahalanobisDetector, OCSVMDetector,
                        RandomDetector)
from .flymodel import FlyFamiliarityDetector
from .preprocess import FrozenStandardizer

FLY_FULL = "Fly model (full)"


def _grid(**axes) -> list[dict]:
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*axes.values())]


@dataclass(frozen=True)
class Spec:
    group: str                                   # conventional | control | fly | ablation | sanity
    build: Callable[[int, dict], object]         # (seed, params) -> detector
    grid: list[dict] | None                      # None = inherit the full fly model's params


def _fly(variant: str):
    return lambda seed, p: FlyFamiliarityDetector(seed=seed, variant=variant, **p)


# Defaults are listed first on every axis so that ties keep the a-priori default.
SPECS: dict[str, Spec] = {
    "Mahalanobis": Spec("conventional", lambda s, p: MahalanobisDetector(), [{}]),
    "kNN (Euclidean)": Spec("conventional", lambda s, p: KNNDetector(metric="euclidean", **p),
                            _grid(k=[5, 1, 3, 10])),
    "One-Class SVM": Spec("conventional", lambda s, p: OCSVMDetector(**p),
                          _grid(nu=[0.05, 0.01, 0.1], gamma_mult=[1.0, 0.1, 10.0])),
    "Isolation Forest": Spec("conventional", lambda s, p: IsolationForestDetector(seed=s, **p),
                             _grid(max_features=[1.0, 0.5])),
    "kNN (cosine)": Spec("control", lambda s, p: KNNDetector(metric="cosine", **p),
                         _grid(k=[5, 1, 3, 10])),
    FLY_FULL: Spec("fly", _fly("full"),
                   _grid(n_kc=[2000], active_frac=[0.05, 0.02, 0.10], conn=[7, 4, 12],
                         beta=[0.0, 0.5, 0.9, 0.99])),
    "Ablation: no expansion (m=128)": Spec("ablation", _fly("no_expansion"), None),
    "Ablation: dense wiring": Spec("ablation", _fly("dense"), None),
    "Ablation: no top-k (fixed threshold)": Spec("ablation", _fly("no_wta"), None),
    "Ablation: no plastic memory (KC nearest neighbour)": Spec("ablation", _fly("kc_nn"), None),
    "Random (sanity)": Spec("sanity", lambda s, p: RandomDetector(seed=s), [{}]),
}


def inner_auroc(df: pd.DataFrame, X: np.ndarray, spec: Spec, params: dict, heldout_gas: int,
                seed: int = 0, train_frac: float = 1 - C.CALIB_FRACTION) -> float:
    batch, gas = df["batch"].to_numpy(), df["gas_id"].to_numpy()
    b1 = batch == C.TRAIN_BATCH
    familiar = [g for g in C.GAS_NAMES if g != heldout_gas and (b1 & (gas == g)).any()]
    aucs = []
    for pseudo in familiar:
        rng = np.random.default_rng(seed)
        tr, ev = [], []
        for g in familiar:
            if g == pseudo:
                continue
            idx = rng.permutation(np.flatnonzero(b1 & (gas == g)))
            n_tr = int(round(train_frac * len(idx)))
            tr.append(idx[:n_tr])
            ev.append(idx[n_tr:])
        tr, ev = np.concatenate(tr), np.concatenate(ev)
        nov = np.flatnonzero(b1 & (gas == pseudo))
        scaler = FrozenStandardizer().fit(X[tr])
        det = spec.build(seed, params).fit(scaler.transform(X[tr]), gas[tr])
        s = np.r_[det.score(scaler.transform(X[ev])), det.score(scaler.transform(X[nov]))]
        y = np.r_[np.zeros(len(ev)), np.ones(len(nov))]
        aucs.append(roc_auc_score(y, s))
    return float(np.mean(aucs))


def tune(df: pd.DataFrame, X: np.ndarray, spec: Spec, heldout_gas: int) -> tuple[dict, list]:
    table = [(p, inner_auroc(df, X, spec, p, heldout_gas)) for p in spec.grid]
    best = max(table, key=lambda t: t[1])  # max returns the first maximum -> default wins ties
    return best[0], table
