"""Fit-freeze-test pipeline shared by every method (baselines now, fly later).

    1. tune      pseudo-novelty inside Batch 1 (no gas G, no later batches)
    2. fit       preprocessing + detector on FIT with the chosen config
    3. threshold theta = 95th percentile of CALIB scores
    4. freeze    nothing above changes after this point
    5. test      score batches 2-10, metrics per batch

`fit_frozen` receives only FIT and CALIB rows, so later batches cannot reach
it. tests/test_protocol.py checks this end to end with a perturbation test.

Pseudo-novelty tuning, for each familiar gas g' in FIT:
    train        FIT rows minus g'   (preprocessing is refitted on these too)
    familiar     CALIB rows minus g'
    novel        all Batch-1 FIT and CALIB rows of g'
    record AUROC; pick the config with the best mean AUROC over g'
    (ties -> first config in grid order).
Using both FIT and CALIB rows of g' as pseudo-novel gives more novel samples
(acetaldehyde has only 30 in Batch 1). The model in that fold never sees g',
so this does not leak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from .baselines import METHODS, MethodSpec
from .data import GAS_NAMES, DriftDataset
from .metrics import evaluate, threshold_from_calib
from .preprocess import Preprocessor
from .protocol import TEST_BATCHES, Split


@dataclass
class FrozenModel:
    method: str
    prep: Preprocessor
    config: dict
    detector: object
    theta: float
    calib_scores: np.ndarray
    tuning: list[dict]  # one row per (config, pseudo-novel gas)

    def score(self, X_raw: np.ndarray) -> np.ndarray:
        return self.detector.score(self.prep.transform(X_raw))


def tune(
    spec: MethodSpec,
    prep_mode: str,
    X_fit: np.ndarray,
    gas_fit: np.ndarray,
    X_calib: np.ndarray,
    gas_calib: np.ndarray,
    seed: int,
) -> tuple[dict, list[dict]]:
    rows = []
    if len(spec.grid) == 1:
        return spec.grid[0], rows
    for g in np.unique(gas_fit):
        tr = gas_fit != g
        pre = Preprocessor(prep_mode).fit(X_fit[tr])
        Xtr = pre.transform(X_fit[tr])
        fam = pre.transform(X_calib[gas_calib != g])
        nov = pre.transform(np.vstack([X_fit[~tr], X_calib[gas_calib == g]]))
        y = np.r_[np.zeros(len(fam)), np.ones(len(nov))]
        for ci, cfg in enumerate(spec.grid):
            det = spec.build(cfg, seed).fit(Xtr, gas_fit[tr])
            s = np.r_[det.score(fam), det.score(nov)]
            rows.append({"config_id": ci, "config": cfg, "pseudo_gas": int(g),
                         "auroc": roc_auc_score(y, s)})
    mean = [np.mean([r["auroc"] for r in rows if r["config_id"] == ci])
            for ci in range(len(spec.grid))]
    return spec.grid[int(np.argmax(mean))], rows


def fit_frozen(
    method: str,
    prep_mode: str,
    X_fit: np.ndarray,
    gas_fit: np.ndarray,
    X_calib: np.ndarray,
    gas_calib: np.ndarray,
    seed: int,
) -> FrozenModel:
    spec = METHODS[method]
    config, tuning_rows = tune(spec, prep_mode, X_fit, gas_fit, X_calib, gas_calib, seed)
    pre = Preprocessor(prep_mode).fit(X_fit)
    det = spec.build(config, seed).fit(pre.transform(X_fit), gas_fit)
    calib_scores = det.score(pre.transform(X_calib))
    return FrozenModel(method, pre, config, det, threshold_from_calib(calib_scores),
                       calib_scores, tuning_rows)


def run_split(ds: DriftDataset, split: Split, method: str, prep_mode: str):
    """Return (metric rows, tuning rows, frozen model) for one method on one split."""
    model = fit_frozen(
        method, prep_mode,
        ds.X[split.fit], ds.gas[split.fit],
        ds.X[split.calib], ds.gas[split.calib],
        split.seed,
    )
    base = {
        "method": method,
        "label_use": METHODS[method].label_use,
        "prep": prep_mode,
        "heldout": split.heldout,
        "heldout_name": GAS_NAMES[split.heldout],
        "seed": split.seed,
        "config": str(model.config),
        "theta": model.theta,
    }
    rows = []
    # Batch-1 reference: CALIB (familiar) vs Batch-1 held-out gas. No drift.
    # FPR here is ~5% by construction because theta was set on CALIB.
    ref_scores = np.r_[model.calib_scores, model.score(ds.X[split.b1_reference])]
    ref_y = np.r_[np.zeros(len(split.calib)), np.ones(len(split.b1_reference))]
    rows.append({**base, "batch": 1, **evaluate(ref_scores, ref_y, model.theta)})

    test_scores = model.score(ds.X[split.test])
    y = split.test_labels(ds)
    test_batch = ds.batch[split.test]
    for b in TEST_BATCHES:
        m = test_batch == b
        rows.append({**base, "batch": b, **evaluate(test_scores[m], y[m], model.theta)})

    tuning = [{**{k: base[k] for k in ("method", "prep", "heldout", "seed")},
               "config_id": r["config_id"], "config": str(r["config"]),
               "pseudo_gas": r["pseudo_gas"], "auroc": r["auroc"],
               "chosen": r["config"] == model.config}
              for r in model.tuning]
    return rows, tuning, model
