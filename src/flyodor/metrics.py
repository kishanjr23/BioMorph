"""Per-batch and pooled evaluation metrics for one RunResult."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .protocol import RunResult


def _metrics(scores, novel, threshold) -> dict:
    pred = scores > threshold
    fam, nov = novel == 0, novel == 1
    both = fam.any() and nov.any()
    return {
        "n_familiar": int(fam.sum()),
        "n_novel": int(nov.sum()),
        "novel_prevalence": float(nov.mean()),
        "auroc": float(roc_auc_score(novel, scores)) if both else np.nan,
        "auprc": float(average_precision_score(novel, scores)) if both else np.nan,
        "fpr_familiar": float(pred[fam].mean()) if fam.any() else np.nan,
        "detection_rate": float(pred[nov].mean()) if nov.any() else np.nan,
    }


def evaluate(r: RunResult) -> pd.DataFrame:
    """One row per test batch, a 'pooled' row over Batches 2-10, and a '1 (ref)' row.

    '1 (ref)' is the no-drift reference: Batch-1 calibration familiar samples vs. the
    held-out gas's Batch-1 samples. Its FPR is ~TARGET_FPR by construction (the threshold
    was set on those same calibration samples); its AUROC/AUPRC are unbiased.
    """
    ref_scores = np.concatenate([r.calib_scores, r.b1_novel_scores])
    ref_novel = np.r_[np.zeros(len(r.calib_scores), int), np.ones(len(r.b1_novel_scores), int)]
    rows = [{"batch": "1 (ref)", **_metrics(ref_scores, ref_novel, r.threshold)}]
    for b in np.unique(r.test_batch):
        m = r.test_batch == b
        rows.append({"batch": str(int(b)), **_metrics(r.test_scores[m], r.test_novel[m], r.threshold)})
    rows.append({"batch": "pooled", **_metrics(r.test_scores, r.test_novel, r.threshold)})
    out = pd.DataFrame(rows)
    out.insert(0, "detector", r.detector)
    out.insert(1, "heldout_gas", r.heldout_gas)
    out.insert(2, "seed", r.seed)
    out["calib_fpr"] = r.calib_fpr
    return out
