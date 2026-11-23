"""Evaluation metrics for one set of scores (higher = more novel).

AUROC   probability that a random unfamiliar sample scores above a random
        familiar one. 0.5 = chance, independent of the class mix.
AUPRC   average precision for the unfamiliar class. Its chance level equals the
        unfamiliar share of the batch, so always report `novel_share` with it.
FPR     share of familiar samples called unfamiliar at the frozen threshold.
TPR     share of unfamiliar samples called unfamiliar at the frozen threshold.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

THRESHOLD_QUANTILE = 0.95  # target ~5% false alarms in the Batch-1 period
LOW_N = 20  # cells with fewer unfamiliar samples are flagged


def threshold_from_calib(calib_scores: np.ndarray) -> float:
    return float(np.quantile(calib_scores, THRESHOLD_QUANTILE))


def evaluate(scores: np.ndarray, y: np.ndarray, theta: float) -> dict:
    y = np.asarray(y).astype(int)
    n_novel = int(y.sum())
    n_fam = int(len(y) - n_novel)
    pred = scores > theta
    both = n_novel > 0 and n_fam > 0
    return {
        "auroc": roc_auc_score(y, scores) if both else np.nan,
        "auprc": average_precision_score(y, scores) if both else np.nan,
        "novel_share": n_novel / len(y) if len(y) else np.nan,
        "fpr": float(pred[y == 0].mean()) if n_fam else np.nan,
        "tpr": float(pred[y == 1].mean()) if n_novel else np.nan,
        "n_fam": n_fam,
        "n_novel": n_novel,
        "low_n": n_novel < LOW_N,
    }
