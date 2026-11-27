"""Conventional novelty detectors. Convention: higher score = more novel.

Hyperparameters are fixed a priori to common defaults (documented below) and are never
tuned on Batches 2-10. A Batch-1-only tuning procedure, applied identically to every
model including the fly model, is a planned follow-up.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM


class MahalanobisDetector:
    """Min Mahalanobis distance to any familiar-gas mean, with one shared covariance.

    Uses gas labels of the training data (class-conditional means). A shared, Ledoit-Wolf
    shrunk covariance is used because Batch 1 has as few as ~21 fit samples for some gases,
    too few for a per-gas 128x128 covariance.
    """

    def fit(self, X, y):
        classes = np.unique(y)
        self.means_ = np.vstack([X[y == c].mean(axis=0) for c in classes])
        centered = np.vstack([X[y == c] - X[y == c].mean(axis=0) for c in classes])
        self.precision_ = LedoitWolf().fit(centered).precision_
        return self

    def score(self, X):
        d = np.empty((X.shape[0], self.means_.shape[0]))
        for j, m in enumerate(self.means_):
            diff = X - m
            d[:, j] = np.einsum("ij,jk,ik->i", diff, self.precision_, diff)
        return np.sqrt(d.min(axis=1))


class KNNDetector:
    """Mean distance to the k nearest familiar training samples (label-free).

    metric="cosine" is the control for the KC code, whose overlap tracks cosine similarity.
    """

    def __init__(self, k: int = 5, metric: str = "euclidean"):
        self.k, self.metric = k, metric

    def fit(self, X, y=None):
        algo = "brute" if self.metric == "cosine" else "auto"
        self.nn_ = NearestNeighbors(n_neighbors=self.k, metric=self.metric, algorithm=algo).fit(X)
        return self

    def score(self, X):
        dist, _ = self.nn_.kneighbors(X)
        return dist.mean(axis=1)


class OCSVMDetector:
    """One-Class SVM, RBF kernel (label-free). gamma = gamma_mult * sklearn's 'scale' value."""

    def __init__(self, nu: float = 0.05, gamma_mult: float = 1.0):
        self.nu, self.gamma_mult = nu, gamma_mult

    def fit(self, X, y=None):
        gamma = self.gamma_mult / (X.shape[1] * X.var())
        self.model_ = OneClassSVM(kernel="rbf", nu=self.nu, gamma=gamma).fit(X)
        return self

    def score(self, X):
        # Samples far from every support vector all get (almost exactly) the maximum score;
        # their remaining differences are floating-point noise (~1e-8). Rounding turns that noise
        # into genuine ties, which AUROC counts as 1/2, instead of an arbitrary ordering.
        # (Under drift this affects up to ~57% of a batch with a narrow kernel; found in Week 5.)
        return np.round(-self.model_.decision_function(X), 6)


class IsolationForestDetector:
    """Isolation Forest, 300 trees (label-free)."""

    def __init__(self, seed: int = 0, n_estimators: int = 300, max_features: float = 1.0):
        self.seed, self.n_estimators, self.max_features = seed, n_estimators, max_features

    def fit(self, X, y=None):
        self.model_ = IsolationForest(n_estimators=self.n_estimators, max_features=self.max_features,
                                      random_state=self.seed).fit(X)
        return self

    def score(self, X):
        return -self.model_.score_samples(X)


class RandomDetector:
    """Sanity check only: random scores. Expected AUROC ~ 0.5, AUPRC ~ novel prevalence."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, y=None):
        return self

    def score(self, X):
        return np.random.default_rng(self.seed + 10_000).random(X.shape[0])


# name -> factory(seed) ; same interface the fly model will use later
BASELINES = {
    "Mahalanobis": lambda seed: MahalanobisDetector(),
    "kNN (k=5)": lambda seed: KNNDetector(k=5),
    "One-Class SVM": lambda seed: OCSVMDetector(nu=0.05),
    "Isolation Forest": lambda seed: IsolationForestDetector(seed=seed),
    "Random (sanity)": lambda seed: RandomDetector(seed=seed),
}
