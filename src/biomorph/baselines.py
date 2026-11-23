"""Conventional novelty detectors.

Every detector has   fit(X, gas=None) -> self   and   score(X) -> array,
and every score is oriented so that HIGHER = MORE NOVEL.

Label-free (main comparison; the fly memory also never uses gas labels):
    mahalanobis   squared Mahalanobis distance to the FIT mean,
                  Ledoit-Wolf shrunk covariance (128 dims, ~250 samples)
    ocsvm         One-Class SVM, RBF kernel, score = -decision_function
    iforest       Isolation Forest, 200 trees, score = -score_samples
    knn           Euclidean distance to the k-th nearest FIT sample

Class-aware (stronger REFERENCE baselines; they use gas labels the fly model
does not get, so they are reported separately and labelled as such):
    centroid_cls  min Euclidean distance to a per-gas FIT centroid
    mahal_cls     min squared Mahalanobis distance over per-gas models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM


class Mahalanobis:
    def fit(self, X, gas=None):
        lw = LedoitWolf().fit(X)
        self.mean_ = lw.location_
        self.precision_ = lw.precision_
        return self

    def score(self, X):
        D = X - self.mean_
        return np.einsum("ij,jk,ik->i", D, self.precision_, D)


class OCSVM:
    def __init__(self, nu: float, gamma_mult: float):
        self.nu = nu
        self.gamma_mult = gamma_mult

    def fit(self, X, gas=None):
        # sklearn's gamma="scale" is 1 / (n_features * X.var()); computed here
        # from FIT so the multiplier grid is relative to it.
        self.gamma_ = self.gamma_mult / (X.shape[1] * X.var())
        self.model_ = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma_).fit(X)
        return self

    def score(self, X):
        return -self.model_.decision_function(X)


class IForest:
    def __init__(self, max_samples: int, max_features: float, seed: int):
        self.max_samples = max_samples
        self.max_features = max_features
        self.seed = seed

    def fit(self, X, gas=None):
        self.model_ = IsolationForest(
            n_estimators=200,
            max_samples=min(self.max_samples, len(X)),
            max_features=self.max_features,
            random_state=self.seed,
        ).fit(X)
        return self

    def score(self, X):
        return -self.model_.score_samples(X)


class KNNDistance:
    def __init__(self, k: int):
        self.k = k

    def fit(self, X, gas=None):
        self.nn_ = NearestNeighbors(n_neighbors=self.k).fit(X)
        return self

    def score(self, X):
        dist, _ = self.nn_.kneighbors(X, n_neighbors=self.k)
        return dist[:, -1]


class ClassCentroid:
    def fit(self, X, gas):
        self.centroids_ = np.vstack([X[gas == g].mean(axis=0) for g in np.unique(gas)])
        return self

    def score(self, X):
        d = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        return d.min(axis=1)


class ClassMahalanobis:
    def fit(self, X, gas):
        self.models_ = [Mahalanobis().fit(X[gas == g]) for g in np.unique(gas)]
        return self

    def score(self, X):
        return np.min(np.vstack([m.score(X) for m in self.models_]), axis=0)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    label_use: str  # "label-free" or "class-aware"
    grid: list[dict] = field(default_factory=lambda: [{}])
    build: Callable[[dict, int], object] = None  # (config, seed) -> detector


def _grid(**axes) -> list[dict]:
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in product(*axes.values())]


METHODS: dict[str, MethodSpec] = {
    "mahalanobis": MethodSpec(
        "mahalanobis", "label-free", [{}], lambda c, s: Mahalanobis()
    ),
    "ocsvm": MethodSpec(
        "ocsvm",
        "label-free",
        _grid(nu=[0.01, 0.05, 0.1, 0.2], gamma_mult=[0.1, 1.0, 10.0]),
        lambda c, s: OCSVM(**c),
    ),
    "iforest": MethodSpec(
        "iforest",
        "label-free",
        _grid(max_samples=[64, 128, 256], max_features=[0.5, 1.0]),
        lambda c, s: IForest(**c, seed=s),
    ),
    "knn": MethodSpec(
        "knn", "label-free", _grid(k=[1, 3, 5, 10]), lambda c, s: KNNDistance(**c)
    ),
    "centroid_cls": MethodSpec(
        "centroid_cls", "class-aware", [{}], lambda c, s: ClassCentroid()
    ),
    "mahal_cls": MethodSpec(
        "mahal_cls", "class-aware", [{}], lambda c, s: ClassMahalanobis()
    ),
}
