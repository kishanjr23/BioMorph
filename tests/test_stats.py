import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from flyodor.stats import bootstrap_p, holm, stratified_weights, weighted_auc


def test_weighted_auc_matches_sklearn_with_unit_weights():
    rng = np.random.default_rng(0)
    s = rng.normal(size=500)
    y = (rng.random(500) < 0.3).astype(int)
    s[y == 1] += 0.7
    assert weighted_auc(s, y, np.ones((1, 500)))[0] == pytest.approx(roc_auc_score(y, s))


def test_weighted_auc_handles_ties_like_sklearn():
    s = np.array([0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0])     # fly scores are heavily tied
    y = np.array([0, 1, 0, 1, 1, 0, 1])
    assert weighted_auc(s, y, np.ones((1, 7)))[0] == pytest.approx(roc_auc_score(y, s))


def test_integer_weights_equal_duplicated_samples():
    rng = np.random.default_rng(1)
    s, y = rng.normal(size=60), np.r_[np.zeros(40, int), np.ones(20, int)]
    w = rng.integers(0, 4, size=60)
    w[0] = w[45] = 1                                          # keep both classes present
    dup = np.repeat(np.arange(60), w)
    assert weighted_auc(s, y, w[None, :])[0] == pytest.approx(roc_auc_score(y[dup], s[dup]))


def test_stratified_weights_preserve_group_sizes():
    rng = np.random.default_rng(2)
    groups = [np.arange(0, 10), np.arange(10, 13)]
    W = stratified_weights(rng, groups, 13, R=200)
    np.testing.assert_array_equal(W[:, :10].sum(1), 10)
    np.testing.assert_array_equal(W[:, 10:].sum(1), 3)


def test_holm_known_example():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])


def test_bootstrap_p():
    assert bootstrap_p(np.full(1000, 0.1)) == pytest.approx(0.001)
    assert bootstrap_p(np.r_[np.full(500, -1.0), np.full(500, 1.0)]) == pytest.approx(1.0)
