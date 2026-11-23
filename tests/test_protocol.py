"""Leakage and correctness tests for the chronological protocol (Phase 2).

The most important test is `test_future_batches_cannot_change_frozen_model`:
it replaces every batch 2-10 feature with random numbers and checks that the
fitted preprocessing, chosen hyperparameters, threshold and detector outputs
are bit-for-bit identical. If a later batch leaked into training anywhere,
this test fails.
"""

import numpy as np
import pytest

from biomorph.baselines import METHODS
from biomorph.data import DEFAULT_DATA_DIR, DriftDataset, load_all
from biomorph.experiment import fit_frozen, run_split
from biomorph.metrics import evaluate
from biomorph.preprocess import Preprocessor, signed_log
from biomorph.protocol import HELDOUT_GASES, TEST_BATCHES, make_split

needs_data = pytest.mark.skipif(
    not (DEFAULT_DATA_DIR / "batch1.dat").exists(),
    reason="dataset not downloaded (python scripts/download_data.py)",
)


@pytest.fixture(scope="module")
def ds():
    return load_all()


# ---------- split ----------

@needs_data
@pytest.mark.parametrize("gas", HELDOUT_GASES)
def test_split_has_no_heldout_gas_or_future_batch(ds, gas):
    sp = make_split(ds, gas, seed=0)
    train = np.r_[sp.fit, sp.calib]
    assert not (ds.gas[train] == gas).any()
    assert (ds.batch[train] == 1).all()
    assert len(np.intersect1d(sp.fit, sp.calib)) == 0
    # FIT + CALIB is exactly the familiar part of Batch 1
    expected = np.flatnonzero((ds.batch == 1) & (ds.gas != gas))
    assert np.array_equal(np.sort(train), expected)
    assert (ds.batch[sp.b1_reference] == 1).all() and (ds.gas[sp.b1_reference] == gas).all()
    assert set(np.unique(ds.batch[sp.test])) == set(TEST_BATCHES)
    assert len(np.intersect1d(sp.test, train)) == 0


@needs_data
def test_split_is_stratified_and_seeded(ds):
    a, b, c = (make_split(ds, 5, s) for s in (0, 0, 1))
    assert np.array_equal(a.fit, b.fit)
    assert not np.array_equal(a.fit, c.fit)
    # every familiar gas appears in CALIB, in roughly the 70/30 proportion
    for g in np.unique(ds.gas[a.fit]):
        n_fit = (ds.gas[a.fit] == g).sum()
        n_cal = (ds.gas[a.calib] == g).sum()
        assert n_cal > 0
        assert abs(n_fit / (n_fit + n_cal) - 0.7) < 0.05


@needs_data
def test_test_labels(ds):
    sp = make_split(ds, 3, 0)
    y = sp.test_labels(ds)
    assert np.array_equal(y == 1, ds.gas[sp.test] == 3)
    for b in TEST_BATCHES:  # every test batch has both classes
        m = ds.batch[sp.test] == b
        assert 0 < y[m].sum() < m.sum()


# ---------- preprocessing ----------

def test_preprocessor_uses_fit_statistics_only():
    rng = np.random.default_rng(0)
    X_fit = rng.normal(5, 3, (50, 4))
    X_other = rng.normal(500, 300, (50, 4))
    pre = Preprocessor("P1").fit(X_fit)
    Z = signed_log(X_fit)
    np.testing.assert_allclose(pre.mean_, Z.mean(0))
    np.testing.assert_allclose(pre.std_, Z.std(0))
    pre.transform(X_other)  # transforming other data must not change statistics
    np.testing.assert_allclose(pre.mean_, Z.mean(0))
    np.testing.assert_allclose(pre.transform(X_fit).mean(0), 0, atol=1e-12)


def test_signed_log_keeps_sign_and_p2_unit_norm():
    X = np.array([[-670_000.0, -1.0, 0.0, 2.0], [3.0, 1.0, -4.0, 5.0], [1, 2, 3, 4]])
    assert np.array_equal(np.sign(signed_log(X)), np.sign(X))
    Z = Preprocessor("P2").fit(X).transform(X)
    np.testing.assert_allclose(np.linalg.norm(Z, axis=1), 1.0)


@needs_data
def test_scaler_equals_fit_rows_statistics(ds):
    sp = make_split(ds, 1, 0)
    model = fit_frozen("mahalanobis", "P1", ds.X[sp.fit], ds.gas[sp.fit],
                       ds.X[sp.calib], ds.gas[sp.calib], 0)
    Z = signed_log(ds.X[sp.fit])
    np.testing.assert_array_equal(model.prep.mean_, Z.mean(0))


# ---------- the perturbation (leakage) test ----------

@needs_data
@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("prep", ["P1", "P2"])
def test_future_batches_cannot_change_frozen_model(ds, method, prep):
    rng = np.random.default_rng(123)
    X_bad = ds.X.copy()
    future = ds.batch >= 2
    X_bad[future] = rng.normal(0, 1e5, size=(future.sum(), ds.X.shape[1]))
    ds_bad = DriftDataset(X_bad, ds.gas, ds.batch, ds.index)

    gas, seed = 4, 7
    sp, sp_bad = make_split(ds, gas, seed), make_split(ds_bad, gas, seed)
    assert np.array_equal(sp.fit, sp_bad.fit) and np.array_equal(sp.calib, sp_bad.calib)

    _, _, m = run_split(ds, sp, method, prep)
    _, _, m_bad = run_split(ds_bad, sp_bad, method, prep)
    np.testing.assert_array_equal(m.prep.mean_, m_bad.prep.mean_)
    np.testing.assert_array_equal(m.prep.std_, m_bad.prep.std_)
    assert m.config == m_bad.config
    assert m.theta == m_bad.theta
    probe = ds.X[ds.batch == 1][:50]
    np.testing.assert_array_equal(m.score(probe), m_bad.score(probe))


# ---------- detectors and metrics ----------

@pytest.mark.parametrize("method", list(METHODS))
def test_scores_are_higher_for_novel(method):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (120, 6))
    gas = np.repeat([1, 2, 3], 40)
    spec = METHODS[method]
    det = spec.build(spec.grid[0], 0).fit(X, gas)
    near = rng.normal(0, 1, (30, 6))
    far = rng.normal(8, 1, (30, 6))
    assert det.score(far).mean() > det.score(near).mean()


def test_evaluate_basic():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.9, 0.8, 0.95])
    r = evaluate(s, y, theta=0.5)
    assert r["auroc"] == pytest.approx(5 / 6)
    assert r["fpr"] == pytest.approx(1 / 3) and r["tpr"] == 1.0
    assert r["n_novel"] == 2 and r["low_n"]
    assert r["novel_share"] == pytest.approx(0.4)
