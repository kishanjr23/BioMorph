"""Leakage and sanity tests for the chronological leave-one-gas-out protocol."""
import dataclasses

import numpy as np
import pytest

from flyodor import config as C
from flyodor.baselines import BASELINES
from flyodor.metrics import evaluate
from flyodor.protocol import check_no_leakage, make_split, run_detector


@pytest.mark.parametrize("gas", C.PRIMARY_HELDOUT_GASES)
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_split_respects_protocol(df, gas, seed):
    s = make_split(df, gas, seed)
    check_no_leakage(df, s)
    # fit + calib together are exactly the Batch-1 familiar samples
    expected = np.flatnonzero((df["batch"] == 1).to_numpy() & (df["gas_id"] != gas).to_numpy())
    np.testing.assert_array_equal(np.sort(np.concatenate([s.fit_idx, s.calib_idx])), expected)
    # every familiar gas appears in both fit and calib
    for part in (s.fit_idx, s.calib_idx):
        assert set(df["gas_id"].to_numpy()[part]) == set(C.GAS_NAMES) - {gas}
    # test set is all of Batches 2-10 and contains the held-out gas
    assert len(s.test_idx) == int((df["batch"] >= 2).sum())
    assert (df["gas_id"].to_numpy()[s.test_idx] == gas).any()


def test_split_is_reproducible(df):
    a, b, c = make_split(df, 5, 3), make_split(df, 5, 3), make_split(df, 5, 4)
    np.testing.assert_array_equal(a.fit_idx, b.fit_idx)
    assert not np.array_equal(a.fit_idx, c.fit_idx)


def test_leakage_check_catches_violations(df):
    s = make_split(df, 1, 0)
    heldout_b1 = np.flatnonzero((df["batch"] == 1).to_numpy() & (df["gas_id"] == 1).to_numpy())
    with pytest.raises(AssertionError, match="held-out gas"):
        check_no_leakage(df, dataclasses.replace(s, fit_idx=np.append(s.fit_idx, heldout_b1[0])))
    with pytest.raises(AssertionError, match="outside Batch 1"):
        check_no_leakage(df, dataclasses.replace(s, fit_idx=np.append(s.fit_idx, s.test_idx[0])))
    with pytest.raises(AssertionError, match="overlap"):
        check_no_leakage(df, dataclasses.replace(s, calib_idx=np.append(s.calib_idx, s.fit_idx[0])))


@pytest.mark.parametrize("name", ["Mahalanobis", "kNN (k=5)", "One-Class SVM", "Isolation Forest"])
def test_future_batches_cannot_influence_model(df, X, name):
    """Corrupting every Batch 2-10 feature must not change the threshold or calibration."""
    s = make_split(df, 4, 0)
    X_bad = X.copy()
    X_bad[s.test_idx] = X_bad[s.test_idx] * 1000 + 12345.0
    r1 = run_detector(name, BASELINES[name], df, X, s)
    r2 = run_detector(name, BASELINES[name], df, X_bad, s)
    assert r1.threshold == r2.threshold
    assert r1.calib_fpr == r2.calib_fpr
    assert not np.allclose(r1.test_scores, r2.test_scores)  # test data really was used for scoring


def test_random_detector_is_at_chance(df, X):
    s = make_split(df, 2, 0)
    pooled = evaluate(run_detector("Random", BASELINES["Random (sanity)"], df, X, s))
    pooled = pooled[pooled["batch"] == "pooled"].iloc[0]
    assert abs(pooled["auroc"] - 0.5) < 0.03
    assert abs(pooled["auprc"] - pooled["novel_prevalence"]) < 0.03


def test_ocsvm_far_away_samples_are_exact_ties():
    from flyodor.baselines import OCSVMDetector
    rng = np.random.default_rng(0)
    det = OCSVMDetector(nu=0.05, gamma_mult=10.0).fit(rng.normal(size=(200, 20)))
    far = det.score(rng.normal(loc=50.0, size=(30, 20)))   # far outside the kernel's reach
    assert np.unique(far).size == 1


def test_calibration_fpr_near_target(df, X):
    s = make_split(df, 3, 0)
    r = run_detector("kNN", BASELINES["kNN (k=5)"], df, X, s)
    assert r.calib_fpr <= C.TARGET_FPR + 0.01
