"""Phase 3 tests: sparse encoder and simple familiarity memory."""

import numpy as np
import pytest

from biomorph.data import DEFAULT_DATA_DIR, DriftDataset, load_all
from biomorph.experiment import run_split
from biomorph.fly import FlyDetector
from biomorph.fly.encoder import FlyEncoder, on_off, top_k_binary
from biomorph.fly.memory import FamiliarityMemory
from biomorph.protocol import make_split

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 128))


def test_on_off_is_nonnegative_and_lossless():
    P = on_off(X)
    assert P.shape == (200, 256) and (P >= 0).all()
    np.testing.assert_array_equal(P[:, :128] - P[:, 128:], X)
    assert not ((P[:, :128] > 0) & (P[:, 128:] > 0)).any()


def test_exactly_s_inputs_per_kc_and_reproducible():
    e1, e2, e3 = FlyEncoder(seed=1), FlyEncoder(seed=1), FlyEncoder(seed=2)
    row_sums = np.asarray(e1.M_.sum(axis=1)).ravel()
    assert (row_sums == 6).all()
    assert e1.M_.max() == 1  # binary, no duplicate input within a KC
    assert (e1.M_ != e2.M_).nnz == 0
    assert (e1.M_ != e3.M_).nnz > 0


def test_exactly_k_active_and_binary():
    enc = FlyEncoder(seed=0)
    H = enc.encode(X)
    assert enc.k == 100
    assert set(np.unique(H)) <= {0, 1}
    assert (H.sum(axis=1) == 100).all()


def test_top_k_tie_break_is_deterministic():
    A = np.array([[1.0, 3.0, 3.0, 3.0, 0.0]])
    np.testing.assert_array_equal(top_k_binary(A, 2), [[0, 1, 1, 0, 0]])


def test_encoder_has_nothing_to_fit():
    # encoding one sample does not depend on which other samples are encoded
    enc = FlyEncoder(seed=0)
    np.testing.assert_array_equal(enc.encode(X)[:5], enc.encode(X[:5]))


def test_similar_inputs_get_overlapping_codes():
    enc = FlyEncoder(seed=0)
    near = X[:1] + 0.05 * rng.normal(size=(1, 128))
    h0, hn, hf = enc.encode(X[:1]), enc.encode(near), enc.encode(X[1:2])
    assert (h0 & hn).sum() > (h0 & hf).sum()


def test_memory_closed_form_and_order_independence():
    H = top_k_binary(rng.normal(size=(50, 40)), 4)
    m1 = FamiliarityMemory(0.8).fit(H)
    m2 = FamiliarityMemory(0.8).fit(H[::-1])
    np.testing.assert_allclose(m1.w_, 0.8 ** H.sum(axis=0))
    np.testing.assert_array_equal(m1.w_, m2.w_)
    # explicit sequential rule gives the same weights
    w = np.ones(40)
    for h in H:
        w[h == 1] *= 0.8
    np.testing.assert_allclose(m1.w_, w)


def test_memory_score_range_and_familiarity():
    Xtr = rng.normal(size=(300, 128))
    det = FlyDetector(eps=0.9, seed=0).fit(Xtr)
    s_seen = det.score(Xtr[:50])
    s_new = det.score(rng.normal(5, 1, size=(50, 128)))
    assert ((s_seen >= 0) & (s_seen <= 1)).all()
    assert s_new.mean() > s_seen.mean()


needs_data = pytest.mark.skipif(
    not (DEFAULT_DATA_DIR / "batch1.dat").exists(),
    reason="dataset not downloaded (python scripts/download_data.py)",
)


@needs_data
@pytest.mark.parametrize("prep", ["P1", "P2"])
def test_fly_future_batches_cannot_change_frozen_model(prep):
    ds = load_all()
    X_bad = ds.X.copy()
    fut = ds.batch >= 2
    X_bad[fut] = np.random.default_rng(1).normal(0, 1e5, size=(fut.sum(), 128))
    ds_bad = DriftDataset(X_bad, ds.gas, ds.batch, ds.index)
    _, _, m = run_split(ds, make_split(ds, 5, 3), "fly", prep)
    _, _, mb = run_split(ds_bad, make_split(ds_bad, 5, 3), "fly", prep)
    assert m.config == mb.config and m.theta == mb.theta
    np.testing.assert_array_equal(m.detector.memory_.w_, mb.detector.memory_.w_)
    probe = ds.X[ds.batch == 1][:50]
    np.testing.assert_array_equal(m.score(probe), mb.score(probe))


def test_top_k_code_is_invariant_to_per_sample_scaling():
    # Why fly results are identical under P1 and P2: P2 rescales each sample by
    # a positive factor, which scales all KC activities alike, so the top-k
    # winners do not change. Top-k acts as built-in gain normalisation.
    enc = FlyEncoder(seed=0)
    scale = rng.uniform(0.01, 100, size=(200, 1))
    np.testing.assert_array_equal(enc.encode(X), enc.encode(X * scale))
