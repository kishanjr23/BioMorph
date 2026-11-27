"""Does the KC encoder do what the biology-to-engineering mapping says it does?"""
import numpy as np
import pytest

from flyodor.encoder import SparseKCEncoder, overlap


@pytest.fixture
def enc():
    return SparseKCEncoder(n_inputs=128, n_kc=2000, conn_per_kc=7, active_frac=0.05, seed=3)


def test_wiring_is_sparse_and_binary(enc):
    assert set(np.unique(enc.W)) == {0.0, 1.0}
    np.testing.assert_array_equal(enc.W.sum(axis=1), 7)       # every KC has exactly 7 inputs
    assert (enc.W.sum(axis=0) > 0).all()                       # every input reaches some KC


def test_exactly_k_cells_active(enc):
    Z = np.random.default_rng(0).normal(size=(300, 128))
    codes = enc.encode(Z)
    assert codes.dtype == bool and codes.shape == (300, 2000)
    np.testing.assert_array_equal(codes.sum(axis=1), enc.k)
    assert enc.k == 100


def test_active_cells_are_the_most_driven(enc):
    Z = np.random.default_rng(1).normal(size=(50, 128))
    a, codes = enc.drive(Z), enc.encode(Z)
    for i in range(50):
        assert a[i, codes[i]].min() >= a[i, ~codes[i]].max()


def test_same_seed_same_wiring_different_seed_differs():
    a = SparseKCEncoder(128, seed=5)
    b = SparseKCEncoder(128, seed=5)
    c = SparseKCEncoder(128, seed=6)
    np.testing.assert_array_equal(a.W, b.W)
    assert not np.array_equal(a.W, c.W)


def test_similar_inputs_share_more_cells(enc):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1, 128))
    near = x + 0.1 * rng.normal(size=(1, 128))
    far = rng.normal(size=(1, 128))
    codes = enc.encode(np.vstack([x, near, far]))
    ov = overlap(codes[:1], codes[1:])[0]
    assert ov[0] > 0.7 and ov[1] < 0.3


def test_identical_input_overlap_is_one(enc):
    x = np.random.default_rng(4).normal(size=(1, 128))
    c = enc.encode(x)
    assert overlap(c, c)[0, 0] == pytest.approx(1.0)


def test_rejects_wrong_feature_count(enc):
    with pytest.raises(ValueError):
        enc.encode(np.zeros((2, 100)))
