"""Familiarity memory, fly detector, ablation fairness, and Week-4 leakage checks."""
import numpy as np
import pytest

from flyodor import config as C
from flyodor.flymodel import VARIANTS, FlyFamiliarityDetector
from flyodor.memory import FamiliarityMemory
from flyodor.protocol import make_split, run_detector
from flyodor.registry import FLY_FULL, SPECS, inner_auroc


# ---------------- memory rule ----------------
def test_memory_depresses_only_active_cells():
    codes = np.zeros((3, 6), bool)
    codes[0, [0, 1]] = codes[1, [1, 2]] = codes[2, [1]] = True
    m = FamiliarityMemory(beta=0.5).fit(codes)
    np.testing.assert_allclose(m.w_, [0.5, 0.125, 0.5, 1, 1, 1])  # w = beta ** n_active


def test_beta_zero_is_seen_before_memory():
    codes = np.zeros((1, 4), bool)
    codes[0, [0, 1]] = True
    m = FamiliarityMemory(beta=0.0).fit(codes)
    probe = np.array([[True, True, False, False], [False, False, True, True],
                      [True, False, True, False]])
    np.testing.assert_allclose(m.score(probe), [0.0, 1.0, 0.5])


def test_empty_code_is_maximally_novel():
    m = FamiliarityMemory(0.5).fit(np.ones((2, 4), bool))
    assert m.score(np.zeros((1, 4), bool))[0] == 1.0


def test_beta_must_be_below_one():
    with pytest.raises(ValueError):
        FamiliarityMemory(beta=1.0)


# ---------------- fly detector ----------------
@pytest.fixture(scope="module")
def toy():
    # Clustered like real gases: 3 familiar "gases" and 1 different novel one.
    # (Isotropic noise would use every KC equally, leaving a usage-based memory nothing to learn.)
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=2.0, size=(4, 128))
    fam = np.vstack([centers[i] + 0.3 * rng.normal(size=(67, 128)) for i in range(3)])
    fam = fam[rng.permutation(len(fam))]
    novel = centers[3] + 0.3 * rng.normal(size=(50, 128))
    return fam, novel


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_separates_obvious_novelty(toy, variant):
    fam, novel = toy
    # graded memory (beta=0.9); with beta=0 isotropic toy data saturates the memory (below)
    det = FlyFamiliarityDetector(seed=1, variant=variant, beta=0.9).fit(fam[:150])
    s_fam, s_nov = det.score(fam[150:]), det.score(novel)
    assert s_nov.mean() > s_fam.mean()


def test_binary_memory_saturates_with_many_diverse_samples(toy):
    """Documented limitation: with beta=0 every KC that fired once is silenced. When training
    data activates (almost) every KC, all scores collapse to 0 and nothing looks novel."""
    rng = np.random.default_rng(5)
    diverse = rng.normal(size=(400, 128))            # unstructured, covers every direction
    det = FlyFamiliarityDetector(seed=1, variant="no_wta", beta=0.0).fit(diverse)
    assert (det.memory_.w_ == 0).mean() > 0.99
    assert det.score(rng.normal(loc=2.0, size=(20, 128))).max() < 0.05


def test_training_samples_are_fully_familiar_with_beta_zero(toy):
    fam, _ = toy
    det = FlyFamiliarityDetector(seed=1, beta=0.0).fit(fam)
    np.testing.assert_allclose(det.score(fam), 0.0)


def test_ablations_hold_other_components_fixed(toy):
    fam, _ = toy
    full = FlyFamiliarityDetector(seed=3).fit(fam)
    noexp = FlyFamiliarityDetector(seed=3, variant="no_expansion").fit(fam)
    dense = FlyFamiliarityDetector(seed=3, variant="dense").fit(fam)
    nowta = FlyFamiliarityDetector(seed=3, variant="no_wta").fit(fam)
    assert full.encoder_.n_kc == dense.encoder_.n_kc == nowta.encoder_.n_kc == 2000
    assert noexp.encoder_.n_kc == 128
    assert (full.encoder_.W.sum(1) == 7).all()               # sparse wiring
    assert (dense.encoder_.W != 0).all()                      # every KC sees every input
    # no-top-k: average sparsity on training data matches the full model's k/m
    assert nowta.codes(fam).mean() == pytest.approx(0.05, abs=0.002)
    # ...but the number of active cells now varies per sample (no per-sample competition)
    assert nowta.codes(fam).sum(1).std() > 0
    np.testing.assert_array_equal(full.codes(fam).sum(1), 100)


# ---------------- protocol / tuning leakage ----------------
@pytest.mark.parametrize("name", [FLY_FULL, "Ablation: no top-k (fixed threshold)",
                                  "Ablation: no plastic memory (KC nearest neighbour)"])
def test_future_batches_cannot_influence_fly_models(df, X, name):
    s = make_split(df, 5, 0)
    X_bad = X.copy()
    X_bad[s.test_idx] = X_bad[s.test_idx] * 1000 + 12345.0
    factory = lambda seed: SPECS[name].build(seed, SPECS[FLY_FULL].grid[0])  # noqa: E731
    r1 = run_detector(name, factory, df, X, s)
    r2 = run_detector(name, factory, df, X_bad, s)
    assert r1.threshold == r2.threshold


def test_tuning_ignores_heldout_gas_and_future_batches(df, X):
    spec, p = SPECS["kNN (Euclidean)"], {"k": 5}
    a = inner_auroc(df, X, spec, p, heldout_gas=2)
    X_bad = X.copy()
    X_bad[(df["batch"] > 1).to_numpy() | (df["gas_id"] == 2).to_numpy()] = 9e9
    assert inner_auroc(df, X_bad, spec, p, heldout_gas=2) == a
