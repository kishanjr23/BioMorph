"""Chronological leave-one-gas-out split.

For held-out gas G and seed:

    Batch 1, gases != G  ->  stratified 70% FIT / 30% CALIB   (seeded)
    Batch 1, gas G       ->  B1 reference (never trained on; optimistic, no drift)
    Batches 2-10         ->  TEST: gas G = unfamiliar (y=1), others familiar (y=0)

FIT   fits preprocessing, model parameters and (via pseudo-novelty inside FIT
      and CALIB) hyperparameters.
CALIB sets the decision threshold theta.
Nothing from batches 2-10 and nothing of gas G may enter FIT or CALIB.

The seed controls only the FIT/CALIB split here, and is shared by every method,
so all methods see identical data for a given (gas, seed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from .data import GAS_NAMES, N_BATCHES, DriftDataset

HELDOUT_GASES = (1, 2, 3, 4, 5)  # toluene (6) excluded: absent from batches 3-5
TRAIN_BATCH = 1
TEST_BATCHES = tuple(range(2, N_BATCHES + 1))
FIT_FRACTION = 0.7


@dataclass(frozen=True)
class Split:
    """Row indices into the full DriftDataset."""

    heldout: int
    seed: int
    fit: np.ndarray
    calib: np.ndarray
    b1_reference: np.ndarray  # Batch-1 rows of the held-out gas
    test: np.ndarray  # all rows of batches 2-10

    @property
    def heldout_name(self) -> str:
        return GAS_NAMES[self.heldout]

    def test_labels(self, ds: DriftDataset) -> np.ndarray:
        """1 = unfamiliar (held-out gas), 0 = familiar."""
        return (ds.gas[self.test] == self.heldout).astype(int)


def make_split(ds: DriftDataset, heldout: int, seed: int) -> Split:
    if heldout not in GAS_NAMES:
        raise ValueError(f"unknown gas {heldout}")
    in_b1 = ds.batch == TRAIN_BATCH
    familiar_b1 = np.flatnonzero(in_b1 & (ds.gas != heldout))
    fit, calib = train_test_split(
        familiar_b1,
        train_size=FIT_FRACTION,
        stratify=ds.gas[familiar_b1],
        random_state=seed,
        shuffle=True,
    )
    return Split(
        heldout=heldout,
        seed=seed,
        fit=np.sort(fit),
        calib=np.sort(calib),
        b1_reference=np.flatnonzero(in_b1 & (ds.gas == heldout)),
        test=np.flatnonzero(np.isin(ds.batch, TEST_BATCHES)),
    )
