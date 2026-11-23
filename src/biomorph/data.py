"""Loading and validating the UCI Gas Sensor Array Drift Dataset.

Dataset: Vergara et al., UCI ML Repository, https://doi.org/10.24432/C5RP6W

File format (libsvm style), one measurement per line:
    <label> 1:<value> 2:<value> ... 128:<value>

Label codes (from the UCI dataset page):
    1 Ethanol, 2 Ethylene, 3 Ammonia, 4 Acetaldehyde, 5 Acetone, 6 Toluene

Feature layout: 16 sensors x 8 features, sensor-major. Feature j (1-based)
belongs to sensor ceil(j / 8). Within each sensor the 8 features are:
    0 dR          steady-state resistance change
    1 |dR|norm    normalized steady-state feature (ratio to baseline)
    2-4 ema_rise  max of exponential moving average of rising transient,
                  alpha = 0.001, 0.01, 0.1 (in that order)
    5-7 ema_decay min of EMA of decaying transient, same alphas and order
Order and alpha values confirmed from the UCI dataset page (Nov 2026).
The `a1/a2/a3` suffixes in FEATURE_KINDS mean alpha 0.001/0.01/0.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_BATCHES = 10
N_SENSORS = 16
FEATURES_PER_SENSOR = 8
N_FEATURES = N_SENSORS * FEATURES_PER_SENSOR  # 128
EXPECTED_TOTAL = 13_910

GAS_NAMES = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene",
}

# Months combined into each batch (UCI dataset page).
BATCH_MONTHS = {
    1: "1-2",
    2: "3, 4, 8, 9, 10",
    3: "11-13",
    4: "14-15",
    5: "16",
    6: "17-20",
    7: "21",
    8: "22-23",
    9: "24, 30",
    10: "36",
}

FEATURE_KINDS = [
    "dR",
    "dR_norm",
    "ema_rise_a1",
    "ema_rise_a2",
    "ema_rise_a3",
    "ema_decay_a1",
    "ema_decay_a2",
    "ema_decay_a3",
]

FEATURE_NAMES = [
    f"s{s + 1:02d}_{kind}" for s in range(N_SENSORS) for kind in FEATURE_KINDS
]

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "Dataset"


@dataclass(frozen=True)
class DriftDataset:
    """All measurements, in file order within each batch.

    X      : (n, 128) float64 feature matrix
    gas    : (n,) int gas label 1..6
    batch  : (n,) int batch number 1..10
    index  : (n,) int row position inside its batch file (0-based)
    """

    X: np.ndarray
    gas: np.ndarray
    batch: np.ndarray
    index: np.ndarray

    def __len__(self) -> int:
        return len(self.gas)

    def subset(self, mask: np.ndarray) -> "DriftDataset":
        return DriftDataset(self.X[mask], self.gas[mask], self.batch[mask], self.index[mask])

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.X, columns=FEATURE_NAMES)
        df.insert(0, "row_in_batch", self.index)
        df.insert(0, "gas_name", [GAS_NAMES[g] for g in self.gas])
        df.insert(0, "gas", self.gas)
        df.insert(0, "batch", self.batch)
        return df


def parse_line(line: str, n_features: int = N_FEATURES) -> tuple[int, np.ndarray]:
    """Parse one libsvm-style line into (label, feature vector).

    Raises ValueError on malformed lines, unknown labels, duplicate or
    missing feature indices, so silent corruption cannot slip through.
    """
    parts = line.split()
    if not parts:
        raise ValueError("empty line")
    label_token = parts[0].split(";")[0]  # tolerate "label;concentration" variant
    label = int(float(label_token))
    if label not in GAS_NAMES:
        raise ValueError(f"unknown gas label {label}")

    x = np.full(n_features, np.nan)
    for tok in parts[1:]:
        idx_s, val_s = tok.split(":")
        idx = int(idx_s)
        if not 1 <= idx <= n_features:
            raise ValueError(f"feature index {idx} out of range")
        if not np.isnan(x[idx - 1]):
            raise ValueError(f"duplicate feature index {idx}")
        x[idx - 1] = float(val_s)
    if np.isnan(x).any():
        missing = (np.flatnonzero(np.isnan(x)) + 1).tolist()
        raise ValueError(f"missing feature indices {missing[:5]}...")
    return label, x


def load_batch(batch: int, data_dir: Path | str = DEFAULT_DATA_DIR) -> DriftDataset:
    path = Path(data_dir) / f"batch{batch}.dat"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    labels, rows = [], []
    with path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                label, x = parse_line(line)
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line_no}: {exc}") from exc
            labels.append(label)
            rows.append(x)
    n = len(labels)
    return DriftDataset(
        X=np.vstack(rows),
        gas=np.asarray(labels, dtype=int),
        batch=np.full(n, batch, dtype=int),
        index=np.arange(n),
    )


def load_all(data_dir: Path | str = DEFAULT_DATA_DIR, validate: bool = True) -> DriftDataset:
    parts = [load_batch(b, data_dir) for b in range(1, N_BATCHES + 1)]
    ds = DriftDataset(
        X=np.vstack([p.X for p in parts]),
        gas=np.concatenate([p.gas for p in parts]),
        batch=np.concatenate([p.batch for p in parts]),
        index=np.concatenate([p.index for p in parts]),
    )
    if validate:
        validate_dataset(ds)
    return ds


def validate_dataset(ds: DriftDataset) -> None:
    """Check the facts our experimental design relies on. Raises AssertionError."""
    assert len(ds) == EXPECTED_TOTAL, f"expected {EXPECTED_TOTAL} rows, got {len(ds)}"
    assert ds.X.shape[1] == N_FEATURES, f"expected {N_FEATURES} features"
    assert np.isfinite(ds.X).all(), "non-finite feature values present"
    assert set(np.unique(ds.batch)) == set(range(1, N_BATCHES + 1))
    assert set(np.unique(ds.gas)) <= set(GAS_NAMES)
    # Toluene gap documented in the project plan: no toluene in batches 3-5.
    for b in (3, 4, 5):
        n_tol = int(((ds.batch == b) & (ds.gas == 6)).sum())
        assert n_tol == 0, f"expected no toluene in batch {b}, found {n_tol}"


def counts_table(ds: DriftDataset) -> pd.DataFrame:
    """Batch x gas sample-count table with totals."""
    df = pd.crosstab(
        pd.Series(ds.batch, name="batch"),
        pd.Series([GAS_NAMES[g] for g in ds.gas], name="gas"),
    )
    df = df.reindex(columns=[GAS_NAMES[k] for k in sorted(GAS_NAMES)], fill_value=0)
    df["Total"] = df.sum(axis=1)
    df.loc["Total"] = df.sum(axis=0)
    return df
