"""Load and validate the UCI Gas Sensor Array Drift Dataset.

Each line of batchN.dat is in sparse 'libsvm' style:
    <label>[;<concentration>] 1:<value> 2:<value> ... 128:<value>
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


def _parse_line(line: str) -> tuple[int, float, np.ndarray]:
    parts = line.split()
    head = parts[0]
    if ";" in head:
        label_s, conc_s = head.split(";", 1)
        label, conc = int(label_s), float(conc_s)
    else:
        label, conc = int(head), np.nan
    x = np.full(C.N_FEATURES, np.nan)
    for tok in parts[1:]:
        idx, val = tok.split(":")
        x[int(idx) - 1] = float(val)
    return label, conc, x


def load_batch(batch: int) -> pd.DataFrame:
    path = C.RAW_DIR / f"batch{batch}.dat"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset into data/raw/ (see README)."
        )
    labels, concs, rows = [], [], []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                label, conc, x = _parse_line(line)
                labels.append(label)
                concs.append(conc)
                rows.append(x)
    df = pd.DataFrame(np.vstack(rows), columns=C.FEATURE_NAMES)
    df.insert(0, "batch", batch)
    df.insert(1, "gas_id", labels)
    df.insert(2, "gas", [C.GAS_NAMES[g] for g in labels])
    df.insert(3, "concentration", concs)
    df.insert(4, "row_in_batch", np.arange(len(df)))
    return df


def load_all() -> pd.DataFrame:
    """All 10 batches in chronological order."""
    return pd.concat(
        [load_batch(b) for b in range(1, C.N_BATCHES + 1)], ignore_index=True
    )


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[C.FEATURE_NAMES].to_numpy(dtype=float)


def counts_table(df: pd.DataFrame) -> pd.DataFrame:
    """Samples per batch (rows) x gas (columns)."""
    t = pd.crosstab(df["batch"], df["gas"]).reindex(
        columns=list(C.GAS_NAMES.values()), fill_value=0
    )
    t["Total"] = t.sum(axis=1)
    return t


@dataclass
class ValidationReport:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append((name, ok, detail))

    @property
    def all_passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.checks, columns=["check", "passed", "detail"])


def validate(df: pd.DataFrame) -> ValidationReport:
    r = ValidationReport()
    n = len(df)
    r.add("Total samples", n == C.EXPECTED_TOTAL_SAMPLES,
          f"{n} (expected {C.EXPECTED_TOTAL_SAMPLES})")

    batches = sorted(int(b) for b in df["batch"].unique())
    r.add("Batches present", batches == list(range(1, C.N_BATCHES + 1)),
          f"{batches}")

    X = feature_matrix(df)
    r.add("Feature count", X.shape[1] == C.N_FEATURES, f"{X.shape[1]} columns")
    n_missing = int(np.isnan(X).sum())
    r.add("No missing feature values", n_missing == 0, f"{n_missing} missing")
    n_inf = int(np.isinf(X).sum())
    r.add("No infinite values", n_inf == 0, f"{n_inf} infinite")

    gases = sorted(int(g) for g in df["gas_id"].unique())
    r.add("Six gas classes", gases == list(C.GAS_NAMES), f"labels {gases}")

    ct = counts_table(df)
    tol_gap = [int(b) for b in ct.index if ct.loc[b, "Toluene"] == 0]
    r.add("Toluene gap (batches 3, 4, 5)", tol_gap == [3, 4, 5],
          f"batches with zero toluene: {tol_gap}")

    primary_missing = [
        (int(b), C.GAS_NAMES[g]) for b in ct.index for g in C.PRIMARY_HELDOUT_GASES
        if ct.loc[b, C.GAS_NAMES[g]] == 0
    ]
    r.add("Primary gases present in every batch", not primary_missing,
          "all present" if not primary_missing else f"missing: {primary_missing}")

    dup = int(df.duplicated(subset=C.FEATURE_NAMES).sum())
    r.add("Duplicate feature rows (informational)", True, f"{dup} exact duplicates")
    return r
