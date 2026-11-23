"""Week-1 data audit: counts, feature ranges, outliers, and temporal shift.

Usage:  python scripts/audit_data.py

Writes reports/data_audit.md and figures in reports/figures/.
Everything in the report is computed from the raw files; nothing is typed in.

The PCA view is fitted on Batch 1 only and later batches are projected into
it, mirroring the train-early / test-late protocol. It is a visual check of
distribution shift, not a modelling step.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biomorph.data import (  # noqa: E402
    BATCH_MONTHS,
    FEATURE_KINDS,
    GAS_NAMES,
    N_BATCHES,
    N_SENSORS,
    counts_table,
    load_all,
)

REPORT = ROOT / "reports" / "data_audit.md"
FIG_DIR = ROOT / "reports" / "figures"


def signed_log(x: np.ndarray) -> np.ndarray:
    """sign(x) * log(1 + |x|): compresses the huge dR range, keeps sign."""
    return np.sign(x) * np.log1p(np.abs(x))


def feature_kind_table(X: np.ndarray) -> pd.DataFrame:
    rows = []
    for k, name in enumerate(FEATURE_KINDS):
        cols = X[:, k::len(FEATURE_KINDS)]
        rows.append(
            {
                "feature kind": name,
                "min": cols.min(),
                "median": np.median(cols),
                "max": cols.max(),
                "fraction < 0": np.mean(cols < 0),
            }
        )
    return pd.DataFrame(rows)


def outlier_table(X: np.ndarray, batch: np.ndarray) -> pd.DataFrame:
    """Values beyond 6 robust SDs (MAD) of their own feature column."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    mad[mad == 0] = np.inf
    z = np.abs(X - med) / mad
    flagged = z > 6
    rows = []
    for k, name in enumerate(FEATURE_KINDS):
        f = flagged[:, k::len(FEATURE_KINDS)]
        rows.append(
            {
                "feature kind": name,
                "flagged values": int(f.sum()),
                "samples with >=1 flag": int(f.any(axis=1).sum()),
            }
        )
    df = pd.DataFrame(rows)
    per_batch = pd.Series(flagged.any(axis=1)).groupby(batch).mean()
    return df, per_batch


def shift_table(X: np.ndarray, gas: np.ndarray, batch: np.ndarray) -> pd.DataFrame:
    """Per gas, per batch: median over samples of mean signed-log dR across sensors.

    Computed within each gas so that a changing gas mix between batches does
    not masquerade as drift. Concentration still varies between batches, so
    this is shift, not proof of sensor ageing.
    """
    dr = signed_log(X[:, 0::len(FEATURE_KINDS)]).mean(axis=1)
    df = pd.DataFrame({"gas": [GAS_NAMES[g] for g in gas], "batch": batch, "dr": dr})
    return df.pivot_table(index="batch", columns="gas", values="dr", aggfunc="median")


PRIMARY_HELDOUT = [1, 2, 3, 4, 5]  # toluene (6) excluded: absent from batches 3-5


def heldout_tables(ct: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each planned held-out-gas experiment: Batch-1 training size and
    the share of unfamiliar (held-out) samples in every test batch."""
    gas_cols = [GAS_NAMES[g] for g in sorted(GAS_NAMES)]
    per_batch = ct.drop(index="Total")[gas_cols]
    train_rows, share = [], {}
    for g in PRIMARY_HELDOUT:
        name = GAS_NAMES[g]
        familiar = [c for c in gas_cols if c != name]
        b1 = per_batch.loc[1]
        train_rows.append(
            {
                "held-out gas": name,
                "Batch-1 familiar samples": int(b1[familiar].sum()),
                "smallest familiar gas in Batch 1": f"{b1[familiar].idxmin()} ({int(b1[familiar].min())})",
                "Batch-1 held-out samples (not trained on)": int(b1[name]),
            }
        )
        share[name] = per_batch[name] / per_batch.sum(axis=1)
    train = pd.DataFrame(train_rows).set_index("held-out gas")
    share_df = pd.DataFrame(share).loc[2:]
    share_df.index.name = "test batch"
    return train, share_df


def plot_counts(ct: pd.DataFrame) -> Path:
    data = ct.drop(index="Total", columns="Total")
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(data))
    for col in data.columns:
        ax.bar(data.index.astype(str), data[col], bottom=bottom, label=col)
        bottom += data[col].to_numpy()
    ax.set_xlabel("Batch (chronological)")
    ax.set_ylabel("Samples")
    ax.set_title("Samples per batch and gas")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    path = FIG_DIR / "counts_by_batch.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_shift(shift: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for col in shift.columns:
        ax.plot(shift.index, shift[col], marker="o", label=col)
    ax.set_xticks(range(1, N_BATCHES + 1))
    ax.set_xlabel("Batch (chronological)")
    ax.set_ylabel("median of mean signed-log dR")
    ax.set_title("Steady-state response over time, per gas")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    path = FIG_DIR / "dr_shift_by_gas.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pca(X: np.ndarray, gas: np.ndarray, batch: np.ndarray) -> tuple[Path, Path, np.ndarray]:
    Z = signed_log(X)
    train = batch == 1
    mu = Z[train].mean(axis=0)
    sd = Z[train].std(axis=0)
    sd[sd == 0] = 1.0
    Zs = (Z - mu) / sd
    # PCA fitted on Batch 1 only
    _, s, vt = np.linalg.svd(Zs[train] - Zs[train].mean(axis=0), full_matrices=False)
    P = Zs @ vt[:2].T
    explained = (s**2 / np.sum(s**2))[:2]

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(P[:, 0], P[:, 1], c=batch, cmap="viridis", s=4, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="Batch")
    ax.set_xlabel(f"PC1 ({explained[0]:.0%} of Batch-1 variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.0%} of Batch-1 variance)")
    ax.set_title("All batches projected onto Batch-1 PCA, coloured by batch")
    fig.tight_layout()
    p1 = FIG_DIR / "pca_by_batch.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5), sharex=True, sharey=True)
    for b, ax in zip(range(1, N_BATCHES + 1), axes.ravel()):
        m = batch == b
        for g, name in GAS_NAMES.items():
            mg = m & (gas == g)
            ax.scatter(P[mg, 0], P[mg, 1], s=4, alpha=0.6, label=name)
        ax.set_title(f"Batch {b} (months {BATCH_MONTHS[b]})", fontsize=9)
    axes[0, 0].legend(fontsize=7, markerscale=3)
    fig.suptitle("Batch-1 PCA projection, one panel per batch, coloured by gas")
    fig.tight_layout()
    p2 = FIG_DIR / "pca_by_gas_per_batch.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return p1, p2, explained


def md_table(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |"]
    lines.append("|" + "---|" * (len(cols) + 1))
    for idx, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append(format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_all(validate=True)  # raises if any documented fact is wrong
    ct = counts_table(ds)
    kinds = feature_kind_table(ds.X).set_index("feature kind")
    outl, outl_batch = outlier_table(ds.X, ds.batch)
    shift = shift_table(ds.X, ds.gas, ds.batch)
    f_counts = plot_counts(ct)
    f_shift = plot_shift(shift)
    f_pca1, f_pca2, explained = plot_pca(ds.X, ds.gas, ds.batch)

    b1 = ct.loc[1].drop("Total")
    smallest_gas = b1.idxmin()
    ho_train, ho_share = heldout_tables(ct)

    months = pd.DataFrame(
        {"months": [BATCH_MONTHS[b] for b in range(1, N_BATCHES + 1)]},
        index=pd.Index(range(1, N_BATCHES + 1), name="batch"),
    )
    outl_batch_df = outl_batch.rename("fraction of samples flagged").to_frame()
    outl_batch_df.index.name = "batch"

    rel = lambda p: p.relative_to(ROOT / "reports").as_posix()  # noqa: E731
    text = f"""# Data audit: UCI Gas Sensor Array Drift Dataset

Generated by `scripts/audit_data.py` on {date.today().isoformat()}. All numbers are
computed from the raw files (`data/raw/Dataset/batch1.dat` to `batch10.dat`).

## 1. Validation checks (all passed)

- Total measurements: **{len(ds)}** (expected 13,910)
- Features per measurement: **{ds.X.shape[1]}** ({N_SENSORS} sensors x {len(FEATURE_KINDS)} features)
- All values finite; no missing or duplicate feature indices
- Duplicate measurement rows: **{len(ds) - len(np.unique(ds.X, axis=0))}**
- Toluene samples in batches 3, 4, 5: **0, 0, 0** (gap confirmed)

## 2. Batch timeline

{md_table(months)}

## 3. Samples per batch and gas

{md_table(ct, ".0f")}

![counts]({rel(f_counts)})

Observations for the experimental design:

- Batch 1 (the training period) has only **{int(ct.loc[1, 'Total'])}** samples.
  The smallest gas there is **{smallest_gas}** with **{int(b1.min())}** samples.
- Some later batches have very few samples of a gas (e.g. batch 4: Ammonia
  {int(ct.loc[4, 'Ammonia'])}, Acetone {int(ct.loc[4, 'Acetone'])}). Per-batch metrics for those cells will be noisy.
- Batch 2 contains only **{int(ct.loc[2, 'Toluene'])}** toluene samples, so toluene is nearly absent
  from batches 2-5.
- The gas mix changes a lot between batches, so the share of "unfamiliar" samples
  differs per batch. AUPRC must be reported alongside that share.

### 3a. What each planned held-out-gas experiment looks like

Training uses Batch 1, familiar gases only (all gases except the held-out one,
toluene included):

{md_table(ho_train)}

Share of **unfamiliar** samples in each test batch. This is the chance level
of AUPRC for that batch, so AUPRC values are only comparable to this baseline:

{md_table(ho_share, ".2f")}

## 4. Feature value ranges

{md_table(kinds)}

- `dR` spans several orders of magnitude and is occasionally negative.
- All `ema_decay` features are (almost) always negative, by construction.
- Implication: some compressive transform (e.g. signed log) is needed before
  scaling, and the fly encoder's non-negative "PN" input must be designed with
  the negative decay features in mind. This is a preprocessing decision for Week 2-3.

## 5. Heavy tails (raw values more than 6 robust SDs from the column median)

{md_table(outl.set_index('feature kind'), ".0f")}

{md_table(outl_batch_df)}

The flagged fraction is large because raw features are strongly right-skewed:
gases were dosed from 5 to 1000 ppmv, so high-concentration samples sit far
from the median. **These are mostly not measurement errors.** The table says
raw-scale distances are dominated by a few large values, which is another
reason for a compressive transform. Nothing is removed. Any clipping rule
must be fitted on Batch-1 training data only and applied to every model.

## 6. Temporal shift (first look)

Median over samples of the mean signed-log steady-state response across the
16 sensors, computed **within each gas** so the changing gas mix does not
create a false trend:

{md_table(shift)}

![shift]({rel(f_shift)})

All batches projected onto a PCA fitted on Batch 1 only
(PC1 {explained[0]:.0%}, PC2 {explained[1]:.0%} of Batch-1 variance):

![pca by batch]({rel(f_pca1)})

![pca by gas per batch]({rel(f_pca2)})

**Caution:** batches also differ in gas concentration and sampling schedule
(see Vergara et al. 2012, Tables 1-2). Changes here show *temporal distribution
shift*; they do not by themselves prove sensor ageing.
"""
    REPORT.write_text(text)
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
