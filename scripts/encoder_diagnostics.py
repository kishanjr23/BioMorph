"""Week 3: verify the KC encoder on real sensor data (descriptive, not a novelty benchmark).

For each of 20 wiring seeds:
  * sparsity: active fraction per sample, per-KC lifetime activity, silent KCs
  * similarity preservation (Batch 1): Spearman(input cosine similarity, KC overlap)
  * gas structure (Batch 1): mean KC overlap within the same gas vs between gases
  * nearest-neighbour gas agreement per batch, KC space vs input space, using Batch 1
    as the reference memory (Batch 1 itself: leave-one-out)

Scaler is fitted on Batch 1 only. Writes results/encoder_diagnostics.json and
results/encoder_nn_by_batch.csv.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from flyodor import config as C  # noqa: E402
from flyodor.data import feature_matrix, load_all  # noqa: E402
from flyodor.encoder import SparseKCEncoder, overlap  # noqa: E402
from flyodor.preprocess import FrozenStandardizer  # noqa: E402


def main() -> None:
    df = load_all()
    X = feature_matrix(df)
    batch, gas = df["batch"].to_numpy(), df["gas_id"].to_numpy()
    b1 = batch == C.TRAIN_BATCH
    Z = FrozenStandardizer().fit(X[b1]).transform(X)
    Zb1, gb1 = Z[b1], gas[b1]
    n1 = len(Zb1)

    # ---- input-space reference (deterministic) ----
    unit = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    iu = np.triu_indices(n1, k=1)
    cos_b1 = (unit[b1] @ unit[b1].T)[iu]
    same_pair = (gb1[:, None] == gb1[None, :])[iu]

    def nn_agreement_input(mask):
        d = ((Z[mask][:, None, :] - Zb1[None, :, :]) ** 2).sum(-1)
        if mask is b1:
            np.fill_diagonal(d, np.inf)
        return gb1[d.argmin(axis=1)] == gas[mask]

    def nn_agreement_cosine(mask):
        sim = unit[mask] @ unit[b1].T
        if mask is b1:
            np.fill_diagonal(sim, -np.inf)
        return gb1[sim.argmax(axis=1)] == gas[mask]

    # Two input-space references: Euclidean, and cosine. KC overlap closely tracks cosine
    # similarity, so cosine is the fair control for "does sparse expansion add anything".
    rows = []
    for b in range(1, C.N_BATCHES + 1):
        m = b1 if b == 1 else batch == b
        rows.append({"batch": b, "space": "Input, Euclidean", "seed": -1,
                     "nn_same_gas": float(nn_agreement_input(m).mean())})
        rows.append({"batch": b, "space": "Input, cosine", "seed": -1,
                     "nn_same_gas": float(nn_agreement_cosine(m).mean())})

    summary = []
    for seed in range(C.N_SEEDS):
        enc = SparseKCEncoder(C.N_FEATURES, seed=seed)
        codes = enc.encode(Z)
        cb1 = codes[b1]
        life = cb1.mean(axis=0)                          # per-KC firing probability (Batch 1)
        ov_b1 = overlap(cb1, cb1)
        rho = spearmanr(cos_b1, ov_b1[iu]).statistic

        for b in range(1, C.N_BATCHES + 1):
            m = batch == b
            ov = ov_b1.copy() if b == 1 else overlap(codes[m], cb1)
            if b == 1:
                np.fill_diagonal(ov, -1)
            agree = gb1[ov.argmax(axis=1)] == gas[m]
            rows.append({"batch": b, "space": "KC code", "seed": seed,
                         "nn_same_gas": float(agree.mean())})

        summary.append({
            "seed": seed,
            "active_frac_per_sample": float(codes.sum(1).mean() / enc.n_kc),
            "silent_kc_frac_batch1": float((life == 0).mean()),
            "silent_kc_frac_all": float((codes.sum(0) == 0).mean()),
            "max_kc_lifetime_activity_batch1": float(life.max()),
            "spearman_cos_vs_overlap_batch1": float(rho),
            "overlap_within_gas_batch1": float(ov_b1[iu][same_pair].mean()),
            "overlap_between_gas_batch1": float(ov_b1[iu][~same_pair].mean()),
        })

    nn = pd.DataFrame(rows)
    s = pd.DataFrame(summary)
    maxz = pd.Series({int(b): float(np.median(np.abs(Z[batch == b]).max(axis=1)))
                      for b in range(1, C.N_BATCHES + 1)})
    out = {
        "encoder": {"n_kc": 2000, "conn_per_kc": 7, "active_frac": 0.05, "n_seeds": C.N_SEEDS},
        "mean": s.drop(columns="seed").mean().round(4).to_dict(),
        "sd": s.drop(columns="seed").std().round(4).to_dict(),
        "median_max_abs_z_by_batch": maxz.round(2).to_dict(),
    }
    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.RESULTS_DIR / "encoder_diagnostics.json").write_text(json.dumps(out, indent=2))
    nn.to_csv(C.RESULTS_DIR / "encoder_nn_by_batch.csv", index=False)

    print(json.dumps(out, indent=2))
    tbl = nn.groupby(["batch", "space"])["nn_same_gas"].agg(["mean", "std"]).unstack().round(3)
    print("\n=== 1-NN gas agreement with Batch-1 memory (Batch 1: leave-one-out) ===")
    print(tbl.to_string())


if __name__ == "__main__":
    main()
