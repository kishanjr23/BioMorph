"""Week 5: uncertainty, paired significance, the response-strength hypothesis, memory saturation,
and report figures.

A. Load the per-sample test scores saved by run_week4.py (results/benchmark_scores.npz) and
   verify that they reproduce every per-batch AUROC in results/benchmark_metrics.csv.
B. Two-level bootstrap (seeds x test samples within batch and class), paired across detectors:
   95% CIs for macro AUROC and familiar-gas FPR, and fly-vs-other paired differences with
   bootstrap p-values and Holm correction.
C. EXPLORATORY (hypothesis formed after Week-4 results): magnitude-sensitive detectors beat
   magnitude-blind ones where the held-out gas differs from the familiar gases in overall
   response strength. Test: across held-out gas x batch cells, correlate
       strength separability  = |AUROC(response strength alone) - 0.5| * 2
       advantage              = AUROC(kNN Euclidean) - AUROC(kNN cosine)
   (same algorithm; differs only in sensitivity to response magnitude).
D. Memory saturation of the full fly model: fraction of KC weights depressed to 0, and the
   share of each test sample's active KCs that are depressed.
E. Report figures -> results/figures/*.png
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from flyodor import config as C  # noqa: E402
from flyodor.data import load_all  # noqa: E402
from flyodor.protocol import make_split  # noqa: E402
from flyodor.registry import FLY_FULL, SPECS  # noqa: E402
from flyodor.stats import bootstrap_p, ci, holm, stratified_weights, weighted_auc  # noqa: E402

R_PER_SEED = 50                       # 50 x 20 seeds = 1,000 bootstrap replicates
OUT, FIG = C.RESULTS_DIR, C.RESULTS_DIR / "figures"
DETS = list(SPECS)
GASES = C.PRIMARY_HELDOUT_GASES
SEEDS = list(range(C.N_SEEDS))


# ------------------------------------------------------------------------------------------
def phase_a():
    """Per-sample test scores saved by run_week4.py (the very same runs as the Week-4 table)."""
    path = OUT / "benchmark_scores.npz"
    if not path.exists():
        raise SystemExit("results/benchmark_scores.npz missing: run scripts/run_week4.py first")
    z = np.load(path, allow_pickle=True)
    return z["scores"].item(), z["thr"].item(), z["sat"].item()


def verify_against_week4(scores, test_batch, novel_by_gas):
    """Recompute every per-batch AUROC from the saved scores and compare with the Week-4 CSV."""
    bm = pd.read_csv(OUT / "benchmark_metrics.csv", dtype={"batch": str})
    bm = bm[bm["batch"].isin([str(b) for b in C.TEST_BATCHES])].set_index(
        ["detector", "heldout_gas", "seed", "batch"])["auroc"]
    worst = 0.0
    for (name, g, seed), s in scores.items():
        for b in C.TEST_BATCHES:
            m = test_batch == b
            worst = max(worst, abs(roc_auc_score(novel_by_gas[g][m], s[m]) - bm[(name, g, seed, str(b))]))
    return worst


# ------------------------------------------------------------------------------------------
def phase_b(scores, thr, test_batch, novel_by_gas):
    rng = np.random.default_rng(20261127)
    n = len(test_batch)
    # auc[name][g] -> (20 seeds, R) macro AUROC; fpr likewise
    auc = {d: {g: np.zeros((len(SEEDS), R_PER_SEED)) for g in GASES} for d in DETS}
    fpr = {d: {g: np.zeros((len(SEEDS), R_PER_SEED)) for g in GASES} for d in DETS}
    batch_masks = {b: np.flatnonzero(test_batch == b) for b in C.TEST_BATCHES}
    for g in GASES:
        y_all = novel_by_gas[g]
        for si, seed in enumerate(SEEDS):
            groups = []
            for b, idx in batch_masks.items():
                groups += [idx[y_all[idx] == 0], idx[y_all[idx] == 1]]
            W = stratified_weights(rng, groups, n, R_PER_SEED)
            for d in DETS:
                s, t = scores[(d, g, seed)], thr[(d, g, seed)]
                a_sum, f_sum = 0.0, 0.0
                for b, idx in batch_masks.items():
                    Wb, yb, sb = W[:, idx], y_all[idx], s[idx]
                    a_sum = a_sum + weighted_auc(sb, yb, Wb)
                    fam = yb == 0
                    f_sum = f_sum + (Wb[:, fam] * (sb[fam] > t)).sum(1) / Wb[:, fam].sum(1)
                auc[d][g][si] = a_sum / len(batch_masks)
                fpr[d][g][si] = f_sum / len(batch_masks)
        print(f"  bootstrapped held-out {C.GAS_NAMES[g]}", flush=True)

    def point(d, g, metric):  # exact (non-bootstrap) estimate: mean over seeds, unit weights
        vals = []
        for seed in SEEDS:
            s, y = scores[(d, g, seed)], novel_by_gas[g]
            per_b = []
            for b, idx in batch_masks.items():
                if metric == "auroc":
                    per_b.append(roc_auc_score(y[idx], s[idx]))
                else:
                    fam = idx[y[idx] == 0]
                    per_b.append(float((s[fam] > thr[(d, g, seed)]).mean()))
            vals.append(np.mean(per_b))
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    rows = []
    for d in DETS:
        all_auc = np.mean([auc[d][g] for g in GASES], axis=0).ravel()
        all_fpr = np.mean([fpr[d][g] for g in GASES], axis=0).ravel()
        pts = {g: (point(d, g, "auroc"), point(d, g, "fpr")) for g in GASES}
        rows.append({"detector": d, "group": SPECS[d].group, "gas": "All five",
                     "auroc": np.mean([pts[g][0][0] for g in GASES]), "auroc_lo": ci(all_auc)[0],
                     "auroc_hi": ci(all_auc)[1], "fpr": np.mean([pts[g][1][0] for g in GASES]),
                     "fpr_lo": ci(all_fpr)[0], "fpr_hi": ci(all_fpr)[1]})
        for g in GASES:
            a, f = auc[d][g].ravel(), fpr[d][g].ravel()
            rows.append({"detector": d, "group": SPECS[d].group, "gas": C.GAS_NAMES[g],
                         "auroc": pts[g][0][0], "auroc_lo": ci(a)[0], "auroc_hi": ci(a)[1],
                         "auroc_seed_sd": pts[g][0][1], "fpr": pts[g][1][0],
                         "fpr_lo": ci(f)[0], "fpr_hi": ci(f)[1]})
    cis = pd.DataFrame(rows)

    prow = []
    others = [d for d in DETS if d != FLY_FULL]
    for scope in ["All five"] + [C.GAS_NAMES[g] for g in GASES]:
        block = []
        for d in others:
            if scope == "All five":
                diff = (np.mean([auc[FLY_FULL][g] for g in GASES], axis=0)
                        - np.mean([auc[d][g] for g in GASES], axis=0)).ravel()
            else:
                g = [k for k in GASES if C.GAS_NAMES[k] == scope][0]
                diff = (auc[FLY_FULL][g] - auc[d][g]).ravel()
            est = (cis[(cis.detector == FLY_FULL) & (cis.gas == scope)]["auroc"].iloc[0]
                   - cis[(cis.detector == d) & (cis.gas == scope)]["auroc"].iloc[0])
            lo, hi = ci(diff)
            block.append({"vs": d, "group": SPECS[d].group, "gas": scope, "diff": est,
                          "lo": lo, "hi": hi, "p": bootstrap_p(diff)})
        adj = holm([b["p"] for b in block])
        for b, a in zip(block, adj):
            b["p_holm"] = a
            b["significant"] = bool(a < 0.05)
        prow += block
    return cis, pd.DataFrame(prow)


# ------------------------------------------------------------------------------------------
def phase_c(df, test_idx, test_batch, novel_by_gas):
    dR = df[[f"S{s:02d}_dR" for s in range(1, C.N_SENSORS + 1)]].to_numpy()
    strength = np.log10(np.abs(dR).mean(axis=1))[test_idx]
    bm = pd.read_csv(OUT / "benchmark_metrics.csv", dtype={"batch": str})
    per = bm[bm["batch"].isin([str(b) for b in C.TEST_BATCHES])].groupby(
        ["detector", "heldout_gas", "batch"])["auroc"].mean()
    rows = []
    for g in GASES:
        y = novel_by_gas[g]
        for b in C.TEST_BATCHES:
            m = test_batch == b
            s_auc = roc_auc_score(y[m], strength[m])
            rows.append({
                "heldout": C.GAS_NAMES[g], "batch": b, "strength_auc": s_auc,
                "strength_separability": abs(s_auc - 0.5) * 2,
                "euclid_minus_cosine": per[("kNN (Euclidean)", g, str(b))] - per[("kNN (cosine)", g, str(b))],
                "mahalanobis_minus_fly": per[("Mahalanobis", g, str(b))] - per[(FLY_FULL, g, str(b))],
            })
    mag = pd.DataFrame(rows)
    rho1, p1 = spearmanr(mag["strength_separability"], mag["euclid_minus_cosine"])
    rho2, p2 = spearmanr(mag["strength_separability"], mag["mahalanobis_minus_fly"])
    # permutation test that respects the gas structure: shuffle batches within each gas
    rng = np.random.default_rng(7)
    null = []
    for _ in range(5000):
        sh = mag.groupby("heldout")["euclid_minus_cosine"].transform(lambda v: rng.permutation(v.values))
        null.append(spearmanr(mag["strength_separability"], sh).statistic)
    p_perm = float((np.abs(null) >= abs(rho1)).mean())
    test = {"rho_euclid_minus_cosine": rho1, "p_spearman": p1, "p_permutation_within_gas": p_perm,
            "rho_mahalanobis_minus_fly": rho2, "p_spearman_mahalanobis_fly": p2, "n_cells": len(mag),
            "note": "exploratory; hypothesis formed after Week-4 results; cells are not independent"}
    return mag, test


# ------------------------------------------------------------------------------------------
def phase_e(cis, paired, mag, test, bm):
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    col = {FLY_FULL: "#B5532A", "Mahalanobis": "#2F3D1C", "kNN (Euclidean)": "#7C8F4E",
           "One-Class SVM": "#C9A646", "Isolation Forest": "#8C6B3F", "kNN (cosine)": "#4F7A6B",
           "Random (sanity)": "#A9AE9C"}
    short = {FLY_FULL: "Fly model (full)", "Ablation: no expansion (m=128)": "− expansion",
             "Ablation: dense wiring": "− sparse wiring", "Ablation: no top-k (fixed threshold)": "− top-k",
             "Ablation: no plastic memory (KC nearest neighbour)": "− plastic memory"}

    # Fig 1: AUROC by batch
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    order = ["1 (ref)"] + [str(b) for b in C.TEST_BATCHES]
    for d, c in col.items():
        sub = bm[(bm.detector == d) & (bm.batch != "pooled")]
        m = sub.groupby(["batch", "seed"])["auroc"].mean().groupby("batch").mean().reindex(order)
        ax.plot(range(len(order)), m.values, "-o", ms=3.5, color=c, lw=2.4 if d == FLY_FULL else 1.2,
                label=d, zorder=3 if d == FLY_FULL else 2)
    ax.axhline(0.5, ls=":", c="grey", lw=1)
    ax.set_xticks(range(len(order)), ["B1\n(ref)"] + [f"B{b}" for b in C.TEST_BATCHES])
    ax.set_ylabel("AUROC (mean over 5 gases, 20 seeds)")
    ax.set_title("Novelty detection over 36 months: all detectors degrade after Batch 1", loc="left")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_auroc_by_batch.png")
    plt.close(fig)

    # Fig 2: forest plot of macro AUROC with 95% CI
    allg = cis[cis.gas == "All five"].sort_values("auroc")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, r in enumerate(allg.itertuples()):
        c = col.get(r.detector, "#D99A6C")
        ax.errorbar(r.auroc, i, xerr=[[r.auroc - r.auroc_lo], [r.auroc_hi - r.auroc]], fmt="o",
                    color=c, ms=5, capsize=2.5)
    ax.set_yticks(range(len(allg)), [short.get(d, d) for d in allg.detector])
    ax.axvline(0.5, ls=":", c="grey")
    ax.set_xlabel("Macro AUROC, Batches 2-10 (95% bootstrap CI: seeds x test samples)")
    ax.set_title("Overall performance with uncertainty", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_forest_auroc.png")
    plt.close(fig)

    # Fig 3: paired differences fly - other (all gases)
    pa = paired[paired.gas == "All five"].sort_values("diff")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, r in enumerate(pa.itertuples()):
        ax.errorbar(r.diff, i, xerr=[[r.diff - r.lo], [r.hi - r.diff]], fmt="o", capsize=2.5,
                    color="#B5532A" if r.significant else "#A9AE9C", ms=5)
    ax.set_yticks(range(len(pa)), [f"vs {short.get(d, d)}" + ("  *" if s else "")
                                   for d, s in zip(pa.vs, pa.significant)])
    ax.axvline(0, c="k", lw=0.8)
    ax.set_xlabel("Fly model minus other, macro AUROC (95% CI)   * Holm p < 0.05")
    ax.set_title("Paired comparisons: positive = fly model better", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_paired_differences.png")
    plt.close(fig)

    # Fig 4: ablations with CI
    abl = [FLY_FULL] + [d for d in SPECS if SPECS[d].group == "ablation"] + ["kNN (cosine)"]
    sub = cis[(cis.gas == "All five") & cis.detector.isin(abl)].set_index("detector").reindex(abl)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colors = ["#B5532A", "#D99A6C", "#E3BF96", "#9C5B3C", "#6E3F2A", "#4F7A6B"]
    ax.bar(range(len(abl)), sub["auroc"], color=colors,
           yerr=[sub["auroc"] - sub["auroc_lo"], sub["auroc_hi"] - sub["auroc"]], capsize=3)
    ax.axhline(0.5, ls=":", c="grey")
    ax.set_ylim(0.4, 0.7)
    ax.set_xticks(range(len(abl)), [short.get(d, "cosine kNN\n(control)") for d in abl], fontsize=8)
    ax.set_ylabel("Macro AUROC, Batches 2-10")
    ax.set_title("Biological ablations (95% CI)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_ablations.png")
    plt.close(fig)

    # Fig 5: response-strength hypothesis
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    gcol = {"Ethanol": "#2F3D1C", "Ethylene": "#7C8F4E", "Ammonia": "#9DB86A",
            "Acetaldehyde": "#C9A646", "Acetone": "#8C6B3F"}
    for gname, sub in mag.groupby("heldout"):
        ax.scatter(sub["strength_separability"], sub["euclid_minus_cosine"], s=26, color=gcol[gname],
                   label=gname, edgecolor="white", lw=0.5)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xlabel("How differently the held-out gas responds in overall strength\n"
                  "|AUROC(strength alone) - 0.5| x 2")
    ax.set_ylabel("AUROC: Euclidean kNN - cosine kNN")
    ax.set_title(f"Exploratory: Spearman rho = {test['rho_euclid_minus_cosine']:.2f}, "
                 f"permutation p = {test['p_permutation_within_gas']:.3f}", loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_strength_hypothesis.png")
    plt.close(fig)

    # Fig 6: false alarms by batch
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for d, c in col.items():
        if d == "Random (sanity)":
            continue
        sub = bm[(bm.detector == d) & (bm.batch != "pooled")]
        m = sub.groupby(["batch", "seed"])["fpr_familiar"].mean().groupby("batch").mean().reindex(order)
        ax.plot(range(len(order)), m.values * 100, "-o", ms=3.5, color=c, lw=2.4 if d == FLY_FULL else 1.2,
                label=d)
    ax.axhline(C.TARGET_FPR * 100, ls=":", c="grey", lw=1)
    ax.text(0.1, C.TARGET_FPR * 100 + 2, "target 5%", fontsize=7, color="grey")
    ax.set_xticks(range(len(order)), ["B1\n(ref)"] + [f"B{b}" for b in C.TEST_BATCHES])
    ax.set_ylabel("Familiar gases flagged as novel (%)")
    ax.set_title("Frozen thresholds do not transfer over time", loc="left")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_false_alarms.png")
    plt.close(fig)


# ------------------------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    df = load_all()
    split0 = make_split(df, GASES[0], 0)
    test_idx = split0.test_idx
    test_batch = df["batch"].to_numpy()[test_idx]
    novel_by_gas = {g: (df["gas_id"].to_numpy()[test_idx] == g).astype(int) for g in GASES}

    print("A. loading Week-4 per-sample scores...", flush=True)
    scores, thr, sat = phase_a()
    worst = verify_against_week4(scores, test_batch, novel_by_gas)
    print(f"   reproducibility vs Week 4: max |AUROC difference| = {worst:.2e}", flush=True)
    assert worst < 1e-9, "saved scores do not reproduce Week-4 results"

    print("B. bootstrap...", flush=True)
    cis, paired = phase_b(scores, thr, test_batch, novel_by_gas)
    cis.to_csv(OUT / "week5_ci.csv", index=False)
    paired.to_csv(OUT / "week5_paired.csv", index=False)

    print("C. response-strength hypothesis...", flush=True)
    mag, test = phase_c(df, test_idx, test_batch, novel_by_gas)
    mag.to_csv(OUT / "week5_strength.csv", index=False)

    print("D. saturation...", flush=True)
    satdf = pd.DataFrame([{"gas": C.GAS_NAMES[g], "seed": s, **v} for (g, s), v in sat.items()])
    sat_sum = satdf.groupby("gas")[["frac_w0", "frac_never_active"]].mean().round(4)
    fly_fam = []
    for g in GASES:
        y = novel_by_gas[g]
        for b in C.TEST_BATCHES:
            m = (test_batch == b) & (y == 0)
            fly_fam.append({"gas": C.GAS_NAMES[g], "batch": b, "mean_novelty_familiar": float(np.mean(
                [scores[(FLY_FULL, g, s)][m].mean() for s in SEEDS]))})
    sat_batch = pd.DataFrame(fly_fam).groupby("batch")["mean_novelty_familiar"].mean().round(4)

    print("E. figures...", flush=True)
    bm = pd.read_csv(OUT / "benchmark_metrics.csv", dtype={"batch": str})
    phase_e(cis, paired, mag, test, bm)

    summary = {
        "bootstrap": {"replicates": R_PER_SEED * len(SEEDS), "resampled": "seeds x test samples "
                      "(within batch and class); paired across detectors", "ci": "95% percentile"},
        "reproducibility_max_abs_diff": worst,
        "strength_hypothesis": test,
        "strength_separability_by_gas": mag.groupby("heldout")["strength_separability"].mean().round(3).to_dict(),
        "saturation_by_gas": sat_sum.to_dict(orient="index"),
        "fly_mean_novelty_of_familiar_by_batch": sat_batch.to_dict(),
        "fly_calib_mean_fpr": float(satdf["calib_fpr"].mean()),
        "runtime_s": round(time.time() - t0),
    }
    (OUT / "week5_summary.json").write_text(json.dumps(summary, indent=1, default=float))

    pd.set_option("display.width", 220)
    print("\n=== Macro AUROC, all five gases (95% CI) ===")
    a = cis[cis.gas == "All five"].sort_values("auroc", ascending=False)
    print(a[["detector", "auroc", "auroc_lo", "auroc_hi", "fpr", "fpr_lo", "fpr_hi"]].round(3).to_string(index=False))
    print("\n=== Paired: Fly(full) minus other, all five gases ===")
    print(paired[paired.gas == "All five"][["vs", "diff", "lo", "hi", "p", "p_holm", "significant"]]
          .round(4).to_string(index=False))
    print("\n=== Per-gas: significant fly differences (Holm within gas) ===")
    pg = paired[(paired.gas != "All five") & paired.significant]
    print(pg[["gas", "vs", "diff", "lo", "hi", "p_holm"]].round(3).to_string(index=False))
    print("\n=== Strength hypothesis (exploratory) ===")
    print(json.dumps(test, indent=1, default=float))
    print(mag.groupby("heldout")[["strength_separability", "euclid_minus_cosine"]].mean().round(3).to_string())
    print("\n=== Saturation (full fly model) ===")
    print(sat_sum.to_string())
    print("mean novelty score of FAMILIAR test samples by batch:\n", sat_batch.to_string())
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
