"""Week 4: fly model + ablations + tuned baselines, chronological leave-one-gas-out.

1. For each held-out gas, tune every tunable detector on Batch 1 only (registry.tune).
   Ablations inherit the tuned full-fly-model settings, so each removes exactly one component.
2. For each held-out gas x seed: fit on Batch-1 familiar data, freeze, score Batches 2-10.

Outputs (results/):
  tuning.json                  chosen settings + inner Batch-1 AUROC for every grid point
  benchmark_metrics.csv        one row per detector x held-out gas x seed x batch
  benchmark_run_info.json      settings and library versions
"""
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from flyodor import config as C  # noqa: E402
from flyodor.data import feature_matrix, load_all  # noqa: E402
from flyodor.metrics import evaluate  # noqa: E402
from flyodor.protocol import make_split, run_detector  # noqa: E402
from flyodor.registry import FLY_FULL, SPECS, tune  # noqa: E402


def main() -> None:
    df = load_all()
    X = feature_matrix(df)
    t0 = time.time()

    # ---------------- 1. Batch-1-only tuning ----------------
    chosen: dict[int, dict[str, dict]] = {}
    tuning_log: dict[str, dict] = {}
    for gas in C.PRIMARY_HELDOUT_GASES:
        chosen[gas], tuning_log[C.GAS_NAMES[gas]] = {}, {}
        for name, spec in SPECS.items():
            if spec.grid is None:
                continue
            best, table = tune(df, X, spec, gas)
            chosen[gas][name] = best
            tuning_log[C.GAS_NAMES[gas]][name] = {
                "chosen": best, "inner_auroc": max(a for _, a in table),
                "grid": [{"params": p, "inner_auroc": round(a, 4)} for p, a in table]}
        for name, spec in SPECS.items():
            if spec.grid is None:
                chosen[gas][name] = chosen[gas][FLY_FULL]
        fly = tuning_log[C.GAS_NAMES[gas]][FLY_FULL]
        print(f"tuned for held-out {C.GAS_NAMES[gas]:<13} fly={fly['chosen']} "
              f"inner AUROC={fly['inner_auroc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.RESULTS_DIR / "tuning.json").write_text(json.dumps(tuning_log, indent=1))

    # ---------------- 2. chronological evaluation ----------------
    frames, scores, thr, sat = [], {}, {}, {}
    for gas in C.PRIMARY_HELDOUT_GASES:
        for seed in range(C.N_SEEDS):
            split = make_split(df, gas, seed)
            for name, spec in SPECS.items():
                built = []

                def factory(s, spec=spec, params=chosen[gas][name], built=built):
                    d = spec.build(s, params)
                    built.append(d)
                    return d
                r = run_detector(name, factory, df, X, split)
                out = evaluate(r)
                out["group"] = spec.group
                frames.append(out)
                scores[(name, gas, seed)] = r.test_scores          # full float64 precision
                thr[(name, gas, seed)] = r.threshold
                if name == FLY_FULL:
                    w = built[0].memory_.w_
                    sat[(gas, seed)] = {"frac_w0": float((w == 0).mean()),
                                        "frac_never_active": float((w == 1).mean()),
                                        "calib_fpr": r.calib_fpr}
        print(f"evaluated held-out {C.GAS_NAMES[gas]:<13} ({time.time() - t0:.0f}s)", flush=True)

    res = pd.concat(frames, ignore_index=True)
    res["heldout_name"] = res["heldout_gas"].map(C.GAS_NAMES)
    res.to_csv(C.RESULTS_DIR / "benchmark_metrics.csv", index=False)
    # per-sample test scores for Week-5 statistics (same runs -> identical numbers by construction)
    np.savez_compressed(C.RESULTS_DIR / "benchmark_scores.npz", scores=np.array(scores, dtype=object),
                        thr=np.array(thr, dtype=object), sat=np.array(sat, dtype=object))
    (C.RESULTS_DIR / "benchmark_run_info.json").write_text(json.dumps({
        "seeds": list(range(C.N_SEEDS)), "heldout_gases": C.PRIMARY_HELDOUT_GASES,
        "train_batch": C.TRAIN_BATCH, "test_batches": C.TEST_BATCHES,
        "calib_fraction": C.CALIB_FRACTION, "target_fpr": C.TARGET_FPR,
        "detectors": {n: s.group for n, s in SPECS.items()},
        "chosen_params": {C.GAS_NAMES[g]: v for g, v in chosen.items()},
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "runtime_s": round(time.time() - t0),
    }, indent=1))

    # ---------------- 3. summary ----------------
    later = res[res["batch"].isin([str(b) for b in C.TEST_BATCHES])]
    macro = later.groupby(["detector", "heldout_name", "seed"])[
        ["auroc", "auprc", "fpr_familiar", "detection_rate"]].mean().reset_index()
    ref = res[res["batch"] == "1 (ref)"].groupby("detector")["auroc"].mean().rename("B1_ref_auroc")
    summ = macro.groupby("detector")[["auroc", "auprc", "fpr_familiar", "detection_rate"]].mean()
    summ = summ.join(ref).sort_values("auroc", ascending=False).round(3)
    pd.set_option("display.width", 200)
    print("\n=== Batches 2-10 (macro over batches), mean over 5 gases x 20 seeds ===")
    print(summ.to_string())
    print("\n=== Macro AUROC by held-out gas ===")
    print(macro.pivot_table(index="detector", columns="heldout_name", values="auroc").round(3).to_string())

    fly = macro[macro["detector"] == FLY_FULL].set_index(["heldout_name", "seed"])["auroc"]
    rows = []
    for name in SPECS:
        if name == FLY_FULL:
            continue
        other = macro[macro["detector"] == name].set_index(["heldout_name", "seed"])["auroc"]
        d = (fly - other).dropna()
        rows.append({"vs": name, "mean_diff": d.mean(), "sd": d.std(),
                     "fly_better_frac": (d > 0).mean(),
                     "gases_fly_better": int((d.groupby(level=0).mean() > 0).sum())})
    print("\n=== Paired: Fly(full) minus other, macro AUROC (100 gas x seed pairs) ===")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
