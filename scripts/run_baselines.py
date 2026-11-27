"""Week 2: conventional baselines under the chronological leave-one-gas-out protocol.

Writes results/baseline_metrics.csv (one row per detector x held-out gas x seed x batch)
and results/baseline_run_info.json (settings + library versions).
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
from flyodor.baselines import BASELINES  # noqa: E402
from flyodor.data import feature_matrix, load_all  # noqa: E402
from flyodor.metrics import evaluate  # noqa: E402
from flyodor.protocol import make_split, run_detector  # noqa: E402


def main() -> None:
    df = load_all()
    X = feature_matrix(df)
    seeds = list(range(C.N_SEEDS))
    frames, t0 = [], time.time()
    for gas in C.PRIMARY_HELDOUT_GASES:
        for seed in seeds:
            split = make_split(df, gas, seed)
            for name, factory in BASELINES.items():
                frames.append(evaluate(run_detector(name, factory, df, X, split)))
        print(f"held-out {C.GAS_NAMES[gas]:<13} done ({time.time() - t0:.0f}s)")

    res = pd.concat(frames, ignore_index=True)
    res["heldout_name"] = res["heldout_gas"].map(C.GAS_NAMES)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(C.RESULTS_DIR / "baseline_metrics.csv", index=False)
    (C.RESULTS_DIR / "baseline_run_info.json").write_text(json.dumps({
        "seeds": seeds, "heldout_gases": C.PRIMARY_HELDOUT_GASES,
        "train_batch": C.TRAIN_BATCH, "test_batches": C.TEST_BATCHES,
        "calib_fraction": C.CALIB_FRACTION, "target_fpr": C.TARGET_FPR,
        "detectors": list(BASELINES), "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
    }, indent=2))

    ref = res[res["batch"] == "1 (ref)"]
    print("\n=== No-drift reference (Batch 1): mean AUROC by held-out gas ===")
    print(ref.pivot_table(index="detector", columns="heldout_name", values="auroc").round(3).to_string())

    per_batch = res[~res["batch"].isin(["pooled", "1 (ref)"])]
    # macro = mean over the 9 test batches (each time period weighted equally), per seed
    macro = per_batch.groupby(["detector", "heldout_name", "seed"])[
        ["auroc", "auprc", "fpr_familiar", "detection_rate"]].mean().reset_index()
    summary = macro.groupby("detector")[["auroc", "auprc", "fpr_familiar", "detection_rate"]].agg(
        ["mean", "std"]).round(3)
    print("\n=== Macro-average over Batches 2-10, then over 5 held-out gases x 20 seeds ===")
    print(summary.to_string())
    print("\n=== Mean macro AUROC by held-out gas ===")
    print(macro.pivot_table(index="detector", columns="heldout_name", values="auroc").round(3).to_string())


if __name__ == "__main__":
    main()
