"""Central configuration: paths, dataset constants, and protocol settings.

Everything an experiment depends on lives here so runs are reproducible.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "Dataset"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---- Dataset constants (UCI Gas Sensor Array Drift Dataset, DOI 10.24432/C5RP6W) ----
N_BATCHES = 10
N_SENSORS = 16
FEATURES_PER_SENSOR = 8
N_FEATURES = N_SENSORS * FEATURES_PER_SENSOR  # 128
EXPECTED_TOTAL_SAMPLES = 13_910

GAS_NAMES = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene",
}

# Per-sensor feature order (Vergara et al., 2012):
#   dR      : steady-state resistance change
#   |dR|n   : normalised steady-state change
#   EMAi_a  : max of exponential moving average on the rising (adsorption) phase, alpha = a
#   EMAd_a  : min of exponential moving average on the decaying (desorption) phase, alpha = a
FEATURE_KINDS = [
    "dR", "dR_norm",
    "EMAi_0.001", "EMAi_0.01", "EMAi_0.1",
    "EMAd_0.001", "EMAd_0.01", "EMAd_0.1",
]
FEATURE_NAMES = [
    f"S{s:02d}_{kind}" for s in range(1, N_SENSORS + 1) for kind in FEATURE_KINDS
]

# Approximate collection months per batch, as listed in the UCI dataset description.
BATCH_MONTHS = {
    1: "1-2", 2: "3-10", 3: "11-13", 4: "14-15", 5: "16",
    6: "17-20", 7: "21", 8: "22-23", 9: "24-30", 10: "36",
}

# ---- Experimental protocol ----
TRAIN_BATCH = 1                      # early calibration period
TEST_BATCHES = list(range(2, 11))    # later, drift-affected periods
PRIMARY_HELDOUT_GASES = [1, 2, 3, 4, 5]  # toluene (6) excluded: absent from batches 3-5
CALIB_FRACTION = 0.3   # share of Batch-1 familiar samples reserved for threshold selection
TARGET_FPR = 0.05      # threshold = 95th percentile of calibration (familiar) scores
DEFAULT_SEED = 0
N_SEEDS = 20
