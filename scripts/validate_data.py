"""Week 1: load the dataset, run integrity checks, print per-batch class counts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flyodor import config as C  # noqa: E402
from flyodor.data import counts_table, load_all, validate  # noqa: E402


def main() -> int:
    df = load_all()
    report = validate(df)
    print("=== Validation ===")
    for name, ok, detail in report.checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print("\n=== Samples per batch x gas ===")
    ct = counts_table(df)
    ct.insert(0, "months", [C.BATCH_MONTHS[b] for b in ct.index])
    print(ct.to_string())
    C.RESULTS_DIR.mkdir(exist_ok=True)
    ct.to_csv(C.RESULTS_DIR / "counts_by_batch.csv")
    report.to_frame().to_csv(C.RESULTS_DIR / "validation_report.csv", index=False)
    print(f"\nSaved tables to {C.RESULTS_DIR}")
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
