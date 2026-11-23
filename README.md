# BioMorph

**Drosophila mushroom-body-inspired familiarity detection under long-term electronic-nose sensor drift**

BT232TD Bioinspired Engineering, experiential-learning project.

## Research question

Does the Drosophila mushroom body's sparse odor representation and familiarity
memory improve the robustness of electronic-nose novelty detection under
long-term changes in sensor data?

This is a hypothesis under test. **No model has been trained yet and no result
is claimed.**

## Status

| Week | Task | Status |
|---|---|---|
| 1 | Dataset download, loader, validation, data audit, glossary, literature notes | Done ([summary](docs/PHASE1_SUMMARY.md)) |
| 2 | Chronological splits, conventional baselines | Not started |
| 3 | Sparse PN→KC encoder + top-k | Not started |
| 4 | Familiarity memory, ablations | Not started |
| 5 | 20+ seed evaluation, figures | Not started |
| 6 | Dashboard, report, slides | Not started |

## Dataset

UCI Gas Sensor Array Drift Dataset (Vergara et al.), https://doi.org/10.24432/C5RP6W.
13,910 measurements, 16 metal-oxide sensors × 8 features = 128 features,
6 gases, 10 chronological batches over 36 months. The raw data is not
committed. Download it with the script below; it verifies a SHA-256 checksum.

See [`reports/data_audit.md`](reports/data_audit.md) for counts per batch and
gas, feature ranges, and a first look at temporal shift.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/audit_data.py
python -m pytest
```

## Documentation

- [`docs/ROADMAP.md`](docs/ROADMAP.md): phase-by-phase plan
- [`docs/PHASE1_SUMMARY.md`](docs/PHASE1_SUMMARY.md): what Phase 1 established, and open team tasks
- [`docs/GLOSSARY.md`](docs/GLOSSARY.md): precise terminology (biology ↔ engineering)
- [`docs/literature/`](docs/literature/): paper notes with reading checklists

## Layout

```
src/biomorph/data.py      load + validate the 10 batch files
scripts/download_data.py  fetch dataset, verify checksum
scripts/audit_data.py     regenerate reports/data_audit.md and figures
tests/                    parser and dataset-fact tests
reports/                  generated audit report and figures
docs/                     roadmap, glossary, literature notes
```

## References

- Caron et al. (2013) Nature, https://doi.org/10.1038/nature12063
- Lin et al. (2014) Nat. Neurosci., https://doi.org/10.1038/nn.3660
- Dasgupta et al. (2018) PNAS, https://doi.org/10.1073/pnas.1814448115
- Vergara et al. (2012) Sens. Actuators B, https://doi.org/10.1016/j.snb.2012.01.074
