# Phase 1 summary: understanding the data

**Status:** code and documentation complete. Three team-only items remain
(bottom of this page). They need people, not code.

## Phase 1 checklist

| Task | Status | Where |
|---|---|---|
| Download dataset with integrity check | ✅ | `scripts/download_data.py` (SHA-256 verified) |
| Inspect format and features | ✅ | `src/biomorph/data.py` docstring; feature order and α confirmed from UCI page |
| Verify sample counts | ✅ | 13,910 rows, 128 features; enforced by `validate_dataset()` and tests |
| Class distribution by batch | ✅ | `reports/data_audit.md` §3 |
| Document toluene gap | ✅ | audit §3; enforced in validation |
| Per-experiment training sizes and novel share | ✅ | audit §3a |
| Visualize temporal shift | ✅ | audit §6 (per-gas ΔR trend, Batch-1-fitted PCA) |
| Glossary | ✅ | `docs/GLOSSARY.md` |
| Dataset paper notes | ✅ partial | `docs/literature/vergara2012_dataset.md`. Paper paywalled; UCI facts verified; reading checklist provided |
| Dasgupta 2018 notes | ✅ partial | `docs/literature/dasgupta2018_novelty.md`. Abstract verified; full-text checklist provided |
| Roadmap for Phases 2–8 | ✅ | `docs/ROADMAP.md` |

## Key facts (all measured, see audit)

| Fact | Consequence for later phases |
|---|---|
| Batch 1 = 445 samples; acetaldehyde only 30 | Small training set; acetaldehyde is a weakly learned familiar gas in 4 of 5 experiments |
| Training sizes 347–415 per experiment | Memory capacity and covariance estimation are limited (use shrinkage) |
| Toluene: 0 in batches 3–5, 5 in batch 2 | Excluded as held-out gas; the familiar mix changes over time |
| Novel share per test batch ranges from 1% to 49% | AUPRC must be compared with that share; acetaldehyde batch 6 is low-n |
| ΔR up to ~670,000; decay features negative | Signed-log transform; ON/OFF channels for non-negative PN input |
| Heavy tails: 1–64% of samples per batch exceed 6 robust SDs on raw scale | Mostly genuine high concentrations (5–1000 ppmv), not errors. Nothing removed |
| Gas clusters move on a Batch-1-fitted PCA; per-gas ΔR levels shift by batch | Temporal shift is real, but concentration and schedule are confounded with time |

## Decisions deferred to Phase 2

1. Preprocessing: P1 (signed log + Batch-1 z-score) as primary, P2 (+ per-sample
   L2) as secondary, applied identically to all models.
2. FIT/CALIB split ratio (planned 70/30, stratified by gas).
3. Whether to clip extreme values (fitted on Batch-1 FIT only, if at all).

## Team-only items (please do and tick)

- [ ] **Everyone:** read `reports/data_audit.md` and be able to explain the
      counts table, §3a, and the PCA-per-batch figure.
- [ ] **One person:** read Vergara et al. 2012 (library access) and fill in
      the checklist in `docs/literature/vergara2012_dataset.md`.
- [ ] **One person:** read Dasgupta et al. 2018
      (https://europepmc.org/articles/PMC6304992) and fill in the checklist in
      `docs/literature/dasgupta2018_novelty.md`. Note every difference from our
      planned formulation.
- [ ] Start `docs/CONTRIBUTIONS.md`: who did what in Week 1.
