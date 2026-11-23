# Phase 2 summary: protocol, preprocessing and baselines

**Status:** code, tests and the 20-seed baseline run are complete. Full tables
are in [`reports/phase2_baselines.md`](../reports/phase2_baselines.md), and every
number there comes from `results/baselines.csv`.

## What was built

| Item | File |
|---|---|
| P1 / P2 preprocessing, fitted on FIT rows only | `src/biomorph/preprocess.py` |
| Chronological leave-one-gas-out split (70/30 stratified, seeded) | `src/biomorph/protocol.py` |
| 4 label-free and 2 class-aware detectors, higher score = more novel | `src/biomorph/baselines.py` |
| Pseudo-novelty tuning, fit, threshold, freeze, test | `src/biomorph/experiment.py` |
| AUROC, AUPRC, novel share, FPR/TPR at θ, low-n flag | `src/biomorph/metrics.py` |
| Leakage tests, including the perturbation test | `tests/test_protocol.py` |
| Runner (5 gases × 20 seeds × 6 methods × 2 preps, ~8 min on 12 cores) | `scripts/run_baselines.py` |

## Decisions taken

- **No clipping.** The signed log already compresses the tails.
- **Baselines also use 20 seeds.** The seed changes the FIT/CALIB split, which
  makes paired fly-versus-baseline comparisons possible in Phase 5.
- **Pseudo-novel samples in tuning** are all Batch-1 FIT and CALIB rows of the
  left-out familiar gas g′. The roadmap said FIT rows only. Using both gives
  more samples, and the model in that fold never sees g′, so nothing leaks.
- **Unequal grid sizes:** Mahalanobis 1, kNN 4, Isolation Forest 6, OC-SVM 12.
  The fly model should get a grid of about 4–12 configs.

## Measured observations (P1, label-free)

1. **The pipeline works where there is no drift.** On the Batch-1 reference,
   kNN and OC-SVM reach AUROC 0.91–0.99 for 4 of the 5 gases.
2. **The frozen threshold collapses under drift.** For kNN, OC-SVM and
   Mahalanobis, FPR rises from about 0.06 in Batch 1 to 0.64–0.81 in batch 2,
   and to about 0.96–1.00 from batch 3 onward. Almost every later sample scores
   as "unfamiliar". Isolation Forest stays lower (0.13–0.77) but also detects
   less.
3. **Ranking quality degrades unevenly.** Mean AUROC over batches 2–10 ranges
   from 0.33 to 0.86, depending on gas and method. No single method is best
   for every gas.
4. **AUROC often falls below 0.5.** Examples: Ethylene in batches 8–9, and
   Acetaldehyde in batch 3. In these batches the held-out gas scores as *more
   familiar* than the drifted familiar gases. A plausible reading is that
   familiar gases drifted further from their Batch-1 positions than the
   held-out gas sits from Batch-1 data, but concentration differences between
   batches could produce the same pattern. This has not been tested.
5. **P2 (L2 normalisation) does not help across the board.** It improves
   Acetaldehyde and Ammonia for some methods and hurts Ethanol and Ethylene.
   For kNN, OC-SVM and Mahalanobis it delays the FPR collapse (batches 3–5:
   about 0.7–0.8 instead of 0.98–1.00), but from batch 6 onward FPR is again
   0.97–1.00.

## What this means for Phases 3–5

- **Primary metric stays AUROC per batch.** FPR at a fixed θ is secondary. It
  mainly shows that a fixed threshold does not survive drift, for every method.
- **A fair H1 test is about ranking**: does the fly code keep familiar gases
  ranked below the novel gas better than raw kNN (A5)? Expect the fly model's
  frozen threshold to fail too, unless its score is bounded in a way that
  happens to help. If it does help, check why before claiming anything.
- **Phase 3 diagnostic:** the within-gas versus between-gas code-overlap plot
  per batch speaks directly to observation 4.
- **Phase 7 concentration check** (C5MK6M) is the best way to separate drift
  from concentration in observation 4.

## Team tasks

- [ ] Everyone: be able to explain, for one gas, why FPR rises over time and
      why AUROC can be below 0.5 (roadmap "done when").
- [ ] Check `results/baselines_tuning.csv`: is the acetaldehyde pseudo-fold
      noisier than the others?
- [ ] Update `docs/CONTRIBUTIONS.md`.
