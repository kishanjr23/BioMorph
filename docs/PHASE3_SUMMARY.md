# Phase 3 summary: fly encoder (PN → KC → top-k) and MVP slice

**Status:** encoder, simple memory, tests (48 pass), diagnostics and the
acetone MVP slice are complete. Numbers: [`reports/phase3_encoder.md`](../reports/phase3_encoder.md)
and [`reports/phase3_mvp.md`](../reports/phase3_mvp.md).

## What was built

| Item | File |
|---|---|
| ON/OFF PN stage, sparse binary projection (m = 2000, s = 6), top-k (k = 100) | `src/biomorph/fly/encoder.py` |
| Familiarity memory w = eps^count, score = mean w of active KCs | `src/biomorph/fly/memory.py` |
| `FlyDetector` + registration in the shared pipeline as method `fly` | `src/biomorph/fly/__init__.py`, `experiment.py` |
| Encoder and memory tests, fly leakage perturbation test, scale-invariance test | `tests/test_fly.py` |
| Encoder diagnostics (4 figures + numbers) | `scripts/encoder_diagnostics.py` |
| MVP: acetone held out, fly vs kNN and Isolation Forest, 20 seeds | `scripts/run_mvp.py` |

The simple memory was pulled forward from Phase 4 because the MVP needs it.
Phase 4 still has to justify it against the paper and add the ablations.

## Encoder verification (measured)

- Exactly 100 of 2000 KCs active for all 13,910 samples; exactly 6 inputs per KC.
- Similarity is preserved: Spearman ρ = 0.943 between PN-input cosine and
  KC-code Jaccard (Batch-1 pairs). The relation is convex: similar inputs
  keep high overlap, dissimilar inputs fall to near-zero overlap.
- Gases produce visibly different codes in the raster.
- **Dead KCs:** 25% of KCs never fire in Batch 1 (461–504 over 5 seeds);
  36% never fire in a single experiment's FIT set. No KC fires in more than
  34% of Batch-1 samples.
- Only 5% of KCs are wired to both the ON and the OFF channel of the same
  feature, so the effective fan-in is 5.95.

## Two findings that change Phase 4

**1. Top-k makes the fly model immune to per-sample scaling.** Multiplying a
sample by any positive number scales all its KC activities equally, so the
same k KCs win. P2 (per-sample L2 normalisation) therefore has **no effect**
on the fly model; P1 and P2 results are identical (verified, and now a test).
Consequence: the fly model always carries built-in gain normalisation, which
loosely corresponds to APL feedback inhibition. **The fair H1 comparison is
fly vs. raw kNN under P2**, where both are scale-invariant. Under P1, part of
any fly advantage may simply be normalisation.

**2. eps is chosen at the edge of the grid.** Pseudo-novelty tuning (Batch 1
only) picked eps = 0.5, the smallest value, in 34 of 40 runs. The optimum may
lie lower. Extend the grid downward for Phase 4 (e.g. 0.1, 0.2, 0.3, 0.5, 0.8,
0.9). This decision is based on Batch-1 tuning only, not on test results.

## MVP slice: acetone held out (pipeline check, not a project result)

Mean over batches 2–10, mean ± SD over 20 seeds:

| prep | method | AUROC | FPR at θ |
|---|---|---|---|
| P1/P2 | fly | 0.590 ± 0.048 | 0.789 |
| P1 | kNN | 0.536 ± 0.033 | 0.956 |
| P2 | kNN | 0.578 ± 0.027 | 0.820 |
| P1 | Isolation Forest | 0.424 ± 0.036 | 0.399 |

Paired by seed: fly − kNN is +0.054 under P1 (15/20 seeds favour fly) but
only **+0.013 under P2 (12/20)**. Fly − Isolation Forest is +0.17 to +0.19
(20/20). Per batch, the fly model is not uniformly better: it falls to AUROC
0.29 in batch 10, where kNN reaches 0.62. Its frozen threshold also fails under
drift (FPR 0.79). On one gas, none of this supports a conclusion.

## Go/no-go (bars fixed before the run)

| check | result |
|---|---|
| G1 leakage tests pass (fly included) | ✅ 48/48 |
| G2 both classes in every test batch | ✅ |
| G3 exactly k active; dead KCs in FIT < 50% | ✅ max 36% |
| G4 seed SD of mean AUROC ≤ 0.05 | ✅ 0.048 (close to the bar) |

**Decision: go to Phase 4.**

## Drift in code space (diagnostic, all gases)

From `encoder_overlap_by_batch.png`: for each gas and batch, overlap with
Batch-1 codes of the same gas vs. of other gases.

- Ethylene and ammonia keep a clear same-gas margin through batch 10.
- **Acetaldehyde** codes overlap *less* with Batch-1 acetaldehyde than with
  other Batch-1 gases from batch 2 onward. In input space (cosine) that
  reversal appears only from batch 6. Top-k sharpens similarity, so the code
  can lose a gas's identity earlier than the raw input. The two measures are
  on different scales, so this compares the *sign* of the margin, not its size.
- **Acetone** loses its code margin from batch 5 onward; **ethanol** in batches 8–9.

This directly matters for H1: if drift moves a familiar gas far enough, sparse
codes may amplify the problem rather than absorb it. Phase 4 ablations
(A1 no expansion, A3 no top-k) test this.

## Team tasks

- [ ] Read Dasgupta et al. 2018: input centering vs. our ON/OFF, their m, s,
      k, and learning rule. Note every difference before Phase 4.
- [ ] Check Caron et al. 2013 for the mean number of PN inputs per KC (we use 6).
- [ ] Be able to explain why P2 cannot change the fly model (finding 1).
- [ ] Update `docs/CONTRIBUTIONS.md`.
