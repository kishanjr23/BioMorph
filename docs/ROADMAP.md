# Roadmap

Six phases (one per week), plus optional Phases 7–8. Each phase lists its goal,
tasks, decisions, outputs, and a "done when" check. Numbers marked *(audit)*
come from `reports/data_audit.md`.

**Ground rules for every phase**
- Batch 1 is the only training period. Batches 2–10 are test-only.
- Nothing from later batches or the held-out gas may influence preprocessing,
  hyperparameters, thresholds, or memory.
- Every reported number must come from a script in this repo writing to a CSV.
- Hypotheses are not results. Negative or mixed outcomes are valid findings.

---

## Phase 1 (Week 1): Understand the data ✅

See [`PHASE1_SUMMARY.md`](PHASE1_SUMMARY.md).

---

## Phase 2 (Week 2): Protocol, preprocessing, baselines

**Goal:** a leakage-free evaluation pipeline and honest baseline numbers,
before any fly code exists.

### Preprocessing (`src/biomorph/preprocess.py`)

| Option | Steps | Notes |
|---|---|---|
| **P1 (primary)** | signed log `sign(x)·log(1+|x|)` → z-score with **Batch-1 FIT statistics only** | Compresses heavy tails *(audit §5)* and keeps decay-feature sign |
| **P2 (secondary)** | P1 → per-sample L2 normalization | Loosely antennal-lobe-like gain normalization; may cancel multiplicative drift on its own |

Every model runs under P1 and P2. If P2 helps everyone equally, the gain is
from normalization, not the fly circuit.

### Split (`src/biomorph/protocol.py`)

For each held-out gas G ∈ {ethanol, ethylene, ammonia, acetaldehyde, acetone}
and each seed (≥20):

```
Batch 1, gases ≠ G  → stratified 70% FIT / 30% CALIB   (seeded)
Batch 1, gas G      → "B1 reference" test only (optimistic, no drift)
Batches 2–10        → TEST: gas G = unfamiliar (1), others = familiar (0)
```

- Familiar set = the 5 other gases, toluene included.
- The seed controls FIT/CALIB for **every** method, so all methods see the same data.
- Training sizes: 347–415 familiar samples; acetaldehyde has only 30 *(audit §3a)*.

### Hyperparameter tuning: pseudo-novelty inside Batch 1

```
for each familiar gas g' in FIT:
    train on FIT minus g'
    AUROC: CALIB (minus g') vs. FIT samples of g'
pick the config with the best mean AUROC; retrain on the full FIT set
```

All methods use this procedure and get a similar number of candidate configs.

**Threshold θ:** the 95th percentile of CALIB scores (target 5% false alarms).

### Baselines (`src/biomorph/baselines.py`): higher score = more novel

| Baseline | Implementation | Grid |
|---|---|---|
| Mahalanobis | mean + Ledoit-Wolf shrunk covariance | none |
| One-Class SVM | RBF | ν ∈ {0.01, 0.05, 0.1, 0.2}; γ ∈ {0.1, 1, 10} × "scale" |
| Isolation Forest | 200 trees, seeded | max_samples ∈ {64, 128, 256}; max_features ∈ {0.5, 1.0} |
| kNN distance | distance to k-th nearest FIT sample | k ∈ {1, 3, 5, 10} |

The fly memory never uses gas labels, so these label-free baselines are the
main comparison. Class-aware versions (per-gas centroid, per-gas Mahalanobis)
are run as clearly labelled *reference* baselines.

### Metrics (`src/biomorph/metrics.py`)

Per method × preprocessing × gas × seed × batch: AUROC, AUPRC, FPR and TPR at
θ, n_familiar, n_novel, novel share. Cells with fewer than 20 novel samples
are flagged "low-n" (e.g. acetaldehyde in batch 6, with 29 samples, 1%).

### Leakage tests (`tests/test_protocol.py`)
- No gas-G sample and no Batch 2–10 sample in FIT or CALIB.
- Scaler statistics equal statistics computed from FIT rows alone.
- Perturbation test: randomize Batch 2–10 features → every fitted parameter
  and θ is unchanged.

**Outputs:** `results/baselines.csv` (long format), AUROC-by-batch figure.

**Done when:** leakage tests pass; baselines exist for all 5 gases; the team
can explain one AUROC/FPR trend. If a gas is trivially easy (AUROC ≈ 1
everywhere), report it and don't rely on it.

---

## Phase 3 (Week 3): Fly encoder

**Goal:** a verified sparse encoder (no memory yet).

### Design (`src/biomorph/fly/encoder.py`)
1. **Non-negative PN input** via ON/OFF channels: `x⁺ = max(x,0)`,
   `x⁻ = max(−x,0)` → 256 inputs. Compare with the input step used by
   Dasgupta et al. once read.
2. **Projection** `M ∈ {0,1}^{m×256}`, each row exactly **s** ones at random.
   Defaults: m = 2,000; s = 6 (check against Caron et al. 2013). Seeded.
3. **Activity** `a = M · x_PN`.
4. **Top-k (APL-like inhibition):** the k = 5% of m (100) most active KCs → 1, rest → 0.

The encoder has nothing to fit.

### Verification
- Tests: exactly k active per sample; s inputs per KC; same seed → same M.
- Figures: input cosine similarity vs. code overlap (should be monotonic);
  KC usage histogram (dead cells); raster of 60 samples × KCs sorted by gas;
  within- vs. between-gas code overlap per batch.

### MVP checkpoint (end of Week 3)
One held-out gas (acetone), encoder + simple memory vs. kNN and Isolation
Forest; AUROC and FPR. Check the go/no-go conditions: no leakage, meaningful
evaluation, truly sparse codes, stable across seeds. Fix before continuing.

---

## Phase 4 (Week 4): Familiarity memory, full model, ablations

### Memory (`src/biomorph/fly/memory.py`)

```
w_i = 1 initially                                 (KC→MBON synapses)
training: for each familiar sample, active KC i:  w_i ← ε · w_i
score:    s(x) = (1/k) · Σ_i h_i(x) · w_i         (high = novel)
```

- Equivalent form: wᵢ = ε^{cᵢ}, where cᵢ is how many training samples
  activated KC i. Order-independent.
- **Saturation risk:** ~350 samples × 100 active / 2,000 KCs ≈ 17 uses per KC.
  With ε = 0, most used KCs hit zero. Tune ε ∈ {0.5, 0.8, 0.9, 0.95, 0.99}
  by pseudo-novelty and plot the trained w distribution.
- Biological basis (verify sources): dopamine-dependent depression of KC→MBON
  synapses; novelty-responsive MBONs whose response weakens with exposure.
- Approximations to state: single MBON, fixed ε, no dopamine signal, no forgetting.

### Ablations (same m, k, s and tuning budget unless noted)

| ID | Variant | Change | Tests |
|---|---|---|---|
| A0 | Full | none | reference |
| A1 | No expansion | m = 256, k = 13 | expansion |
| A2 | Dense connectivity | Gaussian M, all inputs per KC | sparse wiring |
| A3 | No top-k | graded `ã = ReLU(a)/max`; `w_i ← w_i·ε^{ã_i}`; score `Σ ã_i w_i / Σ ã_i` | sparsification (reduces to A0 for binary ã) |
| A4 | No synaptic memory | k-th-NN Hamming distance to stored FIT codes | value of the synaptic memory |
| A5 | Raw kNN | kNN on preprocessed features | H1 comparison |

Confounds to report: A4 stores all examples (more capacity); A1 changes
dimension and active count together.

**Run:** 5 gases × 20 seeds × 6 variants × 2 preprocessings → `results/fly.csv`.

**Done when:** all variants run; the A3 = A0 equivalence test passes; the w
distribution is not saturated.

---

## Phase 5 (Week 5): Rigorous evaluation

1. **Pre-register first.** Commit `docs/PREREGISTRATION.md` before looking at
   full results. Include H1–H4; primary metric = mean AUROC over batches 2–10
   per held-out gas, then averaged over gases; secondary metrics = AUPRC (vs.
   novel share) and FPR at θ; the list of comparisons.
2. **Statistics.**
   - Mean and 95% interval over seeds per method × gas × batch.
   - Paired differences (fly − baseline, same seed and split) and the count of
     seeds favouring fly.
   - Bootstrap over test samples for low-n cells.
   - State the limitation: seeds capture projection/split randomness only, and
     one dataset is not replication.
3. **Figures.**
   - AUROC by batch per gas.
   - FPR at the fixed θ by batch.
   - Ablation bars.
   - Method × gas heatmap.
   - KC overlap over time.
4. **Interpretation.** Positive, mixed, or negative. Explain the result with
   diagnostics (saturation, code overlap), not speculation.

**Done when:** one command (`python -m biomorph.run`) regenerates every number.

---

## Phase 6 (Week 6): Deliverables

- **Streamlit dashboard** (`dashboard/app.py`): offline, loads saved models
  and results.
  - Sample explorer: batch → sample → features → KC raster → score →
    decision → reveal label.
  - Also: results, ablations, biology pages.
  - Rehearse on the presentation laptop, and keep a screen recording as backup.
- **Report** (10–15 pp): abstract, introduction, biology, methods, results,
  ablations, discussion and limitations, conclusion, references, contributions.
- **Slides** (10–12), **2-minute demo script**, **viva Q&A**. Prepare at least:
  - why this is biological and not just random features
  - why not deep learning
  - how leakage was prevented
  - why toluene is excluded
  - drift vs. concentration
  - what if the baselines win
  - what ε means biologically
- **Contributions log**: keep it from Week 1, not written at the end.

---

## Phases 7–8 (optional)

- Sensitivity: m ∈ {500, 1k, 2k, 4k}; k/m ∈ {1, 2, 5, 10, 20}%; s ∈ {3, 6, 12, 24}.
- Concentration-annotated dataset (https://doi.org/10.24432/C5MK6M): do score
  changes follow concentration rather than time? This is the best available
  test of the drift/concentration confound.
- Secondary experiment: online memory updates from later familiar samples.
  Label it as adaptation, a different question from the frozen protocol.

---

## Suggested team split (4 people; adjust)

| Person | Phases | Owns |
|---|---|---|
| A | 2, 5 | protocol, leakage tests, statistics |
| B | 2, 5 | baselines, metrics, figures |
| C | 3, 4 | encoder, memory, ablations |
| D | 6 + docs | dashboard, report, slides, glossary |

Everyone must be able to explain the full pipeline in the viva.
