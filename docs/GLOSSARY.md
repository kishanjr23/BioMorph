# Glossary

Terms we must use precisely in the code, report, and viva. Where two terms are
easy to confuse, the difference is spelled out.

## Task terms

| Term | Meaning in this project | Not to be confused with |
|---|---|---|
| **Odor / gas classification** | Naming *which* known gas a sample is (6-way label). | Novelty detection. We are **not** primarily classifying. |
| **Novelty detection** | Deciding whether a sample comes from something *never learned* during training. Output: a novelty score and a familiar/unfamiliar decision. | Anomaly detection of faulty readings. Our "novel" samples are valid measurements of a gas withheld from training. |
| **Familiarity detection** | The same decision viewed from the memory side: "have I experienced something like this before?" Used for the fly framing. In this project, familiarity score = −novelty score. | Recognition of a *specific* odor identity. |
| **Familiar gas** | A gas whose Batch-1 samples were used for training in a given experiment. | |
| **Unfamiliar / held-out / novel gas** | The single gas excluded from training in a given experiment. Its samples are labelled 1 (novel) at test time. | |
| **Leave-one-gas-out** | Repeat the experiment 5 times, each time holding out a different gas (ethanol, ethylene, ammonia, acetaldehyde, acetone). | Leave-one-out cross-validation over samples. |

## Drift terms

| Term | Meaning | Not to be confused with |
|---|---|---|
| **Sensor drift** | Slow change in a sensor's response to the *same* exposure over months (ageing, poisoning, environment). A physical property of the sensor. | Concept drift in general ML. |
| **Temporal distribution shift** | The measured feature distribution changes between time periods, for *any* reason: sensor drift, different concentrations, different gas mix, procedure changes. This is what we can actually observe in the dataset. | Sensor drift alone. We can't separate causes without extra data. |
| **Drift compensation** | Methods that try to *correct* drifted data or retrain models with later data so performance recovers. | Our project. We do **not** compensate drift and do not repair sensors. |
| **Robustness under shift** | How well a model trained on early data keeps working on later data **without** being updated. This is what we measure. | |
| **Chronological split** | Train only on early data (Batch 1), test only on later data (Batches 2–10). Never shuffle across time. | Random train/test split, which leaks future sensor states. |
| **Data leakage** | Any information from test data (later batches, held-out gas) influencing training, preprocessing, hyperparameters, or thresholds. Invalidates results. | |

## Model terms (biology ↔ engineering)

| Term | Biology | Our engineering version |
|---|---|---|
| **ORN** (olfactory receptor neuron) | Detects odorant molecules. | The 16 metal-oxide gas sensors. |
| **Antennal lobe / glomeruli** | Early processing and normalization of receptor signals. | Preprocessing (signed log, scaling, optional per-sample normalization). |
| **PN** (projection neuron) | Carries odor signal from antennal lobe to mushroom body; firing rates are non-negative. | The preprocessed feature vector, split into non-negative ON/OFF channels. |
| **KC** (Kenyon cell) | ~2,000 neurons in the mushroom body, each receiving a few random PN inputs. | Units of the expanded representation (m ≈ 2,000). |
| **Random sparse projection** | Each KC connects to a small random set of PNs. | Fixed binary matrix M, each row with s ones. Never trained. |
| **Expansion** | Many more KCs than PN types. | m ≫ input dimension. |
| **APL neuron** | Inhibitory neuron giving feedback inhibition to KCs, keeping activity sparse. | Top-k / winner-take-all: keep only the k most active KCs. |
| **Sparse code / tag** | The small set of KCs active for an odor. | Binary vector h with exactly k ones. |
| **Population sparsity** | Fraction of KCs active for one odor. | k / m (we fix it, e.g. 5%). |
| **Lifetime sparsity** | Fraction of odors a single KC responds to. | Measured, not fixed. KCs that never fire are "dead". |
| **MBON** (mushroom body output neuron) | Reads out KC activity through KC→MBON synapses. | A single readout computing the novelty score. |
| **Synaptic depression / plasticity** | Experience-dependent weakening of KC→MBON synapses (dopamine-dependent). | Weight update wᵢ ← ε·wᵢ for active KCs during training. |
| **Familiarity memory** | Reduced output response to previously experienced odors. | The trained weight vector w. |
| **ε (epsilon)** | Strength of depression per exposure. | 0 = one exposure fully silences a synapse; near 1 = gradual. |

## Evaluation terms

| Term | Meaning |
|---|---|
| **Novelty score** | Real number per sample; higher = more likely unfamiliar. All methods are oriented this way. |
| **Threshold θ** | Score cut-off for the familiar/unfamiliar decision. Set on Batch-1 calibration data only (95th percentile → target 5% false alarms). |
| **AUROC** | Probability that a random novel sample scores higher than a random familiar one. 0.5 = chance, 1.0 = perfect ranking. Threshold-free. |
| **AUPRC** | Area under precision-recall curve. Chance level = share of novel samples in that batch, so compare against that share (see data audit §3a). |
| **False-positive rate (FPR)** | Fraction of familiar samples wrongly called unfamiliar at θ ("false alarms"). |
| **Detection rate (TPR)** | Fraction of novel samples correctly called unfamiliar at θ. |
| **Baseline** | A conventional method (Mahalanobis, One-Class SVM, Isolation Forest, kNN) run under the identical protocol. |
| **Ablation** | Removing or replacing one model component to test whether it matters. |
| **Pseudo-novelty tuning** | Choosing hyperparameters inside Batch 1 by temporarily treating one *familiar* gas as novel. Never uses the real held-out gas or later batches. |
| **Seed** | Random number generator setting. Controls the projection matrix and the fit/calibration split. Results are reported over ≥20 seeds. |
