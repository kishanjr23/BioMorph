# Fly-inspired novelty detection: Dasgupta et al. (2018)

**Citation:** S. Dasgupta, T. C. Sheehan, C. F. Stevens, S. Navlakha.
*A neural data structure for novelty detection.* PNAS 115(51) (2018)
13093–13098. https://doi.org/10.1073/pnas.1814448115 (free copy: PMC6304992)

## Source status

- **Read:** the abstract only, via the Europe PMC API (2026-11-23).
- **Not read:** the full text. The free PDF sits behind a browser check that
  this environment cannot pass. Open it in a browser:
  https://europepmc.org/articles/PMC6304992
- Do not cite any algorithm detail from this file until the checklist below is
  filled in from the paper itself.

## Abstract (verified, verbatim)

> Novelty detection is a fundamental biological problem that organisms must
> solve to determine whether a given stimulus departs from those previously
> experienced. In computer science, this problem is solved efficiently using a
> data structure called a Bloom filter. We found that the fruit fly olfactory
> circuit evolved a variant of a Bloom filter to assess the novelty of odors.
> Compared with a traditional Bloom filter, the fly adjusts novelty responses
> based on two additional features: the similarity of an odor to previously
> experienced odors and the time elapsed since the odor was last experienced.
> We elaborate and validate a framework to predict novelty responses of fruit
> flies to given pairs of odors. We also translate insights from the fly
> circuit to develop a class of distance- and time-sensitive Bloom filters
> that outperform prior filters when evaluated on several biological and
> computational datasets. Overall, our work illuminates the algorithmic basis
> of an important neurobiological problem and offers strategies for novelty
> detection in computational systems.

## What the abstract already tells us

- The fly circuit is framed as a **Bloom filter variant**. A Bloom filter is a
  compact data structure that answers "have I seen this item before?"
- Two fly-specific features: **distance sensitivity** (similar odors seem
  less novel) and **time sensitivity** (familiarity fades with time).
- Our project uses the **distance-sensitive** idea. Their *time-sensitive*
  part (forgetting) is **not** in our primary protocol, because our model is
  frozen after Batch 1. Say this explicitly in the report.

## Our planned formulation (our design, to compare against the paper)

```
h  = top-k( M · x_PN )                  binary KC code, exactly k ones
w_i ← ε · w_i   for every active KC i    during training, w starts at 1
s(x) = (1/k) · Σ_i h_i(x) · w_i          novelty score, high = novel
```

When reading the paper, note **every** difference from this.

## Checklist for the team member reading the paper

Fill in with section/figure/equation numbers.

- [ ] Input preprocessing used before projection (centering? normalization?).
- [ ] Projection matrix: binary or weighted; inputs per KC; number of KCs.
- [ ] Winner-take-all: fraction of KCs kept.
- [ ] Learning rule for KC→MBON weights: initial value, update, parameter names.
- [ ] Novelty score formula (binary vs. distance-sensitive version).
- [ ] Time-sensitive version: how weights recover over time.
- [ ] Biological evidence cited for each part. In particular, the MBON
      that responds to novelty and the dopamine-dependent synaptic change.
      Copy the exact references (e.g. Hattori et al. 2017, if cited).
- [ ] Datasets and baselines they evaluated on, and their main result.
- [ ] Limitations they state.
- [ ] One sentence on what *we* add: chronological e-nose drift, held-out-gas
      protocol, conventional baselines, ablations.
