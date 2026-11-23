# Dataset paper: Vergara et al. (2012)

**Citation:** A. Vergara, S. Vembu, T. Ayhan, M. A. Ryan, M. L. Homer, R. Huerta.
*Chemical gas sensor drift compensation using classifier ensembles.*
Sensors and Actuators B: Chemical 166–167 (2012) 320–329.
https://doi.org/10.1016/j.snb.2012.01.074

**Dataset citation:** Vergara, A. (2012). Gas Sensor Array Drift Dataset.
UCI Machine Learning Repository. https://doi.org/10.24432/C5RP6W

## Source status

- The **paper is paywalled** and could not be read from this environment.
  Use the college library / institutional access.
- Everything in "Verified" below was taken from the **UCI dataset page**
  (read 2026-11-23) or **measured from the data files** by `scripts/audit_data.py`.
- Nothing here should be cited as a claim of the paper until someone has
  read the paper and ticked the checklist.

## Verified (UCI page or our own measurement)

**Collection**
- 13,910 measurements, 16 chemical sensors, 6 pure gases: ethanol, ethylene,
  ammonia, acetaldehyde, acetone, toluene.
- Collected January 2007 to February 2011 (36 months) on a computer-controlled
  gas delivery platform at the ChemoSignals Laboratory, BioCircuits Institute,
  UC San Diego.
- Concentrations ranged from **5 to 1000 ppmv**. Per-gas concentrations are
  not in this version of the data; they are in the extension dataset
  (https://doi.org/10.24432/C5MK6M).
- The UCI page states the platform was automated to give accurate,
  reproducible concentrations "making it possible to exclusively concentrate
  on the chemical sensors for compensating real drift". This is the authors'
  claim; batches still differ in concentration and gas mix.

**Features (8 per sensor, sensor-major order)**
1. ΔR: steady-state feature, maximal resistance change minus baseline.
2. |ΔR| (normalized): ratio of maximal resistance to baseline.
3–5. EMAi, α = 0.001, 0.01, 0.1: maximum of the exponential moving average of
     the *rising* transient.
6–8. EMAd, α = 0.001, 0.01, 0.1: minimum of the exponential moving average of
     the *decaying* transient (hence negative values).
- The EMA transform is credited to Muezzinoglu et al. (2009).

**Batches:** 1 (months 1–2), 2 (3, 4, 8–10), 3 (11–13), 4 (14–15), 5 (16),
6 (17–20), 7 (21), 8 (22–23), 9 (24, 30), 10 (36). Batches were formed to
spread classes and months as uniformly as possible. They are **not**
equal-length time windows.

**Data notes**
- `batch10.dat` was corrected on 2013-10-14 (last 120 lines). Our checksummed
  download is the current file.
- The UCI page lists a CC BY 4.0 licence, while the description text says
  "research purposes only". We use it for research and education only.
- Our audit found: no missing values, no duplicate rows, no toluene in
  batches 3–5, and only 30 acetaldehyde samples in Batch 1.

## Checklist for the team member reading the paper

Fill in with page/table numbers so the report can cite them.

- [ ] Sensor models (manufacturer, type) and how many of each type.
- [ ] Operating conditions (heater voltage/temperature, flow rate, exposure
      duration, sampling rate).
- [ ] Table 1 / Table 2: concentrations per gas and measurement schedule.
- [ ] Their stated causes of drift.
- [ ] Their method (classifier ensemble) and evaluation protocol: what they
      trained on and tested on.
- [ ] Their main quantitative result and how accuracy changed over batches
      without compensation.
- [ ] Any limitation they mention.
- [ ] One sentence on how our task differs: they **classify** with
      **compensation**; we **detect novelty** with a **frozen** model.
