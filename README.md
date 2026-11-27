# Drosophila Mushroom-Body-Inspired Familiarity Detection Under E-Nose Drift

BT232TD Bioinspired Engineering. Status: **Week 5, statistical evaluation complete** (bootstrap CIs, paired tests, report figures). Week 6 (report, slides, demo, viva) is next.

## Setup (Windows, PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt pytest
```

Dataset (UCI Gas Sensor Array Drift, DOI 10.24432/C5RP6W). Extract it into `data/raw/` so you get `data/raw/Dataset/batch1.dat` through `batch10.dat`:

```powershell
curl.exe -L -o data\raw\drift.zip "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"
Expand-Archive data\raw\drift.zip -DestinationPath data\raw
```

## Run

```powershell
.\.venv\Scripts\python.exe scripts\validate_data.py       # integrity checks + counts -> results/
.\.venv\Scripts\python.exe -m pytest -q tests             # protocol, leakage, encoder, memory tests
.\.venv\Scripts\python.exe scripts\encoder_diagnostics.py # KC encoder checks -> results/encoder_*
.\.venv\Scripts\python.exe scripts\run_week4.py           # tuning + full benchmark -> results/benchmark_*
.\.venv\Scripts\python.exe scripts\run_week5.py           # bootstrap CIs, paired tests, figures -> results/week5_*, results/figures/
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py    # dashboard at http://localhost:8501
```

Report (department Word format, built from the department's report template, then TOC/PDF via Microsoft Word):

```powershell
.\.venv\Scripts\python.exe report\build_report.py "<path to template .docx>" report\BT232TD_EL_Report.docx
powershell -ExecutionPolicy Bypass -File report\finalize_word.ps1 report\BT232TD_EL_Report.docx
```

`scripts/run_baselines.py` is the older Week-2 run (untuned baselines only); `run_week4.py` supersedes it.

## Layout

| Path | Purpose |
|---|---|
| `src/flyodor/config.py` | Paths, dataset constants, protocol settings, seeds |
| `src/flyodor/data.py` | Loader for `batchN.dat`, counts table, validation checks |
| `src/flyodor/preprocess.py` | Standardizer fitted on Batch 1 only and then frozen |
| `src/flyodor/protocol.py` | Chronological leave-one-gas-out split, leakage check, run/threshold logic |
| `src/flyodor/baselines.py` | Mahalanobis, kNN (Euclidean / cosine), One-Class SVM, Isolation Forest, Random |
| `src/flyodor/encoder.py` | Sparse random PN->KC projection + top-k winner-take-all (binary KC code) |
| `src/flyodor/memory.py` | KC->MBON familiarity memory: w_j <- beta * w_j for active KCs |
| `src/flyodor/flymodel.py` | Complete fly detector + ablations (no expansion, dense, no top-k, no plastic memory) |
| `src/flyodor/registry.py` | All compared detectors, their grids, Batch-1-only tuning |
| `src/flyodor/stats.py` | Weighted AUROC for bootstrapping, stratified resampling, Holm correction |
| `src/flyodor/metrics.py` | Per-batch AUROC, AUPRC, familiar FPR, detection rate, Batch-1 reference |
| `scripts/` | Week-by-week experiment scripts |
| `tests/` | Protocol, leakage, encoder, memory and ablation tests |
| `app/streamlit_app.py` | Dashboard: overview, data, drift, samples, encoder, familiarity demo, benchmark, ablations |
