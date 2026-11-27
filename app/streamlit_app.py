"""FlyNose dashboard: project overview + real-data exploration, baselines and KC encoder.

Run:  .venv\\Scripts\\streamlit run app\\streamlit_app.py
Every number shown here is computed live from the recorded dataset or saved result files.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.decomposition import PCA

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent / "src"))

from flyodor import config as C  # noqa: E402
from flyodor.data import counts_table, feature_matrix, load_all, validate  # noqa: E402
from flyodor.encoder import SparseKCEncoder, overlap  # noqa: E402
from flyodor.preprocess import FrozenStandardizer  # noqa: E402
from flyodor.protocol import make_split  # noqa: E402
from flyodor.registry import FLY_FULL, SPECS  # noqa: E402

st.set_page_config(page_title="FlyNose · Mushroom-body familiarity", page_icon="🪰", layout="wide")
st.logo(str(APP_DIR / "assets" / "logo.svg"), size="large")

# ======================================================================================
# Design tokens (sage / forest clinical-dashboard palette)
# ======================================================================================
FOREST, OLIVE, SAGE, LIME = "#2F3D1C", "#7C8F4E", "#B5C98A", "#DCE6C0"
INK, MUTED, LINE, CARD = "#1C2412", "#6B7358", "#CDD2BF", "#F6F7F0"
OCHRE, BROWN, GREY = "#C9A646", "#8C6B3F", "#A9AE9C"

GAS_ORDER = list(C.GAS_NAMES.values())
GAS_COLORS = {"Ethanol": FOREST, "Ethylene": OLIVE, "Ammonia": "#9DB86A",
              "Acetaldehyde": OCHRE, "Acetone": BROWN, "Toluene": GREY}
FLY_COLOR = "#B5532A"   # terracotta accent: the fly model stands out against the greens
ABLATIONS = [n for n, s in SPECS.items() if s.group == "ablation"]
DET_COLORS = {FLY_FULL: FLY_COLOR, "Mahalanobis": FOREST, "kNN (Euclidean)": OLIVE,
              "kNN (k=5)": OLIVE, "One-Class SVM": OCHRE, "Isolation Forest": BROWN,
              "kNN (cosine)": "#4F7A6B", "Random (sanity)": GREY,
              **dict(zip(ABLATIONS, ["#D99A6C", "#E3BF96", "#9C5B3C", "#6E3F2A"]))}
SHORT = {FLY_FULL: "Full fly model", "Ablation: no expansion (m=128)": "− expansion",
         "Ablation: dense wiring": "− sparse wiring", "Ablation: no top-k (fixed threshold)": "− top-k",
         "Ablation: no plastic memory (KC nearest neighbour)": "− plastic memory",
         "kNN (cosine)": "cosine kNN (control)"}
SPACE_COLORS = {"Input, Euclidean": GREY, "Input, cosine": OLIVE, "KC code": FOREST}
GREEN_SCALE = [[0, "#F3F4EC"], [0.35, SAGE], [0.7, OLIVE], [1, FOREST]]
DIVERGING = [[0, BROWN], [0.5, "#F3F4EC"], [1, FOREST]]

pio.templates["flynose"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, ui-monospace, monospace", color="#3A4429", size=11),
    colorway=list(GAS_COLORS.values()),
    xaxis=dict(gridcolor="rgba(47,61,28,0.09)", zerolinecolor="rgba(47,61,28,0.25)",
               linecolor=LINE, title_font=dict(size=10, color=MUTED)),
    yaxis=dict(gridcolor="rgba(47,61,28,0.09)", zerolinecolor="rgba(47,61,28,0.25)",
               linecolor=LINE, title_font=dict(size=10, color=MUTED)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    hoverlabel=dict(bgcolor=CARD, bordercolor=LINE, font=dict(color=INK, family="JetBrains Mono")),
))
pio.templates.default = "plotly_white+flynose"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter+Tight:wght@300;400;500;600&display=swap');
:root{--bg:#EEF0E6;--card:#F6F7F0;--line:#CDD2BF;--ink:#1C2412;--muted:#6B7358;--forest:#2F3D1C;
  --olive:#7C8F4E;--sage:#B5C98A;--lime:#DCE6C0;--mono:"JetBrains Mono",ui-monospace,monospace;
  --sans:"Inter Tight",Inter,sans-serif}
header[data-testid="stHeader"]{background:var(--card) !important;border-bottom:1px solid var(--line);height:64px}
header[data-testid="stHeader"] a{font-family:var(--mono) !important;text-transform:uppercase;
  font-size:.7rem !important;letter-spacing:.05em;background:var(--lime);border-radius:3px;
  padding:6px 14px !important;margin:0 3px;color:var(--ink) !important}
header[data-testid="stHeader"] a *{font-family:var(--mono) !important;font-size:.7rem !important}
header[data-testid="stHeader"] a[aria-current="page"],
header[data-testid="stHeader"] a.active{background:var(--forest) !important;color:#EEF2E2 !important}
header[data-testid="stHeader"] a[aria-current="page"] *{color:#EEF2E2 !important}
.block-container{padding-top:5.2rem !important;padding-bottom:2.5rem;max-width:1360px}
h1,h2,h3{letter-spacing:-.03em !important}
[data-testid="stMetricLabel"] p{font-family:var(--mono);text-transform:uppercase;letter-spacing:.05em;
  font-size:.66rem !important;color:var(--ink)}
[data-testid="stMetricValue"]{font-family:var(--sans);font-weight:300;letter-spacing:-.03em}
[data-testid="stWidgetLabel"] p{font-family:var(--mono);text-transform:uppercase;font-size:.66rem !important;
  letter-spacing:.05em}
[data-testid="stCaptionContainer"]{font-family:var(--mono);font-size:.68rem}
[class*="st-key-card"]{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:14px 18px 16px}
[class*="st-key-card"][class*="dash"]{border-style:dashed}

/* ---------- html building blocks ---------- */
.fx-mono{font-family:var(--mono)}
.fx-label{font-family:var(--mono);font-size:.68rem;letter-spacing:.05em;text-transform:uppercase;
  color:var(--ink);font-weight:500}
.fx-label.m{color:var(--muted)}
.fx-ph{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;flex-wrap:wrap;margin:0 0 18px}
.fx-title{font-family:var(--sans);font-weight:400;letter-spacing:-.035em;line-height:1;
  font-size:clamp(2.2rem,4vw,3.4rem);color:var(--ink);margin:0}
.fx-sub{font-family:var(--sans);font-size:1rem;color:var(--ink);margin:10px 0 0;max-width:780px;line-height:1.45}
.fx-chips{display:flex;gap:8px;flex-wrap:wrap}
.fx-chip{font-family:var(--mono);font-size:.66rem;text-transform:uppercase;letter-spacing:.05em;
  background:var(--lime);padding:6px 10px;border-radius:3px;display:inline-flex;align-items:center;gap:8px}
.fx-chip b{background:var(--forest);color:#EEF2E2;padding:2px 7px;border-radius:2px;font-weight:500}
.fx-chip.dark{background:var(--forest);color:#EEF2E2}
.fx-grid{display:grid;gap:12px;margin-bottom:12px}
.fx-card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;
  display:flex;flex-direction:column;min-width:0}
.fx-card.dash{border-style:dashed}
.fx-card.dark{background:var(--forest);border-color:var(--forest);color:#E4EAD5}
.fx-card.dark .fx-label{color:#E4EAD5}
.fx-card.lime{background:var(--lime);border-color:#C7D3A6}
.fx-h{display:flex;justify-content:space-between;gap:10px;padding-bottom:8px;border-bottom:1px solid var(--ink);
  margin-bottom:10px}
.fx-card.dark .fx-h{border-color:rgba(228,234,213,.5)}
.fx-big{font-family:var(--sans);font-weight:300;font-size:3.1rem;line-height:1;letter-spacing:-.04em;
  text-align:right;margin:6px 0 12px}
.fx-rows{margin-top:auto}
.fx-row{font-family:var(--mono);font-size:.66rem;text-transform:uppercase;display:flex;gap:10px;
  justify-content:space-between;padding:6px 0;border-top:1px solid var(--line)}
.fx-row:first-child{border-top:1px solid var(--ink)}
.fx-links a{display:block;font-family:var(--mono);font-size:.7rem;text-transform:uppercase;color:#E4EAD5 !important;
  text-decoration:none !important;padding:4px 0;letter-spacing:.04em}
.fx-links a:hover{color:#fff !important;text-decoration:underline !important}
.fx-stacks{display:flex;justify-content:space-between;align-items:flex-end;gap:4px;height:100%;min-height:200px;
  padding:4px 0 0}
.fx-stack{display:flex;flex-direction:column;align-items:center;gap:6px;flex:1}
.fx-dots{display:flex;flex-direction:column-reverse;gap:2px;align-items:center}
.fx-dots i{display:block;width:10px;height:10px;border:1.4px solid var(--forest);border-radius:50%}
.fx-dots i.f{background:var(--forest)}
.fx-dots i.fly{background:#B5532A;border-color:#B5532A}
.fx-pair{display:flex;gap:2px;align-items:flex-end}
.fx-stack span{font-family:var(--mono);font-size:.58rem;color:var(--ink)}
.fx-tiles{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.fx-tile{background:#FBFBF7;border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.fx-tile.on{background:var(--forest);border-color:var(--forest);color:#EEF2E2}
.fx-tile .t{font-family:var(--sans);font-size:1.55rem;font-weight:400;letter-spacing:-.03em;line-height:1}
.fx-tile .s{font-family:var(--mono);font-size:.58rem;text-transform:uppercase;margin-top:4px;opacity:.8}
.fx-heat{display:grid;gap:3px;align-items:center}
.fx-heat .c{height:30px;border-radius:2px;font-family:var(--mono);font-size:.6rem;display:flex;
  align-items:center;justify-content:center}
.fx-heat .rl{font-family:var(--mono);font-size:.6rem;text-transform:uppercase;padding-right:6px}
.fx-heat .cl{font-family:var(--mono);font-size:.6rem;text-align:center}
.fx-legend{display:flex;gap:14px;flex-wrap:wrap;margin:2px 0 12px;font-family:var(--mono);font-size:.62rem;text-transform:uppercase}
.fx-legend span{display:inline-flex;align-items:center;gap:6px}
.fx-legend i{display:inline-block;width:10px;height:10px;border-radius:2px}
.fx-res{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;border-radius:6px;
  background:#FBFBF7;border:1px solid var(--line);margin-bottom:8px}
.fx-res.hl{background:var(--lime);border-color:#C7D3A6}
.fx-res .n{font-family:var(--sans);font-weight:300;font-size:2.5rem;letter-spacing:-.04em;line-height:1}
.fx-res .l{font-family:var(--mono);font-size:.62rem;text-transform:uppercase;text-align:right;max-width:55%}
.fx-steps{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}
.fx-step{border:1px dashed var(--line);border-radius:8px;padding:12px;background:#FBFBF7}
.fx-step .n{font-family:var(--mono);font-size:.62rem;color:var(--muted)}
.fx-step .t{font-family:var(--sans);font-size:1rem;font-weight:500;margin:10px 0 6px;line-height:1.2;color:var(--ink)}
.fx-step .b{font-family:var(--mono);font-size:.6rem;text-transform:uppercase;color:var(--olive)}
.fx-note{font-family:var(--mono);font-size:.72rem;line-height:1.55;color:var(--ink)}
.fx-note ul{margin:0;padding-left:16px}.fx-note li{margin:3px 0}
.fx-note b{font-weight:600}
.fx-stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px}
.fx-stat .v{font-family:var(--sans);font-weight:300;font-size:2.3rem;letter-spacing:-.04em;line-height:1.05;margin-top:10px}
.fx-foot{margin-top:28px;padding-top:12px;border-top:1px dashed var(--line);display:flex;justify-content:space-between;
  gap:16px;flex-wrap:wrap;font-family:var(--mono);font-size:.6rem;text-transform:uppercase;color:var(--muted)}
@media (max-width:900px){
  .fx-grid{grid-template-columns:1fr !important}
  .fx-steps{grid-template-columns:repeat(2,1fr)}
}
</style>
"""
st.html(CSS)


# ======================================================================================
# Data + helpers
# ======================================================================================
@st.cache_data(show_spinner="Loading recorded sensor data...")
def get_data() -> pd.DataFrame:
    return load_all()


@st.cache_data
def get_standardized(df: pd.DataFrame) -> np.ndarray:
    """Standardize ALL batches using Batch-1 statistics only (frozen, no leakage)."""
    X = feature_matrix(df)
    train = (df["batch"] == C.TRAIN_BATCH).to_numpy()
    return FrozenStandardizer().fit(X[train]).transform(X)


@st.cache_data
def get_pca(df: pd.DataFrame, Z: np.ndarray):
    train = (df["batch"] == C.TRAIN_BATCH).to_numpy()
    pca = PCA(n_components=2, random_state=C.DEFAULT_SEED).fit(Z[train])
    return pca.transform(Z), pca.explained_variance_ratio_


@st.cache_data
def load_results(path: str, mtime: float) -> pd.DataFrame:  # mtime busts cache on rerun
    return pd.read_csv(path, dtype={"batch": str})


@st.cache_resource
def get_encoder(n_kc: int, conn: int, active_frac: float, seed: int) -> SparseKCEncoder:
    return SparseKCEncoder(C.N_FEATURES, n_kc=n_kc, conn_per_kc=conn,
                           active_frac=active_frac, seed=seed)


def page_header(title: str, sub: str = "", chips: str = "") -> None:
    st.html(f'<div class="fx-ph"><div><div class="fx-title">{title}</div>'
            f'{f"<p class=fx-sub>{sub}</p>" if sub else ""}</div>'
            f'<div class="fx-chips">{chips}</div></div>')


def chip(text: str, n=None, dark: bool = False) -> str:
    num = f"<b>{n}</b>" if n is not None else ""
    return f'<span class="fx-chip{" dark" if dark else ""}">{num}{text}</span>'


def card(key: str, dashed: bool = False):
    return st.container(key=f"card_{'dash_' if dashed else ''}{key}")


def card_head(label: str, right: str = "") -> None:
    st.html(f'<div class="fx-h"><span class="fx-label">{label}</span>'
            f'<span class="fx-label m">{right}</span></div>')


def callout(label: str, items: list[str], right: str = "") -> None:
    lis = "".join(f"<li>{x}</li>" for x in items)
    st.html(f'<div class="fx-card lime" style="margin:4px 0 12px"><div class="fx-h">'
            f'<span class="fx-label">{label}</span><span class="fx-label m">{right}</span></div>'
            f'<div class="fx-note"><ul>{lis}</ul></div></div>')


def stat_row(stats: list[tuple[str, str]]) -> None:
    cells = "".join(f'<div class="fx-stat"><div class="fx-label">{k}</div><div class="v">{v}</div></div>'
                    for k, v in stats)
    st.html(f'<div class="fx-grid" style="grid-template-columns:repeat({len(stats)},1fr)">{cells}</div>')


def show(fig: go.Figure, height: int = 400, top: int = 10) -> None:
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=top, b=0))
    st.plotly_chart(fig, theme=None, width="stretch")


def footer() -> None:
    st.html('<div class="fx-foot"><span>BT232TD Bioinspired Engineering · FlyNose</span>'
            '<span>UCI Gas Sensor Array Drift Dataset · DOI 10.24432/C5RP6W</span>'
            '<span>Software replay of recorded measurements — not live sensing</span></div>')


@st.cache_data
def get_X(df: pd.DataFrame) -> np.ndarray:
    return feature_matrix(df)


def tuned_params(gas: int) -> dict:
    """Fly settings chosen by Batch-1-only tuning (falls back to the a-priori default)."""
    if TUNING_JSON.exists():
        t = json.loads(TUNING_JSON.read_text())
        return t[C.GAS_NAMES[gas]][FLY_FULL]["chosen"]
    return SPECS[FLY_FULL].grid[0]


@st.cache_resource(show_spinner="Training the fly model on Batch 1...")
def fit_fly_demo(gas: int, seed: int, _mtime: float):
    """Exactly the benchmark protocol for one held-out gas and seed, scored on every sample."""
    X = get_X(df)
    params = tuned_params(gas)
    split = make_split(df, gas, seed)
    scaler = FrozenStandardizer().fit(X[split.fit_idx])
    det = SPECS[FLY_FULL].build(seed, params).fit(scaler.transform(X[split.fit_idx]))
    calib = det.score(scaler.transform(X[split.calib_idx]))
    thr = float(np.quantile(calib, 1.0 - C.TARGET_FPR))
    Zs = scaler.transform(X)
    scores = det.score(Zs)
    role = np.full(len(df), "test (unseen)", dtype=object)
    role[split.fit_idx] = "Batch-1 training sample"
    role[split.calib_idx] = "Batch-1 calibration sample"
    role[((df["batch"] == 1) & (df["gas_id"] == gas)).to_numpy()] = "Batch-1 held-out gas (never trained)"
    return det, Zs, thr, scores, role, params


def load_bench() -> pd.DataFrame | None:
    path = BENCH_CSV if BENCH_CSV.exists() else (BASELINE_CSV if BASELINE_CSV.exists() else None)
    return None if path is None else load_results(str(path), path.stat().st_mtime)


def macro_later(res: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per detector x gas x seed: mean over Batches 2-10 (each period weighted equally)."""
    later = res[res["batch"].isin([str(b) for b in C.TEST_BATCHES])]
    return later.groupby(["detector", "heldout_name", "seed"])[metric].mean().reset_index()


df = get_data()
Z = get_standardized(df)
BASELINE_CSV = C.RESULTS_DIR / "baseline_metrics.csv"
BENCH_CSV = C.RESULTS_DIR / "benchmark_metrics.csv"
TUNING_JSON = C.RESULTS_DIR / "tuning.json"
DIAG_JSON = C.RESULTS_DIR / "encoder_diagnostics.json"
NN_CSV = C.RESULTS_DIR / "encoder_nn_by_batch.csv"


# ======================================================================================
# Overview dashboard
# ======================================================================================
def _heat_color(n: int) -> tuple[str, str]:
    if n == 0:
        return "transparent;border:1px dashed #CDD2BF", MUTED
    if n < 50:
        return LIME, INK
    if n < 200:
        return SAGE, INK
    if n < 500:
        return OLIVE, "#F3F6EA"
    return "#3E5226", "#F3F6EA"


def page_home() -> None:
    report = validate(df)
    n_pass = sum(ok for _, ok, _ in report.checks)
    page_header(
        "Project Overview",
        "A fruit-fly-inspired familiarity detector tested against conventional detectors on three "
        "years of real electronic-nose sensor drift.",
        chip("data checks passed", f"{n_pass}/{len(report.checks)}")
        + chip("held-out gases", len(C.PRIMARY_HELDOUT_GASES)) + chip("seeds", C.N_SEEDS, dark=True))

    ct = counts_table(df)
    totals = ct[GAS_ORDER].sum()
    present = (ct[GAS_ORDER] > 0).sum()

    # ---------- row 1 ----------
    links = "".join(f'<a href="{u}" target="_self">[&gt;] {t}</a>' for t, u in [
        ("Dataset & checks", "data"), ("Drift explorer", "drift"), ("Sample viewer", "samples"),
        ("Kenyon-cell encoder", "encoder"), ("Familiarity demo", "familiarity"),
        ("Benchmark", "benchmark"), ("Ablations", "ablations"), ("Statistics", "statistics")])
    quick = (f'<div class="fx-card dark"><div class="fx-h"><span class="fx-label">Quick access</span>'
             f'</div><div style="margin-top:auto" class="fx-links">{links}</div></div>')

    gas_rows = "".join(f'<div class="fx-row"><span>{g}</span><span>{int(totals[g]):,}</span>'
                       f'<span>{int(present[g])}/10 batches</span></div>' for g in GAS_ORDER)
    data_card = (f'<div class="fx-card"><div class="fx-h"><span class="fx-label">Recorded measurements</span>'
                 f'<span class="fx-label m">36 months</span></div><div class="fx-big">{len(df):,}</div>'
                 f'<div class="fx-rows">{gas_rows}</div></div>')

    ho_rows = "".join(
        f'<div class="fx-row"><span>{C.GAS_NAMES[g]}</span><span>B1: {int(ct.loc[1, C.GAS_NAMES[g]])}</span>'
        f'<span>[held out]</span></div>' for g in C.PRIMARY_HELDOUT_GASES)
    ho_rows += ('<div class="fx-row"><span>Toluene</span><span>absent B3–5</span>'
                '<span>[excluded]</span></div>')
    heldout = (f'<div class="fx-card"><div class="fx-h"><span class="fx-label">Novelty experiments</span>'
               f'<span class="fx-label m">leave-one-gas-out</span></div>'
               f'<div class="fx-big">{len(C.PRIMARY_HELDOUT_GASES):02d}</div><div class="fx-rows">{ho_rows}</div></div>')

    res = load_bench()
    if res is not None:
        week4 = "group" in res.columns
        conv = ([n for n, s in SPECS.items() if s.group == "conventional"] if week4
                else [d for d in res["detector"].unique() if d != "Random (sanity)"])
        real = res[res["detector"].isin(conv)]
        fly = res[res["detector"] == FLY_FULL] if week4 else None
        fpr = real[real["batch"] != "pooled"].groupby("batch")["fpr_familiar"].mean()
        fpr_fly = fly[fly["batch"] != "pooled"].groupby("batch")["fpr_familiar"].mean() if week4 else None
        cols = ""
        for b in ["1 (ref)"] + [str(b) for b in C.TEST_BATCHES]:
            pair = '<div class="fx-dots">' + "<i></i>" * int(round(fpr[b] / 0.05)) + "</div>"
            tip = f"conventional {fpr[b]:.0%}"
            if week4:
                pair += ('<div class="fx-dots">' + '<i class="fly"></i>' * int(round(fpr_fly[b] / 0.05))
                         + "</div>")
                tip += f" · fly {fpr_fly[b]:.0%}"
            cols += (f'<div class="fx-stack" title="familiar samples flagged — {tip}"><div class="fx-pair">'
                     f'{pair}</div><span>B{b.split()[0]}</span></div>')
        legend = "○ conventional · ● fly · each dot = 5%" if week4 else "each dot = 5% of familiar"
        drift_card = (f'<div class="fx-card lime"><div class="fx-h"><span class="fx-label">'
                      f'False alarms on familiar gases</span><span class="fx-label m">{legend}</span>'
                      f'</div><div class="fx-stacks">{cols}</div></div>')
        ref = real[real["batch"] == "1 (ref)"]["auroc"].mean()
        later_df = real[real["batch"].isin([str(b) for b in C.TEST_BATCHES])]
        later = later_df.groupby(["detector", "heldout_gas", "seed"])["auroc"].mean().mean()
        later_fpr = later_df["fpr_familiar"].mean()
        fly_rows = ""
        if week4:
            f_auc = macro_later(res[res["detector"] == FLY_FULL], "auroc")["auroc"].mean()
            f_fpr = macro_later(res[res["detector"] == FLY_FULL], "fpr_familiar")["fpr_familiar"].mean()
            fly_rows = (f'<div class="fx-res" style="border-color:{FLY_COLOR}"><span class="n" '
                        f'style="color:{FLY_COLOR}">{f_auc:.2f}</span><span class="l">fly model AUROC · '
                        f'B2–10<br>false alarms {f_fpr:.0%}</span></div>')
        results = (
            f'<div class="fx-card"><div class="fx-h"><span class="fx-label">Results so far</span>'
            f'<span class="fx-label m">mean of 4 conventional detectors</span></div>'
            f'<div class="fx-res"><span class="n">{ref:.2f}</span><span class="l">AUROC · Batch 1<br>no drift</span></div>'
            f'<div class="fx-res hl"><span class="n">{later_fpr:.0%}</span><span class="l">familiar gases flagged '
            f'novel · B2–10<br>target {C.TARGET_FPR:.0%}</span></div>'
            f'<div class="fx-res"><span class="n">{later:.2f}</span><span class="l">AUROC · Batches 2–10<br>'
            f'0.50 = chance</span></div>{fly_rows}</div>')
    else:
        drift_card = results = ('<div class="fx-card"><span class="fx-label">Run scripts/run_baselines.py'
                                '</span></div>')

    st.html(f'<div class="fx-grid" style="grid-template-columns:1fr 1.3fr 1.3fr 1.7fr">'
            f'{quick}{data_card}{heldout}{drift_card}</div>')

    # ---------- row 2 ----------
    tiles = "".join(
        f'<div class="fx-tile{" on" if b == 1 else ""}"><div class="t">B{b}</div>'
        f'<div class="s">mo {C.BATCH_MONTHS[b]} · {"train" if b == 1 else "test"}</div></div>'
        for b in range(1, 11))
    timeline = (f'<div class="fx-card"><div class="fx-h"><span class="fx-label">Chronology</span>'
                f'<span class="fx-label m">10 batches</span></div><div class="fx-tiles">{tiles}</div></div>')

    head = '<span></span>' + "".join(f'<span class="cl">B{b}</span>' for b in range(1, 11))
    body = ""
    for g in GAS_ORDER:
        body += f'<span class="rl">{g}</span>'
        for b in range(1, 11):
            n = int(ct.loc[b, g])
            bg, fg = _heat_color(n)
            body += f'<span class="c" style="background:{bg};color:{fg}" title="{g}, batch {b}: {n}">{n}</span>'
    legend = "".join(f'<span><i style="background:{c};{x}"></i>{t}</span>' for c, t, x in [
        ("transparent", "none", "border:1px dashed #A9AE9C"), (LIME, "1–49", ""), (SAGE, "50–199", ""),
        (OLIVE, "200–499", ""), ("#3E5226", "500+", "")])
    heat = (f'<div class="fx-card"><div class="fx-h"><span class="fx-label">Samples per batch</span>'
            f'<span class="fx-label m">class density</span></div><div class="fx-legend">{legend}</div>'
            f'<div class="fx-heat" style="grid-template-columns:96px repeat(10,1fr)">{head}{body}</div></div>')

    st.html(f'<div class="fx-grid" style="grid-template-columns:1fr 2.3fr 1.5fr">{timeline}{heat}{results}</div>')

    # ---------- row 3: pipeline ----------
    steps = [("128 sensor features", "olfactory input"),
             ("Frozen standardization", "PN normalization"),
             ("Sparse random projection", "PN → KC wiring"),
             ("Top-k activation · 5%", "APL inhibition"),
             ("Familiarity memory", "KC → MBON plasticity"),
             ("Novelty score → decision", "familiarity output")]
    cells = "".join(f'<div class="fx-step"><div class="n">[{i:02d}] · built'
                    f'</div><div class="t">{e}</div><div class="b">{b}</div></div>'
                    for i, (e, b) in enumerate(steps, 1))
    st.html(f'<div class="fx-card" style="margin-bottom:12px"><div class="fx-h"><span class="fx-label">'
            f'Processing pipeline</span><span class="fx-label m">biology → engineering</span></div>'
            f'<div class="fx-steps">{cells}</div></div>')

    # ---------- row 4: memory + protocol ----------
    c1, c2 = st.columns(2, gap="small")
    with c1, card("memory"):
        card_head("Familiarity memory", "implemented")
        st.markdown(
            "Repeated exposure **depresses** the synapses from an odor's active Kenyon cells onto a "
            "familiarity output neuron (MBON), so familiar odors drive it weakly "
            "(after Dasgupta et al., 2018).")
        st.markdown(r"""
- Each KC $j$ starts with weight $w_j = 1$.
- Each familiar training sample with active set $A$: $w_j \leftarrow \beta\, w_j$ for $j \in A$.
- Novelty score: $\; s(x) = \frac{1}{k}\sum_{j \in A(x)} w_j$
- Threshold $\tau$ from held-out Batch-1 familiar data, then **frozen**.
- $\beta$, $k$ and inputs per KC are chosen by Batch-1-only tuning. Small $\beta$ can **saturate**
  the memory when training data activate almost every KC (tested).
""")
    with c2, card("protocol"):
        card_head("Chronological protocol", "no future data in training")
        st.markdown(
            "1. Withhold one gas (ethanol, ethylene, ammonia, acetaldehyde or acetone).\n"
            "2. Fit everything on **Batch 1 familiar gases only**.\n"
            "3. Freeze scaler, projection, memory and threshold.\n"
            "4. Score Batches 2–10: training gases = familiar, withheld gas = unfamiliar.\n"
            "5. Repeat over 5 held-out gases × 20 seeds; compare with baselines and ablations.")
    footer()


# ======================================================================================
# Dataset
# ======================================================================================
def page_data() -> None:
    report = validate(df)
    n_pass = sum(ok for _, ok, _ in report.checks)
    page_header("Dataset & Validation",
                "UCI Gas Sensor Array Drift Dataset — 16 metal-oxide sensors × 8 features, "
                "10 chronological batches over about 36 months.",
                chip("checks passed", f"{n_pass}/{len(report.checks)}") + chip("DOI 10.24432/C5RP6W"))
    stat_row([("Samples", f"{len(df):,}"), ("Features", str(C.N_FEATURES)),
              ("Batches", str(df["batch"].nunique())), ("Gas classes", str(df["gas_id"].nunique()))])

    c1, c2 = st.columns([1, 1.25], gap="small")
    with c1, card("checks"):
        card_head("Integrity checks", f"{n_pass} passed")
        rep = report.to_frame()
        rep["passed"] = rep["passed"].map({True: "✅", False: "❌"})
        st.dataframe(rep, hide_index=True, width="stretch", height=370)
    with c2, card("counts"):
        card_head("Samples per batch × gas", "class density")
        ct = counts_table(df)
        heat = ct[GAS_ORDER]
        fig = px.imshow(heat.values, x=GAS_ORDER,
                        y=[f"B{b} · mo {C.BATCH_MONTHS[b]}" for b in heat.index],
                        text_auto=True, color_continuous_scale=GREEN_SCALE, aspect="auto")
        fig.update_traces(xgap=3, ygap=3)
        fig.update_layout(coloraxis_showscale=False)
        show(fig, 370)

    callout("Protocol implications", [
        "<b>Toluene is absent from batches 3–5</b> → excluded as a primary held-out gas.",
        f"<b>Batch 1 is small</b> ({int(ct.loc[1, 'Total'])} samples; only "
        f"{int(ct.loc[1, 'Acetaldehyde'])} acetaldehyde) → the training + threshold split is tight.",
        f"<b>Some later batches are tiny per gas</b> (batch 4: {int(ct.loc[4, 'Ammonia'])} ammonia, "
        f"{int(ct.loc[4, 'Acetone'])} acetone) → wide per-batch uncertainty.",
        "<b>Class proportions change across batches</b> → report AUPRC alongside AUROC."],
        right="key finding")
    footer()


# ======================================================================================
# Drift explorer
# ======================================================================================
def page_drift() -> None:
    page_header("Drift Explorer",
                "All batches are standardized with Batch-1 statistics only and projected onto a PCA "
                "fitted on Batch 1 only — the frozen view a deployed model would have.",
                chip("fit on", "B1") + chip("frozen", dark=True))
    with card("pca"):
        card_head("Batch-1 PCA projection", "colour = gas · symbol = batch")
        P, evr = get_pca(df, Z)
        plot_df = pd.DataFrame({"PC1": P[:, 0], "PC2": P[:, 1], "gas": df["gas"], "batch": df["batch"]})
        sel = st.multiselect("Batches to show", list(range(1, 11)), default=[1, 5, 10])
        show_df = plot_df[plot_df["batch"].isin(sel)].copy()
        show_df["batch"] = "B" + show_df["batch"].astype(str)
        fig = px.scatter(show_df, x="PC1", y="PC2", color="gas", symbol="batch", opacity=0.7,
                         color_discrete_map=GAS_COLORS, category_orders={"gas": GAS_ORDER})
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(xaxis_title=f"PC1 ({evr[0]:.0%} of Batch-1 variance)",
                          yaxis_title=f"PC2 ({evr[1]:.0%} of Batch-1 variance)")
        show(fig, 500)

    c1, c2 = st.columns([1.3, 1], gap="small")
    with c1, card("shift"):
        card_head("Centroid shift from Batch 1", "RMS, z-units")
        b1 = df["batch"] == 1
        cent1 = {g: Z[(b1 & (df["gas"] == g)).to_numpy()].mean(axis=0) for g in GAS_ORDER}
        pair = [np.sqrt(np.mean((cent1[a] - cent1[b]) ** 2))
                for i, a in enumerate(GAS_ORDER) for b in GAS_ORDER[i + 1:]]
        rows = []
        for b in range(1, 11):
            for g in GAS_ORDER:
                mask = ((df["batch"] == b) & (df["gas"] == g)).to_numpy()
                if mask.sum():
                    rows.append({"batch": b, "gas": g, "n": int(mask.sum()),
                                 "shift": np.sqrt(np.mean((Z[mask].mean(axis=0) - cent1[g]) ** 2))})
        fig2 = px.line(pd.DataFrame(rows), x="batch", y="shift", color="gas", markers=True,
                       hover_data=["n"], color_discrete_map=GAS_COLORS, category_orders={"gas": GAS_ORDER})
        fig2.add_hline(y=float(np.median(pair)), line_dash="dash", line_color=MUTED,
                       annotation_text="median between-gas distance (B1)")
        fig2.update_layout(xaxis=dict(dtick=1), yaxis_title="RMS centroid shift")
        show(fig2, 380)
    with c2, card("feature"):
        card_head("Single-feature trend", "median per batch")
        s1, s2 = st.columns(2)
        sensor = s1.selectbox("Sensor", list(range(1, 17)), format_func=lambda s: f"S{s:02d}")
        kind = s2.selectbox("Feature", C.FEATURE_KINDS)
        feat = f"S{sensor:02d}_{kind}"
        # Full batch x gas grid so missing combinations (toluene, B3-5) break the line.
        med = (df.groupby(["batch", "gas"])[feat].median()
               .reindex(pd.MultiIndex.from_product([range(1, 11), GAS_ORDER], names=["batch", "gas"]))
               .reset_index())
        fig3 = px.line(med, x="batch", y=feat, color="gas", markers=True,
                       color_discrete_map=GAS_COLORS, category_orders={"gas": GAS_ORDER})
        fig3.update_layout(xaxis=dict(dtick=1), yaxis_title=f"{feat} (median)",
                           legend=dict(orientation="h", y=-0.22, title=None))
        show(fig3, 340)

    callout("Interpretation caution", [
        "Shifts mix sensor ageing with differences in gas concentration, environment and sampling.",
        "They show that the data distribution changes over time — not that every change is sensor drift."])
    footer()


# ======================================================================================
# Sample viewer
# ======================================================================================
def page_sample() -> None:
    page_header("Sample Viewer",
                "One recorded measurement from real metal-oxide sensors — software replay, not live sensing.",
                chip("sensors", 16) + chip("features each", 8))
    c1, c2 = st.columns([1, 2.4], gap="small")
    with c1, card("pick", dashed=True):
        card_head("Select sample")
        batch = st.selectbox("Batch", list(range(1, 11)),
                             format_func=lambda b: f"Batch {b} (months {C.BATCH_MONTHS[b]})")
        idx = np.flatnonzero((df["batch"] == batch).to_numpy())
        pos = st.slider("Sample in batch", 0, len(idx) - 1, 0)
        i = idx[pos]
        if st.toggle("Reveal true gas label"):
            st.html(f'<div class="fx-res hl"><span class="n" style="font-size:1.6rem">{df.at[i, "gas"]}</span>'
                    f'<span class="l">batch {batch}<br>row {df.at[i, "row_in_batch"]}</span></div>')
        st.caption("See the Encoder page for this sample's Kenyon-cell code and the Familiarity page "
                   "for its novelty score.")
    with c2, card("grid"):
        card_head("Sensor response map", "z vs Batch 1 · clipped ±4")
        grid = Z[i].reshape(C.N_SENSORS, C.FEATURES_PER_SENSOR)
        fig = go.Figure(go.Heatmap(z=grid, x=C.FEATURE_KINDS, y=[f"S{s:02d}" for s in range(1, 17)],
                                   colorscale=DIVERGING, zmid=0, zmin=-4, zmax=4, xgap=2, ygap=2,
                                   colorbar=dict(title="z", thickness=10)))
        fig.update_layout(yaxis_autorange="reversed")
        show(fig, 500)
    with st.expander("Raw feature values"):
        st.dataframe(df.loc[[i], C.FEATURE_NAMES].T.rename(columns={i: "value"}), width="stretch")
    footer()


# ======================================================================================
# Baselines
# ======================================================================================
def page_benchmark() -> None:
    res = load_bench()
    if res is None:
        page_header("Benchmark", "No results yet — run scripts/run_week4.py first.")
        return
    week4 = "group" in res.columns
    n_seeds, n_gas = res["seed"].nunique(), res["heldout_gas"].nunique()
    page_header(
        "Benchmark",
        "Every detector: fit on Batch 1 familiar gases → freeze → test on Batches 2–10. Settings "
        "chosen by the same Batch-1-only tuning for every method. Threshold set for "
        f"{C.TARGET_FPR:.0%} false positives on Batch-1 calibration data. “1 (ref)” is the no-drift "
        "reference: Batch-1 calibration samples vs. the held-out gas's Batch-1 samples.",
        chip("held-out gases", n_gas) + chip("seeds", n_seeds)
        + ("" if week4 else chip("untuned baselines", dark=True)))

    metric_labels = {"auroc": "AUROC", "auprc": "AUPRC",
                     "fpr_familiar": "Familiar-gas false-positive rate",
                     "detection_rate": "Novel-gas detection rate"}
    with card("controls", dashed=True):
        c1, c2, c3 = st.columns(3)
        metric = c1.selectbox("Metric", list(metric_labels), format_func=metric_labels.get)
        gas_opt = c2.selectbox("Held-out gas", ["All five (mean)"] + sorted(res["heldout_name"].unique()))
        present = [d for d in DET_COLORS if d in set(res["detector"])]
        dets = c3.multiselect("Detectors", present, default=[d for d in present if d not in ABLATIONS])

    with card("trend"):
        card_head(f"{metric_labels[metric]} by batch", "mean ± sd over seeds")
        sub = res[(res["batch"] != "pooled") & res["detector"].isin(dets)]
        if gas_opt != "All five (mean)":
            sub = sub[sub["heldout_name"] == gas_opt]
        per_seed = sub.groupby(["detector", "batch", "seed"])[metric].mean().reset_index()
        agg = per_seed.groupby(["detector", "batch"])[metric].agg(["mean", "std"]).reset_index()
        order = ["1 (ref)"] + [str(b) for b in C.TEST_BATCHES]
        agg["batch"] = pd.Categorical(agg["batch"], order, ordered=True)
        fig = px.line(agg.sort_values("batch"), x="batch", y="mean", color="detector", error_y="std",
                      markers=True, color_discrete_map=DET_COLORS)
        if metric == "auroc":
            fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED, annotation_text="chance")
        if metric == "fpr_familiar":
            fig.add_hline(y=C.TARGET_FPR, line_dash="dot", line_color=MUTED, annotation_text="target FPR")
        fig.update_layout(xaxis_title="batch (chronological)", yaxis_title=metric_labels[metric])
        show(fig, 420)

    later = res[res["batch"].isin([str(b) for b in C.TEST_BATCHES])]
    ref = res[res["batch"] == "1 (ref)"]
    later_m = later.groupby(["detector", "heldout_name", "seed"])[metric].mean().groupby(
        ["detector", "heldout_name"]).mean().unstack()
    ref_m = ref.groupby(["detector", "heldout_name"])[metric].mean().unstack()

    def styled(t):
        s = t.style.format("{:.3f}")
        return s if metric == "fpr_familiar" else s.background_gradient(cmap="BrBG", vmin=0, vmax=1, axis=None)

    c1, c2 = st.columns(2, gap="small")
    with c1, card("ref"):
        card_head("Batch 1 reference", "no drift")
        st.dataframe(styled(ref_m), width="stretch")
    with c2, card("later"):
        card_head("Batches 2–10", "macro mean")
        st.dataframe(styled(later_m), width="stretch")

    conv = ([n for n, s in SPECS.items() if s.group == "conventional"] if week4
            else [d for d in res["detector"].unique() if d != "Random (sanity)"])
    fpr_conv = later[later["detector"].isin(conv)].groupby("detector")["fpr_familiar"].mean()
    items = [f"Thresholds were set for {C.TARGET_FPR:.0%} false alarms, but on Batches 2–10 the "
             f"conventional detectors flag <b>{fpr_conv.min():.0%}–{fpr_conv.max():.0%}</b> of "
             "familiar-gas samples as novel."]
    if week4:
        auc = macro_later(res, "auroc").groupby("detector")["auroc"].mean()
        fpr = macro_later(res, "fpr_familiar").groupby("detector")["fpr_familiar"].mean()
        best = auc[conv].idxmax()
        items.append(f"Fly model: macro AUROC <b>{auc[FLY_FULL]:.3f}</b> and familiar-gas FPR "
                     f"<b>{fpr[FLY_FULL]:.0%}</b>; best conventional detector ({best}): "
                     f"<b>{auc[best]:.3f}</b>, FPR {fpr[best]:.0%}; cosine-kNN control: "
                     f"<b>{auc['kNN (cosine)']:.3f}</b>.")
    items += ["AUROC below 0.5 means the held-out gas looked <i>more</i> familiar than the drifted "
              "familiar gases.",
              "The Random detector staying at ~0.5 AUROC confirms the evaluation pipeline is unbiased.",
              "Seed-to-seed spread reflects the calibration split and (fly) random wiring only; formal "
              "uncertainty with confidence intervals is on the Statistics page."]
    callout("What these measured numbers say", items, right="measured")
    footer()


# ======================================================================================
# Ablations
# ======================================================================================
def page_ablations() -> None:
    res = load_bench()
    if res is None or "group" not in res.columns:
        page_header("Biological Ablations", "No results yet — run scripts/run_week4.py first.")
        return
    page_header(
        "Biological Ablations",
        "Each variant removes exactly one component of the fly model while keeping the rest — "
        "including the tuned settings, KC count and average sparsity — fixed. A component is "
        "supported only if removing it reliably lowers performance across gases and seeds.",
        chip("variants", len(ABLATIONS)) + chip("pairs per comparison", res["heldout_gas"].nunique()
                                                 * res["seed"].nunique()))
    variants = [FLY_FULL] + ABLATIONS + ["kNN (cosine)"]
    metric_labels = {"auroc": "AUROC · Batches 2–10", "fpr_familiar": "Familiar-gas FPR · Batches 2–10",
                     "detection_rate": "Novel-gas detection · Batches 2–10", "auprc": "AUPRC · Batches 2–10"}
    with card("abl_ctrl", dashed=True):
        metric = st.radio("Metric", list(metric_labels), format_func=metric_labels.get, horizontal=True)

    m = macro_later(res[res["detector"].isin(variants)], metric)
    per_seed = m.groupby(["detector", "seed"])[metric].mean().reset_index()   # mean over 5 gases
    agg = per_seed.groupby("detector")[metric].agg(["mean", "std"]).reindex(variants).reset_index()
    agg["label"] = agg["detector"].map(lambda d: SHORT.get(d, d))
    ref = res[(res["batch"] == "1 (ref)") & res["detector"].isin(variants)]
    ref_auc = ref.groupby("detector")["auroc"].mean().reindex(variants)

    c1, c2 = st.columns([1.5, 1], gap="small")
    with c1, card("abl_bar"):
        card_head(metric_labels[metric], "mean over 5 gases · ± sd over 20 seeds")
        fig = go.Figure(go.Bar(x=agg["label"], y=agg["mean"], error_y=dict(array=agg["std"]),
                               marker_color=[DET_COLORS[d] for d in agg["detector"]],
                               text=[f"{v:.3f}" if metric in ("auroc", "auprc") else f"{v:.0%}"
                                     for v in agg["mean"]], textposition="outside"))
        if metric == "auroc":
            fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED, annotation_text="chance")
            fig.add_scatter(x=agg["label"], y=ref_auc.values, mode="markers", name="Batch 1 (no drift)",
                            marker=dict(symbol="diamond-open", size=11, color=INK))
            fig.update_layout(legend=dict(orientation="h", y=1.08))
        fig.update_layout(yaxis_title=metric_labels[metric], showlegend=metric == "auroc")
        show(fig, 400)
    with c2, card("abl_pair"):
        card_head("Paired vs full model", "same gas · same seed")
        full = m[m["detector"] == FLY_FULL].set_index(["heldout_name", "seed"])[metric]
        rows = []
        for d in ABLATIONS + ["kNN (cosine)"]:
            o = m[m["detector"] == d].set_index(["heldout_name", "seed"])[metric]
            diff = (full - o).dropna()
            better = (diff < 0) if metric == "fpr_familiar" else (diff > 0)
            rows.append({"variant": SHORT[d], "full − variant": round(diff.mean(), 3),
                         "full better": f"{better.mean():.0%}",
                         "gases": f"{int((better.groupby(level=0).mean() > 0.5).sum())}/5"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("“full better” = share of the 100 gas × seed pairs where the full model wins "
                   "(lower is better for FPR). “gases” = held-out gases where it wins in most seeds.")

    with card("abl_gas"):
        card_head(f"{metric_labels[metric]} by held-out gas", "mean over seeds")
        tbl = m.pivot_table(index="detector", columns="heldout_name", values=metric).reindex(variants)
        tbl.index = [SHORT.get(i, i) for i in tbl.index]
        sty = tbl.style.format("{:.3f}")
        if metric != "fpr_familiar":
            sty = sty.background_gradient(cmap="BrBG", vmin=0, vmax=1, axis=None)
        st.dataframe(sty, width="stretch")

    callout("How to read this", [
        "<b>− expansion</b>: 128 KCs instead of 2,000. <b>− sparse wiring</b>: every KC sees every input. "
        "<b>− top-k</b>: fixed threshold with the same average sparsity, no per-sample competition. "
        "<b>− plastic memory</b>: same KC code, but nearest stored code instead of synaptic depression.",
        "<b>cosine kNN</b> is not an ablation but a control: KC overlap tracks cosine "
        "similarity, so the fly model should be judged against it.",
        "A small or inconsistent difference means the data do not support a causal role for that "
        "component under this protocol — that is a valid finding and should be reported as such."])
    footer()


# ======================================================================================
# Statistics (Week 5)
# ======================================================================================
W5_CI, W5_PAIRED = C.RESULTS_DIR / "week5_ci.csv", C.RESULTS_DIR / "week5_paired.csv"
W5_STRENGTH, W5_SUMMARY = C.RESULTS_DIR / "week5_strength.csv", C.RESULTS_DIR / "week5_summary.json"
FIG_DIR = C.RESULTS_DIR / "figures"


def page_stats() -> None:
    if not all(p.exists() for p in (W5_CI, W5_PAIRED, W5_STRENGTH, W5_SUMMARY)):
        page_header("Statistics & Uncertainty", "No results yet — run scripts/run_week5.py first.")
        return
    cis = load_results(str(W5_CI), W5_CI.stat().st_mtime)
    paired = load_results(str(W5_PAIRED), W5_PAIRED.stat().st_mtime)
    mag = load_results(str(W5_STRENGTH), W5_STRENGTH.stat().st_mtime)
    summ = json.loads(W5_SUMMARY.read_text())
    nrep = summ["bootstrap"]["replicates"]
    page_header(
        "Statistics & Uncertainty",
        "95% confidence intervals from a two-level bootstrap: random seeds (split + wiring) and the "
        "test measurements themselves (resampled within each batch and class). Detectors are compared "
        "on identical resamples, so differences are paired; p-values are Holm-corrected.",
        chip("bootstrap replicates", f"{nrep:,}") + chip("Holm-corrected", dark=True))

    with card("st_ctrl", dashed=True):
        c1, c2 = st.columns(2)
        scope = c1.selectbox("Held-out gas", ["All five"] + [C.GAS_NAMES[g] for g in C.PRIMARY_HELDOUT_GASES])
        metric = c2.radio("Metric", ["auroc", "fpr"], horizontal=True,
                          format_func={"auroc": "Macro AUROC", "fpr": "Familiar-gas FPR"}.get)

    c1, c2 = st.columns([1.15, 1], gap="small")
    with c1, card("st_forest"):
        card_head(f"{'Macro AUROC' if metric == 'auroc' else 'Familiar-gas FPR'} · Batches 2–10",
                  "point = mean · bar = 95% CI")
        sub = cis[cis["gas"] == scope].sort_values(metric)
        fig = go.Figure()
        for r in sub.itertuples():
            v, lo, hi = getattr(r, metric), getattr(r, f"{metric}_lo"), getattr(r, f"{metric}_hi")
            fig.add_trace(go.Scatter(
                x=[v], y=[SHORT.get(r.detector, r.detector)], mode="markers",
                marker=dict(size=11 if r.detector == FLY_FULL else 8, color=DET_COLORS.get(r.detector, GREY)),
                error_x=dict(type="data", symmetric=False, array=[hi - v], arrayminus=[v - lo], thickness=1.5),
                hovertemplate=f"{r.detector}<br>{v:.3f} [{lo:.3f}, {hi:.3f}]<extra></extra>", showlegend=False))
        ref = 0.5 if metric == "auroc" else C.TARGET_FPR
        fig.add_vline(x=ref, line_dash="dot", line_color=MUTED,
                      annotation_text="chance" if metric == "auroc" else "target")
        if metric == "fpr":
            fig.update_xaxes(tickformat=".0%")
        show(fig, 430)
    with c2, card("st_paired"):
        card_head("Fly model vs each detector", f"paired · {scope}")
        p = paired[paired["gas"] == scope].copy()

        def verdict(r):
            if not r["significant"]:
                return "no detectable difference"
            return "fly better" if r["diff"] > 0 else "fly worse"
        p["verdict"] = p.apply(verdict, axis=1)
        p["95% CI"] = p.apply(lambda r: f"[{r['lo']:+.3f}, {r['hi']:+.3f}]", axis=1)
        p["vs"] = p["vs"].map(lambda d: SHORT.get(d, d))
        st.dataframe(p[["vs", "diff", "95% CI", "p_holm", "verdict"]].rename(
            columns={"diff": "Δ AUROC", "p_holm": "p (Holm)"}).style.format(
            {"Δ AUROC": "{:+.3f}", "p (Holm)": "{:.3f}"}), hide_index=True, width="stretch", height=430)

    allp = paired[paired["gas"] == "All five"]
    better = [SHORT.get(d, d) for d in allp[(allp.significant) & (allp["diff"] > 0)]["vs"]]
    worse = [SHORT.get(d, d) for d in allp[(allp.significant) & (allp["diff"] < 0)]["vs"]]
    tie = [SHORT.get(d, d) for d in allp[~allp.significant]["vs"]]
    callout("What the statistics say (all five gases)", [
        f"Fly model significantly <b>better</b> than: {', '.join(better) or 'none'}.",
        f"Fly model significantly <b>worse</b> than: {', '.join(worse) or 'none'}.",
        f"No detectable difference from: {', '.join(tie) or 'none'}.",
        "“No detectable difference” is not proof of equality — the interval shows the range of "
        "differences still compatible with the data."], right="Holm-corrected, α = 0.05")

    # ---- strength hypothesis ----
    t = summ["strength_hypothesis"]
    c1, c2 = st.columns([1.3, 1], gap="small")
    with c1, card("st_strength"):
        card_head("Exploratory: does response strength explain the per-gas pattern?",
                  "one point = held-out gas × batch")
        fig = px.scatter(mag, x="strength_separability", y="euclid_minus_cosine", color="heldout",
                         color_discrete_map=GAS_COLORS, hover_data=["batch", "strength_auc"])
        fig.add_hline(y=0, line_color=INK, line_width=1)
        fig.update_traces(marker=dict(size=9, line=dict(color="white", width=0.5)))
        fig.update_layout(xaxis_title="how differently the held-out gas responds in overall strength",
                          yaxis_title="AUROC: Euclidean kNN − cosine kNN",
                          legend=dict(orientation="h", y=-0.25, title=None))
        show(fig, 400)
    with c2:
        stat_row([("Spearman ρ", f"{t['rho_euclid_minus_cosine']:.2f}"),
                  ("Permutation p", f"{t['p_permutation_within_gas']:.3f}")])
        with card("st_strength_tbl"):
            card_head("By held-out gas", "mean over batches")
            g = mag.groupby("heldout")[["strength_separability", "euclid_minus_cosine"]].mean()
            g.columns = ["strength separability", "Euclid − cosine"]
            st.dataframe(g.style.format("{:.3f}"), width="stretch")
        callout("How to read this", [
            "If the explanation is right, points should rise to the right: the more the held-out "
            "gas differs in <b>strength</b>, the more a magnitude-sensitive method (Euclidean) beats a "
            "magnitude-blind one (cosine — and the fly model's top-k code).",
            "Exploratory: the idea came after seeing the benchmark results, and the 45 points are not independent "
            "(the permutation test shuffles batches within each gas)."], right="post hoc")

    # ---- saturation ----
    sat = summ["saturation_by_gas"]
    w0 = np.mean([v["frac_w0"] for v in sat.values()])
    never = np.mean([v["frac_never_active"] for v in sat.values()])
    nov = summ["fly_mean_novelty_of_familiar_by_batch"]
    c1, c2 = st.columns([1, 1.3], gap="small")
    with c1:
        stat_row([("KCs depressed to w = 0", f"{w0:.0%}"), ("KCs never active", f"{never:.0%}")])
        callout("Memory saturation", [
            "With the tuned β = 0, any Kenyon cell active for even one training sample is silenced.",
            f"After Batch-1 training, about <b>{w0:.0%}</b> of cells are silenced and {never:.0%} are "
            "untouched; a familiar test sample looks novel only to the extent its winners fall on "
            "untouched cells."])
    with c2, card("st_sat"):
        card_head("Fly novelty score of FAMILIAR gases by batch", "higher = looks less familiar")
        s = pd.Series(nov).rename(index=int).sort_index()
        fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines+markers", line=dict(color=FLY_COLOR)))
        fig.update_layout(xaxis=dict(dtick=1, title="batch"), yaxis_title="mean novelty score")
        show(fig, 280)

    if FIG_DIR.exists():
        with st.expander("Report figures (PNG, saved in results/figures)"):
            figs = sorted(FIG_DIR.glob("fig*.png"))
            for a, b in zip(figs[::2], figs[1::2] + [None]):
                x1, x2 = st.columns(2)
                x1.image(str(a), caption=a.name)
                if b is not None:
                    x2.image(str(b), caption=b.name)
    footer()


# ======================================================================================
# Familiarity demo
# ======================================================================================
def _gauge(score: float, thr: float) -> str:
    return (f'<div style="position:relative;height:16px;background:linear-gradient(90deg,{LIME},#F1E2C9);'
            f'border:1px solid {LINE};border-radius:3px;margin:22px 0 6px">'
            f'<div style="position:absolute;left:{thr*100:.1f}%;top:-18px;bottom:-4px;border-left:2px dashed {INK}">'
            f'<span class="fx-label" style="position:absolute;left:4px;top:-2px;white-space:nowrap">'
            f'threshold {thr:.3f}</span></div>'
            f'<div style="position:absolute;left:calc({score*100:.1f}% - 7px);top:1px;width:14px;height:14px;'
            f'border-radius:50%;background:{FLY_COLOR};border:2px solid #fff"></div></div>'
            f'<div class="fx-label m" style="display:flex;justify-content:space-between">'
            f'<span>0 · familiar</span><span>1 · completely new</span></div>')


def page_familiarity() -> None:
    page_header(
        "Familiarity Detector",
        "The complete fly model, trained live on Batch-1 familiar gases with one gas withheld. "
        "Pick any recorded measurement: the model reports how novel it looks and calls it familiar or "
        "unfamiliar. Same protocol and threshold rule as the benchmark — nothing is fitted on later batches.",
        chip("live inference on recorded data", dark=True))

    c1, c2 = st.columns([1, 2.3], gap="small")
    with c1, card("fam_ctrl", dashed=True):
        card_head("Experiment")
        gas = st.selectbox("Withheld (unfamiliar) gas", C.PRIMARY_HELDOUT_GASES, index=4,
                           format_func=C.GAS_NAMES.get)
        seed = st.number_input("Seed (split + wiring)", 0, C.N_SEEDS - 1, 0)
        mtime = TUNING_JSON.stat().st_mtime if TUNING_JSON.exists() else 0.0
        det, Zs, thr, scores, role, params = fit_fly_demo(gas, int(seed), mtime)
        st.caption("Tuned on Batch 1: " + ", ".join(f"{k}={v}" for k, v in params.items()))
        st.divider()
        batch = st.selectbox("Batch", list(range(1, 11)), index=9, key="fam_b",
                             format_func=lambda b: f"Batch {b} (months {C.BATCH_MONTHS[b]})")
        idx = np.flatnonzero((df["batch"] == batch).to_numpy())

        def _random():
            st.session_state["fam_s"] = int(np.random.default_rng().integers(len(idx)))
        st.button("Random sample", on_click=_random, width="stretch")
        if st.session_state.get("fam_s", 0) > len(idx) - 1:   # batch changed to a smaller one
            st.session_state["fam_s"] = 0
        pos = st.slider("Sample in batch", 0, len(idx) - 1, 0, key="fam_s")
        i = idx[min(pos, len(idx) - 1)]

    s, unfamiliar = float(scores[i]), bool(scores[i] > thr)
    truth_novel = int(df.at[i, "gas_id"]) == gas
    with c2:
        with card("fam_result"):
            card_head("Model output", f"batch {batch} · row {df.at[i, 'row_in_batch']}")
            st.html(f'<div class="fx-grid" style="grid-template-columns:1fr 1fr;margin:0">'
                    f'<div class="fx-res" style="margin:0"><span class="n">{s:.3f}</span>'
                    f'<span class="l">novelty score<br>mean memory weight of active KCs</span></div>'
                    f'<div class="fx-res {"hl" if unfamiliar else ""}" style="margin:0;'
                    f'{"background:" + FOREST + ";color:#EEF2E2" if not unfamiliar else ""}">'
                    f'<span class="n" style="font-size:1.9rem">{"UNFAMILIAR" if unfamiliar else "FAMILIAR"}</span>'
                    f'<span class="l">prediction<br>score {">" if unfamiliar else "≤"} threshold</span></div></div>'
                    + _gauge(min(max(s, 0), 1), min(max(thr, 0), 1)))
            if st.toggle("Reveal true gas label", key="fam_reveal"):
                ok = unfamiliar == truth_novel
                st.html(f'<div class="fx-res {"hl" if ok else ""}" style="margin-top:10px">'
                        f'<span class="n" style="font-size:1.6rem">{df.at[i, "gas"]}</span>'
                        f'<span class="l">{"withheld gas → should be UNFAMILIAR" if truth_novel else "trained gas → should be FAMILIAR"}'
                        f'<br><b>{"✔ correct" if ok else "✘ wrong"}</b> · {role[i]}</span></div>')
        g1, g2 = st.columns(2, gap="small")
        with g1, card("fam_kc"):
            code = det.codes(Zs[i][None, :])[0]
            w = det.memory_.w_[code]
            card_head("Active Kenyon cells", f"{int(code.sum())} active · memory weight each")
            fig = go.Figure(go.Bar(y=np.sort(w)[::-1], marker_color=[FLY_COLOR if v > 0.5 else SAGE
                                                                    for v in np.sort(w)[::-1]]))
            fig.update_layout(yaxis_range=[0, 1.05], xaxis_title="active KC (sorted)",
                              yaxis_title="synaptic weight w", bargap=0.05)
            show(fig, 260)
            st.caption("w = 1: this KC was (almost) never active during training → evidence of novelty. "
                       "The novelty score is the mean of these bars.")
        with g2, card("fam_dist"):
            card_head(f"Batch {batch} score distribution", "trained vs withheld gas")
            b = (df["batch"] == batch).to_numpy()
            nov = (df["gas_id"] == gas).to_numpy()
            dd = pd.DataFrame({"score": scores[b], "gas": np.where(nov[b], "withheld gas", "trained gases")})
            fig2 = px.histogram(dd, x="score", color="gas", nbins=40, barmode="overlay", opacity=0.75,
                                color_discrete_map={"withheld gas": FLY_COLOR, "trained gases": OLIVE})
            fig2.add_vline(x=thr, line_dash="dash", line_color=INK)
            fig2.update_layout(legend=dict(orientation="h", y=1.12, title=None), yaxis_title="samples")
            show(fig2, 260)
            fam_b, nov_b = b & ~nov, b & nov
            fpr = float((scores[fam_b] > thr).mean()) if fam_b.any() else float("nan")
            det_rate = float((scores[nov_b] > thr).mean()) if nov_b.any() else float("nan")
            st.caption(f"In this batch: {fpr:.0%} of trained-gas samples flagged (false alarms), "
                       f"{det_rate:.0%} of withheld-gas samples detected.")
    footer()


# ======================================================================================
# Kenyon-cell encoder
# ======================================================================================
def page_encoder() -> None:
    page_header("Kenyon-Cell Encoder",
                "Standardized sensor features (≈ projection neurons) → fixed random sparse wiring → "
                "top-k winner-take-all (≈ APL inhibition) → binary sparse code. This page shows the "
                "representation; the Familiarity page adds the memory and the novelty score.")
    c1, c2 = st.columns([1, 2.4], gap="small")
    with c1, card("enc_ctrl", dashed=True):
        card_head("Encoder settings")
        n_kc = st.select_slider("Kenyon cells (m)", [500, 1000, 2000, 4000], value=2000)
        conn = st.slider("Inputs per KC", 2, 20, 7)
        act_pct = st.slider("Active KCs (%)", 1, 20, 5)
        wseed = st.number_input("Wiring seed", 0, 999, 0)
        st.divider()
        batch = st.selectbox("Batch", list(range(1, 11)), key="kc_b",
                             format_func=lambda b: f"Batch {b} (months {C.BATCH_MONTHS[b]})")
        idx = np.flatnonzero((df["batch"] == batch).to_numpy())
        pos = st.slider("Sample in batch", 0, len(idx) - 1, 0, key="kc_s")
    enc = get_encoder(n_kc, conn, act_pct / 100, int(wseed))
    i = idx[pos]
    drive = enc.drive(Z[i])[0]
    code = enc.encode(Z[i])[0]

    with c2:
        stat_row([("Kenyon cells", f"{enc.n_kc:,}"), ("Active cells (k)", str(int(code.sum()))),
                  ("Fraction active", f"{code.mean():.1%}")])
        with card("kcmap"):
            card_head(f"KC activity map", f"{enc.k} of {enc.n_kc:,} cells on")
            cols = 50
            rows_ = int(np.ceil(enc.n_kc / cols))
            grid = np.full(rows_ * cols, np.nan)
            grid[: enc.n_kc] = code.astype(float)
            fig2 = go.Figure(go.Heatmap(z=grid.reshape(rows_, cols), showscale=False, xgap=2, ygap=2,
                                        colorscale=[[0, "#E3E7D6"], [1, FOREST]]))
            fig2.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False, autorange="reversed"))
            show(fig2, 330)

    g1, g2 = st.columns(2, gap="small")
    with g1, card("wta"):
        card_head("Winner-take-all", "only the top-k drives survive")
        order = np.argsort(-drive)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=drive[order], mode="lines", line=dict(color=GREY)))
        fig.add_trace(go.Scatter(x=np.arange(enc.k), y=drive[order[:enc.k]], mode="markers",
                                 marker=dict(color=FOREST, size=4)))
        fig.add_hline(y=drive[order[enc.k - 1]], line_dash="dot", line_color=MUTED,
                      annotation_text="k-th largest drive")
        fig.update_layout(showlegend=False, xaxis_title="KC rank", yaxis_title="summed input (W·z)")
        show(fig, 320)
    with g2, card("overlap"):
        card_head("Shared cells with Batch-1 gases", "overlap = |A ∩ B| / k")
        b1_mask = (df["batch"] == 1).to_numpy()
        ov = overlap(code[None, :], enc.encode(Z[b1_mask]))[0]
        stats = pd.DataFrame({"gas": df.loc[b1_mask, "gas"].to_numpy(), "overlap": ov}).groupby(
            "gas")["overlap"].agg(["mean", "max"]).reindex(GAS_ORDER).reset_index()
        fig3 = go.Figure([
            go.Bar(x=stats["gas"], y=stats["mean"], name="mean overlap", marker_color=SAGE),
            go.Bar(x=stats["gas"], y=stats["max"], name="best single match", marker_color=FOREST)])
        fig3.update_layout(barmode="group", yaxis_range=[0, 1], yaxis_title="shared active KCs / k",
                           legend=dict(orientation="h", y=1.1))
        show(fig3, 320)
        if st.toggle("Reveal true gas label", key="kc_reveal"):
            st.html(f'<div class="fx-res hl"><span class="n" style="font-size:1.6rem">{df.at[i, "gas"]}</span>'
                    f'<span class="l">batch {batch}<br>row {df.at[i, "row_in_batch"]}</span></div>')

    if not (DIAG_JSON.exists() and NN_CSV.exists()):
        callout("Diagnostics missing", ["Run <b>python scripts/encoder_diagnostics.py</b> first."])
        return
    mu = json.loads(DIAG_JSON.read_text())["mean"]
    st.html('<div class="fx-label" style="margin:18px 0 8px">Diagnostics on real data · default encoder · '
            '20 wiring seeds</div>')
    stat_row([("Active per sample", f"{mu['active_frac_per_sample']:.1%}"),
              ("Similarity preserved ρ", f"{mu['spearman_cos_vs_overlap_batch1']:.3f}"),
              ("Overlap same / diff gas",
               f"{mu['overlap_within_gas_batch1']:.2f} / {mu['overlap_between_gas_batch1']:.2f}"),
              ("Silent KCs on Batch 1", f"{mu['silent_kc_frac_batch1']:.0%}")])
    with card("nn"):
        card_head("Nearest Batch-1 neighbour is the same gas", "B1 = leave-one-out")
        nn = load_results(str(NN_CSV), NN_CSV.stat().st_mtime)
        agg = nn.groupby(["batch", "space"])["nn_same_gas"].agg(["mean", "std"]).reset_index()
        agg["batch"] = agg["batch"].astype(int)
        fig4 = px.line(agg, x="batch", y="mean", color="space", error_y="std", markers=True,
                       color_discrete_map=SPACE_COLORS)
        fig4.update_layout(xaxis=dict(dtick=1), yaxis_tickformat=".0%", yaxis_title="agreement")
        show(fig4, 380)
    callout("Reading this fairly", [
        "KC overlap behaves much like cosine similarity on the input (ρ ≈ 0.96), so <b>Input, cosine</b> "
        "is the right control.",
        "Against Euclidean distance the KC code looks better in batches 2–5, but most of that gap "
        "disappears against cosine.",
        "This is a descriptive check of the representation, not the novelty-detection benchmark."])
    footer()


# ======================================================================================
nav = st.navigation([
    st.Page(page_home, title="Overview", default=True),
    st.Page(page_data, title="Dataset", url_path="data"),
    st.Page(page_drift, title="Drift", url_path="drift"),
    st.Page(page_sample, title="Samples", url_path="samples"),
    st.Page(page_encoder, title="Encoder", url_path="encoder"),
    st.Page(page_familiarity, title="Familiarity", url_path="familiarity"),
    st.Page(page_benchmark, title="Benchmark", url_path="benchmark"),
    st.Page(page_ablations, title="Ablations", url_path="ablations"),
    st.Page(page_stats, title="Statistics", url_path="statistics"),
], position="top")
nav.run()
