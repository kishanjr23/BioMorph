import numpy as np
import pytest

from biomorph.data import (
    DEFAULT_DATA_DIR,
    FEATURE_NAMES,
    N_FEATURES,
    counts_table,
    load_all,
    parse_line,
)

LINE = "3 " + " ".join(f"{i}:{i * 0.5}" for i in range(1, N_FEATURES + 1))


def test_parse_line_ok():
    label, x = parse_line(LINE)
    assert label == 3
    assert x.shape == (N_FEATURES,)
    assert x[0] == 0.5 and x[-1] == N_FEATURES * 0.5


def test_parse_line_accepts_concentration_suffix():
    label, _ = parse_line("5;50.000000 " + LINE.split(" ", 1)[1])
    assert label == 5


@pytest.mark.parametrize(
    "bad",
    [
        "7 " + LINE.split(" ", 1)[1],  # unknown label
        "1 1:0.1 2:0.2",  # missing features
        LINE + " 1:9.9",  # duplicate index
        LINE + " 129:1.0",  # out of range
    ],
)
def test_parse_line_rejects_bad(bad):
    with pytest.raises(ValueError):
        parse_line(bad)


def test_feature_names():
    assert len(FEATURE_NAMES) == N_FEATURES
    assert FEATURE_NAMES[0] == "s01_dR" and FEATURE_NAMES[8] == "s02_dR"


needs_data = pytest.mark.skipif(
    not (DEFAULT_DATA_DIR / "batch1.dat").exists(),
    reason="dataset not downloaded (python scripts/download_data.py)",
)


@needs_data
def test_full_dataset_validates():
    ds = load_all(validate=True)
    ct = counts_table(ds)
    assert ct.loc["Total", "Total"] == 13_910
    assert ct.loc[1, "Total"] == 445
    assert (ct.loc[[3, 4, 5], "Toluene"] == 0).all()
    assert np.unique(ds.batch).tolist() == list(range(1, 11))
