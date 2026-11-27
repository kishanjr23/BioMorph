import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flyodor.data import feature_matrix, load_all  # noqa: E402


@pytest.fixture(scope="session")
def df():
    return load_all()


@pytest.fixture(scope="session")
def X(df):
    return feature_matrix(df)
