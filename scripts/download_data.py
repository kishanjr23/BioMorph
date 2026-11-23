"""Download the UCI Gas Sensor Array Drift Dataset into data/raw/.

Usage:  python scripts/download_data.py

Source: https://doi.org/10.24432/C5RP6W
The SHA-256 below is of the zip downloaded on 2026-11-23; a mismatch means
UCI changed the file and the audit should be re-checked.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"
SHA256 = "91e8f466f202e7a093d657673ce47311c3e90416f7df3057966058961c351fe4"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
ZIP_PATH = RAW_DIR / "gas_sensor_array_drift.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, ZIP_PATH)
    digest = sha256(ZIP_PATH)
    if digest != SHA256:
        print(f"WARNING: checksum mismatch\n  expected {SHA256}\n  got      {digest}")
    else:
        print("Checksum OK")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(RAW_DIR)
    files = sorted((RAW_DIR / "Dataset").glob("batch*.dat"))
    print(f"Extracted {len(files)} batch files to {RAW_DIR / 'Dataset'}")
    return 0 if len(files) == 10 else 1


if __name__ == "__main__":
    sys.exit(main())
