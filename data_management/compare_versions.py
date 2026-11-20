"""
compare_versions.py — compare a dataset built by the old downloader (rows labelled at
bar OPEN) with one built by the current version (rows labelled at bar CLOSE).

With the same source data the two must agree:
  - 15min columns: new[t] == old[t - 15min]   (same bar, new label = its close time)
  - higher-TF columns: new[t] == old[t]       (old: shift(1) on open labels; new: close
                                               labels without shift -> same value at t)

Columns expected to differ are reported apart:
  - spread_mean/max_15min: one-sided minutes now have an unknown spread instead of 0;
  - volume / vwp: if the old file was built before the one-sided volume fix;
  - long EMAs on 1D (and their slopes): if the old file was extended with the old
    `update`, whose warm-up was too short for them to converge.

Usage:
    python -m data_management.compare_versions --old D:/DataOld --new D:/Data --assets EURUSD
    python -m data_management.compare_versions --old-file old.parquet --new-file new.parquet
Exits with status 1 if a column outside the expected list differs.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, storage

EXPECTED_DIFF = re.compile(
    r"^(spread_(mean|max)_15min|volume_.*|vwp_.*|EMA\d+(_slope)?_1D)$", re.IGNORECASE
)
RTOL, ATOL = 1e-7, 1e-12
MIN_MATCH = 0.999  # share of rows that must agree


def compare(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Per-column agreement between an old (open-labelled) and a new (close-labelled) frame."""
    rows = []
    shifted = old.copy()
    shifted.index = shifted.index + pd.Timedelta(minutes=15)
    for c in new.columns.intersection(old.columns):
        ref = shifted[c] if c.endswith("_15min") else old[c]
        idx = new.index.intersection(ref.index)
        a = new.loc[idx, c].to_numpy(dtype=float)
        b = ref.loc[idx].to_numpy(dtype=float)
        same = np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)
        diff = np.abs(a - b)
        rows.append(
            {
                "column": c,
                "rows": len(idx),
                "match": float(same.mean()) if len(idx) else np.nan,
                "max_abs_diff": float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0,
                "expected_diff": bool(EXPECTED_DIFF.match(c)),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def report(name: str, old: pd.DataFrame, new: pd.DataFrame, old_meta: dict, new_meta: dict):
    print(f"\n=== {name} ===")
    if old_meta.get("label") == "close":
        print("  the 'old' file is already close-labelled: nothing to compare")
        return True
    if new_meta.get("label") != "close":
        print("  the 'new' file is not close-labelled (built by the old downloader?)")
        return False
    res = compare(old, new)
    only_old = old.columns.difference(new.columns)
    only_new = new.columns.difference(old.columns)
    if len(only_old) or len(only_new):
        print(f"  columns only in old: {list(only_old)[:6]}  only in new: {list(only_new)[:6]}")
    print(f"  common rows: {res['rows'].max():,}  common columns: {len(res)}")

    bad = res[(res["match"] < MIN_MATCH) & ~res["expected_diff"]]
    expected = res[(res["match"] < MIN_MATCH) & res["expected_diff"]]
    print(f"  columns that agree: {int((res['match'] >= MIN_MATCH).sum())}/{len(res)}")
    if len(expected):
        print("  expected differences (see module docstring):")
        print(expected[["match", "max_abs_diff"]].round(6).to_string().replace("\n", "\n    "))
    if len(bad):
        print("  UNEXPECTED differences:")
        print(bad[["match", "max_abs_diff"]].round(6).to_string().replace("\n", "\n    "))
    return bad.empty


def _latest(folder: Path):
    files = sorted(folder.glob("*_multiTF_*.parquet"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main():
    p = argparse.ArgumentParser(description="Compare old (open-labelled) and new datasets.")
    p.add_argument("--old", type=Path, help="data root built by the old downloader")
    p.add_argument("--new", type=Path, help="data root built by the current downloader")
    p.add_argument("--old-file", type=Path)
    p.add_argument("--new-file", type=Path)
    p.add_argument("--group", nargs="*")
    p.add_argument("--assets", nargs="*")
    args = p.parse_args()

    if args.old_file and args.new_file:
        pairs = [(args.new_file.stem, args.old_file, args.new_file)]
    elif args.old and args.new:
        pairs = [
            (
                f"{g}/{a}",
                _latest(args.old / "Raw" / g / a),
                _latest(args.new / "Raw" / g / a),
            )
            for g, a in config.iter_assets(args.group, args.assets)
        ]
    else:
        p.error("give --old and --new roots, or --old-file and --new-file")

    ok = True
    for name, old_path, new_path in pairs:
        if old_path is None or new_path is None:
            print(f"\n=== {name} === missing file (old: {old_path}, new: {new_path})")
            ok = False
            continue
        ok &= report(
            name,
            pd.read_parquet(old_path),
            pd.read_parquet(new_path),
            storage.read_meta(old_path),
            storage.read_meta(new_path),
        )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
