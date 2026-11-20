"""
storage.py — parquet I/O with a small metadata block.

Every parquet written by the pipeline carries a JSON entry in its schema metadata
(key ``forex_rl_pipeline``) with the schema version, the timestamp convention and,
for feature files, the indicators it contains. It travels with the file, so a reader
can tell how it was built without guessing from the column names.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

META_KEY = b"forex_rl_pipeline"
# 2 = rows labelled at bar CLOSE + separate 15min bars cache (1 = open-labelled rows)
SCHEMA_VERSION = 2


def write_parquet(df: pd.DataFrame, path, meta: dict | None = None):
    table = pa.Table.from_pandas(df)
    md = dict(table.schema.metadata or {})
    md[META_KEY] = json.dumps({"schema": SCHEMA_VERSION, **(meta or {})}).encode()
    pq.write_table(table.replace_schema_metadata(md), path)


def read_meta(path) -> dict:
    """Pipeline metadata of a parquet ({} for files written before schema 2)."""
    raw = (pq.read_schema(path).metadata or {}).get(META_KEY)
    return json.loads(raw) if raw else {}


def latest(folder, prefix: str):
    """Most recently modified <prefix>*.parquet in folder, or None."""
    if not os.path.isdir(folder):
        return None
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.startswith(prefix) and f.endswith(".parquet")
    ]
    return max(files, key=os.path.getmtime) if files else None


def save_single(df: pd.DataFrame, folder, prefix: str, meta: dict | None = None) -> str:
    """
    Save as <prefix><first row>_to_<last row>.parquet and remove every other
    <prefix>*.parquet in the folder: always exactly one file per kind and asset.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(
        folder, f"{prefix}{df.index.min():%Y_%m_%d}_to_{df.index.max():%Y_%m_%d}.parquet"
    )
    write_parquet(df, path, meta)
    print(f"Saved: {path}")
    for f in os.listdir(folder):
        old = os.path.join(folder, f)
        if f.startswith(prefix) and f.endswith(".parquet") and old != path:
            os.remove(old)
            print(f"Removed old file: {f}")
    return path
