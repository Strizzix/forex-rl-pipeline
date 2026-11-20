"""
scale.py — per-column scaling of the multi-TF parquets made by downloader.py.

Every column gets its own sklearn transformer, chosen from the feature name
(the TF suffix doesn't matter):
  - StandardScaler     : prices (OHLC, EMA, WMA, VWP, pivots), ATR, returns, MACD
  - RobustScaler       : slopes, CCI, BB width, candle shapes, volatility regime
  - MinMaxScaler       : RSI, ADX, Stoch
  - log1p + Robust     : volume
  - not scaled         : spread_mean / spread_max -> only kept as <col>_raw

Raw copies: open/high/low/close/volume/ATR14 at 15min are also kept as <col>_raw
(real prices needed downstream for returns and targets). Higher-TF prices can be
recovered with inverse_transform from the saved scalers.

LEAK-SAFE: scalers are fit ONLY on rows <= FIT_CUTOFF_DATE (2023+ holdout never
seen). If saved scalers cover all the columns they are reused as they are
(no refit), so new data goes through exactly the same transform.

Saved as <root>/Scaled/<group>/<asset>/all_scalers.pkl (joblib):
    {"meta": {...}, "col_scalers": {col: fitted transformer}, "cols": [...]}

Usage:
    python -m data_management.scale --root D:/Data                # reuse saved scalers if complete
    python -m data_management.scale --root D:/Data --refit        # refit on <= cutoff
    python -m data_management.scale --root D:/Data --diagnose     # only print the scaler per column
    python -m data_management.scale --apply all_scalers.pkl --input new.parquet
"""

from __future__ import annotations

import argparse
import datetime
import logging
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, RobustScaler, StandardScaler

from . import config

FIT_CUTOFF_DATE = config.FIT_CUTOFF_DATE
JOBLIB_COMPRESS = 3
VERSION = "3.4"  # kept for compatibility with the existing all_scalers.pkl

# 15min columns also kept as <col>_raw before being overwritten with scaled values
RAW_COLUMNS_TO_REPLACE = {
    "open_15min",
    "high_15min",
    "low_15min",
    "close_15min",
    "volume_15min",
    "atr14_15min",
}
# never scaled, only kept as <col>_raw (original removed). Substring match.
RAW_ONLY_TOKENS = {"spread_mean", "spread_max"}

TF_SUFFIX = re.compile(r"_(15min|30min|1h|4h|1d)$")

# scaler by feature root (lowercase, no TF suffix)
SCALER_BY_ROOT = {
    **dict.fromkeys(
        [
            "open",
            "high",
            "low",
            "close",
            "vwp",
            "momentum_proxy",
            "atr14",
            "atr50",
            "atr_norm",
            "volatility",
            "log_return",
            "abs_return",
            "macd",
            "macd_signal",
            "macd_diff",
            "pivot",
            "r1",
            "s1",
            "r2",
            "s2",
        ],
        "StandardScaler",
    ),
    **dict.fromkeys(
        [
            "cci20",
            "bb_width",
            "volatility_regime",
            "high_low_range",
            "close_open_diff",
            "candle_body",
            "candle_shadow",
            "candle_pressure",
            "range_ratio",
        ],
        "RobustScaler",
    ),
    **dict.fromkeys(["adx14", "rsi14", "stoch"], "MinMaxScaler"),
    "volume": "Log1p+RobustScaler",
    "timestamp": "Passthrough",
}


def _is_raw_only(col):
    return any(tok in col.lower() for tok in RAW_ONLY_TOKENS)


def scaler_name(col: str, series: pd.Series | None = None) -> str:
    """Scaler for a column: by root name, EMA/WMA by pattern, otherwise by skewness."""
    if _is_raw_only(col):
        return "Passthrough"
    root = TF_SUFFIX.sub("", col.lower())
    if root in SCALER_BY_ROOT:
        return SCALER_BY_ROOT[root]
    if re.fullmatch(r"(ema|wma)\d+", root):
        return "StandardScaler"
    if re.fullmatch(r"ema\d+_slope", root):
        return "RobustScaler"
    # unknown column (e.g. DI+/DI-): robust if skewed, standard otherwise
    try:
        skew = float(series.dropna().skew()) if series is not None else 0.0
    except Exception:
        skew = 0.0
    return "RobustScaler" if abs(skew) > 1.0 else "StandardScaler"


def make_pipeline(name: str):
    if name == "Passthrough":
        return None
    if name == "Log1p+RobustScaler":
        return Pipeline(
            [("log1p", FunctionTransformer(np.log1p, validate=False)), ("robust", RobustScaler())]
        )
    scaler = {
        "StandardScaler": StandardScaler,
        "RobustScaler": RobustScaler,
        "MinMaxScaler": MinMaxScaler,
    }[name]()
    return Pipeline([("scaler", scaler)])


def feature_columns(df: pd.DataFrame) -> list:
    """Numeric columns to scale (no *_raw, no raw-only)."""
    return [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not c.lower().endswith("_raw")
        and not _is_raw_only(c)
    ]


# ── fit / transform ─────────────────────────────────────────────────────────
def fit_column_scalers(df_fit: pd.DataFrame, cols: list) -> dict:
    col_scalers = {}
    for c in cols:
        pipe = make_pipeline(scaler_name(c, df_fit[c]))
        if pipe is None:
            continue
        try:
            col_scalers[c] = pipe.fit(df_fit[[c]])
        except Exception as e:
            logging.exception("Fit error on column %s: %s", c, e)
    return col_scalers


def transform_with_column_scalers(df: pd.DataFrame, col_scalers: dict) -> pd.DataFrame:
    """
    Overwrite every column that has a scaler with its scaled values.
    RAW_COLUMNS_TO_REPLACE also get a <col>_raw copy, raw-only columns are moved
    to <col>_raw. New columns are added with a single concat (no fragmentation).
    """
    scaled, new_raw = {}, {}
    for c, scaler in col_scalers.items():
        if c not in df.columns:
            logging.warning("Column %s missing in the data, skipped.", c)
            continue
        if c.lower() in RAW_COLUMNS_TO_REPLACE and f"{c}_raw" not in df.columns:
            new_raw[f"{c}_raw"] = df[c].rename(f"{c}_raw")
        scaled[c] = scaler.transform(df[[c]]).ravel()

    raw_only = [c for c in df.columns if _is_raw_only(c)]
    for c in raw_only:
        if f"{c}_raw" not in df.columns:
            new_raw[f"{c}_raw"] = df[c].rename(f"{c}_raw")

    out = df.copy()
    if scaled:
        cols = list(scaled)
        out[cols] = pd.DataFrame(scaled, index=df.index)[cols].values
    if new_raw:
        out = pd.concat([out, pd.DataFrame(new_raw, index=df.index)], axis=1)
    return out.drop(columns=raw_only)


def fit_leak_safe(df: pd.DataFrame, cols: list, cutoff: str = FIT_CUTOFF_DATE) -> dict:
    """Fit ONLY on rows <= cutoff. No silent fallback on the whole dataset."""
    cut = pd.Timestamp(cutoff)
    if df.index.tz is not None and cut.tzinfo is None:
        cut = cut.tz_localize(df.index.tz)
    df_fit = df[df.index <= cut]
    if df_fit.empty:
        raise RuntimeError(f"no rows <= cutoff {cutoff}: can't fit without using the future.")
    logging.info("Fit on <= %s (%d rows), transform on all (%d rows).", cut, len(df_fit), len(df))
    return fit_column_scalers(df_fit, cols)


def load_scalers(path: Path):
    if path.exists():
        try:
            return joblib.load(path)
        except Exception as e:
            logging.warning("Can't load scalers from %s: %s", path, e)
    return None


def _ensure_datetime_index(df):
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


# ── per asset ───────────────────────────────────────────────────────────────
def scale_and_save(
    df: pd.DataFrame,
    asset: str,
    class_name: str,
    save_root: Path,
    refit: bool = False,
    cutoff: str = FIT_CUTOFF_DATE,
):
    """Scale one asset and save <asset>_scaled_*.parquet + all_scalers.pkl."""
    df = _ensure_datetime_index(df)
    cols = feature_columns(df)
    save_path = Path(save_root) / class_name / asset
    save_path.mkdir(parents=True, exist_ok=True)
    scalers_path = save_path / "all_scalers.pkl"

    saved = None if refit else load_scalers(scalers_path)
    if saved and set(cols).issubset(saved.get("cols", [])):
        logging.info("Using saved scalers for %s", asset)
        col_scalers = saved["col_scalers"]
        strategy = "use_saved_if_available"
    else:
        if not refit:
            logging.info("Saved scalers missing or incomplete for %s -> fit", asset)
        col_scalers = fit_leak_safe(df, cols, cutoff)
        strategy = "fit_then_transform"

    df_scaled = transform_with_column_scalers(df, col_scalers)

    name = f"{asset}_scaled_{df.index[0]:%Y_%m_%d}_to_{df.index[-1]:%Y_%m_%d}.parquet"
    out_file = save_path / name
    df_scaled.to_parquet(out_file)
    logging.info("Scaled parquet saved: %s", out_file)
    # one scaled parquet per asset: old ones would break glob("*.parquet") downstream
    for old in save_path.glob(f"{asset}_scaled_*.parquet"):
        if old.resolve() != out_file.resolve():
            old.unlink()
            logging.info("Removed old scaled parquet: %s", old.name)

    meta = {
        "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "asset": asset,
        "class_name": class_name,
        "strategy": strategy,
        "fit_cutoff_date": cutoff,
        "version": VERSION,
    }
    joblib.dump(
        {"meta": meta, "col_scalers": col_scalers, "cols": list(col_scalers)},
        scalers_path,
        compress=JOBLIB_COMPRESS,
    )
    logging.info("Scalers saved: %s (cols=%d)", scalers_path, len(col_scalers))
    return out_file


def apply_saved_scalers(scalers_path: Path, input_parquet: Path, output: Path | None = None):
    """Transform a new parquet with saved scalers (no fit)."""
    saved = joblib.load(scalers_path)
    df = _ensure_datetime_index(pd.read_parquet(input_parquet))
    col_scalers = {c: s for c, s in saved["col_scalers"].items() if not _is_raw_only(c)}
    missing = [c for c in col_scalers if c not in df.columns]
    if missing:
        logging.warning("%d scaler columns missing in the input: %s", len(missing), missing[:10])
    out = output or input_parquet.with_name(input_parquet.stem + "_with_scalers.parquet")
    transform_with_column_scalers(df, col_scalers).to_parquet(out)
    logging.info("Transformed parquet saved: %s", out)
    return out


def latest_parquet(folder: Path):
    files = sorted(folder.glob("*.parquet"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Scale the multi-TF parquets (per-column scalers, leak-safe fit)."
    )
    p.add_argument(
        "--root", type=Path, default=config.DATA_ROOT, help="data root with Raw/ and Scaled/"
    )
    p.add_argument("--group", nargs="*")
    p.add_argument("--assets", nargs="*")
    p.add_argument(
        "--refit", action="store_true", help="ignore saved scalers and refit on <= cutoff"
    )
    p.add_argument("--cutoff", default=FIT_CUTOFF_DATE)
    p.add_argument(
        "--diagnose", action="store_true", help="only print the scaler chosen for each column"
    )
    p.add_argument("--apply", type=Path, help="all_scalers.pkl to apply to --input")
    p.add_argument("--input", type=Path, help="parquet to transform with --apply")
    p.add_argument(
        "--output", type=Path, help="output of --apply (default <input>_with_scalers.parquet)"
    )
    args = p.parse_args()

    if args.apply:
        if not args.input:
            p.error("--apply needs --input")
        apply_saved_scalers(args.apply, args.input, args.output)
        return

    for group, asset in config.iter_assets(args.group, args.assets):
        path = latest_parquet(args.root / "Raw" / group / asset)
        if path is None:
            logging.warning("No parquet for %s/%s", group, asset)
            continue
        logging.info("Processing %s", path.name)
        try:
            df = pd.read_parquet(path)
            if args.diagnose:
                for c in df.columns:
                    logging.info("%-28s -> %s", c, scaler_name(c, df[c]))
                continue
            scale_and_save(
                df, asset, group, args.root / "Scaled", refit=args.refit, cutoff=args.cutoff
            )
        except Exception as e:
            logging.exception("Error on %s: %s", path, e)


if __name__ == "__main__":
    main()
