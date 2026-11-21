"""
checks.py — quality checks on the real datasets (raw and scaled).

Raw, for every asset in <root>/Raw/<group>/<ASSET>/:
  - exactly one bars cache and one features file, both of the current schema;
  - bars: tz-aware, sorted, unique, on the 15min grid, OHLC consistent, no inf;
  - features: exactly the columns of the indicators listed in its metadata;
  - bar-close labels: close_15min equals the cached bar close, and at every 1H / 4H / 1D
    boundary the higher-TF close equals the 15min close (true only with close labels);
  - closed-market bars: none left in the features when they are dropped (weekly close or
    long volume-0 runs), while quiet bars of open sessions are still there;
  - NaN (overall and in the first rows = warm-up too short), zero-volume and unknown-spread
    shares, time gaps classified as weekends, holidays or anomalies.

Scaled, for every asset in <root>/Scaled/<group>/<ASSET>/:
  - one scaled file + all_scalers.pkl of the current schema, same rows and indicators as raw;
  - *_raw columns identical to the raw values, spread only as *_raw;
  - every feature column has a saved scaler, no NaN/inf added by the scaling;
  - the scalers were fit on the cutoff window: on the rows up to the saved cutoff the
    Standard columns have mean 0 / std 1, the Robust ones median 0, the MinMax ones span [0, 1].

Usage:
    python -m data_management.checks --root D:/Data
    python -m data_management.checks --root D:/Data --only raw --assets EURUSD
Exits with status 1 if any asset has errors.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config, indicators, scale, storage
from .downloader import BIN_ORIGIN, HIGHER_TFS, closed_market_mask

# gaps between consecutive 15min rows (hours)
WEEKEND_HOURS = (30, 80)  # starting Friday/Saturday: normal weekly close
HOLIDAY_HOURS = (4, 100)  # any other day: bank holidays, up to a 4-day closure
# a duration repeated on the same weekday more than this many times, almost identical
# (std < SUSPICIOUS_STD_HOURS), is the signature of a systematic download bug
# (e.g. the old one-day-every-week hole); weekends are excluded from this test
SUSPICIOUS_RECURRENCE = 15
SUSPICIOUS_STD_HOURS = 1.0

BOUNDARIES = {"1H": "1h", "4H": "4h", "1D": "1D"}
FIT_TOL = 0.05  # tolerance on the fit-window statistics of the scaled columns


@dataclass
class Report:
    group: str
    asset: str
    kind: str
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def ok(self):
        return not self.errors


# ── helpers ─────────────────────────────────────────────────────────────────
def _files(folder: Path, kind: str):
    return sorted(folder.glob(f"*_{kind}_*.parquet"))


def _check_index(rep, idx, name):
    if not isinstance(idx, pd.DatetimeIndex):
        rep.errors.append(f"{name}: index is not a DatetimeIndex")
        return False
    if idx.tz is None:
        rep.warnings.append(f"{name}: index without timezone ({config.TIMEZONE} expected)")
    if not idx.is_monotonic_increasing:
        rep.errors.append(f"{name}: index not sorted")
    if not idx.is_unique:
        rep.errors.append(f"{name}: {int(idx.duplicated().sum())} duplicated timestamps")
    return True


def _n_inf(df):
    num = df.select_dtypes(include=[np.number])
    return int(np.isinf(num.to_numpy()).sum()) if not num.empty else 0


def expected_columns(names) -> set:
    """Columns of a features file built with these indicators (names don't depend on data)."""
    n = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="15min", tz=config.TIMEZONE)
    c = pd.Series(1.0 + 0.001 * np.sin(np.arange(n) / 7.0), index=idx)
    dummy = pd.DataFrame(
        {"open": c, "high": c + 1e-4, "low": c - 1e-4, "close": c, "volume": 1.0}, index=idx
    )
    cols = {"spread_mean_15min", "spread_max_15min"}
    for tf in ("15min", *HIGHER_TFS):
        cols |= {f"{col}_{tf}" for col in indicators.compute(dummy, tf, names).columns}
    return cols


def boundary_mask(idx: pd.DatetimeIndex, tf: str) -> np.ndarray:
    """Rows labelled exactly at the close of a 1H / 4H / 1D bar."""
    if tf == "1D":
        return (idx.hour == 0) & (idx.minute == 0)
    origin = pd.Timestamp(BIN_ORIGIN, tz=idx.tz)
    return ((idx - origin) % pd.Timedelta(BOUNDARIES[tf])) == pd.Timedelta(0)


def analyze_gaps(idx: pd.DatetimeIndex) -> dict:
    """
    Gaps between consecutive rows longer than one bar, classified as weekend closes,
    holidays, short intraday gaps (illiquid hours) or anomalies.
    """
    ts = idx.to_series()
    hours = ts.diff().dt.total_seconds() / 3600.0
    mask = hours > 0.25 + 1e-9
    gaps = pd.DataFrame(
        {"start": ts.shift(1)[mask], "end": ts[mask], "hours": hours[mask]}
    ).reset_index(drop=True)
    if gaps.empty:
        return {"weekend": 0, "holiday": 0, "short": 0, "anomalies": gaps}

    weekday = pd.DatetimeIndex(gaps["start"]).day_of_week  # local time
    weekend = weekday.isin([4, 5]) & gaps["hours"].between(*WEEKEND_HOURS)
    short = gaps["hours"] < HOLIDAY_HOURS[0]
    holiday = ~weekend & ~short & gaps["hours"].between(*HOLIDAY_HOURS)

    suspicious = np.zeros(len(gaps), dtype=bool)
    other = ~weekend & ~short
    for _, grp in gaps[other].groupby(weekday[other]):
        if len(grp) > SUSPICIOUS_RECURRENCE and grp["hours"].std() < SUSPICIOUS_STD_HOURS:
            suspicious[grp.index] = True

    anomalies = gaps[(~weekend & ~short & ~holiday) | suspicious].copy()
    anomalies["weekday"] = pd.DatetimeIndex(anomalies["start"]).day_name()
    return {
        "weekend": int(weekend.sum()),
        "holiday": int((holiday & ~suspicious).sum()),
        "short": int(short.sum()),
        "anomalies": anomalies,
    }


# ── raw ─────────────────────────────────────────────────────────────────────
def check_raw(pair_dir: Path, group: str, asset: str) -> Report:
    rep = Report(group, asset, "raw")
    bars_files, feat_files = _files(pair_dir, "bars15min"), _files(pair_dir, "multiTF")
    for kind, files in (("bars15min", bars_files), ("multiTF", feat_files)):
        if len(files) != 1:
            rep.errors.append(f"{len(files)} *_{kind}_* files (exactly 1 expected)")
    if not bars_files or not feat_files:
        return rep
    bars_path, feat_path = bars_files[-1], feat_files[-1]

    meta = storage.read_meta(feat_path)
    if meta.get("schema") != storage.SCHEMA_VERSION or meta.get("label") != "close":
        rep.errors.append(
            f"{feat_path.name}: schema {meta.get('schema', '1')}, label {meta.get('label')} "
            f"(expected schema {storage.SCHEMA_VERSION}, close labels): rebuild with --mode full"
        )
        return rep

    bars, feats = pd.read_parquet(bars_path), pd.read_parquet(feat_path)

    # bars cache
    if _check_index(rep, bars.index, "bars"):
        origin = pd.Timestamp(BIN_ORIGIN, tz=bars.index.tz)
        off_grid = int((((bars.index - origin) % pd.Timedelta("15min")) != pd.Timedelta(0)).sum())
        if off_grid:
            rep.errors.append(f"bars: {off_grid} timestamps not on the 15min grid")
    o, h, lo, c = bars["open"], bars["high"], bars["low"], bars["close"]
    bad_ohlc = int(((h < np.maximum(o, c)) | (lo > np.minimum(o, c)) | (h < lo)).sum())
    if bad_ohlc:
        rep.errors.append(f"bars: {bad_ohlc} bars with inconsistent OHLC")
    if (bars["volume"] < 0).any():
        rep.errors.append("bars: negative volume")
    if (bars["spread_mean"] < 0).any():
        rep.errors.append(f"bars: {int((bars['spread_mean'] < 0).sum())} negative spreads")
    if _n_inf(bars):
        rep.errors.append(f"bars: {_n_inf(bars)} infinite values")
    rep.info["pct_vol0"] = float((bars["volume"] == 0).mean() * 100)
    rep.info["pct_spread_nan"] = float(bars["spread_mean"].isna().mean() * 100)

    # features
    if not _check_index(rep, feats.index, "features") or not feats.index.is_unique:
        return rep
    rep.info.update(rows=len(feats), start=feats.index.min(), end=feats.index.max())
    expected = expected_columns(meta["indicators"])
    missing, extra = expected - set(feats.columns), set(feats.columns) - expected
    if missing:
        rep.errors.append(f"{len(missing)} expected columns missing: {sorted(missing)[:6]}")
    if extra:
        rep.warnings.append(
            f"{len(extra)} columns not produced by its indicators: {sorted(extra)[:6]}"
        )

    # bar-close labels: features vs bars cache, higher-TF boundaries
    not_in_bars = feats.index.difference(bars.index)
    if len(not_in_bars):
        rep.errors.append(f"{len(not_in_bars)} feature rows with no bar in the cache")
    common = feats.index.intersection(bars.index)
    if not np.allclose(feats.loc[common, "close_15min"], bars.loc[common, "close"], equal_nan=True):
        rep.errors.append("close_15min differs from the bars cache")
    if feats.index.max() != bars.index.max():
        rep.warnings.append(
            f"features end at {feats.index.max()}, bars at {bars.index.max()}: "
            f"run --mode features"
        )
    for tf in BOUNDARIES:
        col = f"close_{tf}"
        if col not in feats:
            continue
        at = boundary_mask(feats.index, tf)
        same = np.isclose(feats.loc[at, col], feats.loc[at, "close_15min"])
        if at.sum() and same.mean() < 1.0:
            rep.errors.append(
                f"{col} != close_15min on {(~same).sum()}/{at.sum()} bar-close rows "
                f"(timestamps not at bar close?)"
            )

    # closed-market bars
    vol = feats[["volume_15min"]].rename(columns={"volume_15min": "volume"})
    zero_kept = int(vol["volume"].eq(0).sum())
    rep.info["pct_vol0_kept"] = zero_kept / max(len(feats), 1) * 100
    if meta.get("drop_closed"):
        closed = int(closed_market_mask(vol, meta.get("closed_run_hours")).sum())
        if closed:
            rep.errors.append(f"{closed} closed-market bars left in the features")
        dropped = closed_market_mask(bars)
        dropped = dropped[dropped.index > feats.index.min()]
        lost = feats.index.intersection(dropped[dropped].index)
        if len(lost):
            rep.errors.append(f"{len(lost)} closed-market bars of the cache in the features")
        span = (bars.index >= feats.index.min()) & (bars.index <= feats.index.max())
        missing_open = bars.index[span & ~closed_market_mask(bars)]
        missing_open = missing_open.difference(feats.index)
        if len(missing_open):
            rep.errors.append(
                f"{len(missing_open)} open-market bars of the cache missing from the features"
            )

    # NaN / inf
    nan_pct = feats.isna().mean() * 100
    high_nan = nan_pct[nan_pct > 5].sort_values(ascending=False)
    if len(high_nan):
        rep.warnings.append(
            f"{len(high_nan)} columns with >5% NaN (worst {high_nan.index[0]} {high_nan.iloc[0]:.1f}%)"
        )
    head_nan = feats.iloc[:20].isna().sum()
    head_nan = head_nan[head_nan > 0]
    if len(head_nan):
        rep.errors.append(
            f"NaN in the first 20 rows in {len(head_nan)} columns (warm-up too short?): "
            f"{list(head_nan.index[:5])}"
        )
    if _n_inf(feats):
        rep.errors.append(f"features: {_n_inf(feats)} infinite values")

    # gaps
    g = analyze_gaps(feats.index)
    rep.info.update(gap_weekend=g["weekend"], gap_holiday=g["holiday"], gap_short=g["short"])
    rep.info["gap_anomalous"] = len(g["anomalies"])
    if len(g["anomalies"]):
        first = g["anomalies"].iloc[0]
        rep.errors.append(
            f"{len(g['anomalies'])} anomalous gaps, e.g. {first['start']} -> {first['end']} "
            f"({first['hours']:.1f}h, {first['weekday']})"
        )
    return rep


# ── scaled ──────────────────────────────────────────────────────────────────
def check_scaled(scaled_dir: Path, pair_dir: Path, group: str, asset: str) -> Report:
    rep = Report(group, asset, "scaled")
    files = sorted(scaled_dir.glob(f"{asset}_scaled_*.parquet"))
    pkl = scaled_dir / "all_scalers.pkl"
    if len(files) != 1:
        rep.errors.append(f"{len(files)} scaled files (exactly 1 expected)")
    if not pkl.exists():
        rep.errors.append("all_scalers.pkl missing")
    raw_files = _files(pair_dir, "multiTF")
    if not files or not pkl.exists() or not raw_files:
        if not raw_files:
            rep.errors.append("no raw features file to compare with")
        return rep

    saved = joblib.load(pkl)
    smeta = saved.get("meta", {})
    meta, raw_meta = storage.read_meta(files[-1]), storage.read_meta(raw_files[-1])
    if (
        smeta.get("schema") != storage.SCHEMA_VERSION
        or meta.get("schema") != storage.SCHEMA_VERSION
    ):
        rep.errors.append("scaled data or scalers of an older schema: run scale --refit")
        return rep
    if smeta.get("asset") != asset:
        rep.warnings.append(f"all_scalers.pkl is for {smeta.get('asset')!r}")
    if meta.get("indicators") != raw_meta.get("indicators"):
        rep.errors.append("indicators differ from the raw file: run scale again")

    df, raw = pd.read_parquet(files[-1]), pd.read_parquet(raw_files[-1])
    rep.info.update(rows=len(df), cutoff=smeta.get("fit_cutoff_date"))
    if not df.index.equals(raw.index):
        rep.errors.append(f"rows differ from raw ({len(df):,} vs {len(raw):,}): run scale again")
        return rep

    # raw columns
    for c in raw.columns:
        if scale._is_raw_only(c):
            if c in df:
                rep.errors.append(f"raw-only column {c} still present")
            if f"{c}_raw" not in df or not df[f"{c}_raw"].equals(raw[c]):
                rep.errors.append(f"{c}_raw missing or different from raw")
        elif c.lower() in scale.RAW_COLUMNS_TO_REPLACE:
            if f"{c}_raw" not in df or not np.allclose(df[f"{c}_raw"], raw[c], equal_nan=True):
                rep.errors.append(f"{c}_raw missing or different from raw")

    # every feature column scaled, nothing broken by the scaling
    cols = [c for c in scale.feature_columns(df) if c in raw]
    unscaled = [c for c in cols if c not in saved["col_scalers"]]
    if unscaled:
        rep.errors.append(f"{len(unscaled)} columns without a saved scaler: {unscaled[:5]}")
    more_nan = [c for c in cols if df[c].isna().sum() > raw[c].isna().sum()]
    if more_nan:
        rep.errors.append(f"NaN added by the scaling in {len(more_nan)} columns: {more_nan[:5]}")
    if _n_inf(df):
        rep.errors.append(f"{_n_inf(df)} infinite values after scaling")

    # fit-window statistics: the scalers must have been fit on rows <= cutoff
    cutoff = smeta.get("fit_cutoff_date")
    fit = df[scale.fit_mask(df.index, cutoff)] if cutoff else df.iloc[:0]
    if len(fit) < 100:
        rep.warnings.append(f"only {len(fit)} rows up to the cutoff {cutoff}: fit stats skipped")
        return rep
    bad = []
    for c in cols:
        if c not in saved["col_scalers"] or raw.loc[fit.index, c].nunique() < 3:
            continue
        x = fit[c].to_numpy(dtype=float)
        kind = scale.scaler_name(c)
        if kind == "StandardScaler":
            ok = abs(np.nanmean(x)) < FIT_TOL and abs(np.nanstd(x) - 1) < FIT_TOL
        elif kind in ("RobustScaler", "Log1p+RobustScaler"):
            ok = abs(np.nanmedian(x)) < FIT_TOL
        elif kind == "MinMaxScaler":
            ok = abs(np.nanmin(x)) < FIT_TOL and abs(np.nanmax(x) - 1) < FIT_TOL
        else:
            continue
        if not ok:
            bad.append(c)
    if bad:
        rep.errors.append(
            f"{len(bad)} columns not centred on the fit window <= {cutoff} "
            f"(scalers fit on other data?): {bad[:5]}"
        )
    return rep


# ── output ──────────────────────────────────────────────────────────────────
def print_report(rep: Report):
    icon = "OK " if rep.ok else "ERR"
    print(f"\n[{icon}] {rep.kind:<6} {rep.group}/{rep.asset}")
    if "rows" in rep.info:
        span = f" | {rep.info['start']} -> {rep.info['end']}" if "start" in rep.info else ""
        print(f"      rows: {rep.info['rows']:,}{span}")
    for w in rep.warnings:
        print(f"      warning: {w}")
    for e in rep.errors:
        print(f"      ERROR:   {e}")


def print_summary(reports: list):
    raw = [r for r in reports if r.kind == "raw" and "pct_vol0" in r.info]
    if raw:
        tbl = pd.DataFrame(
            [
                {
                    "asset": f"{r.group}/{r.asset}",
                    "rows": r.info.get("rows", 0),
                    "%vol0": r.info["pct_vol0"],
                    "%vol0 kept": r.info.get("pct_vol0_kept", np.nan),
                    "%spreadNaN": r.info["pct_spread_nan"],
                    "weekends": r.info.get("gap_weekend", 0),
                    "holidays": r.info.get("gap_holiday", 0),
                    "short": r.info.get("gap_short", 0),
                    "anomalous": r.info.get("gap_anomalous", 0),
                    "status": "OK" if r.ok else "ERRORS",
                }
                for r in raw
            ]
        ).set_index("asset")
        # outliers on %vol0 (median +/- 3 MAD): partial downloads or merge problems
        v = tbl["%vol0"]
        mad = (v - v.median()).abs().median()
        if len(v) >= 3 and mad > 0:
            tbl["outlier"] = np.where((v - v.median()).abs() > 3 * mad, "<- %vol0", "")
        print("\n" + tbl.round(2).to_string())
    n_bad = sum(not r.ok for r in reports)
    print(f"\n{len(reports) - n_bad}/{len(reports)} checks OK")


def run(root: Path, groups=None, assets=None, only=None) -> list:
    reports = []
    for group, asset in config.iter_assets(groups, assets):
        pair = root / "Raw" / group / asset
        if only != "scaled":
            reports.append(check_raw(pair, group, asset))
        if only != "raw":
            reports.append(check_scaled(root / "Scaled" / group / asset, pair, group, asset))
    for r in reports:
        print_report(r)
    print_summary(reports)
    return reports


def main():
    p = argparse.ArgumentParser(description="Quality checks on the raw and scaled datasets.")
    p.add_argument("--root", type=Path, default=config.DATA_ROOT)
    p.add_argument("--group", nargs="*")
    p.add_argument("--assets", nargs="*")
    p.add_argument("--only", choices=["raw", "scaled"])
    args = p.parse_args()
    reports = run(args.root, args.group, args.assets, args.only)
    if any(not r.ok for r in reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
