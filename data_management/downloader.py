"""
downloader.py — Dukascopy 1min bid/ask -> multi-timeframe parquet with indicators.

Two files per asset in <root>/Raw/<group>/<ASSET>/:
    <ASSET>_bars15min_<start>_to_<end>.parquet   15min OHLCV + spread, warm-up included (cache)
    <ASSET>_multiTF_<start>_to_<end>.parquet     features of every TF on the 15min grid

Timestamps are bar CLOSE times: the row labelled 10:15 holds the bar [10:00, 10:15)
and everything in it is known at 10:15. Higher timeframes (30min, 1H, 4H, 1D) are
also close-labelled and forward-filled on the 15min grid, so a row only ever sees
bars that closed at or before its own timestamp.

The indicators come from indicators.py; pick them with config.INDICATORS or
--indicators.

Modes:
  full      download from start_date (- warm-up) to end_date, build bars + features.
  update    download only the new data (plus a 1-day overlap), append it to the bars
            cache and rebuild the features from the whole cache. The result is
            identical to a full build. Falls back to full if there is no cache.
  features  no download: rebuild the features from the bars cache (e.g. after
            changing the indicator selection).

Usage:
    python -m data_management.downloader --root D:/Data --mode update
    python -m data_management.downloader --root D:/Data --mode full --group Minors --end 2026-08-26
    python -m data_management.downloader --root D:/Data --mode features --indicators ema rsi atr
    python -m data_management.downloader --list-indicators
"""

import argparse
import os
import sys
import time
from datetime import timedelta

import pandas as pd

from . import config, indicators, storage

OHLC = ["open", "high", "low", "close"]
HIGHER_TFS = {"30min": "30min", "1H": "1h", "4H": "4h", "1D": "1D"}

HEAD_TRIM = 500  # rows dropped at the start of the features (indicator warm-up noise)
UPDATE_OVERLAP = timedelta(days=1)  # re-downloaded before the last cached bar on update

# Fixed origin for the intraday bins. Without it pandas anchors the bins to the
# midnight of the FIRST downloaded day: if that day is in summer time (CEST) the
# 4H bars come out shifted by 1 hour compared to a download that starts in winter
# (01/05/09... instead of 00/04/08...), so an update could mix two alignments.
# A winter date keeps the alignment of the full builds (which start in winter).
BIN_ORIGIN = "2000-01-01 00:00"


# ── pure transforms (no I/O) ────────────────────────────────────────────────
def merge_bid_ask(df_bid, df_ask, label=""):
    """
    bid + ask 1min -> mid OHLCV + spread.

    OUTER join: a minute published on one side only (common at volume=0 with the
    market closed) is kept, using the other side for that same timestamp only
    (never a forward fill in time: no bar is invented). Its spread is unknown and
    stays NaN, so it does not drag the 15min spread mean/max towards 0.
    """
    df = pd.merge(
        df_bid, df_ask, left_index=True, right_index=True, suffixes=("_bid", "_ask"), how="outer"
    )
    n_bid_only, n_ask_only = df["close_ask"].isna().sum(), df["close_bid"].isna().sum()
    if n_bid_only or n_ask_only:
        print(
            f"[Merge] {label}: {n_bid_only} bid-only minutes, {n_ask_only} ask-only minutes "
            f"(filled from the other side, no rows dropped)"
        )
    # computed before the fill: NaN where one side is missing
    df["spread"] = df["close_ask"] - df["close_bid"]
    # volume: a missing side contributes 0 (filling it from the other side would
    # double-count the volume of one-sided minutes)
    df["volume"] = df["volume_bid"].fillna(0.0) + df["volume_ask"].fillna(0.0)
    for col in OHLC:
        df[f"{col}_bid"] = df[f"{col}_bid"].fillna(df[f"{col}_ask"])
        df[f"{col}_ask"] = df[f"{col}_ask"].fillna(df[f"{col}_bid"])

    df["close"] = (df["close_bid"] + df["close_ask"]) / 2
    df["open"] = (df["open_bid"] + df["open_ask"]) / 2
    df["high"] = df[["high_bid", "high_ask"]].max(axis=1)
    df["low"] = df[["low_bid", "low_ask"]].min(axis=1)
    return df[["open", "high", "low", "close", "volume", "spread"]]


def make_bars(df_1min: pd.DataFrame) -> pd.DataFrame:
    """
    1min mid data (labelled at the minute's open, as Dukascopy does) -> 15min bars
    labelled at their CLOSE: bar [10:00, 10:15) -> 10:15. The spread mean/max skip
    the minutes whose spread is unknown.

    No partial bar: a bar is kept only if its last minute is in the data, otherwise
    it would be stored incomplete.
    """
    origin = pd.Timestamp(BIN_ORIGIN, tz=df_1min.index.tz)
    bars = df_1min.resample("15min", origin=origin, closed="left", label="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "spread": ["mean", "max"],
        }
    )
    bars.columns = ["open", "high", "low", "close", "volume", "spread_mean", "spread_max"]
    bars = bars.dropna(subset=OHLC)
    return bars[bars.index <= df_1min.index.max() + pd.Timedelta(minutes=1)]


def resample_bars(bars15: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Close-labelled 15min bars -> close-labelled bars of a higher TF. Aggregating the
    15min bars gives exactly the same OHLCV as resampling the 1min data, since the
    15min bins nest in every higher TF. closed="right": the 1H bar labelled 11:00
    takes the 15min bars labelled 10:15..11:00, i.e. the minutes [10:00, 11:00).
    """
    kw = {} if tf == "1D" else {"origin": pd.Timestamp(BIN_ORIGIN, tz=bars15.index.tz)}
    out = bars15.resample(HIGHER_TFS[tf], closed="right", label="right", **kw).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=OHLC)


def build_features(bars15, start=None, names=None, trim_head=True, warmup_days=None):
    """
    15min bars -> features of every TF aligned on the 15min grid.

    start     : first bar OPEN time kept (rows labelled > start); default: everything.
    names     : indicators to include (default config.INDICATORS).
    trim_head : also drop the first HEAD_TRIM rows (indicator warm-up noise).
    """
    names = indicators.validate(config.INDICATORS if names is None else names)
    base = indicators.compute(bars15, "15min", names).add_suffix("_15min")
    parts = [base]
    for tf in HIGHER_TFS:
        bars = resample_bars(bars15, tf)
        if bars.empty:
            print(f"[Features] TF {tf}: no closed bars available")
            continue
        feats = indicators.compute(bars, tf, names)
        # close-labelled: at row t the last bar with label <= t has already closed
        parts.append(feats.reindex(base.index, method="ffill").add_suffix(f"_{tf}"))

    combined = pd.concat(parts, axis=1)
    # fill the NaN left by the alignment (and unknown spreads) with the last known value;
    # only NaNs are filled, legit zeros are untouched, no rows removed
    combined = combined.ffill()
    if start is not None:
        start = pd.Timestamp(start)
        start = (
            start.tz_localize(combined.index.tz)
            if start.tzinfo is None
            else start.tz_convert(combined.index.tz)
        )
        combined = combined[combined.index > start]

    if trim_head and not combined.empty:
        combined = combined.iloc[HEAD_TRIM:]
        nan_cols = combined.iloc[:20].isna().sum()
        nan_cols = nan_cols[nan_cols > 0]
        if len(nan_cols):
            print(
                f"[Warning] NaN in the first 20 rows after trim ({len(nan_cols)} cols): "
                f"{list(nan_cols.index[:8])} ... Dukascopy data probably missing for "
                f"part of the warm-up ({warmup_days} days)."
            )
    return combined


# ── download + files ────────────────────────────────────────────────────────
class DukascopyMultiTFDownloader:
    def __init__(
        self,
        start_date,
        end_date,
        output_dir,
        instrument_name="EURUSD",
        instrument=None,
        timezone=config.TIMEZONE,
        warmup_days="auto",
        indicator_names=None,
    ):
        self.start_date = pd.to_datetime(start_date).tz_localize("UTC")
        self.end_date = pd.to_datetime(end_date).tz_localize("UTC")
        self.output_dir = output_dir
        self.instrument_name = instrument_name
        self.instrument = instrument
        self.timezone = timezone
        self.indicators = indicators.validate(
            config.INDICATORS if indicator_names is None else indicator_names
        )
        if warmup_days in ("auto", None):
            self.warmup_days = indicators.warmup_days(self.indicators)
            print(f"[Warmup] auto warmup_days = {self.warmup_days}")
        else:
            self.warmup_days = int(warmup_days)

    @property
    def pair_dir(self):
        return os.path.join(self.output_dir, self.instrument_name)

    @property
    def bars_prefix(self):
        return f"{self.instrument_name}_bars15min_"

    @property
    def features_prefix(self):
        return f"{self.instrument_name}_multiTF_"

    # ── download ────────────────────────────────────────────────────────────
    def weekly_ranges(self, fetch_start):
        # contiguous 7-day windows at 1min granularity: no holes, no overlaps
        cur = fetch_start
        while cur < self.end_date:
            end = min(cur + timedelta(days=7) - timedelta(minutes=1), self.end_date)
            yield cur, end
            cur = end + timedelta(minutes=1)

    def safe_fetch(self, start, end, offer_side, retries=10, backoff_s=1.0):
        import dukascopy_python

        for i in range(retries):
            try:
                df = dukascopy_python.fetch(
                    start=start,
                    end=end,
                    instrument=self.instrument,
                    offer_side=offer_side,
                    interval=dukascopy_python.INTERVAL_MIN_1,
                )
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                print(f"Fetch error ({i+1}/{retries}): {e}")
            # back off before retrying (linear), instead of hammering the server
            if i < retries - 1:
                time.sleep(backoff_s * (i + 1))
        return None

    def fetch_weekly_data(self, fetch_start=None):
        """1min mid OHLCV + spread from fetch_start (default start_date - warmup) to end_date."""
        from dukascopy_python import OFFER_SIDE_ASK, OFFER_SIDE_BID

        if fetch_start is None:
            fetch_start = self.start_date - timedelta(days=self.warmup_days)

        all_data = []
        for start, end in self.weekly_ranges(fetch_start):
            print(f"Downloading {start.date()} - {end.date()}")
            df_bid = self.safe_fetch(start, end, OFFER_SIDE_BID)
            df_ask = self.safe_fetch(start, end, OFFER_SIDE_ASK)
            if df_bid is None or df_ask is None:
                print(f"Missing data for {start.date()} - {end.date()}")
                continue
            all_data.append(merge_bid_ask(df_bid, df_ask, label=f"{start.date()}-{end.date()}"))

        if not all_data:
            print("No data downloaded in the requested range.")
            return None

        df = pd.concat(all_data).sort_index()
        df = df[~df.index.duplicated(keep="first")]  # window edges
        n_zero = int((df["volume"] == 0).sum())
        print(
            f"[Check] volume=0 bars (market closed) kept: {n_zero:,} / {len(df):,} "
            f"({n_zero / len(df) * 100:.2f}%)"
        )
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(self.timezone)
        return df

    # ── files ───────────────────────────────────────────────────────────────
    def _load(self, path):
        df = pd.read_parquet(path)
        df.index = (
            df.index.tz_localize(self.timezone)
            if df.index.tz is None
            else df.index.tz_convert(self.timezone)
        )
        return df

    def _save_bars(self, bars, start):
        meta = {"kind": "bars15min", "label": "close", "start": str(start)}
        return storage.save_single(bars, self.pair_dir, self.bars_prefix, meta)

    def _save_features(self, bars, start):
        feats = build_features(bars, start, self.indicators, warmup_days=self.warmup_days)
        if feats.empty:
            print(f"No valid features for {self.instrument_name}")
            return None
        meta = {
            "kind": "features",
            "label": "close",
            "start": str(start),
            "indicators": self.indicators,
        }
        return storage.save_single(feats, self.pair_dir, self.features_prefix, meta)

    # ── modes ───────────────────────────────────────────────────────────────
    def run(self):
        """Full download + build."""
        df_1min = self.fetch_weekly_data()
        if df_1min is None:
            return None
        bars = make_bars(df_1min)
        if bars.empty:
            print(f"No complete bars for {self.instrument_name}")
            return None
        self._save_bars(bars, self.start_date)
        return self._save_features(bars, self.start_date)

    def rebuild_features(self):
        """Rebuild the features from the bars cache, without downloading."""
        path = storage.latest(self.pair_dir, self.bars_prefix)
        if path is None:
            raise FileNotFoundError(
                f"no bars cache for {self.instrument_name}: run --mode full first"
            )
        start = storage.read_meta(path).get("start", str(self.start_date))
        return self._save_features(self._load(path), start)

    def update_parquet(self):
        """Append the new data to the bars cache and rebuild the features (== full build)."""
        bars_path = storage.latest(self.pair_dir, self.bars_prefix)
        if bars_path is None:
            if storage.latest(self.pair_dir, self.features_prefix):
                print(
                    f"[Update] {self.instrument_name}: dataset from an older version "
                    f"(no bars cache, open-labelled rows) -> full rebuild."
                )
            else:
                print(f"[Update] no data for {self.instrument_name}, doing a full download.")
            return self.run()

        bars_old = self._load(bars_path)
        # the start of the dataset is fixed by the full build, not by this run's --start
        start = storage.read_meta(bars_path).get("start", str(self.start_date))
        last = bars_old.index.max()
        if last >= self.end_date.tz_convert(self.timezone):
            print(f"[Update] bars already up to date ({last}).")
            feats_path = storage.latest(self.pair_dir, self.features_prefix)
            if feats_path and storage.read_meta(feats_path).get("indicators") == self.indicators:
                return feats_path
            print("[Update] indicator selection changed -> rebuilding the features.")
            return self._save_features(bars_old, start)

        fetch_from = last - UPDATE_OVERLAP
        print(f"[Update] last bar {last} -> downloading {fetch_from} to {self.end_date.date()}")
        df_1min = self.fetch_weekly_data(fetch_start=fetch_from.tz_convert("UTC"))
        if df_1min is None or df_1min.empty:
            print(f"[Update] no 1min data downloaded for {self.instrument_name}.")
            return None
        new = make_bars(df_1min)
        bars = pd.concat([bars_old[bars_old.index < new.index.min()], new])
        print(
            f"[Update] {self.instrument_name}: {len(bars) - len(bars_old):+,} bars "
            f"(last {bars.index.max()})"
        )
        self._save_bars(bars, start)
        return self._save_features(bars, start)


def main():
    p = argparse.ArgumentParser(
        description="Download Dukascopy data and build the multi-TF parquets."
    )
    p.add_argument(
        "--root",
        type=str,
        default=str(config.DATA_ROOT),
        help="data root (parquets go in <root>/Raw)",
    )
    p.add_argument("--mode", choices=["update", "full", "features"], default="update")
    p.add_argument("--group", nargs="*", help="Majors / Minors (default: all)")
    p.add_argument("--assets", nargs="*", help="e.g. EURUSD GBPUSD (default: all)")
    p.add_argument("--start", default=config.START_DATE)
    p.add_argument("--end", default=pd.Timestamp.now("UTC").strftime("%Y-%m-%d"))
    p.add_argument("--tz", default=config.TIMEZONE)
    p.add_argument(
        "--indicators",
        nargs="+",
        help="indicators to include (default: config.INDICATORS)",
    )
    p.add_argument("--list-indicators", action="store_true", help="list the indicators and exit")
    args = p.parse_args()

    if args.list_indicators:
        for name, ind in indicators.REGISTRY.items():
            tfs = ", ".join(ind.timeframes) if ind.timeframes else "all TFs"
            print(f"{name:<11} {ind.description} ({tfs})")
        return

    todo = list(config.iter_assets(args.group, args.assets))
    failed = []
    for i, (group, name) in enumerate(todo, 1):
        print(f"\n=== [{i}/{len(todo)}] {group}/{name} | MODE={args.mode} ===", flush=True)
        try:
            dl = DukascopyMultiTFDownloader(
                args.start,
                args.end,
                os.path.join(args.root, "Raw", group),
                name,
                instrument=config.dukascopy_instrument(name),
                timezone=args.tz,
                indicator_names=args.indicators,
            )
            if args.mode == "update":
                dl.update_parquet()
            elif args.mode == "features":
                dl.rebuild_features()
            else:
                dl.run()
        except Exception as e:
            print(f"Error {name}: {e}", file=sys.stderr)
            failed.append(name)

    if failed:
        print(f"\n{len(failed)}/{len(todo)} assets failed: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
