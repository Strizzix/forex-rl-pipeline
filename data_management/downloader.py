"""
downloader.py — Dukascopy 1min bid/ask -> multi-timeframe parquet with indicators.

One parquet per asset, on a 15min grid, with the features of every TF
(15min, 30min, 1H, 4H, 1D) aligned on it:
    <ASSET>_multiTF_<start>_to_<end>.parquet

Anti-lookahead: the TFs above 15min are shifted by one bar before being
forward-filled on the 15min grid, so a row only sees higher-TF bars that are
already CLOSED.

Modes:
  full    -> download from start_date (- warmup) to end_date and build everything.
  update  -> load the existing parquet, re-download from (last_ts - warmup) so the
             indicators restart stable, keep only the rows after last_ts and append.
             If there is no parquet yet it does a full download.

Usage:
    python -m data_management.downloader --root D:/Data --mode update
    python -m data_management.downloader --root D:/Data --mode full --group Minors --end 2026-08-26
"""

import argparse
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, CCIIndicator, EMAIndicator, WMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from . import config

# EMA / WMA periods per TF
EMA_CONFIG = {
    "1min": [20, 50],
    "15min": [10, 20, 50],
    "30min": [10, 20, 50],
    "1H": [20, 50],
    "4H": [20, 50],
    "1D": [50, 200],
    "1W": [20],
}

VOLA_WINDOW = {
    "1min": 200,
    "5min": 200,
    "15min": 200,
    "30min": 200,
    "1H": 100,
    "4H": 50,
    "1D": 20,
    "1W": 10,
}

TIMEFRAMES = {"15min": "15min", "30min": "30min", "1H": "1h", "4H": "4h", "1D": "1D"}

HEAD_TRIM = 500  # rows dropped at the start of a full build (warmup noise)

# Fixed origin for the intraday bins. Without it pandas anchors the bins to the
# midnight of the FIRST downloaded day: if that day is in summer time (CEST) the
# 4H bars come out shifted by 1 hour compared to a download that starts in winter
# (01/05/09... instead of 00/04/08...), so an update could mix two alignments.
# A winter date keeps the alignment of the full builds (which start in winter).
BIN_ORIGIN = "2000-01-01 00:00"


def required_warmup_days(safety_margin_days=25):
    """
    Calendar days of history needed before start_date so every indicator is ready.
    1D windows are in TRADING days (~252/year): EMA200 on 1D ≈ 290 calendar days.
    Volatility_regime is a double rolling, so its window counts twice.
    Sub-daily windows are in bars and always shorter than the 1D ones.
    """
    trading_to_cal = 365 / 252
    candidates = [int(p * trading_to_cal) for p in EMA_CONFIG["1D"]]
    candidates.append(int(VOLA_WINDOW["1D"] * 2 * trading_to_cal))
    candidates += [int(p * 7) for p in EMA_CONFIG["1W"]]
    return max(candidates) + safety_margin_days


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
    ):
        self.start_date = pd.to_datetime(start_date).tz_localize("UTC")
        self.end_date = pd.to_datetime(end_date).tz_localize("UTC")
        self.output_dir = output_dir
        self.instrument_name = instrument_name
        self.instrument = instrument
        self.timezone = timezone
        if warmup_days in ("auto", None):
            self.warmup_days = required_warmup_days()
            print(f"[Warmup] auto warmup_days = {self.warmup_days}")
        else:
            self.warmup_days = int(warmup_days)

    @property
    def pair_dir(self):
        return os.path.join(self.output_dir, self.instrument_name)

    # ── download ────────────────────────────────────────────────────────────
    def weekly_ranges(self, fetch_start):
        # contiguous 7-day windows at 1min granularity: no holes, no overlaps
        cur = fetch_start
        while cur < self.end_date:
            end = min(cur + timedelta(days=7) - timedelta(minutes=1), self.end_date)
            yield cur, end
            cur = end + timedelta(minutes=1)

    def safe_fetch(self, start, end, offer_side, retries=10):
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

    # ── features ────────────────────────────────────────────────────────────
    def add_features(self, df, tf):
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.dropna(subset=["open", "high", "low", "close"], how="any").copy()
        if df.empty:
            return pd.DataFrame()
        o, h, lo, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

        for p in EMA_CONFIG.get(tf, []):
            ema = EMAIndicator(c, window=p).ema_indicator()
            df[f"EMA{p}"] = ema
            df[f"EMA{p}_slope"] = ema.diff()
            df[f"WMA{p}"] = WMAIndicator(c, window=p).wma()

        adx = ADXIndicator(h, lo, c, window=14)
        df["ADX14"] = adx.adx()
        df["DI_plus14"] = adx.adx_pos()
        df["DI_minus14"] = adx.adx_neg()
        df["RSI14"] = RSIIndicator(c, window=14).rsi()
        df["CCI20"] = CCIIndicator(h, lo, c, window=20).cci()
        if tf in ("1min", "5min", "15min", "30min"):
            df["Stoch"] = StochasticOscillator(h, lo, c).stoch()
        if tf == "1H":
            macd = MACD(c)
            df["MACD"] = macd.macd()
            df["MACD_signal"] = macd.macd_signal()
            df["MACD_diff"] = macd.macd_diff()

        df["ATR14"] = AverageTrueRange(h, lo, c, window=14).average_true_range()
        df["ATR50"] = AverageTrueRange(h, lo, c, window=50).average_true_range()
        df["ATR_norm"] = df["ATR14"] / (c + 1e-12)
        bb = BollingerBands(c, window=20, window_dev=2)
        df["BB_width"] = bb.bollinger_hband() - bb.bollinger_lband()

        win = int(VOLA_WINDOW.get(tf, 50))
        ret = c.pct_change()
        vol_roll = ret.rolling(win, min_periods=win).std()
        df["volatility"] = vol_roll
        df["Volatility_regime"] = vol_roll / (vol_roll.rolling(win, min_periods=win).mean() + 1e-12)

        body = c - o
        df["high_low_range"] = h - lo
        df["close_open_diff"] = body
        df["log_return"] = np.log(c / c.shift()).fillna(0.0)
        df["abs_return"] = ret.abs()
        df["candle_body"] = body.abs()
        df["candle_shadow"] = (h - lo) - df["candle_body"]
        df["candle_pressure"] = body / (h - lo + 1e-8)
        df["range_ratio"] = df["candle_body"] / (df["high_low_range"] + 1e-8)
        df["vwp"] = (c * v).rolling(50, min_periods=50).sum() / (
            v.rolling(50, min_periods=50).sum() + 1e-8
        )
        df["momentum_proxy"] = c.diff(3) / (c.shift(3) + 1e-12)

        if tf == "1D":
            pivot = (h + lo + c) / 3.0
            df["Pivot"] = pivot
            df["R1"] = 2 * pivot - lo
            df["S1"] = 2 * pivot - h
            df["R2"] = pivot + (h - lo)
            df["S2"] = pivot - (h - lo)
        return df

    def resample_timeframes(self, df, trim_head=True, cutoff=None):
        """
        1min -> features of every TF aligned on the 15min grid.

        trim_head : drop the first HEAD_TRIM rows (full build only).
        cutoff    : first timestamp kept (default start_date). In update mode it's
                    the start of the warmup, so no warmup row gets cut here.
        """
        cutoff = self.start_date if cutoff is None else pd.Timestamp(cutoff)
        cutoff = (
            cutoff.tz_localize(self.timezone)
            if cutoff.tzinfo is None
            else cutoff.tz_convert(self.timezone)
        )

        origin = pd.Timestamp(BIN_ORIGIN, tz=self.timezone)
        ohlc_15 = df.resample("15min", origin=origin).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "spread": ["mean", "max"],
            }
        )
        ohlc_15.columns = ["open", "high", "low", "close", "volume", "spread_mean", "spread_max"]
        base = self.add_features(ohlc_15, "15min").add_suffix("_15min")
        parts = [base]

        for tf, rule in TIMEFRAMES.items():
            if tf == "15min":
                continue
            kw = {} if tf == "1D" else {"origin": origin}  # 1D is calendar based already
            ohlc = df.resample(rule, **kw).agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            )
            ohlc = ohlc.dropna(subset=["open", "high", "low", "close"], how="any")
            if ohlc.empty:
                print(f"[Resample] TF {tf}: no closed bars available")
                continue
            # ANTI-LOOKAHEAD: shift one bar -> only closed bars reach the 15min grid
            feats = self.add_features(ohlc, tf).shift(1).sort_index()
            parts.append(feats.reindex(base.index, method="ffill").add_suffix(f"_{tf}"))

        combined = pd.concat(parts, axis=1, join="outer").dropna(axis=1, how="all")
        # ffill BEFORE the cutoff: the warmup fills any NaN left by shift + reindex
        # (only NaNs are filled, legit zeros are untouched, no rows removed)
        combined = combined.ffill()
        combined = combined[combined.index >= cutoff]
        # NO PARTIAL BAR: the last 15min bar is kept only if its last minute is in
        # the data, otherwise it's saved incomplete and the update never fixes it
        # (it only appends rows after last_ts). It comes back complete next update.
        last_complete = df.index.max() - pd.Timedelta(minutes=14)
        combined = combined[combined.index <= last_complete]

        if trim_head and not combined.empty:
            combined = combined.iloc[HEAD_TRIM:]
            nan_cols = combined.iloc[:20].isna().sum()
            nan_cols = nan_cols[nan_cols > 0]
            if len(nan_cols):
                print(
                    f"[Warning] NaN in the first 20 rows after trim ({len(nan_cols)} cols): "
                    f"{list(nan_cols.index[:8])} ... Dukascopy data probably missing for "
                    f"part of the warmup ({self.warmup_days} days)."
                )
        return combined

    # ── save / load ─────────────────────────────────────────────────────────
    def find_existing_parquet(self):
        """Most recently modified <asset>*.parquet in the asset folder, or None."""
        if not os.path.exists(self.pair_dir):
            return None
        files = [
            f
            for f in os.listdir(self.pair_dir)
            if f.startswith(self.instrument_name) and f.endswith(".parquet")
        ]
        if not files:
            return None
        files.sort(key=lambda f: os.path.getmtime(os.path.join(self.pair_dir, f)), reverse=True)
        return os.path.join(self.pair_dir, files[0])

    def save_parquet(self, df, start=None, end=None, old_path=None):
        """Save as <asset>_multiTF_<start>_to_<end>.parquet (and remove old_path if renamed)."""
        os.makedirs(self.pair_dir, exist_ok=True)
        start = (start or self.start_date).strftime("%Y_%m_%d")
        end = (end or self.end_date).strftime("%Y_%m_%d")
        path = os.path.join(
            self.pair_dir, f"{self.instrument_name}_multiTF_{start}_to_{end}.parquet"
        )
        df.to_parquet(path)
        print(f"Saved: {path}")
        if old_path and os.path.abspath(old_path) != os.path.abspath(path):
            os.remove(old_path)
            print(f"Removed old file: {os.path.basename(old_path)}")
        return path

    # ── modes ───────────────────────────────────────────────────────────────
    def run(self):
        """Full download + build."""
        df_1min = self.fetch_weekly_data()
        if df_1min is None:
            return
        df = self.resample_timeframes(df_1min)
        if df.empty:
            print(f"No valid resampled data for {self.instrument_name}")
            return
        return self.save_parquet(df)

    def update_parquet(self):
        """Append the new rows (up to end_date) to the existing parquet."""
        existing_path = self.find_existing_parquet()
        if existing_path is None:
            print(f"[Update] no parquet for {self.instrument_name}, doing a full download.")
            return self.run()

        print(f"[Update] loading {existing_path}")
        df_old = pd.read_parquet(existing_path)
        df_old.index = (
            df_old.index.tz_localize(self.timezone)
            if df_old.index.tz is None
            else df_old.index.tz_convert(self.timezone)
        )
        last_ts = df_old.index.max()
        if last_ts >= self.end_date.tz_convert(self.timezone):
            print(f"[Update] already up to date ({last_ts}).")
            return

        # download warmup + new block, build features on all of it
        fetch_from = last_ts - timedelta(days=self.warmup_days)
        print(
            f"[Update] last row {last_ts} -> downloading from {fetch_from.date()} "
            f"(warmup incl.) to {self.end_date.date()}"
        )
        df_1min = self.fetch_weekly_data(fetch_start=fetch_from.tz_convert("UTC"))
        if df_1min is None or df_1min.empty:
            print(f"[Update] no 1min data downloaded for {self.instrument_name}.")
            return
        df_new = self.resample_timeframes(df_1min, trim_head=False, cutoff=fetch_from)
        df_new = df_new[df_new.index > last_ts]
        if df_new.empty:
            print(f"[Update] no new rows after {last_ts}.")
            return
        print(f"[Update] new rows: {len(df_new):,} ({df_new.index.min()} -> {df_new.index.max()})")

        # NaN here = the warmup was not enough for some indicator
        nan_cols = df_new.iloc[:10].isna().sum()
        nan_cols = nan_cols[nan_cols > 0]
        if len(nan_cols):
            print(
                f"[Update Warning] NaN in the first new rows ({len(nan_cols)} cols): "
                f"{list(nan_cols.index[:8])} ... try a bigger warmup_days ({self.warmup_days})."
            )

        common = df_old.columns.intersection(df_new.columns)
        for label, cols in (
            ("new cols not in the old parquet", df_new.columns.difference(df_old.columns)),
            ("old cols missing in the new data", df_old.columns.difference(df_new.columns)),
        ):
            if len(cols):
                print(f"[Update] {label} ({len(cols)}): {list(cols)[:5]} ...")

        df_all = pd.concat([df_old[common], df_new[common]])
        df_all = df_all[~df_all.index.duplicated(keep="last")].sort_index()
        self.save_parquet(df_all, df_all.index.min(), df_all.index.max(), old_path=existing_path)
        print(f"[Update] {self.instrument_name}: {len(df_old):,} -> {len(df_all):,} rows")


def merge_bid_ask(df_bid, df_ask, label=""):
    """
    bid + ask 1min -> mid OHLCV + spread.

    OUTER join: a minute published on one side only (common at volume=0 with the
    market closed) is kept, using the other side for that same timestamp only
    (never a forward fill in time: no bar is invented).
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
    for col in ["open", "high", "low", "close", "volume"]:
        df[f"{col}_bid"] = df[f"{col}_bid"].fillna(df[f"{col}_ask"])
        df[f"{col}_ask"] = df[f"{col}_ask"].fillna(df[f"{col}_bid"])

    df["close"] = (df["close_bid"] + df["close_ask"]) / 2
    df["open"] = (df["open_bid"] + df["open_ask"]) / 2
    df["high"] = df[["high_bid", "high_ask"]].max(axis=1)
    df["low"] = df[["low_bid", "low_ask"]].min(axis=1)
    df["volume"] = df["volume_bid"] + df["volume_ask"]
    # one side missing -> real spread unknown: 0.0 (closed book), row kept
    df["spread"] = (df["close_ask"] - df["close_bid"]).fillna(0.0)
    return df[["open", "high", "low", "close", "volume", "spread"]]


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
    p.add_argument("--mode", choices=["update", "full"], default="update")
    p.add_argument("--group", nargs="*", help="Majors / Minors (default: all)")
    p.add_argument("--assets", nargs="*", help="e.g. EURUSD GBPUSD (default: all)")
    p.add_argument("--start", default=config.START_DATE)
    p.add_argument("--end", default=pd.Timestamp.now("UTC").strftime("%Y-%m-%d"))
    p.add_argument("--tz", default=config.TIMEZONE)
    args = p.parse_args()

    todo = list(config.iter_assets(args.group, args.assets))
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
            )
            dl.update_parquet() if args.mode == "update" else dl.run()
        except Exception as e:
            print(f"Error {name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
