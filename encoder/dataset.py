"""
dataset.py — pretraining dataset of the market encoder + threshold utilities.

Builds, for every 15min bar, the input window (scaled multi-TF features +
session / structural features) and all the labels of the heads: trend momentum,
continuous regime, market structure and the strategy outcomes (session
breakout, PDH/PDL breakout, regime quality, breakeven reachability).

Outcome labels: 0=SL, 1=TP, 2=NONE (neither TP nor SL within the horizon),
-1 (skip) only for missing / out-of-range data.
Labels are precomputed once in _ensure_scalers_and_precompute.
__getitem__ returns outcome_long / outcome_short as int64 in {-1, 0, 1, 2}.
"""

import glob
import logging
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import joblib
except Exception:
    joblib = None


logger = logging.getLogger("encoder_trainer")

# Rows of the scaled data are labelled at bar CLOSE (data_management, schema 2): the row
# 10:15 is the bar [10:00, 10:15). Everything that depends on the time of day (sessions,
# Asian range, days, weekday) is computed on the bar OPEN time, so a bar is assigned to the
# session it trades in.
BAR_MINUTES = 15
# 4H (and 30min / 1H) bars are aligned on this origin by data_management.downloader
BIN_ORIGIN = "2000-01-01 00:00"
DATA_TZ = "Europe/Rome"
# The FX trading day rolls at 17:00 New York time: Sunday evening belongs to Monday.
FX_DAY_TZ = "America/New_York"
FX_DAY_ROLL_HOUR = 17


def bar_open_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC open time of close-labelled 15min bars."""
    idx = index.tz_localize(DATA_TZ) if index.tz is None else index
    return idx.tz_convert("UTC") - pd.Timedelta(minutes=BAR_MINUTES)


def fx_trading_weekday(open_utc: pd.DatetimeIndex) -> np.ndarray:
    """0=Monday .. 4=Friday of the FX trading day (rolls at 17:00 New York time)."""
    shifted = open_utc.tz_convert(FX_DAY_TZ) + pd.Timedelta(hours=24 - FX_DAY_ROLL_HOUR)
    return np.clip(shifted.day_of_week.values, 0, 4)


def is_tf_close(ts: pd.Timestamp, tf: str) -> bool:
    """True if a bar of `tf` closes at the close-labelled timestamp `ts`."""
    t = str(tf).lower()
    if t in ("15min", "15m"):
        return True
    if t in ("1d", "1day"):
        local = ts.tz_convert(DATA_TZ) if ts.tzinfo is not None else ts
        return local.hour == 0 and local.minute == 0
    minutes = {"30min": 30, "30m": 30, "1h": 60, "1hour": 60, "4h": 240, "4hour": 240}.get(t)
    if minutes is None:
        return False
    ts = ts.tz_localize(DATA_TZ) if ts.tzinfo is None else ts
    origin = pd.Timestamp(BIN_ORIGIN, tz=DATA_TZ)
    return (ts - origin) % pd.Timedelta(minutes=minutes) == pd.Timedelta(0)


_SESSION_FEATURE_NAMES = [
    "session_hour_sin",
    "session_hour_cos",
    "session_london",
    "session_ny",
    "session_overlap",
    "session_dow_sin",
    "session_dow_cos",
]

# v30: structural features (order = columns added by _compute_structural_features)
_STRUCTURAL_FEATURE_NAMES = [
    "prebreak_range_atr",
    "breakout_impulsivity",
    "day_extension_atr",
    "asian_cleanliness",
    "range_compression",
]


# =============================================================
# Utilities
# =============================================================
# ── Trade simulation for the strategy labels ─────────────────────────────────
W_BE_CUT = 0.35  # weight of a "win" the breakeven stop would have closed
W_AMBIGUOUS = 0.30  # weight when the order of TP/SL inside one bar is unknown


def previous_trading_days(dates, skip_sunday: bool = True) -> dict:
    """
    Map each date to the previous trading day, for PDH/PDL. With ``skip_sunday`` the
    Sunday dates (only the evening reopening stub, in UTC) are left out, so Monday's
    previous day is Friday.
    """
    days = [d for d in sorted(dates) if not (skip_sunday and d.weekday() == 6)]
    return {days[k]: days[k - 1] for k in range(1, len(days))}


def stop_order_filled(
    hi: np.ndarray,
    lo: np.ndarray,
    close: np.ndarray,
    break_after: np.ndarray,
    i: int,
    entry: float,
    direction: int,
) -> bool:
    """
    Strict fill of a stop order in bar ``i``. The order must be resting at the bar
    open — the previous close still on the near side of the entry, with no market
    break in between (a bar that opens beyond the entry is a gap, not a fill at the
    entry) — and the bar must reach the entry price itself, not just the level.
    """
    if i < 1 or break_after[i - 1]:
        return False
    prev_close = float(close[i - 1])
    if not np.isfinite(prev_close):
        return False
    if direction == 1:
        return prev_close < entry and float(hi[i]) >= entry
    return prev_close > entry and float(lo[i]) <= entry


def simulate_trade(
    hi: np.ndarray,
    lo: np.ndarray,
    e_idx: int,
    h_eff: int,
    tp: float,
    sl: float,
    direction: int,
    *,
    strict: bool = True,
    last_row: Optional[int] = None,
    h_req: Optional[int] = None,
    be_rr: float = 0.0,
    be_trigger_r: float = 0.25,
    be_offset_r: float = 0.07,
) -> tuple[np.int8, np.int8, np.int8, float]:
    """
    Outcome of a trade filled in bar ``e_idx``: ``(hit, side, be_reachable, weight)``.

    hit 1 = TP or SL reached, 0 = neither within the horizon, -1 = unknown (strict:
    horizon cut by a market break or by the end of the data); side 1 = TP first,
    0 = SL first, -1 = no outcome. ``weight`` is the sample weight of the side label.

    Strict (``strict=True``): the scan covers the fill bar and the ``h_eff`` bars
    after it, up to ``last_row`` (the last row before the next market break). In the
    fill bar a TP is certain — price opened on the near side of the entry, so it
    crossed the entry before reaching the target — while an SL is ambiguous, since
    the low may come before the fill: it counts as a loss with a low weight, as do
    TP and SL inside the same bar. Historical rules (``strict=False``): the scan
    starts at the bar after the fill and TP/SL in the same bar count as a win.

    Breakeven (``be_rr`` > 0 = reward/risk of the strategy): the entry is rebuilt as
    (tp + rr*sl) / (1 + rr) and R = |entry - sl|. be_reachable = the trigger at
    ``be_trigger_r`` R is touched strictly before the original SL. A loss with
    be_reachable live exits at break-even and gets a lower weight; in strict mode a
    win gets it only if, after the trigger bar, price came back to the BE stop
    (entry + ``be_offset_r`` R) before or in the TP bar.
    """
    n = len(hi)
    if strict:
        start = e_idx
        stop = e_idx + h_eff + 1
        if last_row is not None:
            stop = min(stop, last_row + 1)
        stop = min(stop, n)
        truncated = stop < e_idx + (h_req if h_req is not None else h_eff) + 1
    else:
        start = e_idx + 1
        stop = start + h_eff
        truncated = False
    fh = hi[start:stop]
    fl = lo[start:stop]
    no_outcome = np.int8(-1 if truncated else 0)
    if len(fh) == 0:
        return no_outcome, np.int8(-1), np.int8(0), 0.0
    if direction == 1:
        tp_h = np.flatnonzero(fh >= tp)
        sl_h = np.flatnonzero(fl <= sl)
    else:
        tp_h = np.flatnonzero(fl <= tp)
        sl_h = np.flatnonzero(fh >= sl)
    tb = int(tp_h[0]) if len(tp_h) else None
    sb = int(sl_h[0]) if len(sl_h) else None
    sb_or_inf = sb if sb is not None else 10**9

    be_reach = np.int8(0)
    be_cut_win = False
    if be_rr and be_rr > 0:
        entry = (tp + be_rr * sl) / (1.0 + be_rr)
        r_price = abs(entry - sl)
        if direction == 1:
            trig_h = np.flatnonzero(fh >= entry + be_trigger_r * r_price)
            be_stop_hit = fl <= entry + be_offset_r * r_price
        else:
            trig_h = np.flatnonzero(fl <= entry - be_trigger_r * r_price)
            be_stop_hit = fh >= entry - be_offset_r * r_price
        # conservative: trigger and SL in the same bar -> SL, not saved
        if len(trig_h) and int(trig_h[0]) < sb_or_inf:
            be_reach = np.int8(1)
            if strict and tb is not None and tb < sb_or_inf:
                be_cut_win = bool(be_stop_hit[int(trig_h[0]) + 1 : tb + 1].any())

    if tb is None and sb is None:
        return no_outcome, np.int8(-1), be_reach, 0.0
    hit = np.int8(1)
    loss_w = W_BE_CUT if be_reach else 1.0

    if strict:
        if sb is not None and (sb == 0 or sb == tb):
            return hit, np.int8(0), be_reach, W_AMBIGUOUS
        if tb is not None and tb < sb_or_inf:
            return hit, np.int8(1), be_reach, (W_BE_CUT if be_cut_win else 1.0)
        return hit, np.int8(0), be_reach, loss_w

    # historical rules
    if tb is not None and sb is not None:
        if tb == sb:
            return hit, np.int8(1), be_reach, W_AMBIGUOUS
        if tb < sb:
            return hit, np.int8(1), be_reach, (W_BE_CUT if be_reach else 1.0)
        # historical quirk: a BE-saved loss kept full weight when TP came later
        return hit, np.int8(0), be_reach, 1.0
    if tb is not None:
        return hit, np.int8(1), be_reach, 1.0
    return hit, np.int8(0), be_reach, loss_w


def session_features(index: pd.DatetimeIndex) -> np.ndarray:
    """
    The 7 session features of close-labelled 15min bars, [n, 7] float32: hour sin/cos,
    London / New York / overlap flags (on the bar OPEN time, the session it trades in),
    weekday sin/cos of the FX trading day. Shared by the dataset and the trading agent.
    """
    n = len(index)
    try:
        idx = index
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError("Index is not DatetimeIndex")
        idx_utc = bar_open_utc(idx)  # the session a bar trades in = its open time
        h = idx_utc.hour.values.astype(np.float32)
        m = idx_utc.minute.values.astype(np.float32)
        h_frac = h + m / 60.0
        hour_sin = np.sin(2.0 * np.pi * h_frac / 24.0).astype(np.float32)
        hour_cos = np.cos(2.0 * np.pi * h_frac / 24.0).astype(np.float32)
        h_int = idx_utc.hour.values
        session_london = ((h_int >= 7) & (h_int < 16)).astype(np.float32)
        session_ny = ((h_int >= 13) & (h_int < 21)).astype(np.float32)
        session_overlap = ((h_int >= 13) & (h_int < 16)).astype(np.float32)
        # FX trading day, 0..4: Sunday evening counts as Monday (the calendar weekday
        # made Sunday 6 -> same sin/cos as Tuesday with a 5-day period)
        dow = fx_trading_weekday(idx_utc).astype(np.float32)
        dow_sin = np.sin(2.0 * np.pi * dow / 5.0).astype(np.float32)
        dow_cos = np.cos(2.0 * np.pi * dow / 5.0).astype(np.float32)
    except Exception as e:
        logger.warning("Session feature computation failed (%s). Using zeros.", e)
        return np.zeros((n, len(_SESSION_FEATURE_NAMES)), dtype=np.float32)
    return np.stack(
        [hour_sin, hour_cos, session_london, session_ny, session_overlap, dow_sin, dow_cos],
        axis=1,
    )


def structural_features(real_col, n: int, stats: Optional[dict] = None):
    """v30 — 5 structural features computed on EVERY bar as a time series (they're
    symmetric, they don't depend on the trade direction), on REAL prices: the 15min
    *_raw columns and the inverse-transformed daily levels (high_1D / low_1D / ATR14_1D).

    Feature order: [prebreak_range_atr, breakout_impulsivity,
                     day_extension_atr, asian_cleanliness, range_compression]

    Robust normalization (x - median) / IQR, clipped to [-5, 5]. The median/IQR come
    from `stats` when given (the training window's, see the trainer) and are otherwise
    computed on this data and returned, so the validation and live data are normalized
    with the training statistics. real_col(name) gives the real (unscaled) values of a
    column or None; returns (features [n, 5] float32, stats).
    Default OFF; with the flag ON they are continuous encoder inputs.
    """
    NF = len(_STRUCTURAL_FEATURE_NAMES)
    hi = real_col("high_15min")
    lo = real_col("low_15min")
    cl = real_col("close_15min")
    op = real_col("open_15min")
    if hi is None or lo is None or cl is None:
        raise ValueError("structural features need the real 15min high/low/close")
    if op is None:
        op = cl
    # reference ATR: real daily if available, otherwise 15min
    atr = real_col("atr14_1d")
    if atr is None:
        atr = real_col("atr14_15min")
    if atr is None:
        raise ValueError("structural features need ATR14_1D or ATR14_15min")
    atr = np.where((atr > 0) & np.isfinite(atr), atr, np.nan)
    hi1d = real_col("high_1d")
    lo1d = real_col("low_1d")

    hi_s, lo_s = pd.Series(hi), pd.Series(lo)
    out = np.full((n, NF), np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        # prebreak_range_atr: range of the last 8 CLOSED bars / ATR
        PRE, WIDE = 8, 24
        pre_hi = hi_s.rolling(PRE).max().shift(1).values
        pre_lo = lo_s.rolling(PRE).min().shift(1).values
        out[:, 0] = (pre_hi - pre_lo) / atr
        # breakout_impulsivity: sum|body| / sum range of the last 3 bars
        body_s = pd.Series(np.abs(cl - op)).rolling(3).sum().shift(1).values
        rng_s = pd.Series(hi - lo).rolling(3).sum().shift(1).values
        out[:, 1] = np.where(rng_s > 0, body_s / rng_s, np.nan)
        # day_extension_atr: range of the last closed daily bar / ATR
        if hi1d is not None and lo1d is not None:
            out[:, 2] = (hi1d - lo1d) / atr
        else:
            DAY = 96  # fallback: range of the last 96 bars
            out[:, 2] = (
                hi_s.rolling(DAY).max().shift(1).values - lo_s.rolling(DAY).min().shift(1).values
            ) / atr
        # asian_cleanliness: net move / total move of close over the last 32 bars
        ASIA = 32
        net = np.abs(pd.Series(cl).diff(ASIA - 1).values)
        tot = pd.Series(np.abs(np.diff(cl, prepend=np.nan))).rolling(ASIA - 1).sum().values
        out[:, 3] = pd.Series(np.where(tot > 0, net / tot, np.nan)).shift(1).values
        # range_compression: 8-bar range / 24-bar range
        wide = hi_s.rolling(WIDE).max().shift(1).values - lo_s.rolling(WIDE).min().shift(1).values
        out[:, 4] = np.where(wide > 0, (pre_hi - pre_lo) / wide, np.nan)

    # Robust normalization with the given (training) statistics, or this data's.
    # The parquet features are pre-scaled and the session ones bounded: raw ratios
    # (prebreak / day_extension can be 3-5) would unbalance training.
    if stats is None:
        stats = {}
        for j, name in enumerate(_STRUCTURAL_FEATURE_NAMES):
            finite = out[:, j][np.isfinite(out[:, j])]
            med, scale = 0.0, 1.0
            if finite.size > 100:
                med = float(np.median(finite))
                q1, q3 = np.percentile(finite, [25.0, 75.0])
                sd = float(np.std(finite))
                scale = float(q3 - q1) if q3 - q1 > 1e-8 else (sd if sd > 1e-8 else 1.0)
            stats[name] = (med, scale)
    for j, name in enumerate(_STRUCTURAL_FEATURE_NAMES):
        med, scale = stats[name]
        out[:, j] = np.clip((out[:, j] - med) / scale, -5.0, 5.0)
    # leftover NaN (rolling warm-up) -> 0 = "typical" value after centering
    return np.where(np.isfinite(out), out, 0.0).astype(np.float32), stats


def winsorize_series(a: np.ndarray, lower_pct=1.0, upper_pct=99.0) -> np.ndarray:
    if a is None:
        return a
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a
    mask = np.isnan(a)
    valid = a[~mask]
    if valid.size == 0:
        return a
    lo = np.percentile(valid, lower_pct)
    hi = np.percentile(valid, upper_pct)
    out = a.copy()
    out[~mask] = np.clip(valid, lo, hi)
    return out


def zscore_series(a: np.ndarray, eps=1e-12) -> np.ndarray:
    if a is None:
        return a
    a = np.asarray(a, dtype=float)
    mask = np.isnan(a)
    valid = a[~mask]
    if valid.size == 0:
        return a
    mu = float(np.mean(valid))
    sigma = float(np.std(valid))
    if sigma <= eps:
        out = np.full_like(a, 0.0, dtype=float)
        out[mask] = np.nan
        return out
    out = (a - mu) / (sigma + eps)
    out[mask] = np.nan
    return out


def slice_df_by_period(
    df: pd.DataFrame, start_date: str, end_date: str, tz="Europe/Rome"
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(tz)
    else:
        start_ts = start_ts.tz_convert(tz)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize(tz)
    else:
        end_ts = end_ts.tz_convert(tz)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("DataFrame must have DatetimeIndex for slice_df_by_period")
    df2 = df.tz_localize(tz) if df.index.tz is None else df.tz_convert(tz)
    return df2.loc[start_ts:end_ts].copy()


def tf_to_15m_steps(tf: str) -> int:
    t = str(tf).strip().lower()
    if t in ("15min", "15m"):
        return 1
    if t in ("30min", "30m"):
        return 2
    if t in ("1h", "1hour", "60min", "60m"):
        return 4
    if t in ("4h", "4hour", "240min", "240m"):
        return 16
    if t in ("1d", "1day", "24h"):
        return 96
    m = re.match(r"^\s*(\d+)\s*min\s*$", t)
    if m:
        return max(1, int(m.group(1)) // 15)
    m = re.match(r"^\s*(\d+)\s*h(?:our)?s?\s*$", t)
    if m:
        return max(1, (int(m.group(1)) * 60) // 15)
    return 1


def default_trend_horizons(tf_list: list[str]) -> dict[str, int]:
    return {tf: tf_to_15m_steps(tf) for tf in tf_list}


# =============================================================
# Config
# =============================================================
@dataclass
class MarketPretrainConfig:
    seq_len: int = 128
    # ── WHICH BAR THE MODEL SEES ─────────────────────────────────────────────
    #   0 = up to the signal bar INCLUDED (historical behaviour)
    #   1 = up to the PREVIOUS bar
    #
    # With 0 the model judges a trade having seen the bar where price breaks
    # the level. But that trade opens AT THE LEVEL, i.e. at a price touched
    # inside that bar before it closed: information that doesn't exist yet
    # live, when the decision is taken.
    # With 1 the model only sees closed bars. The decision can be taken at the
    # end of bar i-1, placing a STOP order at the level that gets filled
    # exactly there during bar i — as the backtest assumes.
    #
    # The LABELS stay anchored to the signal bar: the model isn't asked to
    # predict IF the break will happen (the pending order takes care of that),
    # but how the trade goes IF it happens.
    score_offset: int = 0
    step_stride: int = 1
    mask_prob: float = 0.25
    tf_list: tuple[str, ...] = ("15min", "30min", "1h", "4h", "1d")

    trend_horizons: Optional[dict[str, int]] = None
    permute_mode: str = "shuffle"
    seed: int = 42

    trend_threshold_scaled: float = 1e-3
    trend_threshold_scaled_tf: Optional[dict[str, float]] = None
    trend_quantile_tail: float = 0.20

    # ATR-adjusted trend labeling
    # mode: "absolute" (legacy, 80% FLAT) | "atr_adjusted" (recommended)
    # atr_mult: threshold in ATR units — 0.4 gives ~25/50/25 distribution on FX majors
    # atr_mult_tf: per-TF override (higher TFs have larger natural moves vs 15m ATR)
    trend_label_mode: str = "atr_adjusted"
    trend_atr_mult: float = 0.4
    trend_atr_mult_tf: Optional[dict[str, float]] = None
    atr_lookback: int = 14

    use_session_features: bool = True

    # ── v30: structural features (3-fence research, 2025 holdout confirmed) ──
    # 5 features validated OOS as CONTINUOUS encoder inputs (not gates/thresholds):
    #   prebreak_range_atr, breakout_impulsivity  -> useful for the sessions
    #   day_extension_atr, asian_cleanliness, range_compression -> useful for pdhl
    # They're symmetric (don't depend on the trade direction), so they are
    # computed on EVERY bar as a time series, like the session features.
    # They use the inverse-scaled daily levels (high_1D_real, ATR14_1D_real, ...).
    #
    # DEFAULT OFF: with use_structural_features=False the dataset is IDENTICAL to v28/v29
    # (instant rollback, nothing rewritten). Set True to train with the new set
    # at WF8 and compare val AUC + encoder_filter_sweep against the baseline.
    use_structural_features: bool = False

    # ── v28: SessionBreakoutHead labels ───────────────────────────────────────
    # Majors: tp_r=2.5 (RR=2.5:1) — best for Majors (from the sweep)
    # Minors: tp_r=1.8 (RR=1.8:1) — best for Minors (from the sweep)
    # picked by asset_name in the precompute
    outcome_session_tp_r_majors: float = 2.5
    outcome_session_tp_r_minors: float = 1.8
    outcome_session_sl_r: float = 1.0
    strategy_session_horizon_bars: int = 32

    # ── v28: PDHLBreakoutHead labels ──────────────────────────────────────────
    # Both universes: tp_r=1.8 (RR=1.8:1) — best for Majors and Minors
    outcome_pdhl_tp_r: float = 1.8
    outcome_pdhl_sl_r: float = 1.0
    strategy_pdhl_horizon_bars: int = 64
    # DI gate of the PDHL strategy (as the simulator): long only if
    # tanh((DI+ - DI-) / 20) on 4H > thresh, short only if < -thresh, using the 4H DI
    # known when the order is placed (the bar before the signal).
    outcome_pdhl_di_gate: bool = True
    outcome_pdhl_di_thresh: float = 0.15
    # Executable trade simulation for the strategy labels (session + PDHL):
    # - the stop order fills only if it was resting at the bar open (previous
    #   close on the right side of the entry) and the bar reaches the ENTRY price,
    #   not just the level;
    # - the fill bar is part of the TP/SL scan (an SL there is ambiguous);
    # - TP and SL in the same bar count as a loss (low confidence);
    # - a BE-protected win is down-weighted only if price came back to the BE stop;
    # - the horizon stops at weekend/holiday gaps (unresolved -> skipped);
    # - Monday's previous day for PDH/PDL is Friday, not the Sunday-evening stub.
    # False reproduces the historical labels.
    outcome_strict_execution: bool = True

    # ── v28: RegimeQualityHead labels ─────────────────────────────────────────
    # label=1 if the session (32 bars after end_idx) has at least 1 TP over N entries
    # approximated by: hit_session_long OR hit_session_short (something moves)
    regime_quality_horizon_bars: int = 32  # bars used to judge the quality

    regime_rolling_n: int = 500
    regime_rolling_by_tf: Optional[dict[str, int]] = None
    regime_low_thr: float = 0.33
    label_exclude_tfs: Optional[list] = None
    regime_high_thr: float = 0.67
    # ── v27: side-label denoising for StrategyOutcomeHead ─────────────────────
    # side head is trained only on "clean hit" events.
    # A hit is considered clean when:
    # - the winning side exceeds the losing side by at least
    #   strategy_side_margin_atr × ATR
    # - and the hit happens after at least strategy_side_min_hit_bars bars
    #   (avoid same-bar / next-bar noise)
    # If not clean -> side_label = -1 (skip), while hit_label remains valid.
    strategy_side_margin_atr: float = 0.15
    strategy_side_min_hit_bars: int = 2

    # ── v29: breakeven label (optional, backward compatible) ──────────────────
    # With outcome_be_enabled=True, besides (hit, side) a third label is built,
    # be_reachable: 1 if — with the BE rule at outcome_be_trigger_r × R — the
    # trade would reach breakeven (SL moved to BE+costs) BEFORE dying at the
    # original SL; 0 otherwise.
    #
    # DESIGN CHOICE: BE does NOT enter the side label (P(TP first)), which stays
    # computed only on trades with a clean TP/SL outcome — so the side_auc
    # metric already validated on 6 WFs isn't touched. be_reachable is a SEPARATE
    # dimension the filter can use for a drawdown-oriented selection
    # (prefer trades protected by BE) without degrading P(TP).
    #
    # Default OFF: with outcome_be_enabled=False the dataset is identical to v28.
    outcome_be_enabled: bool = False
    outcome_be_trigger_r: float = 0.25  # BE trigger in fractions of R (= 25%R, best OOS)
    outcome_be_offset_r: float = 0.07  # where the BE goes (= COST_R, breakeven+commissions)

    # Optional stricter rule for MR, where micro-noise is more dangerous

    # MktStructure per-TF thresholds — set by trainer based on asset class
    # dist_thresh: |close-EMA|/ATR for RANGING/TRENDING boundary
    # bo_window:   breakout window in candles
    mktstructure_dist_thresh_by_tf: Optional[dict[str, float]] = None
    mktstructure_bo_window_by_tf: Optional[dict[str, int]] = None

    def __post_init__(self):
        if self.trend_threshold_scaled_tf is None:
            self.trend_threshold_scaled_tf = {}
        if self.trend_atr_mult_tf is None:
            # Higher TFs have moves naturally larger than 15min ATR.
            # These multipliers keep the threshold "economically equivalent"
            # across TFs relative to 15min volatility.
            # sqrt scaling: 30min≈sqrt(2)×, 1h≈2×, 4h≈4×, 1d≈8×
            self.trend_atr_mult_tf = {
                "15min": 1.0,
                "30min": 1.4,
                "1h": 2.0,
                "1H": 2.0,
                "4h": 4.0,
                "4H": 4.0,
                "1d": 8.0,
                "1D": 8.0,
            }
        if self.label_exclude_tfs is None:
            self.label_exclude_tfs = ["15min"]
        if self.regime_rolling_by_tf is None:
            self.regime_rolling_by_tf = {
                "15min": 500,
                "30min": 500,
                "1h": 300,
                "1H": 300,
                "4h": 100,
                "4H": 100,
                "1d": 250,
                "1D": 250,
            }
        if self.trend_horizons is None:
            self.trend_horizons = default_trend_horizons(list(self.tf_list))


# =============================================================
# Scaler helpers
# =============================================================
def _load_all_scalers_file(path: str) -> Optional[dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        obj = joblib.load(path) if joblib is not None else None
        if obj is None:
            raise ValueError("joblib not available")
    except Exception:
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            logger.warning(f"Unable to load scaler file {path}: {e}")
            return None
    if not isinstance(obj, dict):
        logger.warning(f"Scaler file {path} does not contain expected dict structure.")
        return None
    col_scalers = obj.get("col_scalers") or obj.get("col_scalers_", None) or {}
    return col_scalers


def _normalize_scaler_map(col_scalers: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    if isinstance(col_scalers, dict):
        for k, v in col_scalers.items():
            normalized[k] = v
            kl = str(k).lower()
            normalized.setdefault(kl, v)
            normalized.setdefault(kl.replace(" ", "_"), v)
            normalized.setdefault("_".join([p for p in kl.replace(" ", "_").split("_") if p]), v)
    return normalized


def _inverse_transform_if_possible(
    df: pd.DataFrame, col: str, scaler_map: dict[str, Any]
) -> np.ndarray:
    arr_orig = pd.to_numeric(df[col], errors="coerce").values.astype(float)
    transformer = scaler_map.get(col) or scaler_map.get(col.lower()) or None
    if transformer is None:
        return arr_orig
    try:
        Xcol = df[[col]]
        if hasattr(transformer, "inverse_transform"):
            inv = np.asarray(transformer.inverse_transform(Xcol)).reshape(-1)
            return inv.astype(float)
    except Exception as e:
        logger.warning(f"Failed inverse_transform for {col}: {e}")
    return arr_orig


# =============================================================
# Discovery helpers
# =============================================================
def autodetect_feature_columns(df: pd.DataFrame, tf_list: list[str]) -> list[str]:
    cols = []
    all_cols = [str(c) for c in df.columns]

    def add_unique(cand_list):
        for c in cand_list:
            if str(c).lower().endswith("_raw"):
                continue
            if c not in cols:
                cols.append(c)

    for tf in tf_list:
        suf = f"_{tf}"
        matched = [
            c
            for c in all_cols
            if c.lower().endswith(suf.lower()) and not c.lower().endswith("_raw")
        ]
        add_unique(matched)

    if len(cols) == 0:
        numeric_cols = [c for c in all_cols if not c.lower().endswith("_raw")]
        try:
            numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(df[c])]
        except Exception:
            pass
        add_unique(numeric_cols)

    return [c for c in cols if c in all_cols]


def encoder_input_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    use_session_features: bool = True,
    use_structural_features: bool = False,
    structural_stats: Optional[dict] = None,
    scaler_map: Optional[dict] = None,
) -> np.ndarray:
    """
    Encoder input of every row of a scaled parquet, [n, n_features] float32, built as
    MarketPretrainDataset builds it: the feature columns (lowercase names, in the given
    order), the session features, the structural features. For the structural ones pass
    the training-window `structural_stats` of the encoder and the asset's scalers
    (all_scalers.pkl) for the daily levels without a *_raw copy.
    """
    d = df.copy()
    d.columns = [str(c).lower() for c in d.columns]
    missing = [c for c in feature_cols if c not in d.columns]
    if missing:
        raise KeyError(f"encoder input columns missing from the data: {missing[:5]}")
    parts = [d[feature_cols].values.astype(np.float32)]
    if use_session_features:
        parts.append(session_features(d.index))
    if use_structural_features:
        smap = _normalize_scaler_map(scaler_map or {})

        def real_col(name):
            if f"{name}_raw" in d.columns:
                return pd.to_numeric(d[f"{name}_raw"], errors="coerce").values.astype(np.float64)
            if name not in d.columns or smap.get(name) is None:
                return None
            return _inverse_transform_if_possible(d, name, smap).astype(np.float64)

        if structural_stats is None:
            logger.warning("structural features normalized with this data's own statistics")
        parts.append(structural_features(real_col, len(d), structural_stats)[0])
    return np.concatenate(parts, axis=1)


def discover_scaler_files_auto(base_dir: str, class_name: str) -> dict[str, str]:
    search_root = os.path.join(base_dir, class_name)
    if not os.path.exists(search_root):
        return {}
    patterns = [
        os.path.join(search_root, "**", "all_scalers.*"),
        os.path.join(search_root, "**", "*_all_scalers.*"),
    ]
    found = {}
    prefer_ext_order = [".pkl", ".joblib", ".plk"]

    def score(p):
        e = os.path.splitext(p)[1].lower()
        try:
            return prefer_ext_order.index(e)
        except Exception:
            return 99

    for patt in patterns:
        for fp in glob.glob(patt, recursive=True):
            if not os.path.isfile(fp):
                continue
            fname = os.path.basename(fp)
            name, _ext = os.path.splitext(fname)
            if name.endswith("_all_scalers"):
                asset = name[: -len("_all_scalers")]
            else:
                parent = os.path.basename(os.path.dirname(fp))
                asset = parent if parent.lower() not in ("scalers", "scaler", "") else name
            prev = found.get(asset)
            if prev is None or score(fp) < score(prev):
                found[asset] = fp

    if not found:
        logger.info("No scaler files auto-discovered under %s", search_root)
    else:
        logger.info(f"Auto-discovered scalers for: {list(found.keys())}")
    return found


def _pick_data_file(files: list[str]) -> Optional[str]:
    """The asset's input file: never the bars cache, the *_scaled_* file if there is one."""
    files = [f for f in files if "_bars15min_" not in os.path.basename(f)]
    scaled = [f for f in files if "_scaled_" in os.path.basename(f)]
    files = sorted(scaled or files)
    return files[0] if files else None


def discover_asset_parquets(
    base_dir: str, class_name: str, symbols: Optional[list[str]] = None
) -> dict[str, str]:
    class_dir = os.path.join(base_dir, class_name)
    if not os.path.exists(class_dir):
        raise RuntimeError(f"Class directory not found: {class_dir}")

    result: dict[str, str] = {}

    if symbols:
        for s in symbols:
            for ext in (".parquet", ".csv"):
                p = os.path.join(class_dir, f"{s}{ext}")
                if os.path.exists(p):
                    result[s] = p
                    break
            if s in result:
                continue
            folder = os.path.join(class_dir, s)
            if os.path.isdir(folder):
                pick = _pick_data_file(
                    [
                        os.path.join(folder, f)
                        for f in os.listdir(folder)
                        if f.endswith(".parquet") or f.endswith(".csv")
                    ]
                )
                if pick:
                    result[s] = pick
                    continue
            files = [
                f
                for f in os.listdir(class_dir)
                if f.lower().startswith(s.lower())
                and (f.endswith(".parquet") or f.endswith(".csv"))
            ]
            if files:
                result[s] = os.path.join(class_dir, files[0])
        if not result:
            raise RuntimeError(f"No asset files found for requested symbols under {class_dir}")
        return result

    subdirs = [d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))]
    for d in subdirs:
        folder = os.path.join(class_dir, d)
        pick = _pick_data_file(
            [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith(".parquet") or f.endswith(".csv")
            ]
        )
        if pick:
            result[d] = pick

    if not result:
        for f in os.listdir(class_dir):
            if f.endswith(".parquet") or f.endswith(".csv"):
                result[os.path.splitext(f)[0]] = os.path.join(class_dir, f)

    if not result:
        raise RuntimeError(f"No asset parquet/csv files discovered under {class_dir}")
    return result


# =============================================================
# Threshold computation
# =============================================================
def compute_trend_thresholds_for_asset(
    parquet_path: str,
    train_start: str,
    train_end: str,
    tf_list: list[str],
    config: dict[str, Any],
    scaler_path: Optional[str] = None,
) -> dict[str, tuple[float, float]]:
    if parquet_path.endswith(".parquet"):
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(parquet_path, parse_dates=True, index_col=0)

    if not isinstance(df.index, pd.DatetimeIndex):
        possible = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
        if possible:
            df[possible[0]] = pd.to_datetime(df[possible[0]])
            df.set_index(possible[0], inplace=True)

    try:
        df.index = (
            df.index.tz_localize("Europe/Rome")
            if df.index.tz is None
            else df.index.tz_convert("Europe/Rome")
        )
    except Exception:
        pass
    df.columns = [c.lower() for c in df.columns]

    try:
        tr_df = slice_df_by_period(df, train_start, train_end)
    except Exception:
        tr_df = df[
            (df.index >= pd.to_datetime(train_start)) & (df.index <= pd.to_datetime(train_end))
        ]

    scaler_map = {}
    if scaler_path:
        col_scalers = _load_all_scalers_file(scaler_path) or {}
        scaler_map = _normalize_scaler_map(col_scalers)

    p_tail = float(config.get("trend_quantile_tail", 0.20))
    p_tail = min(max(p_tail, 0.01), 0.49)
    epsv = 1e-12

    out: dict[str, tuple[float, float]] = {}
    for tf in tf_list:
        h = int(tf_to_15m_steps(tf))
        close_col = f"close_{tf}"
        if close_col not in tr_df.columns:
            out[tf] = (0.0, 0.0)
            continue

        close = (
            _inverse_transform_if_possible(tr_df, close_col, scaler_map)
            if scaler_map
            else pd.to_numeric(tr_df[close_col], errors="coerce")
            .ffill()
            .bfill()
            .values.astype(float)
        )
        close = pd.Series(close).ffill().bfill().values.astype(float)

        if close.size <= (h + 1):
            out[tf] = (0.0, 0.0)
            continue

        cur = close[:-h]
        fut = close[h:]
        denom = np.maximum(np.abs(cur), epsv)
        ret = (fut - cur) / denom
        ret = ret[np.isfinite(ret)]

        if ret.size == 0:
            out[tf] = (0.0, 0.0)
            continue

        q_low = float(np.percentile(ret, 100.0 * p_tail))
        q_high = float(np.percentile(ret, 100.0 * (1.0 - p_tail)))
        out[tf] = (q_low, q_high)

    return out


def analyze_threshold_impact(
    parquet_path: str, tf_list: list[str], scaler_path: Optional[str] = None
):
    if parquet_path.endswith(".parquet"):
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(parquet_path, parse_dates=True, index_col=0)
    df.columns = [c.lower() for c in df.columns]

    scaler_map = {}
    if scaler_path:
        col_scalers = _load_all_scalers_file(scaler_path) or {}
        scaler_map = _normalize_scaler_map(col_scalers)

    candidates = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0015, 0.0020]
    for tf in tf_list:
        h = tf_to_15m_steps(tf)
        col = f"close_{tf}"
        if col not in df.columns:
            col = next((c for c in df.columns if c.lower() == f"close_{tf.lower()}"), None)
        if col is None:
            continue
        close = (
            _inverse_transform_if_possible(df, col, scaler_map)
            if scaler_map
            else pd.to_numeric(df[col], errors="coerce").values.astype(float)
        )
        close = pd.Series(close).ffill().bfill().values.astype(float)
        if len(close) <= h:
            continue
        from_idx = np.where(np.diff(close) != 0)[0] + 1
        if len(from_idx) < 2:
            continue
        cur_c = close[from_idx[:-1]]
        fut_c = close[from_idx[1:]]
        ret = (fut_c - cur_c) / np.maximum(np.abs(cur_c), 1e-12)
        ret = ret[np.isfinite(ret)]
        print(f"\n=== {tf} (n={len(ret):,}) ===")
        for thr in candidates:
            pct_up = (ret > thr).mean() * 100
            pct_down = (ret < -thr).mean() * 100
            pct_flat = 100 - pct_up - pct_down
            marker = " ← TARGET (UP ~25%)" if 20 <= pct_up <= 35 else ""
            print(
                f"  thr={thr:.4f}: UP={pct_up:5.1f}% "
                f"FLAT={pct_flat:5.1f}% DOWN={pct_down:5.1f}%{marker}"
            )


class MarketPretrainDataset(Dataset):
    def __init__(
        self,
        df_or_path: Union[pd.DataFrame, str],
        config: MarketPretrainConfig,
        feature_cols: list[str],
        asset_name: Optional[str] = None,
        scaler_path: Optional[str] = None,
        scaler_map: Optional[dict[str, Any]] = None,
        slice_period: Optional[tuple[str, str]] = None,
        regime_thresholds: Optional[dict[str, tuple[float, float]]] = None,
        trend_thresholds: Optional[dict[str, tuple[float, float]]] = None,
        structural_stats: Optional[dict[str, tuple[float, float]]] = None,
    ):
        super().__init__()
        # median / IQR of the structural features; None = computed on this data
        self._structural_stats = (
            {k: tuple(v) for k, v in structural_stats.items()} if structural_stats else None
        )
        self._df_path = df_or_path if isinstance(df_or_path, str) else None
        self._df: Optional[pd.DataFrame] = (
            df_or_path if isinstance(df_or_path, pd.DataFrame) else None
        )
        self._slice_period = slice_period
        self.config = config

        self.seq_len = config.seq_len
        self.step_stride = config.step_stride
        self.mask_prob = config.mask_prob
        self.tf_list = list(config.tf_list)
        self.trend_horizons = config.trend_horizons or default_trend_horizons(self.tf_list)
        self.permute_mode = config.permute_mode
        self.asset_name = asset_name

        self.scaler_path = scaler_path
        self.scaler_map = scaler_map if scaler_map is not None else None
        self._scalers_loaded = False
        self._loaded = False

        self.feature_cols = [c for c in feature_cols if not c.lower().endswith("_raw")]
        self.data = None
        self.indices = None
        self.close_col_by_tf = {}
        self.close_unscaled = {}
        self.inverse_applied = {}
        self._next_tf_close_by_tf = {}
        self._regime_continuous_by_tf = {}
        self._mktstructure_arr: Optional[np.ndarray] = None  # 15min market structure labels
        self._atr_by_tf: dict[str, Optional[np.ndarray]] = {}  # per-TF ATR unscaled arrays
        self._regime_cont_by_tf: dict[str, dict[str, Optional[np.ndarray]]] = (
            {}
        )  # per-TF {vol_regime, atr_norm, adx}
        # NumPy price caches (populated in _ensure_scalers_and_precompute)
        self._close15_np: Optional[np.ndarray] = None  # close_15min unscaled
        self._atr_raw_np: Optional[np.ndarray] = None  # ATR14_15min_raw
        self._hi_raw_np: Optional[np.ndarray] = None  # high_15min_raw
        self._lo_raw_np: Optional[np.ndarray] = None  # low_15min_raw

        self._provided_regime_thresholds = regime_thresholds or {}
        self._provided_trend_thresholds = trend_thresholds or {}
        self._asset_id = 0
        self._warned_tfs: set = set()
        self._rng = np.random.default_rng(config.seed)

        # v26: strategy-specific outcome arrays — 2-head decomposition (hit / side)
        # hit_label:  0=NONE (price reaches neither TP nor SL), 1=hit (something touched)
        # side_label: 1=TP_first, 0=SL_first, -1=skip (only when hit=0)
        # v28: SessionBreakoutHead
        self._strat_hit_session_long: Optional[np.ndarray] = None
        self._strat_hit_session_short: Optional[np.ndarray] = None
        # v29: be_reachable arrays (filled only if outcome_be_enabled)
        self._strat_be_session_long: Optional[np.ndarray] = None
        self._strat_be_session_short: Optional[np.ndarray] = None
        self._strat_be_pdhl_long: Optional[np.ndarray] = None
        self._strat_be_pdhl_short: Optional[np.ndarray] = None
        self._strat_side_session_long: Optional[np.ndarray] = None
        self._strat_side_session_short: Optional[np.ndarray] = None
        # v28: PDHLBreakoutHead
        self._strat_hit_pdhl_long: Optional[np.ndarray] = None
        self._strat_hit_pdhl_short: Optional[np.ndarray] = None
        self._strat_side_pdhl_long: Optional[np.ndarray] = None
        self._strat_side_pdhl_short: Optional[np.ndarray] = None
        # v28: RegimeQualityHead
        self._regime_quality_label: Optional[np.ndarray] = None

        if self._df is not None:
            self._postprocess_loaded_df()

    def _safe_load_df_from_path(self, path: str) -> pd.DataFrame:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path, parse_dates=True, index_col=0)

    def _normalize_df_index_and_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            possible = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
            if possible:
                try:
                    df[possible[0]] = pd.to_datetime(df[possible[0]])
                    df.set_index(possible[0], inplace=True)
                except Exception:
                    pass
        try:
            df.index = (
                df.index.tz_localize("Europe/Rome")
                if df.index.tz is None
                else df.index.tz_convert("Europe/Rome")
            )
        except Exception:
            pass
        df.columns = [c.lower() for c in df.columns]
        return df

    def _postprocess_loaded_df(self):
        df = self._normalize_df_index_and_columns(self._df)
        self.feature_cols = [c for c in self.feature_cols if c in df.columns]
        if not self.feature_cols:
            raise ValueError("No feature columns after excluding *_raw and intersecting with df")

        self.data = df[self.feature_cols].values.astype(np.float32)

        if getattr(self.config, "use_session_features", True):
            session_arr = self._compute_session_features(df)
            self.data = np.concatenate([self.data, session_arr], axis=1)
            self._n_session_features = session_arr.shape[1]
        else:
            self._n_session_features = 0

        # ── v30: structural features (default OFF -> dataset identical to before) ──
        if getattr(self.config, "use_structural_features", False):
            struct_arr = self._compute_structural_features(df)
            self.data = np.concatenate([self.data, struct_arr], axis=1)
            self._n_structural_features = struct_arr.shape[1]
        else:
            self._n_structural_features = 0

        n = len(df)
        max_h = max(list(self.trend_horizons.values()) + [1])
        self.indices = [end_idx for end_idx in range(self.seq_len - 1, n - max_h, self.step_stride)]

        _col_lower = {c.lower(): c for c in df.columns}
        for tf in self.tf_list:
            cand_exact = f"close_{tf}"
            cand_lower = cand_exact.lower()
            if cand_exact in df.columns:
                self.close_col_by_tf[tf] = cand_exact
            elif cand_lower in _col_lower:
                self.close_col_by_tf[tf] = _col_lower[cand_lower]
            else:
                matches = [
                    c
                    for c in df.columns
                    if c.lower().startswith("close_")
                    and c.lower().endswith(f"_{tf.lower()}")
                    and not c.lower().endswith("_raw")
                ]
                self.close_col_by_tf[tf] = matches[0] if matches else None

        self._df = df
        self._loaded = True
        # Eagerly precompute scalers after loading to avoid per-sample overhead
        # (especially critical with multiprocessing DataLoader workers)
        try:
            self._ensure_scalers_and_precompute()
        except Exception as _e:
            logger.debug("Eager scaler precompute deferred: %s", _e)

    @property
    def n_input_features(self) -> int:
        self._ensure_loaded()
        return (
            len(self.feature_cols)
            + getattr(self, "_n_session_features", 0)
            + getattr(self, "_n_structural_features", 0)
        )

    def _real_col(self, df: pd.DataFrame, name: str) -> Optional[np.ndarray]:
        """
        Real (unscaled) values of a column: its *_raw copy if the scaler kept one, else the
        inverse transform with the asset's scalers. None if neither is possible.
        """
        name = name.lower()
        if f"{name}_raw" in df.columns:
            return pd.to_numeric(df[f"{name}_raw"], errors="coerce").values.astype(np.float64)
        if name not in df.columns:
            return None
        if self.scaler_map is None and self.scaler_path:
            self.scaler_map = _load_all_scalers_file(self.scaler_path) or {}
        smap = _normalize_scaler_map(self.scaler_map or {})
        if smap.get(name) is None:
            return None
        return _inverse_transform_if_possible(df, name, smap).astype(np.float64)

    def _compute_structural_features(self, df: pd.DataFrame) -> np.ndarray:
        out, self._structural_stats = structural_features(
            lambda name: self._real_col(df, name), len(df), self._structural_stats
        )
        return out

    @property
    def structural_stats(self) -> Optional[dict]:
        """Median / IQR used to normalize the structural features (None if off)."""
        self._ensure_loaded()
        return self._structural_stats

    def _compute_session_features(self, df: pd.DataFrame) -> np.ndarray:
        return session_features(df.index)

    def _ensure_loaded(self):
        if self._loaded:
            return
        if self._df is None:
            if self._df_path is None:
                raise RuntimeError("Dataset has no DataFrame or path")
            df = self._safe_load_df_from_path(self._df_path)
            if self._slice_period is not None:
                start, end = self._slice_period
                try:
                    df = slice_df_by_period(df, start, end)
                except Exception:
                    try:
                        df = df[
                            (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))
                        ]
                    except Exception:
                        pass
            self._df = df
        self._postprocess_loaded_df()

    def __len__(self):
        self._ensure_loaded()
        return len(self.indices) if self.indices is not None else 0

    def _get_window(self, end_idx: int) -> np.ndarray:
        """Input window. With score_offset>0 it ends BEFORE the signal."""
        off = int(getattr(self.config, "score_offset", 0))
        last = max(self.seq_len - 1, end_idx - off)
        start = last - self.seq_len + 1
        return self.data[start : last + 1].copy()

    def _mask_input(self, seq: np.ndarray):
        mask = self._rng.random(seq.shape) < self.mask_prob
        masked = seq.copy()
        masked[mask] = 0.0
        return masked.astype(np.float32), mask.astype(np.bool_)

    def _is_tf_close(self, ts_pd: pd.Timestamp, tf: str) -> bool:
        # 4H bars are aligned on BIN_ORIGIN in absolute time: in summer time they close
        # at 01/05/09.. Rome, so `hour % 4 == 0` was wrong half of the year
        return is_tf_close(ts_pd, tf)

    def _ensure_scalers_and_precompute(self):
        if self._scalers_loaded:
            return

        col_scalers = {}
        if isinstance(self.scaler_map, dict) and self.scaler_map:
            col_scalers = self.scaler_map
        elif self.scaler_path is not None:
            try:
                col_scalers = _load_all_scalers_file(self.scaler_path) or {}
            except Exception as e:
                logger.warning(f"Failed scaler load {self.scaler_path}: {e}")
                col_scalers = {}

        self.scaler_map = _normalize_scaler_map(col_scalers) if col_scalers else {}
        df = self._df

        for tf, col in self.close_col_by_tf.items():
            if col is None:
                logger.warning(
                    "[DATASET] close_col_by_tf[%s] = None — " "no trend label for this TF.",
                    tf,
                )
                continue

            raw_vals = pd.to_numeric(df[col], errors="coerce").values.astype(float)

            if self.scaler_map:
                inv = _inverse_transform_if_possible(df, col, self.scaler_map)
                inv_median = (
                    float(np.nanmedian(inv[~np.isnan(inv)])) if np.any(~np.isnan(inv)) else 0.0
                )
                if inv_median < 1e-4:
                    logger.warning(
                        "[DATASET] inverse_transform close_%s gives odd values "
                        "(median=%.2e) — scaler missing or wrong.",
                        tf,
                        inv_median,
                    )
                self.close_unscaled[col] = pd.Series(inv).ffill().bfill().values.astype(float)
                self.inverse_applied[col] = True
            else:
                logger.warning(
                    "[DATASET] close_%s: no scaler_map — thresholds not valid "
                    "for asset=%s. Pass scaler_path= to the dataset.",
                    tf,
                    getattr(self, "asset_name", "?"),
                )
                self.close_unscaled[col] = pd.Series(raw_vals).ffill().bfill().values.astype(float)
                self.inverse_applied[col] = False

        for _extra_col in ["high_15min", "low_15min"]:
            _raw_col = _extra_col + "_raw"
            if _extra_col in df.columns and _extra_col not in self.close_unscaled:
                if _raw_col in df.columns:
                    self.close_unscaled[_extra_col] = (
                        pd.to_numeric(df[_raw_col], errors="coerce")
                        .ffill()
                        .bfill()
                        .values.astype(float)
                    )
                elif self.scaler_map:
                    inv = _inverse_transform_if_possible(df, _extra_col, self.scaler_map)
                    self.close_unscaled[_extra_col] = (
                        pd.Series(inv).ffill().bfill().values.astype(float)
                    )
                else:
                    self.close_unscaled[_extra_col] = (
                        pd.to_numeric(df[_extra_col], errors="coerce")
                        .ffill()
                        .bfill()
                        .values.astype(float)
                    )

        self._next_tf_close_by_tf = {}
        for tf, col in self.close_col_by_tf.items():
            if col is None:
                continue
            ca = self.close_unscaled.get(col)
            if ca is None:
                continue
            self._next_tf_close_by_tf[tf] = self._build_next_tf_close(ca)

        _rolling_by_tf = getattr(self.config, "regime_rolling_by_tf", None) or {}
        _rolling_default = int(getattr(self.config, "regime_rolling_n", 500))
        for tf, col in self.close_col_by_tf.items():
            if col is None:
                self._regime_continuous_by_tf[tf] = (np.full(len(df), 0.5), np.full(len(df), 0.5))
                continue
            ca = self.close_unscaled.get(col)
            if ca is None:
                self._regime_continuous_by_tf[tf] = (np.full(len(df), 0.5), np.full(len(df), 0.5))
                continue
            atr_c = None
            for atr_cand in (f"atr14_{tf}", f"ATR14_{tf}", f"atr_norm_{tf}", f"ATR_norm_{tf}"):
                if atr_cand in df.columns:
                    atr_c = _inverse_transform_if_possible(df, atr_cand, self.scaler_map)
                    break
            if atr_c is None:
                atr_c = np.full(len(ca), np.nan)
            rolling_n_for_tf = int(
                _rolling_by_tf.get(
                    tf,
                    _rolling_by_tf.get(
                        tf.upper(), _rolling_by_tf.get(tf.lower(), _rolling_default)
                    ),
                )
            )
            atr_pct, vol_zc = self._build_regime_continuous_signals(ca, atr_c, rolling_n_for_tf)
            self._regime_continuous_by_tf[tf] = (atr_pct, vol_zc)

        self._true_vol_by_tf = {}
        self._regime_score_by_tf = {}
        self._regime_thresholds_by_tf = {}

        regime_low_thr = float(getattr(self.config, "regime_low_thr", 0.33))
        regime_high_thr = float(getattr(self.config, "regime_high_thr", 0.67))

        for tf, col in self.close_col_by_tf.items():
            cont = self._regime_continuous_by_tf.get(tf)
            if cont is None or col is None:
                self._true_vol_by_tf[tf] = np.full(len(df), np.nan)
                self._regime_score_by_tf[tf] = np.full(len(df), np.nan)
                self._regime_thresholds_by_tf[tf] = (regime_low_thr, regime_high_thr)
                continue

            atr_pct_15m, vol_zclip_15m = cont
            self._regime_score_by_tf[tf] = atr_pct_15m
            self._regime_thresholds_by_tf[tf] = (regime_low_thr, regime_high_thr)
            self._true_vol_by_tf[tf] = vol_zclip_15m

        # ── Cache NumPy arrays for fast _outcome_for_idx (zero Pandas at call time) ──
        _col15 = self.close_col_by_tf.get("15min")
        self._close15_np = self.close_unscaled.get(_col15) if _col15 else None

        # ATR14 raw — try all common column name variants
        self._atr_raw_np = None
        for _atr_cand in ("ATR14_15min_raw", "atr14_15min_raw", "ATR14_15min", "atr14_15min"):
            if _atr_cand in df.columns:
                _v = pd.to_numeric(df[_atr_cand], errors="coerce").values.astype(np.float64)
                # Inverse-transform if not a _raw column and scaler exists
                if not _atr_cand.endswith("_raw") and self.scaler_map:
                    _v = _inverse_transform_if_possible(df, _atr_cand, self.scaler_map)
                self._atr_raw_np = _v.astype(np.float64)
                logger.debug("[DATASET] ATR cache: %s", _atr_cand)
                break

        # high/low raw
        self._hi_raw_np = None
        self._lo_raw_np = None
        for _hc, _lc in [("high_15min_raw", "low_15min_raw"), ("high_15min", "low_15min")]:
            if _hc in df.columns and _lc in df.columns:
                _hi = df[_hc].values.astype(np.float64)
                _lo = df[_lc].values.astype(np.float64)
                if not _hc.endswith("_raw") and self.scaler_map:
                    _hi = _inverse_transform_if_possible(df, _hc, self.scaler_map)
                    _lo = _inverse_transform_if_possible(df, _lc, self.scaler_map)
                self._hi_raw_np = _hi.astype(np.float64)
                self._lo_raw_np = _lo.astype(np.float64)
                logger.debug("[DATASET] hi/lo cache: %s/%s", _hc, _lc)
                break

        # Cache per-TF ATR arrays (inverse-transformed, same pattern as close_unscaled)
        self._atr_by_tf = {}
        for _tf in self.tf_list:
            _tf_key = _tf.lower()
            _cols = self._TF_COL_MAP.get(_tf_key)
            if _cols is None:
                continue
            _atr_col = _cols.get("atr")
            if _atr_col is None:
                continue
            if _tf_key == "15min" and self._atr_raw_np is not None:
                self._atr_by_tf[_tf] = self._atr_raw_np  # already cached raw
            else:
                _arr = self._get_unscaled_col(_atr_col)
                self._atr_by_tf[_tf] = _arr
        # Cache per-TF regime-continuous arrays (inverse-transformed)
        # Four signals per TF: vol_regime, atr_norm, adx14, di_direction (v25)
        self._regime_cont_by_tf = {}
        for _tf in self.tf_list:
            _tf_key = _tf.lower()
            _cols = self._TF_COL_MAP.get(_tf_key, {})
            _tf_signals: dict[str, Optional[np.ndarray]] = {}
            for _sig in ("vol_regime", "atr_norm", "adx"):
                _col = _cols.get(_sig)
                if _col:
                    _arr = self._get_unscaled_col(_col)
                    _tf_signals[_sig] = _arr
                else:
                    _tf_signals[_sig] = None
            # DI direction: tanh((DI+ - DI-) / 20) → [-1, +1]
            # +1 = strong up trend, -1 = strong down trend, 0 = neutral
            _col_dip = _cols.get("di_plus")
            _col_dim = _cols.get("di_minus")
            if _col_dip and _col_dim:
                _dip = self._get_unscaled_col(_col_dip)
                _dim = self._get_unscaled_col(_col_dim)
                if _dip is not None and _dim is not None:
                    _di_dir = np.tanh((_dip - _dim) / 20.0)
                    _di_dir = np.where(np.isfinite(_di_dir), _di_dir, 0.0)
                    _tf_signals["di_direction"] = _di_dir
                else:
                    _tf_signals["di_direction"] = None
            else:
                _tf_signals["di_direction"] = None  # old parquets without DI+/DI-
            self._regime_cont_by_tf[_tf] = _tf_signals
        # Strategy labels AFTER _regime_cont_by_tf: they use the cached di_direction
        self._precompute_strategy_outcome_labels()
        self._precompute_mktstructure_labels()
        self._log_trend_label_distribution()

        self._scalers_loaded = True

    # Number of 15min bars in each TF candle — for semantically correct rolling windows
    _TF_BARS_15M: dict[str, int] = {
        "15min": 1,
        "30min": 2,
        "1h": 4,
        "4h": 16,
        "1d": 96,
    }
    # Column name templates per TF (matching parquet naming)
    # Column names after _normalize_df_index_and_columns (all lowercase)
    _TF_COL_MAP: dict[str, dict[str, str]] = {
        "15min": {
            "high": "high_15min",
            "low": "low_15min",
            "ema": "ema20_15min",
            "atr": "atr14_15min",
            "vol_regime": "volatility_regime_15min",
            "atr_norm": "atr_norm_15min",
            "adx": "adx14_15min",
            "di_plus": "di_plus14_15min",
            "di_minus": "di_minus14_15min",
        },
        "30min": {
            "high": "high_30min",
            "low": "low_30min",
            "ema": "ema20_30min",
            "atr": "atr14_30min",
            "vol_regime": "volatility_regime_30min",
            "atr_norm": "atr_norm_30min",
            "adx": "adx14_30min",
            "di_plus": "di_plus14_30min",
            "di_minus": "di_minus14_30min",
        },
        "1h": {
            "high": "high_1h",
            "low": "low_1h",
            "ema": "ema20_1h",
            "atr": "atr14_1h",
            "vol_regime": "volatility_regime_1h",
            "atr_norm": "atr_norm_1h",
            "adx": "adx14_1h",
            "di_plus": "di_plus14_1h",
            "di_minus": "di_minus14_1h",
        },
        "4h": {
            "high": "high_4h",
            "low": "low_4h",
            "ema": "ema20_4h",
            "atr": "atr14_4h",
            "vol_regime": "volatility_regime_4h",
            "atr_norm": "atr_norm_4h",
            "adx": "adx14_4h",
            "di_plus": "di_plus14_4h",
            "di_minus": "di_minus14_4h",
        },
        "1d": {
            "high": "high_1d",
            "low": "low_1d",
            "ema": "ema50_1d",
            "atr": "atr14_1d",
            "vol_regime": "volatility_regime_1d",
            "atr_norm": "atr_norm_1d",
            "adx": "adx14_1d",
            "di_plus": "di_plus14_1d",
            "di_minus": "di_minus14_1d",
        },
    }
    # Regime-continuous signal indices within the [n_tfs, 3] tensor
    _REGIME_CONT_IDX = {
        "vol_regime": 0,
        "atr_norm": 1,
        "adx": 2,
        "di_direction": 3,
    }  # v25: DI direction tanh((DI+−DI−)/20)

    def _get_unscaled_col(self, col: str) -> Optional[np.ndarray]:
        """Inverse-transform a scaled column, or use raw variant if present.
        Returns float64 array or None. Same logic as close_unscaled."""
        df = self._df
        if df is None or col not in df.columns:
            return None
        raw_col = col + "_raw"
        if raw_col in df.columns:
            return (
                pd.to_numeric(df[raw_col], errors="coerce")
                .ffill()
                .bfill()
                .values.astype(np.float64)
            )
        if self.scaler_map:
            inv = _inverse_transform_if_possible(df, col, self.scaler_map)
            return pd.Series(inv).ffill().bfill().values.astype(np.float64)
        return pd.to_numeric(df[col], errors="coerce").ffill().bfill().values.astype(np.float64)

    def _precompute_mktstructure_labels(self):
        """
        Precomputes the market structure label PER TF from the parquet columns.

        15min: uses _hi_raw_np / _lo_raw_np / _atr_raw_np (already cached, O(1)).
        Other TFs: inverse-transform of high/low/EMA/ATR from the scaled column.

        Rolling window in the right units: 20 candles × 15min bars per candle.
        E.g. 4H breakout = rolling_max(high_4H, 20×16=320 15min bars).

        Label per TF:
          0 = RANGING        |close - EMA| / ATR_pct < 0.5 AND no breakout
          1 = LONG_BREAKOUT  close >= rolling_max(high, 20 candles) * 0.9995
          2 = SHORT_BREAKOUT close <= rolling_min(low,  20 candles) * 1.0005
          (v25: TRENDING removed, BREAKOUT split into LONG/SHORT for direction)
                     OR close <= rolling_min(low,  20 candles)
         -1 = skip (not enough data)

        Output: self._mktstructure_by_tf — Dict[str, np.ndarray of int8]
        """
        df = self._df
        n = len(df) if df is not None else 0
        self._mktstructure_by_tf: dict[str, np.ndarray] = {}

        for tf in self.tf_list:
            # Keys in _TF_COL_MAP are lowercase (matching _normalize_df_index_and_columns)
            tf_key = tf.lower()
            if tf_key not in self._TF_COL_MAP:
                self._mktstructure_by_tf[tf] = np.full(n, -1, dtype=np.int8)
                continue

            cols = self._TF_COL_MAP[tf_key]
            bars = self._TF_BARS_15M.get(tf.lower(), 1)
            # Per-TF breakout window (candles): shorter on longer TFs to increase BREAKOUT%
            # 4H window=10 → ~160 15min bars (~2.5 days) instead of 320 (~5 days)
            _bo_candles_by_tf = self.config.mktstructure_bo_window_by_tf or {
                "15min": 20,
                "30min": 20,
                "1h": 20,
                "4h": 10,
                "1d": 10,
            }
            _bo_candles = _bo_candles_by_tf.get(tf.lower(), 20)
            win_bo = _bo_candles * bars  # rolling breakout window in 15min bars
            win_ma = 5 * bars  # min bars before MA is meaningful

            # ── Load arrays ──────────────────────────────────────────────────
            # 15min: prefer raw cached arrays
            if bars == 1 and self._close15_np is not None:
                close = self._close15_np
                hi = (
                    self._hi_raw_np
                    if self._hi_raw_np is not None
                    else self._get_unscaled_col(cols["high"])
                )
                lo = (
                    self._lo_raw_np
                    if self._lo_raw_np is not None
                    else self._get_unscaled_col(cols["low"])
                )
                atr = (
                    self._atr_raw_np
                    if self._atr_raw_np is not None
                    else self._get_unscaled_col(cols["atr"])
                )
                ema = self._get_unscaled_col(cols["ema"])
            else:
                close_col = self.close_col_by_tf.get(tf)
                close = self.close_unscaled.get(close_col) if close_col else None
                hi = self._get_unscaled_col(cols["high"])
                lo = self._get_unscaled_col(cols["low"])
                atr = self._get_unscaled_col(cols["atr"])
                ema = self._get_unscaled_col(cols["ema"])

            if close is None or len(close) < win_bo + 1:
                self._mktstructure_by_tf[tf] = np.full(n, -1, dtype=np.int8)
                continue

            # Fallback: if EMA not available from parquet (column not in _df),
            # compute rolling MA from close_unscaled — guaranteed real prices.
            if ema is None or not np.any(np.isfinite(ema)):
                _ma_win = 20 * bars
                ema = (
                    pd.Series(close.astype(np.float64))
                    .rolling(_ma_win, min_periods=max(5, _ma_win // 4))
                    .mean()
                    .values
                )
                logger.debug(
                    "[MKTSTR] %s %s: EMA column unavailable, using rolling MA%d from close",
                    getattr(self, "asset_name", "?"),
                    tf,
                    _ma_win // bars,
                )

            n_arr = len(close)
            labels = np.full(n_arr, -1, dtype=np.int8)

            # ── Rolling max high / min low — O(n × win_bo) ────────────────
            # For short TF (15min, 30min) win_bo is small (20, 40) → fast
            # For 4H win_bo=320, 1D=1920 → use pandas rolling (C extension)
            hi_arr = hi if hi is not None else close
            lo_arr = lo if lo is not None else close

            # pd.Series.rolling() uses C extension — fastest for all window sizes
            _min_p = max(4, win_bo // 4)
            # shift(1): compare the current close with max/min of the N PREVIOUS bars
            # no data leakage (the current bar can't break itself)
            roll_hi = pd.Series(hi_arr).rolling(win_bo, min_periods=_min_p).max().shift(1).values
            roll_lo = pd.Series(lo_arr).rolling(win_bo, min_periods=_min_p).min().shift(1).values

            # ── Vectorized labeling (numpy — replaces O(n) Python loop) ────────
            # Log one sample for verification before vectorizing
            _vlog_i = win_ma
            while _vlog_i < n_arr and not (
                np.isfinite(close[_vlog_i]) and np.isfinite(roll_hi[_vlog_i])
            ):
                _vlog_i += 1
            if _vlog_i < n_arr:
                _c0 = float(close[_vlog_i])
                _ea0 = (
                    float(ema[_vlog_i])
                    if ema is not None and _vlog_i < len(ema) and np.isfinite(ema[_vlog_i])
                    else -1
                )
                _at0 = (
                    float(atr[_vlog_i])
                    if atr is not None
                    and _vlog_i < len(atr)
                    and np.isfinite(atr[_vlog_i])
                    and atr[_vlog_i] > 0
                    else -1
                )
                _ap0 = (_at0 / _c0) if (_at0 > 0 and _c0 > 1e-8) else 0.001
                _d0 = abs(_c0 - _ea0) / (_ap0 * _c0 + 1e-12) if _ea0 > 0 else -1
                logger.info(
                    "[MKTSTR SAMPLE] %s %s bar=%d | close=%.6f ema=%.6f atr=%.6f "
                    "roll_hi=%.6f roll_lo=%.6f | atr_pct=%.5f dist_atr=%.3f",
                    getattr(self, "asset_name", "?"),
                    tf,
                    _vlog_i,
                    _c0,
                    _ea0,
                    _at0,
                    float(roll_hi[_vlog_i]),
                    float(roll_lo[_vlog_i]),
                    _ap0,
                    _d0,
                )

            _dist_thresh_by_tf = self.config.mktstructure_dist_thresh_by_tf or {
                "15min": 0.30,
                "30min": 0.30,
                "1h": 0.50,
                "4h": 0.70,
                "1d": 0.70,
            }
            c_arr = close.astype(np.float64)
            rh_arr = roll_hi.astype(np.float64)
            rl_arr = roll_lo.astype(np.float64)
            valid = np.isfinite(c_arr) & np.isfinite(rh_arr) & np.isfinite(rl_arr) & (c_arr > 1e-8)
            valid[:win_ma] = False

            # Breakout: close >= roll_hi * 0.9995 or close <= roll_lo * 1.0005
            # v25: BREAKOUT split into LONG_BREAKOUT (1) and SHORT_BREAKOUT (2)
            # TRENDING removed — the backbone learns the direction, not only the strength
            is_long_breakout = (c_arr >= rh_arr * 0.9995) & valid
            is_short_breakout = (c_arr <= rl_arr * 1.0005) & valid
            # on a conflict (rare) LONG wins (price > max -> more relevant)
            is_short_breakout = is_short_breakout & ~is_long_breakout

            # Classi: 0=RANGING, 1=LONG_BREAKOUT, 2=SHORT_BREAKOUT
            labels = np.where(
                ~valid, -1, np.where(is_long_breakout, 1, np.where(is_short_breakout, 2, 0))
            ).astype(np.int8)

            self._mktstructure_by_tf[tf] = labels

            # Stats log
            valid = labels[labels >= 0]
            n_v = max(1, len(valid))
            logger.info(
                "[DATASET MKTSTR] %s %s: RANGING=%.1f%% LONG_BRK=%.1f%% SHORT_BRK=%.1f%% skip=%.1f%%",
                getattr(self, "asset_name", "?"),
                tf,
                (valid == 0).sum() / n_v * 100,
                (valid == 1).sum() / n_v * 100,
                (valid == 2).sum() / n_v * 100,
                (labels == -1).sum() / max(1, n_arr) * 100,
            )

        # backward compat: keep _mktstructure_arr pointing to 15min
        _ms15 = self._mktstructure_by_tf.get("15min")
        self._mktstructure_arr = _ms15 if _ms15 is not None else np.full(n, -1, dtype=np.int8)

    def _log_trend_label_distribution(self):
        """Samples ~2000 indices and logs the DOWN/FLAT/UP distribution per TF.
        To check that ATR-adjusted gives ~25/50/25 instead of 10/80/10."""
        if self.indices is None or len(self.indices) == 0:
            return
        mode = getattr(self.config, "trend_label_mode", "atr_adjusted")
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(self.indices, size=min(2000, len(self.indices)), replace=False)
        for tf in self.tf_list:
            labels = np.array([self._trend_for_tf(int(i), tf) for i in sample_idx], dtype=np.int8)
            n = len(labels)
            n_d = int((labels == 0).sum())
            n_f = int((labels == 1).sum())
            n_u = int((labels == 2).sum())
            logger.info(
                "[DATASET TREND DIST] %s %s (mode=%s n=%d): " "DOWN=%.1f%% FLAT=%.1f%% UP=%.1f%%",
                getattr(self, "asset_name", "?"),
                tf,
                mode,
                n,
                n_d / n * 100,
                n_f / n * 100,
                n_u / n * 100,
            )

    def _precompute_strategy_outcome_labels(self):
        """
        v28: real labels computed ON THE FLY with the same logic as multi_strat_simulator.

        Uses high_15min_raw / low_15min_raw to compute:
          SESSION: Asian range (00:00-07:00 UTC) -> exact TP/SL of sess_sl_third
          PDHL:    PDH/PDL of the previous day   -> exact TP/SL of pdhl_london_di

        No external file. Correct labels = real AUC instead of ~0.48.

        Output arrays (n_rows, int8):
          _strat_hit_{combo}:  1=TP or SL touched, 0=timeout, -1=skip
          _strat_side_{combo}: 1=TP_first, 0=SL_first, -1=skip
        """
        if self._df is None or self.indices is None:
            return

        n = len(self._df)

        def _empty():
            return np.full(n, -1, dtype=np.int8)

        # ── Raw OHLC arrays (not scaled) ─────────────────────────────────────
        # uses _hi_raw_np / _lo_raw_np already cached by the dataset (real values)
        # _get_raw(col) reads self._df, which can hold scaled versions
        def _get_raw(col):
            if self._df is not None and col in self._df.columns:
                return self._df[col].values.astype(np.float64)
            return None

        # priority: the cached numpy arrays (raw, not scaled)
        hi_raw = self._hi_raw_np if self._hi_raw_np is not None else _get_raw("high_15min_raw")
        lo_raw = self._lo_raw_np if self._lo_raw_np is not None else _get_raw("low_15min_raw")
        # raw close: needed by the EXECUTABLE ENTRY gate (level still ahead of
        # price). If missing, the gate switches itself off and the labels go back
        # to the historical behaviour instead of crashing.
        cl_raw = _get_raw("close_15min_raw")
        if cl_raw is None:
            cl_raw = self._get_unscaled_col("close_15min")
        if cl_raw is None:
            import numpy as _np

            cl_raw = _np.full(len(hi_raw) if hi_raw is not None else 0, _np.nan, dtype=float)

        if hi_raw is None or lo_raw is None or len(self.indices) == 0:
            for attr in (
                "_strat_hit_session_long",
                "_strat_hit_session_short",
                "_strat_side_session_long",
                "_strat_side_session_short",
                "_strat_hit_pdhl_long",
                "_strat_hit_pdhl_short",
                "_strat_side_pdhl_long",
                "_strat_side_pdhl_short",
            ):
                setattr(self, attr, _empty())
            self._regime_quality_label = np.full(n, -1, dtype=np.int8)
            logger.warning("[DATASET] high/low_15min_raw not found — v28 labels skipped")
            return

        cfg = self.config
        _majors_set = {"EURUSD", "GBPUSD", "USDCAD", "USDCHF", "AUDUSD"}
        _asset_upper = str(getattr(self, "asset_name", "")).upper()
        _is_major = _asset_upper in _majors_set

        # ── Strategy parameters (same as multi_strat_simulator) ───────────────
        # SL = range/3 (same as the simulator: sl_type="range_third")
        # TP = tp_r × (range/3) to get the wanted RR
        # RR = tp_r / sl_r = tp_r (with sl_r=1.0)
        sess_tp_r = (
            float(getattr(cfg, "outcome_session_tp_r_majors", 2.5))
            if _is_major
            else float(getattr(cfg, "outcome_session_tp_r_minors", 1.8))
        )
        # SL = sl_r x (range/3), TP = tp_r x (range/3): RR = tp_r / sl_r
        # (sl_r = 1.0 by default = the simulator's sl_type="range_third")
        sess_sl_r = float(getattr(cfg, "outcome_session_sl_r", 1.0))
        pdhl_sl_r = float(getattr(cfg, "outcome_pdhl_sl_r", 1.0))
        h_sess = int(getattr(cfg, "strategy_session_horizon_bars", 32))

        pdhl_tp_r = float(getattr(cfg, "outcome_pdhl_tp_r", 1.8))
        h_pdhl = int(getattr(cfg, "strategy_pdhl_horizon_bars", 64))

        logger.debug(
            "[DATASET] %s (%s): sess_tp=%.1f× pdhl_tp=%.1f×",
            _asset_upper,
            "Major" if _is_major else "Minor",
            sess_tp_r,
            pdhl_tp_r,
        )

        # ── Bar OPEN time in UTC: sessions, Asian range and days by the time a bar
        #    trades in (rows are labelled at bar close) ────────────────────────
        try:
            _ts_idx = bar_open_utc(self._df.index)
            _hours = np.array(_ts_idx.hour, dtype=np.int32)
            _mins = np.array(_ts_idx.minute, dtype=np.int32)
            _dates = np.array([t.date() for t in _ts_idx])
            _hm = _hours * 60 + _mins
            # rows followed by a market break (weekend, holiday): more than an
            # hour between this bar and the next one
            _step_min = np.diff(_ts_idx.asi8) / 6e10
            _break_after = np.append(_step_min > 60, True)
        except Exception as _te:
            logger.warning("[DATASET] timestamp parse failed: %s — label v28 skip", _te)
            for attr in (
                "_strat_hit_session_long",
                "_strat_hit_session_short",
                "_strat_side_session_long",
                "_strat_side_session_short",
                "_strat_hit_pdhl_long",
                "_strat_hit_pdhl_short",
                "_strat_side_pdhl_long",
                "_strat_side_pdhl_short",
            ):
                setattr(self, attr, _empty())
            self._regime_quality_label = np.full(n, -1, dtype=np.int8)
            return

        # ── Timestamp diagnostics (once per asset) ───────────────────────────
        _n_bars = len(_hm)
        _asian_count = int(((_hm >= 0) & (_hm < 7 * 60)).sum())
        _london_count = int(((_hm >= 8 * 60) & (_hm < 12 * 60)).sum())
        _pdhl_count = int(((_hm >= 13 * 60) & (_hm < 20 * 60)).sum())
        _sample_hours = _hours[:5].tolist() if _n_bars >= 5 else _hours.tolist()
        logger.info(
            "[DATASET] %s timestamp UTC: first_hours=%s | asian_bars=%d london_bars=%d pdhl_bars=%d (out of %d)",
            _asset_upper,
            _sample_hours,
            _asian_count,
            _london_count,
            _pdhl_count,
            _n_bars,
        )
        if _asian_count == 0:
            logger.warning(
                "[DATASET] %s: ZERO Asian bars (00-07 UTC)! "
                "Probably a timezone mismatch. Hour range: %d-%d",
                _asset_upper,
                int(_hours.min()),
                int(_hours.max()),
            )

        # ── Maschere finestre orarie ──────────────────────────────────────────
        _asian_mask = (_hm >= 0) & (_hm < 7 * 60)  # 00:00-06:59 UTC (Asian range)

        # ── Asian range and PDH/PDL per date, precomputed ─────────────────────
        from collections import defaultdict

        _bars_by_date = defaultdict(list)
        for _i in range(n):
            _bars_by_date[_dates[_i]].append(_i)

        _asian_by_date = {}  # date → (asian_high, asian_low, range)
        _pdh_by_date = {}  # date → (pdh, pdl, day_range)  [PREVIOUS day]

        _strict = bool(getattr(cfg, "outcome_strict_execution", True))
        _strict_fill = _strict and bool(np.isfinite(cl_raw).any())
        if _strict and not _strict_fill:
            logger.warning(
                "[DATASET] %s: no real close, stop fills checked on the level only",
                str(getattr(self, "asset_name", "")).upper(),
            )
        # last row before the next market break, for every row
        _next_break = np.empty(n, dtype=np.int64)
        _nb = n - 1
        for _i in range(n - 1, -1, -1):
            if _break_after[_i]:
                _nb = _i
            _next_break[_i] = _nb

        _sorted_dates = sorted(_bars_by_date.keys())
        # previous trading day for PDH/PDL. Strict: skip Sundays, whose bars are
        # only the evening reopening stub — Monday's previous day is Friday.
        _prev_day = previous_trading_days(_sorted_dates, skip_sunday=_strict)
        for _i, _d in enumerate(_sorted_dates):
            _all_bars = _bars_by_date[_d]
            _asian_bars = [_b for _b in _all_bars if _asian_mask[_b]]
            if _asian_bars:
                _ah = float(np.nanmax(hi_raw[_asian_bars]))
                _al = float(np.nanmin(lo_raw[_asian_bars]))
                _ar = _ah - _al
                if _ar > 0 and np.isfinite(_ah) and np.isfinite(_al):
                    _asian_by_date[_d] = (_ah, _al, _ar)

            _prev_d = _prev_day.get(_d)
            if _prev_d is not None:
                _prev_bars = _bars_by_date[_prev_d]
                if _prev_bars:
                    _pdh = float(np.nanmax(hi_raw[_prev_bars]))
                    _pdl = float(np.nanmin(lo_raw[_prev_bars]))
                    _dr = _pdh - _pdl
                    if _dr > 0 and np.isfinite(_pdh) and np.isfinite(_pdl):
                        _pdh_by_date[_d] = (_pdh, _pdl, _dr)

        # ── DI direction 4H (for the PDHL gate) ──────────────────────────────
        # Real (inverse-transformed) DI, as the simulator's di_dir_4h = tanh((DI+ - DI-)/20).
        # Before, the columns were looked up as "DI_plus14_4H" in a lower-cased frame:
        # never found, so the gate was silently always open.
        _di_dir = None
        if bool(getattr(cfg, "outcome_pdhl_di_gate", True)):
            _dip = self._get_unscaled_col("di_plus14_4h")
            _dim = self._get_unscaled_col("di_minus14_4h")
            if _dip is not None and _dim is not None:
                _di_dir = np.tanh((_dip - _dim) / 20.0)
            else:
                logger.warning(
                    "[DATASET] %s: DI+/DI- 4H missing, PDHL labels without the DI gate",
                    _asset_upper,
                )
        _has_di = _di_dir is not None
        _DI_THRESH = float(getattr(cfg, "outcome_pdhl_di_thresh", 0.15))

        # ── Label arrays ──────────────────────────────────────────────────────
        hit_sl = _empty()
        side_sl = _empty()  # session long
        hit_ss = _empty()
        side_ss = _empty()  # session short
        hit_pl = _empty()
        side_pl = _empty()  # pdhl long
        hit_ps = _empty()
        side_ps = _empty()  # pdhl short
        qual = np.full(n, -1, dtype=np.int8)

        # ── v29: be_reachable arrays (one per strategy/direction) ─────────────
        be_sl = _empty()
        be_ss = _empty()
        be_pl = _empty()
        be_ps = _empty()

        # Outcome confidence per head: how clean the result is.
        # float16 to keep memory down (one array per head over millions
        # of bars); 0 = no outcome, 1 = unambiguous outcome.
        def _cf():
            return np.zeros(n, dtype=np.float16)

        cnf_sl = _cf()
        cnf_ss = _cf()
        cnf_pl = _cf()
        cnf_ps = _cf()
        # closure state: _be_last[0] = last be_reachable computed by _outcome;
        # _be_rr[0] = rr of the current strategy (set before each group).
        _be_last = [np.int8(0)]
        _be_rr = [0.0]
        # confidence of the last computed outcome (see _conf in _outcome):
        # same trick as _be_last, so the signature of _outcome (called
        # from several places) doesn't change.
        _conf_last = [1.0]

        # ── Outcome helper ────────────────────────────────────────────────────
        # ── v29: BE parameters (read once) ────────────────────────────────────
        _be_on = bool(getattr(cfg, "outcome_be_enabled", False))
        _be_trig_r = float(getattr(cfg, "outcome_be_trigger_r", 0.25))
        _be_off_r = float(getattr(cfg, "outcome_be_offset_r", 0.07))

        def _outcome(e_idx, h_eff, tp, sl, direction, h_req=None):
            """(hit, side) of one trade; be_reachable and confidence go to
            _be_last / _conf_last so the call sites stay unchanged."""
            hit, side, be_reach, conf = simulate_trade(
                hi_raw,
                lo_raw,
                e_idx,
                h_eff,
                tp,
                sl,
                direction,
                strict=_strict,
                last_row=int(_next_break[e_idx]),
                h_req=h_req,
                be_rr=_be_rr[0] if _be_on else 0.0,
                be_trigger_r=_be_trig_r,
                be_offset_r=_be_off_r,
            )
            _be_last[0] = be_reach
            _conf_last[0] = conf
            return hit, side

        # ── ATR per day: uses _hi_raw_np/_lo_raw_np (real raw values) ─────────
        # hi_raw and lo_raw already use self._hi_raw_np when available
        # fallback: hi-lo range of every Asian bar -> mean = ATR proxy
        _atr_by_date = {}
        if hi_raw is not None and lo_raw is not None:
            for _d, _bars in _bars_by_date.items():
                _asian_bars_d = [_b for _b in _bars if _asian_mask[_b]]
                # strict: Asian bars only — the whole day would use bars that
                # come after the entry
                _candidate_bars = _asian_bars_d if (_asian_bars_d or _strict) else _bars
                if _candidate_bars:
                    _tr = hi_raw[_candidate_bars] - lo_raw[_candidate_bars]
                    _tr = _tr[(_tr > 0) & np.isfinite(_tr)]
                    if len(_tr) > 0:
                        _atr_by_date[_d] = float(np.mean(_tr))

        logger.info(
            "[DATASET] %s ATR by date: %d/%d days filled (hi-lo proxy)",
            _asset_upper,
            len(_atr_by_date),
            len(_bars_by_date),
        )
        _use_bar_atr = len(_atr_by_date) == 0

        # ── London direction for PDHL, precomputed (same as the simulator) ────
        # London (08:00-13:00 UTC) must break the Asian range the right way
        _london_mask = (_hm >= 8 * 60) & (_hm < 13 * 60)
        _london_dir = {}  # date → +1 up / -1 down / 0 no break
        for _d, _bars in _bars_by_date.items():
            _as_bars = [_b for _b in _bars if _asian_mask[_b]]
            _lo_bars = [_b for _b in _bars if _london_mask[_b]]
            if _as_bars and _lo_bars:
                _a_h = float(np.nanmax(hi_raw[_as_bars]))
                _a_l = float(np.nanmin(lo_raw[_as_bars]))
                _l_h = float(np.nanmax(hi_raw[_lo_bars]))
                _l_l = float(np.nanmin(lo_raw[_lo_bars]))
                if _l_h > _a_h:
                    _london_dir[_d] = +1
                elif _l_l < _a_l:
                    _london_dir[_d] = -1
                else:
                    _london_dir[_d] = 0
            else:
                _london_dir[_d] = 0

        # ── Session masks (same as the simulator) ─────────────────────────────
        # SESSION: London open 08:00-12:00 UTC (london_start=8, london_end=12)
        _sess_mask_sim = (_hm >= 8 * 60) & (_hm < 12 * 60)
        # PDHL: entry 13:00-20:00 UTC (entry_start=13, entry_end=20)
        _pdhl_mask_sim = (_hm >= 13 * 60) & (_hm < 20 * 60)

        # ── SESSION: one label per day per direction ──────────────────────────
        # the simulator takes the FIRST breakout bar and breaks
        # -> label only the first useful bar, not all of them
        _sess_traded_long = set()  # dates already labelled LONG session
        _sess_traded_short = set()  # dates already labelled SHORT session
        # ── PDHL: one label per day per direction ─────────────────────────────
        _pdhl_traded_long = set()
        _pdhl_traded_short = set()
        # cumulative EXECUTABLE ENTRY: dates where the level has already been AHEAD
        # of price inside the window. One previous bar is enough — there the
        # pending order could be armed and the next break fills it. With the
        # instantaneous check the "breaks and closes beyond" bars were lost, and
        # they're valid fills: samples thrown away, and samples matter for the encoder.
        _sess_davanti_long = set()
        _sess_davanti_short = set()
        _pdhl_davanti_long = set()
        _pdhl_davanti_short = set()

        # ── Diagnostic counters ──────────────────────────────────────────────
        _dbg_sess_checked = 0  # SESSION bars in the loop
        _dbg_sess_no_ar = 0  # no Asian range for that date
        _dbg_sess_no_atr = 0  # ATR zero/missing
        _dbg_sess_no_brk = 0  # high/low don't go past asian_high/low
        _dbg_sess_labeled = 0  # actually labelled
        _dbg_pdhl_checked = 0
        _dbg_pdhl_no_ldir = 0  # london_dir == 0
        _dbg_pdhl_labeled = 0

        def _stop_filled(i, entry, direction):
            return stop_order_filled(hi_raw, lo_raw, cl_raw, _break_after, i, entry, direction)

        # ── Main loop, in chronological order ─────────────────────────────────
        for end_idx in sorted(self.indices):
            remaining = n - end_idx - 1
            if remaining < 4 or end_idx >= n:
                continue

            _date = _dates[end_idx]
            h_eff_sess = min(h_sess, remaining)
            h_eff_pdhl = min(h_pdhl, remaining)

            # ── SESSION labels ─────────────────────────────────────────────────
            # Same logic as session_breakout_signals:
            #   entry_px = asian_high/low ± 0.05×ATR
            #   TP = entry_px ± tp_mult × range
            #   SL = entry_px ∓ range/3
            #   Filter: min_range_atr=0.5, max_range_atr=6.0
            #   One entry per day per direction (first breakout bar)
            if _sess_mask_sim[end_idx] and h_eff_sess >= 8:
                _dbg_sess_checked += 1
                _ar = _asian_by_date.get(_date)
                if _ar is None:
                    _dbg_sess_no_ar += 1
                else:
                    _ah, _al, _asian_range = _ar
                    # ATR: daily mean first, then hi-lo of the current bar
                    _atr_i = _atr_by_date.get(_date, 0.0)
                    if not (_atr_i > 0) and end_idx < len(hi_raw) and end_idx < len(lo_raw):
                        _hl = float(hi_raw[end_idx]) - float(lo_raw[end_idx])
                        if _hl > 0:
                            _atr_i = _hl
                    if not (_atr_i > 0):
                        _dbg_sess_no_atr += 1
                    else:
                        # no range filter: the labels want every context
                        # EXECUTABLE ENTRY: the level must still be AHEAD of
                        # price (close below _ah), otherwise live the pending
                        # order can't be placed and the label would describe
                        # an impossible entry.
                        # armable: low below asian_high (long) / high above
                        # asian_low (short), same logic as the simulator. The
                        # close check threw away the breaks closing beyond (the best ones).
                        if float(lo_raw[end_idx]) < _ah:
                            _sess_davanti_long.add(_date)
                        if float(hi_raw[end_idx]) > _al:
                            _sess_davanti_short.add(_date)
                        _ep_l = _ah + 0.05 * _atr_i
                        _ep_s = _al - 0.05 * _atr_i
                        if _strict_fill:
                            _long_ok = _date not in _sess_traded_long and _stop_filled(
                                end_idx, _ep_l, +1
                            )
                            _short_ok = _date not in _sess_traded_short and _stop_filled(
                                end_idx, _ep_s, -1
                            )
                        else:
                            _long_ok = (
                                _date not in _sess_traded_long
                                and float(hi_raw[end_idx]) > _ah
                                and _date in _sess_davanti_long
                            )
                            _short_ok = (
                                _date not in _sess_traded_short
                                and float(lo_raw[end_idx]) < _al
                                and _date in _sess_davanti_short
                            )
                        if not _long_ok and not _short_ok:
                            _dbg_sess_no_brk += 1
                        if _long_ok:
                            _be_rr[0] = sess_tp_r / sess_sl_r
                            hit_sl[end_idx], side_sl[end_idx] = _outcome(
                                end_idx,
                                h_eff_sess,
                                tp=_ep_l
                                + sess_tp_r * (_asian_range / 3.0),  # RR-based TP (range/3 unit)
                                sl=_ep_l - sess_sl_r * _asian_range / 3.0,  # range_third SL
                                direction=+1,
                                h_req=h_sess,
                            )
                            be_sl[end_idx] = _be_last[0]
                            cnf_sl[end_idx] = _conf_last[0]
                            _sess_traded_long.add(_date)
                            _dbg_sess_labeled += 1
                        if _short_ok:
                            _be_rr[0] = sess_tp_r / sess_sl_r
                            hit_ss[end_idx], side_ss[end_idx] = _outcome(
                                end_idx,
                                h_eff_sess,
                                tp=_ep_s - sess_tp_r * (_asian_range / 3.0),  # RR-based TP
                                sl=_ep_s + sess_sl_r * _asian_range / 3.0,  # range_third SL
                                direction=-1,
                                h_req=h_sess,
                            )
                            be_ss[end_idx] = _be_last[0]
                            cnf_ss[end_idx] = _conf_last[0]
                            _sess_traded_short.add(_date)
                            _dbg_sess_labeled += 1

            # ── PDHL labels ────────────────────────────────────────────────────
            # Same logic as pdhl_london_confluence_signals:
            #   entry window: 13:00-20:00 UTC
            #   PDH/PDL = high/low of the PREVIOUS day
            #   London confluence: London (08-13) must have broken the Asian range
            #   entry_px = pdh/pdl ± 0.05×ATR
            #   TP = entry_px ± tp_mult × day_range
            #   SL = entry_px ∓ day_range/3
            #   Range filter: min_range_atr=0.5, max_range_atr=25.0
            #   DI gate: di_dir_4h >= 0.15 (LONG) or <= -0.15 (SHORT)
            #   One entry per day per direction
            if _pdhl_mask_sim[end_idx] and h_eff_pdhl >= 12:
                _dbg_pdhl_checked += 1
                _pd = _pdh_by_date.get(_date)
                _l_dir = _london_dir.get(_date, 0)
                # current bar ATR: daily mean first, then hi-lo of the bar
                _atr_i_p = _atr_by_date.get(_date, 0.0)
                if not (_atr_i_p > 0) and end_idx < len(hi_raw) and end_idx < len(lo_raw):
                    _hl = float(hi_raw[end_idx]) - float(lo_raw[end_idx])
                    if _hl > 0:
                        _atr_i_p = _hl
                if _l_dir == 0:
                    _dbg_pdhl_no_ldir += 1

                if _pd is not None and _l_dir != 0 and _atr_i_p > 0:
                    _pdh_v, _pdl_v, _day_range = _pd
                    # DI known when the pending order is placed: the row before the
                    # signal (its 4H value = bars closed by the signal bar's open)
                    _di_dir_val = 0.0
                    if _has_di and end_idx > 0:
                        _di_dir_val = float(_di_dir[end_idx - 1])
                        if not np.isfinite(_di_dir_val):
                            _di_dir_val = 0.0

                    # LONG: PDH breakout + London UP + DI bullish
                    # EXECUTABLE ENTRY (see sessions): without it we
                    # labelled bars where PDH had been broken for hours, where
                    # the entry at the theoretical level gives a 20-25 pip
                    # edge that doesn't exist live. It caused the collapse of
                    # pdhl with real entries (WR 73% -> 17%).
                    if float(lo_raw[end_idx]) < _pdh_v:
                        _pdhl_davanti_long.add(_date)
                    if float(hi_raw[end_idx]) > _pdl_v:
                        _pdhl_davanti_short.add(_date)
                    _ep_l = _pdh_v + 0.05 * _atr_i_p
                    _ep_s = _pdl_v - 0.05 * _atr_i_p
                    if _strict_fill:
                        _brk_l = _stop_filled(end_idx, _ep_l, +1)
                        _brk_s = _stop_filled(end_idx, _ep_s, -1)
                    else:
                        _brk_l = float(hi_raw[end_idx]) > _pdh_v and _date in _pdhl_davanti_long
                        _brk_s = float(lo_raw[end_idx]) < _pdl_v and _date in _pdhl_davanti_short
                    if (
                        _l_dir == 1
                        and _date not in _pdhl_traded_long
                        and _brk_l
                        and (not _has_di or _di_dir_val > _DI_THRESH)
                    ):
                        _be_rr[0] = pdhl_tp_r / pdhl_sl_r
                        hit_pl[end_idx], side_pl[end_idx] = _outcome(
                            end_idx,
                            h_eff_pdhl,
                            tp=_ep_l + pdhl_tp_r * (_day_range / 3.0),  # RR-based TP
                            sl=_ep_l - pdhl_sl_r * _day_range / 3.0,
                            direction=+1,
                            h_req=h_pdhl,
                        )
                        be_pl[end_idx] = _be_last[0]
                        cnf_pl[end_idx] = _conf_last[0]
                        _pdhl_traded_long.add(_date)
                        _dbg_pdhl_labeled += 1

                    # SHORT: PDL breakout + London DOWN + DI bearish
                    elif (
                        _l_dir == -1
                        and _date not in _pdhl_traded_short
                        and _brk_s
                        and (not _has_di or _di_dir_val < -_DI_THRESH)
                    ):
                        _be_rr[0] = pdhl_tp_r / pdhl_sl_r
                        hit_ps[end_idx], side_ps[end_idx] = _outcome(
                            end_idx,
                            h_eff_pdhl,
                            tp=_ep_s - pdhl_tp_r * (_day_range / 3.0),
                            sl=_ep_s + pdhl_sl_r * _day_range / 3.0,
                            direction=-1,
                            h_req=h_pdhl,
                        )
                        be_ps[end_idx] = _be_last[0]
                        cnf_ps[end_idx] = _conf_last[0]
                        _pdhl_traded_short.add(_date)
                        _dbg_pdhl_labeled += 1

            # ── REGIME QUALITY ─────────────────────────────────────────────────
            _valid_sides = []
            for _arr in (side_sl, side_ss, side_pl, side_ps):
                _v = int(_arr[end_idx])
                if _v >= 0:
                    _valid_sides.append(_v)
            if _valid_sides:
                qual[end_idx] = int(any(v == 1 for v in _valid_sides))

        # ── Log diagnostici ──────────────────────────────────────────────────
        logger.info(
            "[DATASET] %s SESSION diagnostics: checked=%d no_ar=%d no_atr=%d "
            "no_brk=%d labeled=%d",
            _asset_upper,
            _dbg_sess_checked,
            _dbg_sess_no_ar,
            _dbg_sess_no_atr,
            _dbg_sess_no_brk,
            _dbg_sess_labeled,
        )
        logger.info(
            "[DATASET] %s PDHL diagnostics: checked=%d no_ldir=%d labeled=%d",
            _asset_upper,
            _dbg_pdhl_checked,
            _dbg_pdhl_no_ldir,
            _dbg_pdhl_labeled,
        )
        # sample of the ATR/range ratio, to see if the filter is too tight
        logger.info(
            "[DATASET] %s _atr_by_date populated=%d dates, _asian_by_date=%d dates",
            _asset_upper,
            len(_atr_by_date),
            len(_asian_by_date),
        )
        if _atr_by_date and _asian_by_date:
            _sample_dates = sorted(_asian_by_date.keys())[:5]
            for _sd in _sample_dates:
                _sdr = _asian_by_date.get(_sd)
                _sat = _atr_by_date.get(_sd)
                if _sdr and _sat and _sat > 0:
                    logger.info(
                        "[DATASET] %s sample %s: asian_range=%.5f atr=%.5f ratio=%.1f (ok if 0.5-6.0)",
                        _asset_upper,
                        _sd,
                        _sdr[2],
                        _sat,
                        _sdr[2] / _sat,
                    )
                elif _sdr and not _sat:
                    logger.info(
                        "[DATASET] %s sample %s: asian_range=%.5f atr=MISSING",
                        _asset_upper,
                        _sd,
                        _sdr[2],
                    )

        # v28
        self._strat_hit_session_long = hit_sl
        self._strat_side_session_long = side_sl
        self._strat_hit_session_short = hit_ss
        self._strat_side_session_short = side_ss
        self._strat_hit_pdhl_long = hit_pl
        self._strat_side_pdhl_long = side_pl
        self._strat_hit_pdhl_short = hit_ps
        self._strat_side_pdhl_short = side_ps
        self._regime_quality_label = qual
        # v29: be_reachable arrays
        self._strat_be_session_long = be_sl
        self._strat_be_session_short = be_ss
        self._strat_be_pdhl_long = be_pl
        self._strat_be_pdhl_short = be_ps
        # confidence per head: weight of the sample in the strategy loss
        self._strat_cnf_session_long = cnf_sl
        self._strat_cnf_session_short = cnf_ss
        self._strat_cnf_pdhl_long = cnf_pl
        self._strat_cnf_pdhl_short = cnf_ps

        def _pct(arr, v):
            valid = arr[arr >= 0]
            return (valid == v).sum() / max(1, len(valid)) * 100

        def _side_stats(hit_arr, side_arr):
            hit_mask = hit_arr == 1
            side_valid = side_arr[hit_mask]
            tp_pct = (side_valid == 1).sum() / max(1, (side_valid >= 0).sum()) * 100.0
            skip_pct = (side_valid == -1).sum() / max(1, side_valid.size) * 100.0
            return tp_pct, skip_pct

        # v28: log session, pdhl, quality stats
        _tp_sl, _skip_sl = _side_stats(hit_sl, side_sl)
        _tp_ss, _skip_ss = _side_stats(hit_ss, side_ss)
        _tp_pl, _skip_pl = _side_stats(hit_pl, side_pl)
        _tp_ps, _skip_ps = _side_stats(hit_ps, side_ps)

        # count the bars actually labelled (not -1)
        _n_sess = int((hit_sl >= 0).sum() + (hit_ss >= 0).sum())
        _n_pdhl = int((hit_pl >= 0).sum() + (hit_ps >= 0).sum())

        # ── Label statistics, for debugging ──────────────────────────────────
        def _lbl_stats(name, hit_arr, side_arr):
            valid_side = side_arr[side_arr >= 0]
            tp_rate = 100.0 * ((valid_side == 1).sum() / max(1, len(valid_side)))
            logger.info(
                "[LABEL STATS] %s | side_valid=%d tp_rate=%.1f%%", name, len(valid_side), tp_rate
            )

        _lbl_stats("session_long", hit_sl, side_sl)
        _lbl_stats("session_short", hit_ss, side_ss)
        _lbl_stats("pdhl_long", hit_pl, side_pl)
        _lbl_stats("pdhl_short", hit_ps, side_ps)
        qual_valid = qual[qual >= 0]
        qual_pos_rate = 100.0 * ((qual_valid == 1).sum() / max(1, len(qual_valid)))
        logger.info(
            "[LABEL STATS] tradability | valid=%d pos_rate=%.1f%%", len(qual_valid), qual_pos_rate
        )

        logger.info(
            "[DATASET] v28 labels (%s) — "
            "labelled bars: sess=%d pdhl=%d (out of %d) | "
            "SESS_L TP_rate=%.1f%% skip=%.1f%% | "
            "SESS_S TP_rate=%.1f%% skip=%.1f%% | "
            "PDHL_L TP_rate=%.1f%% skip=%.1f%% | "
            "PDHL_S TP_rate=%.1f%% skip=%.1f%% | "
            "QUAL pos=%.1f%%",
            getattr(self, "asset_name", "?"),
            _n_sess,
            _n_pdhl,
            n,
            _tp_sl,
            _skip_sl,
            _tp_ss,
            _skip_ss,
            _tp_pl,
            _skip_pl,
            _tp_ps,
            _skip_ps,
            _pct(qual, 1),
        )

    @staticmethod
    def _build_regime_continuous_signals(
        close_arr: np.ndarray,
        atr_arr: np.ndarray,
        rolling_n: int = 500,
    ) -> "tuple[np.ndarray, np.ndarray]":
        n = len(close_arr)
        vals = close_arr.astype(np.float64)

        diff_mask = np.zeros(n, dtype=bool)
        diff_mask[1:] = (~np.isnan(vals[1:])) & (~np.isnan(vals[:-1])) & (vals[1:] != vals[:-1])
        change_idx = np.where(diff_mask)[0]
        if len(change_idx) < 2:
            change_idx = np.arange(1, n)

        n_c = len(change_idx)
        candle_closes = vals[change_idx]

        lr = np.full(n_c, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            prev = np.concatenate(([vals[0]], candle_closes[:-1]))
            mask = (candle_closes > 0) & (prev > 0)
            lr[mask] = np.log(candle_closes[mask]) - np.log(prev[mask])

        atr_c = atr_arr[change_idx].astype(np.float64)

        lr_s = pd.Series(lr)
        vol_s = lr_s.rolling(window=rolling_n, min_periods=2).std()
        vol_c = vol_s.values

        atr_s = pd.Series(atr_c)
        # Expanding rank (Cython) → percentile rank in [0,1], O(n log n)
        # For large rolling_n this is nearly equivalent and 100x faster than apply(lambda)
        atr_pct = atr_s.expanding(min_periods=2).rank(pct=True).fillna(0.5).values

        vol_ps = pd.Series(vol_c)
        roll = vol_ps.rolling(window=rolling_n, min_periods=4)
        mu_r = roll.mean()
        sd_r = roll.std()
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (vol_ps - mu_r) / sd_r.replace(0, np.nan)
        vol_zclip = np.clip(z.values, -2.0, 2.0) / 4.0 + 0.5
        vol_zclip = np.where(np.isnan(vol_zclip), 0.5, vol_zclip)

        atr_pct_15m = np.full(n, 0.5)
        vol_zclip_15m = np.full(n, 0.5)
        atr_pct_15m[: change_idx[0] + 1] = atr_pct[0]
        vol_zclip_15m[: change_idx[0] + 1] = vol_zclip[0]
        for ci in range(n_c):
            i_from = change_idx[ci]
            i_to = change_idx[ci + 1] if ci + 1 < n_c else n
            atr_pct_15m[i_from:i_to] = atr_pct[ci]
            vol_zclip_15m[i_from:i_to] = vol_zclip[ci]

        return atr_pct_15m, vol_zclip_15m

    @staticmethod
    def _build_next_tf_close(close_arr: np.ndarray) -> np.ndarray:
        n = len(close_arr)
        next_close = np.full(n, np.nan, dtype=np.float64)
        if n < 2:
            return next_close

        vals = close_arr.astype(np.float64)

        # Vectorized numpy — no Python loop
        _v = vals
        diff_mask = np.zeros(n, dtype=bool)
        diff_mask[1:] = (~np.isnan(_v[1:])) & (~np.isnan(_v[:-1])) & (_v[1:] != _v[:-1])
        change_idx = np.where(diff_mask)[0]

        if len(change_idx) == 0:
            return next_close

        insert_pos = np.searchsorted(change_idx, np.arange(n) + 1, side="left")
        valid_mask = insert_pos < len(change_idx)
        next_close[valid_mask] = vals[change_idx[insert_pos[valid_mask]]]

        return next_close

    def _trend_for_tf(self, end_idx: int, tf: str) -> int:
        col = self.close_col_by_tf.get(tf, None)
        if col is None:
            return 1
        close_arr = self.close_unscaled.get(col, None)
        if close_arr is None:
            close_arr = pd.to_numeric(self._df[col], errors="coerce").values.astype(float)

        cur = close_arr[end_idx] if end_idx < len(close_arr) else np.nan

        next_arr = self._next_tf_close_by_tf.get(tf)
        if next_arr is not None and end_idx < len(next_arr):
            fut = next_arr[end_idx]
        else:
            h = int(self.trend_horizons.get(tf, 1))
            fut = close_arr[end_idx + h] if end_idx + h < len(close_arr) else np.nan

        if np.isnan(cur) or np.isnan(fut):
            return 1

        ret = (fut - cur) / max(abs(cur), 1e-12)

        # ── Provided quantile thresholds (from compute_trend_thresholds_for_asset) ──
        if (
            tf in self._provided_trend_thresholds
            and self._provided_trend_thresholds[tf] is not None
        ):
            try:
                q_low, q_high = self._provided_trend_thresholds[tf]
                q_low = float(q_low)
                q_high = float(q_high)
                if ret <= q_low:
                    return 0
                elif ret >= q_high:
                    return 2
                else:
                    return 1
            except Exception:
                pass

        # ── ATR-adjusted mode (default) ─────────────────────────────────────────────
        # Normalises return by current volatility → consistent threshold across regimes.
        # A move of 0.4 ATR is always "significant" regardless of vol level.
        # Distribution target: ~25/50/25 instead of 10/80/10 with absolute threshold.
        if getattr(self.config, "trend_label_mode", "atr_adjusted") == "atr_adjusted":
            atr = self._get_atr_at(end_idx)
            if atr is not None and atr > 0 and np.isfinite(atr):
                atr_pct = atr / max(abs(cur), 1e-12)  # ATR as % of price
                # TF-specific multiplier: longer TF → naturally larger move vs 15m ATR
                _tf_mult = float(
                    self.config.trend_atr_mult_tf.get(
                        tf, self.config.trend_atr_mult_tf.get(tf.lower(), 1.0)
                    )
                )
                _base = float(getattr(self.config, "trend_atr_mult", 0.4))
                thresh = _base * _tf_mult * atr_pct
                if ret > thresh:
                    return 2
                elif ret < -thresh:
                    return 0
                else:
                    return 1
            # Fall through to absolute if ATR not available

        # ── Absolute mode (legacy fallback) ─────────────────────────────────────────
        scaled_thresh = float(
            self.config.trend_threshold_scaled_tf.get(tf, self.config.trend_threshold_scaled)
        )
        if ret > scaled_thresh:
            return 2
        elif ret < -scaled_thresh:
            return 0
        else:
            return 1

    def _trend_momentum_for_tf(self, end_idx: int, tf: str) -> float:
        """
        Returns log_return_future / ATR_current, clipped to [-3, 3].
        Skip = -999.0 (sentinel, left out of the loss).

        Same horizon as _trend_for_tf (trend_horizons[tf]).
        The ATR normalization makes TFs and volatility regimes comparable.
        """
        _SKIP = -999.0
        col = self.close_col_by_tf.get(tf)
        if col is None:
            return _SKIP
        close_arr = self.close_unscaled.get(col)
        if close_arr is None:
            return _SKIP

        cur = close_arr[end_idx] if end_idx < len(close_arr) else np.nan
        next_arr = self._next_tf_close_by_tf.get(tf)
        if next_arr is not None and end_idx < len(next_arr):
            fut = next_arr[end_idx]
        else:
            h = int(self.trend_horizons.get(tf, 1))
            fut = close_arr[end_idx + h] if end_idx + h < len(close_arr) else np.nan

        if not (np.isfinite(cur) and np.isfinite(fut) and cur > 1e-8):
            return _SKIP

        log_ret = float(np.log(fut / cur))

        # ATR in % of price — use native TF ATR (inverse-transformed from parquet).
        # Falls back to 15min ATR × sqrt(bars) if TF ATR unavailable.
        _tf_atr_arr = self._atr_by_tf.get(tf)
        if (
            _tf_atr_arr is not None
            and end_idx < len(_tf_atr_arr)
            and np.isfinite(_tf_atr_arr[end_idx])
            and _tf_atr_arr[end_idx] > 0
        ):
            atr_raw = float(_tf_atr_arr[end_idx])
        else:
            # fallback: 15min ATR × sqrt(bars) — Brownian motion approximation
            atr_raw = self._get_atr_at(end_idx)
            if atr_raw is None or atr_raw <= 0 or not np.isfinite(atr_raw):
                return _SKIP
            bars = self._TF_BARS_15M.get(tf.lower(), 1)
            atr_raw = atr_raw * float(bars) ** 0.5

        atr_pct = atr_raw / cur  # ATR as fraction of price
        if atr_pct < 1e-8:
            return _SKIP

        signal = log_ret / atr_pct
        return float(np.clip(signal, -3.0, 3.0))

    # Normalization clips for regime-continuous targets (after inverse transform)
    _REGIME_CONT_CLIPS = {
        "vol_regime": (0.0, 5.0),  # vol/vol_mean: >5 is extreme outlier
        "atr_norm": (0.0, 0.05),  # ATR14/price: 5% max for FX
        "adx": (0.0, 100.0),  # ADX14 natural range
        "di_direction": (-1.0, 1.0),  # tanh output, already in [-1,1]
    }
    # Scale factors → map each signal to roughly [0, 1] for loss stability
    _REGIME_CONT_SCALE = {
        "vol_regime": 1.0,  # already ~[0,3], no scaling
        "atr_norm": 100.0,  # 0.001 → 0.1, 0.01 → 1.0
        "adx": 0.01,  # 50 → 0.5
        "di_direction": 1.0,  # already in [-1,1], no scaling
    }

    def _regime_continuous_for_tf(self, end_idx: int, tf: str) -> np.ndarray:
        """Returns [4] float32 array: [vol_regime, atr_norm_scaled, adx_scaled, di_direction].
        v25: added di_direction = tanh((DI+ - DI-) / 20) -> [-1, +1]
        Skip sentinel = -999.0 for signals that aren't available (old parquets).
        """
        _SKIP = -999.0
        n_sigs = len(self._REGIME_CONT_IDX)  # 4 in v25, 3 in v24
        sigs = self._regime_cont_by_tf.get(tf, {})
        out = np.full(n_sigs, _SKIP, dtype=np.float32)
        for _k, _idx in self._REGIME_CONT_IDX.items():
            _arr = sigs.get(_k)
            if _arr is None or end_idx >= len(_arr):
                continue
            v = float(_arr[end_idx])
            if not np.isfinite(v):
                continue
            lo, hi = self._REGIME_CONT_CLIPS[_k]
            v = float(np.clip(v, lo, hi))
            v *= self._REGIME_CONT_SCALE[_k]
            out[_idx] = v
        return out

    def _regime_for_tf(self, end_idx: int, tf: str) -> int:
        if not self._scalers_loaded:
            try:
                self._ensure_scalers_and_precompute()
            except Exception:
                pass
        series = getattr(self, "_regime_score_by_tf", {}).get(tf, None)
        if series is None or end_idx >= len(series):
            return -1
        v = series[end_idx]
        if np.isnan(v):
            return -1
        p20, p80 = self._regime_thresholds_by_tf.get(tf, (0.0, 0.0))
        if v <= p20:
            return 0
        elif v > p80:
            return 2
        else:
            return 1

    def _permute_seq(self, seq: np.ndarray) -> np.ndarray:
        if self.permute_mode == "reverse":
            return seq[::-1].copy()
        idxs = np.arange(len(seq))
        self._rng.shuffle(idxs)
        return seq[idxs].copy()

    def _get_atr_at(self, idx: int) -> Optional[float]:
        """
        Returns ATR14 at bar `idx` using ONLY cached NumPy arrays — no pandas.
        Priority:
        1. _atr_raw_np[idx]       — O(1), precached array
        2. manual TR from _hi/_lo — O(lookback), precached arrays
        3. rolling std(logret)    — last-resort fallback
        """
        # ── 1. ATR cached array — O(1) ──
        if self._atr_raw_np is not None and idx < len(self._atr_raw_np):
            val = float(self._atr_raw_np[idx])
            if np.isfinite(val) and val > 0:
                # FX ATR in real price: 0.00005 < ATR < 0.05
                # below the threshold it's probably still scaled
                if val > 1e-5:
                    return val

        # ── 2. manual TR from cached hi/lo — O(lookback) ──
        lookback = 14
        start = max(0, idx - lookback + 1)
        close_arr = self._close15_np
        if (
            close_arr is not None
            and self._hi_raw_np is not None
            and self._lo_raw_np is not None
            and idx < len(close_arr)
        ):
            try:
                hi_w = self._hi_raw_np[start : idx + 1]
                lo_w = self._lo_raw_np[start : idx + 1]
                cl_w = close_arr[start : idx + 1]
                prev = np.empty_like(cl_w)
                prev[0] = close_arr[start - 1] if start > 0 else cl_w[0]
                prev[1:] = cl_w[:-1]
                tr = np.maximum(hi_w - lo_w, np.maximum(np.abs(hi_w - prev), np.abs(lo_w - prev)))
                valid = tr[np.isfinite(tr) & (tr > 0)]
                if len(valid) >= 3:
                    return float(np.mean(valid))
            except Exception:
                pass

        # ── 3. Rolling std(logret) * price — fallback estremo ──
        if close_arr is not None and idx >= 3 and idx < len(close_arr):
            try:
                sl = close_arr[start : idx + 1]
                sl = sl[np.isfinite(sl) & (sl > 0)]
                if len(sl) >= 3:
                    std = float(np.nanstd(np.diff(np.log(sl))))
                    mid = float(sl[-1])
                    if std > 0 and mid > 0:
                        # std of log returns, times sqrt(14) to approximate ATR14
                        if not getattr(self, "_warned_atr_fallback", False):
                            logger.warning(
                                "[ATR] using std(logret) fallback for asset=%s",
                                getattr(self, "asset_name", "?"),
                            )
                            self._warned_atr_fallback = True
                        return std * mid * (14**0.5)
            except Exception:
                pass

        return None

    def _safe_ret(self, close_arr: np.ndarray, i0: int, i1: int) -> float:
        if close_arr is None:
            return 0.0
        n = len(close_arr)
        if i0 < 0 or i1 < 0 or i0 >= n or i1 >= n:
            return 0.0
        c0 = float(close_arr[i0])
        c1 = float(close_arr[i1])
        if not np.isfinite(c0) or not np.isfinite(c1) or abs(c0) < 1e-12:
            return 0.0
        return float(np.clip((c1 - c0) / abs(c0), -0.05, 0.05))

    def _range_pos(self, close_arr: np.ndarray, end_idx: int, lookback: int) -> float:
        if close_arr is None:
            return 0.5
        s = max(0, end_idx - lookback + 1)
        w = close_arr[s : end_idx + 1].astype(float)
        w = w[np.isfinite(w)]
        if w.size < 5:
            return 0.5
        lo = float(np.min(w))
        hi = float(np.max(w))
        if hi - lo < 1e-12:
            return 0.5
        cur = float(close_arr[end_idx])
        if not np.isfinite(cur):
            return 0.5
        return float(np.clip((cur - lo) / (hi - lo), 0.0, 1.0))

    def _build_regime_cont_vec(self, end_idx: int) -> np.ndarray:
        """Returns [n_tfs * 4] float32: [vol_regime, atr_norm, adx, di_direction] per TF.
        v25: moved from 2 signals/TF (legacy _regime_continuous_by_tf) to 4 signals/TF
        via _regime_continuous_for_tf (usa _regime_cont_by_tf). Skip sentinel = -999.0.
        """
        out = []
        for tf in self.tf_list:
            out.append(self._regime_continuous_for_tf(end_idx, tf))  # [4]
        return np.concatenate(out).astype(np.float32)  # [n_tfs * 4]

    def __getitem__(self, idx: int):
        self._ensure_loaded()
        try:
            self._ensure_scalers_and_precompute()
        except Exception as e:
            logger.debug("Scaler lazy-load failed: %s", e)

        end_idx = self.indices[idx]
        # Bar the INPUTS come from. The labels stay on end_idx: it's the
        # signal bar, and its outcome is what has to be predicted.
        # If this distinction gets lost, the model is trained again with
        # information it wouldn't have live — exactly the problem
        # score_offset exists to solve.
        _off = int(getattr(self.config, "score_offset", 0))
        feat_idx = max(self.seq_len - 1, end_idx - _off)
        seq = self._get_window(end_idx)
        masked_seq, mask = self._mask_input(seq)
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        masked_seq = np.nan_to_num(masked_seq, nan=0.0, posinf=0.0, neginf=0.0)

        trend_labels = []
        regime_labels = []
        is_close = []

        ts = self._df.index[end_idx]
        ts_pd = pd.Timestamp(ts)
        try:
            ts_pd = (
                ts_pd.tz_localize("Europe/Rome")
                if ts_pd.tzinfo is None
                else ts_pd.tz_convert("Europe/Rome")
            )
            ts_str = ts_pd.isoformat()
        except Exception:
            ts_str = str(ts)

        _label_exclude_tfs = set(getattr(self.config, "label_exclude_tfs", ["15min"]))

        for tf in self.tf_list:
            isc = bool(self._is_tf_close(ts_pd, tf))
            is_close.append(1 if isc else 0)

            if tf in _label_exclude_tfs:
                trend_labels.append(-1)
                regime_labels.append(-1)
                continue

            if isc:
                trend_labels.append(self._trend_for_tf(end_idx, tf))

                regime_labels.append(self._regime_for_tf(end_idx, tf))
            else:
                trend_labels.append(-1)
                regime_labels.append(-1)

        seq_perm = self._permute_seq(seq)
        asset_id = int(getattr(self, "_asset_id", 0))

        def _sget(arr):
            return int(arr[end_idx]) if arr is not None and end_idx < len(arr) else -1

        # v28: dedicated heads
        strat_hit_sl = _sget(self._strat_hit_session_long)
        strat_side_sl = _sget(self._strat_side_session_long)
        strat_hit_ss = _sget(self._strat_hit_session_short)
        strat_side_ss = _sget(self._strat_side_session_short)
        strat_hit_pl = _sget(self._strat_hit_pdhl_long)
        strat_side_pl = _sget(self._strat_side_pdhl_long)
        strat_hit_ps = _sget(self._strat_hit_pdhl_short)
        strat_side_ps = _sget(self._strat_side_pdhl_short)
        regime_qual = _sget(self._regime_quality_label)

        # v29: be_reachable labels (-1 if not available -> BE off)
        def _cget(arr):
            """Sample confidence: 1.0 if the array is missing (backward compatible
            with datasets built before this change)."""
            if arr is None or end_idx >= len(arr):
                return 1.0
            return float(arr[end_idx])

        strat_be_sl = _sget(self._strat_be_session_long)
        strat_be_ss = _sget(self._strat_be_session_short)
        strat_be_pl = _sget(self._strat_be_pdhl_long)
        strat_be_ps = _sget(self._strat_be_pdhl_short)

        col15 = self.close_col_by_tf.get("15min")
        close15 = self.close_unscaled.get(col15) if col15 else None
        # timing_15m: these are model INPUTS, so they follow feat_idx.
        # They were the sneakiest case: _ret_15m measures the return OF the
        # signal bar, i.e. the breakout move itself. Leaving it on end_idx
        # would have handed the model the most valuable information we're
        # trying to take away from it, undoing everything else.
        _ret_15m = self._safe_ret(close15, feat_idx - 1, feat_idx)
        _ret_1h = self._safe_ret(close15, feat_idx - 4, feat_idx)
        _range_pos_1d = self._range_pos(close15, feat_idx, lookback=96)
        _rc = self._regime_continuous_for_tf(feat_idx, "15min")
        # _REGIME_CONT_IDX: vol_regime=0, atr_norm=1, adx=2, di_direction=3  (skip=-999.0)
        _atr_pct_15m = float(_rc[1]) if _rc[1] != -999.0 else 0.5
        _vol_zclip_15m = float(_rc[0]) if _rc[0] != -999.0 else 0.5

        return {
            "asset": self.asset_name,
            "asset_id": asset_id,
            # StrategyOutcomeHead labels — hit/side decomposition (v26)
            # v28: dedicated heads
            "strat_hit_sl": strat_hit_sl,
            "strat_side_sl": strat_side_sl,
            "strat_hit_ss": strat_hit_ss,
            "strat_side_ss": strat_side_ss,
            "strat_hit_pl": strat_hit_pl,
            "strat_side_pl": strat_side_pl,
            "strat_hit_ps": strat_hit_ps,
            "strat_side_ps": strat_side_ps,
            # v29: be_reachable (1=BE reachable before the SL, 0=no, -1=n/a)
            "strat_be_sl": strat_be_sl,
            "strat_be_ss": strat_be_ss,
            "strat_be_pl": strat_be_pl,
            "strat_be_ps": strat_be_ps,
            # Weight of the sample in the strategy loss: how clean the outcome is.
            # 0 = no outcome within the horizon, 1 = unambiguous.
            "strat_cnf_sl": _cget(getattr(self, "_strat_cnf_session_long", None)),
            "strat_cnf_ss": _cget(getattr(self, "_strat_cnf_session_short", None)),
            "strat_cnf_pl": _cget(getattr(self, "_strat_cnf_pdhl_long", None)),
            "strat_cnf_ps": _cget(getattr(self, "_strat_cnf_pdhl_short", None)),
            "regime_quality": regime_qual,
            "timestamp": ts_str,
            "seq": torch.from_numpy(seq),
            "masked_seq": torch.from_numpy(masked_seq),
            "mask": torch.from_numpy(mask.astype(np.uint8)),
            "trend": torch.tensor(np.array(trend_labels, dtype=np.int64), dtype=torch.int64),
            "regime": torch.tensor(np.array(regime_labels, dtype=np.int64)),
            "regime_cont": torch.from_numpy(self._build_regime_cont_vec(feat_idx)),
            "timing_15m": torch.tensor(
                np.array(
                    [_ret_15m, _ret_1h, _range_pos_1d, _atr_pct_15m, _vol_zclip_15m],
                    dtype=np.float32,
                )
            ),
            "seq_perm": torch.from_numpy(seq_perm),
            "is_close": torch.tensor(np.array(is_close, dtype=np.uint8)),
            "regime_continuous": torch.tensor(
                np.stack(
                    [self._regime_continuous_for_tf(feat_idx, tf) for tf in self.tf_list], axis=0
                ),
                dtype=torch.float32,
            ),  # [n_tfs, 4]: [vol_regime, atr_norm, adx, di_direction] skip=-999
            "trend_momentum": torch.tensor(
                np.array(
                    [self._trend_momentum_for_tf(feat_idx, tf) for tf in self.tf_list],
                    dtype=np.float32,
                ),
                dtype=torch.float32,
            ),  # [n_tfs], skip=-999.0   # [n_tfs] FUTURE-derived training target, skip=-999.0
            # DON'T use as a feature in build_strategy_context -> future leakage!
            "mktstructure": torch.tensor(
                np.array(
                    [
                        (
                            int(self._mktstructure_by_tf[tf][end_idx])
                            if (
                                hasattr(self, "_mktstructure_by_tf")
                                and tf in self._mktstructure_by_tf
                                and end_idx < len(self._mktstructure_by_tf[tf])
                            )
                            else -1
                        )
                        for tf in self.tf_list
                    ],
                    dtype=np.int64,
                ),
                dtype=torch.int64,
            ),  # [n_tfs]
        }
