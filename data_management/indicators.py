"""
indicators.py — the technical indicators that can go into the dataset.

Every indicator is registered once with:
  - the function computing its columns from the OHLCV bars of one timeframe,
  - the scaler each of its columns gets in scale.py,
  - how many bars of history it needs (sizes the download warm-up),
  - the timeframes it applies to (None = all).

Choose which ones go into the dataset with ``config.INDICATORS`` or
``downloader --indicators ...``. To add a new indicator, write a function that returns
``{column: Series}`` and decorate it with ``@register(...)``: the downloader and the
scaler pick it up automatically.

Columns are named without the timeframe suffix here (e.g. ``RSI14``); the downloader
appends it (``RSI14_1H``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, CCIIndicator, EMAIndicator, WMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

TIMEFRAMES = ("15min", "30min", "1H", "4H", "1D")
TF_MINUTES = {"15min": 15, "30min": 30, "1H": 60, "4H": 240, "1D": 1440}

# moving-average periods per timeframe
MA_PERIODS = {
    "15min": [10, 20, 50],
    "30min": [10, 20, 50],
    "1H": [20, 50],
    "4H": [20, 50],
    "1D": [50, 200],
}
# rolling window (bars) of the realised volatility
VOLA_WINDOW = {"15min": 200, "30min": 200, "1H": 100, "4H": 50, "1D": 20}

# scalers of the price/volume columns that every timeframe has (not indicators)
BASE_SCALERS = {
    "open": "StandardScaler",
    "high": "StandardScaler",
    "low": "StandardScaler",
    "close": "StandardScaler",
    "volume": "Log1p+RobustScaler",
    "spread_mean": "Passthrough",
    "spread_max": "Passthrough",
}


@dataclass(frozen=True)
class Indicator:
    name: str
    compute: Callable[[pd.DataFrame, str], dict]
    scalers: dict  # regex (full match, case-insensitive, no TF suffix) -> scaler name
    lookback: Callable[[str], int]  # bars of history needed on a timeframe
    timeframes: tuple | None = None
    description: str = ""

    def applies_to(self, tf: str) -> bool:
        return self.timeframes is None or tf in self.timeframes


REGISTRY: dict[str, Indicator] = {}


def register(name, *, scalers, lookback, timeframes=None, description=""):
    """Decorator adding an indicator function to REGISTRY (registration order = column order)."""

    def deco(fn):
        lb = lookback if callable(lookback) else (lambda tf, n=lookback: n)
        REGISTRY[name] = Indicator(
            name, fn, scalers, lb, tuple(timeframes) if timeframes else None, description
        )
        return fn

    return deco


# ── indicators ──────────────────────────────────────────────────────────────
@register(
    "ema",
    scalers={r"ema\d+": "StandardScaler", r"ema\d+_slope": "RobustScaler"},
    lookback=lambda tf: max(MA_PERIODS[tf]),
    description="EMA and its one-bar slope for each period in MA_PERIODS",
)
def ema(b, tf):
    out = {}
    for p in MA_PERIODS[tf]:
        e = EMAIndicator(b["close"], window=p).ema_indicator()
        out[f"EMA{p}"] = e
        out[f"EMA{p}_slope"] = e.diff()
    return out


@register(
    "wma",
    scalers={r"wma\d+": "StandardScaler"},
    lookback=lambda tf: max(MA_PERIODS[tf]),
    description="weighted moving average for each period in MA_PERIODS",
)
def wma(b, tf):
    return {f"WMA{p}": WMAIndicator(b["close"], window=p).wma() for p in MA_PERIODS[tf]}


@register(
    "adx",
    scalers={r"adx14|di_plus14|di_minus14": "MinMaxScaler"},
    lookback=28,
    description="ADX(14) with DI+ / DI-",
)
def adx(b, tf):
    a = ADXIndicator(b["high"], b["low"], b["close"], window=14)
    return {"ADX14": a.adx(), "DI_plus14": a.adx_pos(), "DI_minus14": a.adx_neg()}


@register("rsi", scalers={r"rsi14": "MinMaxScaler"}, lookback=14, description="RSI(14)")
def rsi(b, tf):
    return {"RSI14": RSIIndicator(b["close"], window=14).rsi()}


@register("cci", scalers={r"cci20": "RobustScaler"}, lookback=20, description="CCI(20)")
def cci(b, tf):
    return {"CCI20": CCIIndicator(b["high"], b["low"], b["close"], window=20).cci()}


@register(
    "stoch",
    scalers={r"stoch": "MinMaxScaler"},
    lookback=14,
    timeframes=("15min", "30min"),
    description="stochastic oscillator %K(14)",
)
def stoch(b, tf):
    return {"Stoch": StochasticOscillator(b["high"], b["low"], b["close"]).stoch()}


@register(
    "macd",
    scalers={r"macd|macd_signal|macd_diff": "StandardScaler"},
    lookback=35,
    timeframes=("1H",),
    description="MACD(12, 26, 9)",
)
def macd(b, tf):
    m = MACD(b["close"])
    return {"MACD": m.macd(), "MACD_signal": m.macd_signal(), "MACD_diff": m.macd_diff()}


@register(
    "atr",
    scalers={r"atr14|atr50|atr_norm": "StandardScaler"},
    lookback=50,
    description="ATR(14), ATR(50) and ATR(14) / close",
)
def atr(b, tf):
    h, lo, c = b["high"], b["low"], b["close"]
    a14 = AverageTrueRange(h, lo, c, window=14).average_true_range()
    return {
        "ATR14": a14,
        "ATR50": AverageTrueRange(h, lo, c, window=50).average_true_range(),
        "ATR_norm": a14 / (c + 1e-12),
    }


@register(
    "bollinger",
    scalers={r"bb_width": "RobustScaler"},
    lookback=20,
    description="Bollinger band width (20, 2)",
)
def bollinger(b, tf):
    bb = BollingerBands(b["close"], window=20, window_dev=2)
    return {"BB_width": bb.bollinger_hband() - bb.bollinger_lband()}


@register(
    "volatility",
    scalers={r"volatility": "StandardScaler", r"volatility_regime": "RobustScaler"},
    lookback=lambda tf: 2 * VOLA_WINDOW[tf],  # double rolling window
    description="realised volatility and its ratio to its own rolling mean",
)
def volatility(b, tf):
    win = VOLA_WINDOW[tf]
    vol = b["close"].pct_change().rolling(win, min_periods=win).std()
    return {
        "volatility": vol,
        "Volatility_regime": vol / (vol.rolling(win, min_periods=win).mean() + 1e-12),
    }


@register(
    "returns",
    scalers={r"log_return|abs_return|momentum_proxy": "StandardScaler"},
    lookback=3,
    description="log return, absolute return, 3-bar momentum",
)
def returns(b, tf):
    c = b["close"]
    return {
        "log_return": np.log(c / c.shift()).fillna(0.0),
        "abs_return": c.pct_change().abs(),
        "momentum_proxy": c.diff(3) / (c.shift(3) + 1e-12),
    }


@register(
    "candles",
    scalers={
        r"high_low_range|close_open_diff|candle_body|candle_shadow|candle_pressure|range_ratio": (
            "RobustScaler"
        )
    },
    lookback=1,
    description="candle range, body, shadows and body/range ratios",
)
def candles(b, tf):
    o, h, lo, c = b["open"], b["high"], b["low"], b["close"]
    body = c - o
    rng = h - lo
    return {
        "high_low_range": rng,
        "close_open_diff": body,
        "candle_body": body.abs(),
        "candle_shadow": rng - body.abs(),
        "candle_pressure": body / (rng + 1e-8),
        "range_ratio": body.abs() / (rng + 1e-8),
    }


@register(
    "vwp",
    scalers={r"vwp": "StandardScaler"},
    lookback=50,
    description="volume-weighted price over 50 bars",
)
def vwp(b, tf):
    c, v = b["close"], b["volume"]
    return {
        "vwp": (c * v).rolling(50, min_periods=50).sum()
        / (v.rolling(50, min_periods=50).sum() + 1e-8)
    }


@register(
    "pivots",
    scalers={r"pivot|r1|s1|r2|s2": "StandardScaler"},
    lookback=1,
    timeframes=("1D",),
    description="classic daily pivot levels",
)
def pivots(b, tf):
    h, lo, c = b["high"], b["low"], b["close"]
    p = (h + lo + c) / 3.0
    return {"Pivot": p, "R1": 2 * p - lo, "S1": 2 * p - h, "R2": p + (h - lo), "S2": p - (h - lo)}


# ── helpers ─────────────────────────────────────────────────────────────────
ALL = tuple(REGISTRY)


def validate(names) -> list:
    """Indicator names in registry order; raises on unknown names."""
    names = ALL if names is None else names
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown indicators {unknown}; available: {list(REGISTRY)}")
    return [n for n in REGISTRY if n in set(names)]


def compute(bars: pd.DataFrame, tf: str, names=None) -> pd.DataFrame:
    """
    Bars of one timeframe + the selected indicators, built in one concat
    (no column-by-column inserts, so the frame is not fragmented).
    """
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    cols = {}
    for n in validate(names):
        ind = REGISTRY[n]
        if ind.applies_to(tf):
            cols.update(ind.compute(bars, tf))
    if not cols:
        return bars.copy()
    return pd.concat([bars, pd.DataFrame(cols, index=bars.index)], axis=1)


def lookback_bars(tf: str, names=None) -> int:
    """Bars of history the selected indicators need on a timeframe."""
    inds = [REGISTRY[n] for n in validate(names)]
    return max([ind.lookback(tf) for ind in inds if ind.applies_to(tf)], default=0)


def warmup_days(names=None, safety_margin_days=25) -> int:
    """
    Calendar days of history needed before the start date so every selected indicator
    is ready. 1D windows count TRADING days (~252 per year): EMA200 on 1D is ~290
    calendar days. Intraday windows count bars of a 5-day trading week.
    """
    days = []
    for tf in TIMEFRAMES:
        n = lookback_bars(tf, names)
        if tf == "1D":
            days.append(int(n * 365 / 252))
        else:
            days.append(int(n * TF_MINUTES[tf] / 1440 * 7 / 5))
    return max(days) + safety_margin_days


_TF_SUFFIX = re.compile(r"_(15min|30min|1h|4h|1d)$", re.IGNORECASE)


def scaler_for(column: str):
    """
    (scaler name, source) for a dataset column, from BASE_SCALERS or the indicator that
    produces it; (None, None) if no registered indicator produces it.
    """
    root = _TF_SUFFIX.sub("", column).lower()
    if root in BASE_SCALERS:
        return BASE_SCALERS[root], "base"
    for ind in REGISTRY.values():
        for pattern, scaler in ind.scalers.items():
            if re.fullmatch(pattern, root, flags=re.IGNORECASE):
                return scaler, ind.name
    return None, None
