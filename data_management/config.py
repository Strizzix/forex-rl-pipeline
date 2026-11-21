"""
config.py — settings shared by downloader.py and scale.py.

Data layout (root = --root or the FX_DATA_ROOT env var, default ./data):

    <root>/Raw/<Majors|Minors>/<ASSET>/<ASSET>_bars15min_<start>_to_<end>.parquet   15min bars cache
    <root>/Raw/<Majors|Minors>/<ASSET>/<ASSET>_multiTF_<start>_to_<end>.parquet     features
    <root>/Scaled/<Majors|Minors>/<ASSET>/<ASSET>_scaled_<start>_to_<end>.parquet
    <root>/Scaled/<Majors|Minors>/<ASSET>/all_scalers.pkl
"""

import os
from pathlib import Path

ASSETS = {
    "Majors": ["AUDUSD", "EURUSD", "GBPUSD", "USDCAD", "USDCHF"],
    "Minors": ["EURAUD", "GBPAUD", "EURGBP", "EURNZD", "NZDUSD"],
}

DATA_ROOT = Path(os.environ.get("FX_DATA_ROOT", "data"))

START_DATE = "2013-01-01"
TIMEZONE = "Europe/Rome"
FIT_CUTOFF_DATE = "2022-12-31"  # scalers fit on <= 2022, the 2023+ holdout is never seen

# Closed-market bars. Dukascopy sometimes publishes flat volume-0 bars while the market is
# closed (some weekends and holidays, mostly in older years) and sometimes nothing at all.
# Indicators count bars, so those flat bars would change ATR / RSI / EMA after some
# weekends and not after others. With DROP_CLOSED_BARS they are left out of the features
# (they stay in the bars cache). Only volume-0 bars are ever dropped, and only when the
# market is closed:
#   - weekly close: Friday 17:00 -> Sunday 17:00 New York time (DST handled), or
#   - a holiday: a run of consecutive volume-0 bars lasting >= CLOSED_RUN_HOURS.
# A quiet bar with no ticks during an open session is always kept.
DROP_CLOSED_BARS = True
MARKET_TZ = "America/New_York"
WEEKLY_CLOSE_HOUR = 17  # FX week closes Friday and reopens Sunday at 17:00 New York time
CLOSED_RUN_HOURS = 4

# Indicators that go into the dataset (see data_management/indicators.py for the list
# and what each one computes). Remove a name to leave it out; `--indicators` on the
# downloader command line overrides this. Changing the list does not need a new
# download: `downloader --mode features` rebuilds the features from the bars cache.
INDICATORS = [
    "ema",
    "wma",
    "adx",
    "rsi",
    "cci",
    "stoch",
    "macd",
    "atr",  # ATR14_15min is also kept unscaled by scale.py (needed downstream)
    "bollinger",
    "volatility",
    "returns",
    "candles",
    "vwp",
    "pivots",
]


def iter_assets(groups=None, assets=None):
    """Yield (group, asset), optionally filtered."""
    for group, names in ASSETS.items():
        if groups and group not in groups:
            continue
        for a in names:
            if not assets or a in assets:
                yield group, a


def dukascopy_instrument(name):
    """'EURUSD' -> dukascopy_python instrument constant (majors / crosses / metals)."""
    from dukascopy_python import instruments

    base, quote = name[:3], name[3:]
    for family in ("MAJORS", "CROSSES", "METALS"):
        const = f"INSTRUMENT_FX_{family}_{base}_{quote}"
        if hasattr(instruments, const):
            return getattr(instruments, const)
    raise KeyError(f"no Dukascopy instrument for {name}")
