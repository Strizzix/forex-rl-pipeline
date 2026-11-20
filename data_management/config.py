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
