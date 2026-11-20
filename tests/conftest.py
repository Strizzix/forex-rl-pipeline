import numpy as np
import pandas as pd
import pytest


def synth_1min(start="2021-01-01", days=450, seed=0, tz="Europe/Rome"):
    """Random-walk 1min mid OHLCV + spread (same format as fetch_weekly_data), weekends closed."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 1440, freq="1min", tz="UTC")
    idx = idx[idx.day_of_week < 5]
    n = len(idx)
    o = 1.1 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    c = o * np.exp(rng.normal(0, 5e-5, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 3e-5, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 3e-5, n)))
    df = pd.DataFrame(
        {
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "volume": rng.integers(0, 100, n).astype(float),
            "spread": np.abs(rng.normal(1e-4, 3e-5, n)),
        },
        index=idx,
    )
    df.index = df.index.tz_convert(tz)
    return df


@pytest.fixture(scope="session")
def df_1min():
    return synth_1min()


@pytest.fixture(scope="session")
def bars15(df_1min):
    from data_management.downloader import make_bars

    return make_bars(df_1min)


@pytest.fixture(scope="session")
def multi_tf(bars15):
    from data_management.downloader import build_features

    return build_features(bars15, start="2021-12-01")
