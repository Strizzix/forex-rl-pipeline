"""
Downloader checks on synthetic data (no Dukascopy calls).
"""

import os

import numpy as np
import pandas as pd
import pytest
from conftest import synth_1min

from data_management import indicators, storage
from data_management.downloader import (
    DukascopyMultiTFDownloader,
    build_features,
    make_bars,
    merge_bid_ask,
    resample_bars,
)

DATA = synth_1min()
TZ = "Europe/Rome"


class FakeDownloader(DukascopyMultiTFDownloader):
    """fetch_weekly_data served from the synthetic data instead of Dukascopy."""

    def fetch_weekly_data(self, fetch_start=None):
        if fetch_start is None:
            fetch_start = self.start_date - pd.Timedelta(days=self.warmup_days)
        return DATA[(DATA.index >= fetch_start) & (DATA.index <= self.end_date)]


def _files(folder, kind):
    return sorted(p.name for p in folder.glob(f"*_{kind}_*.parquet"))


def test_warmup_covers_ema200_daily():
    assert indicators.warmup_days() == int(200 * 365 / 252) + 25
    # without the EMA the daily lookback is the double volatility window (2 x 20 bars)
    assert indicators.warmup_days(["rsi", "volatility"]) == int(40 * 365 / 252) + 25


def test_columns(multi_tf):
    cols = set(multi_tf.columns)
    for tf in ["15min", "30min", "1H", "4H", "1D"]:
        assert {f"close_{tf}", f"RSI14_{tf}", f"ATR14_{tf}", f"vwp_{tf}"} <= cols
    assert {"spread_mean_15min", "MACD_1H", "Pivot_1D", "EMA200_1D", "Stoch_30min"} <= cols
    assert multi_tf.index.is_monotonic_increasing and multi_tf.index.is_unique
    assert not multi_tf.iloc[:20].isna().any().any()


def test_rows_are_labelled_at_bar_close(bars15):
    t = pd.Timestamp("2022-02-09 10:15", tz=TZ)
    minutes = DATA[(DATA.index >= t - pd.Timedelta(minutes=15)) & (DATA.index < t)]
    assert bars15.loc[t, "open"] == minutes["open"].iloc[0]
    assert bars15.loc[t, "close"] == minutes["close"].iloc[-1]
    assert bars15.loc[t, "volume"] == minutes["volume"].sum()


def test_higher_tf_bars_match_direct_1min_resampling(bars15):
    """Aggregating the 15min bars gives the same OHLCV as resampling the 1min data."""
    origin = pd.Timestamp("2000-01-01", tz=TZ)
    for tf, rule in [("1H", "1h"), ("4H", "4h"), ("1D", "1D")]:
        kw = {} if tf == "1D" else {"origin": origin}
        direct = DATA.resample(rule, closed="left", label="right", **kw).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        direct = direct.dropna(subset=["close"])
        via15 = resample_bars(bars15, tf)
        common = via15.index[:-1]  # the last bar may be partial in `direct`
        pd.testing.assert_frame_equal(via15.loc[common], direct.loc[common], check_freq=False)


def test_no_lookahead():
    """Cutting the 1min data at T must not change any row labelled <= T."""
    full = build_features(make_bars(DATA), start="2021-12-01", trim_head=False)
    for t in ["2022-02-09 13:37", "2022-03-04 23:50", "2022-03-21 10:05", "2022-03-25 00:00"]:
        T = pd.Timestamp(t, tz=TZ)
        trunc = build_features(make_bars(DATA[DATA.index < T]), start="2021-12-01", trim_head=False)
        assert trunc.index.max() <= T
        pd.testing.assert_frame_equal(trunc, full.loc[trunc.index])


def test_update_is_identical_to_full_build(tmp_path):
    upd, full = tmp_path / "upd", tmp_path / "full"
    FakeDownloader("2022-01-01", "2022-02-15", str(upd), "EURUSD").run()
    FakeDownloader("2022-01-01", "2022-03-20", str(upd), "EURUSD").update_parquet()
    FakeDownloader("2022-01-01", "2022-03-20", str(full), "EURUSD").run()

    for kind in ("bars15min", "multiTF"):
        assert len(_files(upd / "EURUSD", kind)) == 1  # old files replaced
        a = pd.read_parquet(upd / "EURUSD" / _files(upd / "EURUSD", kind)[0])
        b = pd.read_parquet(full / "EURUSD" / _files(full / "EURUSD", kind)[0])
        pd.testing.assert_frame_equal(a, b)  # exact, EMA200_1D included


def test_update_rebuilds_old_format_dataset(tmp_path):
    pair = tmp_path / "EURUSD"
    pair.mkdir()
    pd.DataFrame({"close_15min": [1.0]}).to_parquet(pair / "EURUSD_multiTF_old.parquet")
    FakeDownloader("2022-01-01", "2022-02-15", str(tmp_path), "EURUSD").update_parquet()
    assert len(_files(pair, "multiTF")) == 1 and len(_files(pair, "bars15min")) == 1
    meta = storage.read_meta(pair / _files(pair, "multiTF")[0])
    assert meta["schema"] == storage.SCHEMA_VERSION and meta["label"] == "close"


def test_indicator_selection_and_features_mode(tmp_path):
    FakeDownloader("2022-01-01", "2022-02-15", str(tmp_path), "EURUSD").run()
    path = FakeDownloader(
        "2022-01-01", "2022-02-15", str(tmp_path), "EURUSD", indicator_names=["rsi", "atr"]
    ).rebuild_features()
    df = pd.read_parquet(path)
    roots = {c.rsplit("_", 1)[0] for c in df.columns}
    assert roots == {
        "open", "high", "low", "close", "volume", "spread_mean", "spread_max",
        "RSI14", "ATR14", "ATR50", "ATR_norm",
    }  # fmt: skip
    assert storage.read_meta(path)["indicators"] == ["rsi", "atr"]
    assert len(_files(tmp_path / "EURUSD", "multiTF")) == 1


def test_unknown_indicator_is_rejected():
    with pytest.raises(ValueError, match="unknown indicators"):
        indicators.validate(["rsi", "nope"])


def test_4h_bins_dont_depend_on_download_start():
    winter = resample_bars(make_bars(DATA[DATA.index >= "2021-02-01"]), "4H")
    summer = resample_bars(make_bars(DATA[DATA.index >= "2021-05-01"]), "4H")
    T = pd.Timestamp("2021-12-01", tz=TZ)
    pd.testing.assert_frame_equal(winter[winter.index > T], summer[summer.index > T])


def test_no_partial_last_bar():
    data = DATA[DATA.index <= pd.Timestamp("2022-02-09 10:31", tz=TZ)]
    # minutes up to 10:31 included: [10:15, 10:30) is complete, [10:30, 10:45) is not
    assert make_bars(data).index[-1] == pd.Timestamp("2022-02-09 10:30", tz=TZ)


def test_merge_bid_ask_keeps_one_sided_minutes():
    idx = pd.date_range("2022-01-03", periods=100, freq="1min", tz="UTC")
    bid = pd.DataFrame({k: np.full(100, 1.0) for k in ["open", "high", "low", "close"]}, index=idx)
    bid["volume"] = 1.0
    ask = bid + 0.0002
    out = merge_bid_ask(bid.drop(idx[:10]), ask.drop(idx[-5:]))
    assert len(out) == 100
    # one-sided minutes: spread unknown (NaN), not 0
    assert out.loc[idx[:10], "spread"].isna().all() and out.loc[idx[-5:], "spread"].isna().all()
    assert np.allclose(out.loc[idx[10:-5], "spread"], 0.0002)
    # one-sided minutes count only the volume of the side that exists (no double count)
    assert np.allclose(out.loc[idx[:10], "volume"], ask["volume"].iloc[0])
    assert np.allclose(out.loc[idx[-5:], "volume"], bid["volume"].iloc[0])
    assert np.allclose(out.loc[idx[10:-5], "volume"], bid["volume"].iloc[0] + ask["volume"].iloc[0])


def test_spread_mean_ignores_unknown_minutes():
    idx = pd.date_range("2022-01-03 10:00", periods=15, freq="1min", tz=TZ)
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "spread": 2e-4},
        index=idx,
    )
    df.loc[idx[:5], "spread"] = np.nan  # one-sided minutes
    bar = make_bars(df).iloc[0]
    assert bar["spread_mean"] == pytest.approx(2e-4) and bar["spread_max"] == pytest.approx(2e-4)


def test_files_named_after_the_data(tmp_path):
    pair = tmp_path / "EURUSD"
    pair.mkdir()
    (pair / "EURUSD_multiTF_2013_01_01_to_2020_01_01.parquet").write_bytes(b"stale")
    path = FakeDownloader("2022-01-01", "2022-02-15", str(tmp_path), "EURUSD").run()
    df = pd.read_parquet(path)
    assert _files(pair, "multiTF") == [os.path.basename(path)]
    assert os.path.basename(path) == (
        f"EURUSD_multiTF_{df.index.min():%Y_%m_%d}_to_{df.index.max():%Y_%m_%d}.parquet"
    )


def test_safe_fetch_backs_off_between_retries(monkeypatch):
    import dukascopy_python

    import data_management.downloader as dlmod

    calls, sleeps = [], []

    def flaky_fetch(**kw):
        calls.append(kw)
        if len(calls) < 3:
            raise ConnectionError("boom")
        return pd.DataFrame({"close": [1.0]})

    monkeypatch.setattr(dukascopy_python, "fetch", flaky_fetch)
    monkeypatch.setattr(dlmod.time, "sleep", sleeps.append)
    dl = DukascopyMultiTFDownloader("2022-01-01", "2022-02-01", "unused", "EURUSD", warmup_days=0)
    assert dl.safe_fetch(None, None, None, retries=5, backoff_s=2.0) is not None
    assert len(calls) == 3 and sleeps == [2.0, 4.0]


def test_update_rebuilds_features_when_the_selection_changes(tmp_path):
    FakeDownloader("2022-01-01", "2022-02-15", str(tmp_path), "EURUSD").run()
    path = FakeDownloader(
        "2022-01-01", "2022-02-15", str(tmp_path), "EURUSD", indicator_names=["rsi"]
    ).update_parquet()  # bars already up to date: no download, features rebuilt
    assert storage.read_meta(path)["indicators"] == ["rsi"]
