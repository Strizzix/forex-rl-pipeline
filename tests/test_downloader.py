"""
Downloader checks on synthetic data (no Dukascopy calls).
"""
import numpy as np
import pandas as pd

from data_management.downloader import DukascopyMultiTFDownloader, merge_bid_ask, required_warmup_days
from conftest import synth_1min

DATA = synth_1min()


class FakeDownloader(DukascopyMultiTFDownloader):
    """fetch_weekly_data served from the synthetic data instead of Dukascopy."""
    def fetch_weekly_data(self, fetch_start=None):
        if fetch_start is None:
            fetch_start = self.start_date - pd.Timedelta(days=self.warmup_days)
        return DATA[(DATA.index >= fetch_start) & (DATA.index <= self.end_date)]


def test_warmup_covers_ema200_daily():
    assert required_warmup_days() == int(200 * 365 / 252) + 25


def test_columns(multi_tf):
    cols = set(multi_tf.columns)
    for tf in ["15min", "30min", "1H", "4H", "1D"]:
        assert {f"close_{tf}", f"RSI14_{tf}", f"ATR14_{tf}", f"vwp_{tf}"} <= cols
    assert {"spread_mean_15min", "MACD_1H", "Pivot_1D", "EMA200_1D", "Stoch_30min"} <= cols
    assert multi_tf.index.is_monotonic_increasing and multi_tf.index.is_unique
    assert not multi_tf.iloc[:20].isna().any().any()


def test_higher_tf_no_lookahead():
    """Cutting the 1min data at T must not change any row whose 15min bar closed by T."""
    dl = DukascopyMultiTFDownloader("2021-12-01", "2022-04-01", "unused", "EURUSD")
    cutoff = pd.Timestamp("2021-12-01", tz="Europe/Rome")
    full = dl.resample_timeframes(DATA, trim_head=False, cutoff=cutoff)
    for t in ["2022-02-09 13:37", "2022-03-04 23:50", "2022-03-21 10:05"]:
        T = pd.Timestamp(t, tz="Europe/Rome")
        trunc = dl.resample_timeframes(DATA[DATA.index < T], trim_head=False, cutoff=cutoff)
        rows = trunc.index[trunc.index + pd.Timedelta(minutes=15) <= T]
        pd.testing.assert_frame_equal(trunc.loc[rows], full.loc[rows])


def test_update_matches_full_build(tmp_path):
    upd, full = tmp_path / "upd", tmp_path / "full"
    FakeDownloader("2022-01-01", "2022-02-15", str(upd), "EURUSD").run()
    FakeDownloader("2022-01-01", "2022-03-20", str(upd), "EURUSD").update_parquet()
    FakeDownloader("2022-01-01", "2022-03-20", str(full), "EURUSD").run()

    a = pd.read_parquet(list((upd / "EURUSD").glob("*.parquet"))[0])
    b = pd.read_parquet(list((full / "EURUSD").glob("*.parquet"))[0])
    assert len(list((upd / "EURUSD").glob("*.parquet"))) == 1   # old file replaced
    assert a.index.is_unique and a.index.equals(b.index)
    # EMA/ATR have infinite memory: with the warmup the difference is negligible, not 0.
    # The 1D long EMAs don't fully converge in 314 days of warmup (~220 daily bars),
    # so they're left out here (known limit, see README).
    slow = ["EMA200_1D", "EMA200_slope_1D", "EMA50_slope_1D"]
    pd.testing.assert_frame_equal(a.drop(columns=slow), b.drop(columns=slow), rtol=1e-4, atol=1e-6)
    assert np.allclose(a["EMA200_1D"], b["EMA200_1D"], rtol=1e-2)


def test_4h_bins_dont_depend_on_download_start():
    dl = DukascopyMultiTFDownloader("2022-01-01", "2022-03-01", "unused", "EURUSD")
    T = pd.Timestamp("2021-12-01", tz="Europe/Rome")
    winter = dl.resample_timeframes(DATA[DATA.index >= "2021-02-01"], trim_head=False, cutoff=T)
    summer = dl.resample_timeframes(DATA[DATA.index >= "2021-05-01"], trim_head=False, cutoff=T)
    pd.testing.assert_series_equal(winter["close_4H"], summer["close_4H"])


def test_no_partial_last_bar():
    dl = DukascopyMultiTFDownloader("2022-01-01", "2022-03-01", "unused", "EURUSD")
    data = DATA[DATA.index <= pd.Timestamp("2022-02-09 10:31", tz="Europe/Rome")]
    out = dl.resample_timeframes(data, trim_head=False)
    assert out.index[-1] == pd.Timestamp("2022-02-09 10:15", tz="Europe/Rome")


def test_merge_bid_ask_keeps_one_sided_minutes():
    idx = pd.date_range("2022-01-03", periods=100, freq="1min", tz="UTC")
    bid = pd.DataFrame({k: np.full(100, 1.0) for k in ["open", "high", "low", "close"]}, index=idx)
    bid["volume"] = 1.0
    ask = bid + 0.0002
    out = merge_bid_ask(bid.drop(idx[:10]), ask.drop(idx[-5:]))
    assert len(out) == 100
    assert (out.loc[idx[:10], "spread"] == 0).all() and (out.loc[idx[-5:], "spread"] == 0).all()
    assert np.allclose(out.loc[idx[10:-5], "spread"], 0.0002)
