import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from data_management import indicators, scale, storage
from data_management.downloader import build_features

CUTOFF = "2022-02-15"


def _cut():
    return pd.Timestamp(CUTOFF, tz="Europe/Rome")


def test_scaler_choice():
    assert scale.scaler_name("close_4H") == "StandardScaler"
    assert scale.scaler_name("EMA200_1D") == "StandardScaler"
    assert scale.scaler_name("EMA20_slope_1H") == "RobustScaler"
    assert scale.scaler_name("RSI14_15min") == "MinMaxScaler"
    assert scale.scaler_name("DI_plus14_4H") == "MinMaxScaler"
    assert scale.scaler_name("DI_minus14_1D") == "MinMaxScaler"
    assert scale.scaler_name("volume_30min") == "Log1p+RobustScaler"
    assert scale.scaler_name("spread_max_15min") == "Passthrough"


def test_fit_ignores_future(multi_tf):
    cols = scale.feature_columns(multi_tf)
    a = scale.fit_leak_safe(multi_tf, cols, CUTOFF)
    wrecked = multi_tf.copy()
    wrecked.loc[~scale.fit_mask(wrecked.index, CUTOFF), cols] *= 1000.0
    b = scale.fit_leak_safe(wrecked, cols, CUTOFF)
    x = multi_tf.iloc[:50]
    for c in cols:
        assert np.allclose(a[c].transform(x[[c]]), b[c].transform(x[[c]]), equal_nan=True)


def test_scale_and_save(multi_tf, tmp_path):
    out = scale.scale_and_save(multi_tf, "EURUSD", "Majors", tmp_path, refit=True, cutoff=CUTOFF)
    df = pd.read_parquet(out)
    saved = joblib.load(tmp_path / "Majors" / "EURUSD" / "all_scalers.pkl")

    # raw copies and raw-only columns
    assert np.allclose(df["close_15min_raw"], multi_tf["close_15min"])
    assert "spread_mean_15min" not in df and np.allclose(
        df["spread_mean_15min_raw"], multi_tf["spread_mean_15min"]
    )
    # standard columns centered on the fit window, prices recoverable
    fit = df[scale.fit_mask(df.index, CUTOFF)]
    assert abs(fit["close_1H"].mean()) < 1e-6
    back = saved["col_scalers"]["close_4H"].inverse_transform(df[["close_4H"]].to_numpy())
    assert np.allclose(back.ravel(), multi_tf["close_4H"])
    assert (
        saved["cols"] == list(saved["col_scalers"]) and saved["meta"]["fit_cutoff_date"] == CUTOFF
    )

    # second run reuses the saved scalers -> same output, still one parquet
    out2 = scale.scale_and_save(multi_tf, "EURUSD", "Majors", tmp_path, cutoff="2099-01-01")
    pd.testing.assert_frame_equal(pd.read_parquet(out2), df)
    meta = joblib.load(tmp_path / "Majors" / "EURUSD" / "all_scalers.pkl")["meta"]
    assert meta["fit_cutoff_date"] == CUTOFF  # cutoff of the reused scalers, not "2099-01-01"
    assert len(list((tmp_path / "Majors" / "EURUSD").glob("*.parquet"))) == 1


def test_fit_mask_date_cutoff_includes_whole_day():
    idx = pd.date_range("2022-12-30 22:00", "2023-01-01 02:00", freq="15min", tz="Europe/Rome")
    m = scale.fit_mask(idx, "2022-12-31")
    assert m[idx < pd.Timestamp("2023-01-01", tz="Europe/Rome")].all()
    assert not m[idx >= pd.Timestamp("2023-01-01", tz="Europe/Rome")].any()
    # a cutoff with a time is exact
    m2 = scale.fit_mask(idx, "2022-12-31 12:00")
    assert idx[m2].max() == pd.Timestamp("2022-12-31 12:00", tz="Europe/Rome")


def test_apply_warns_on_unscaled_columns(multi_tf, tmp_path, caplog):
    scale.scale_and_save(multi_tf, "EURUSD", "Majors", tmp_path, refit=True, cutoff=CUTOFF)
    src = tmp_path / "new.parquet"
    multi_tf.assign(new_feature_15min=1.0).to_parquet(src)
    with caplog.at_level("WARNING"):
        scale.apply_saved_scalers(tmp_path / "Majors" / "EURUSD" / "all_scalers.pkl", src)
    assert "new_feature_15min" in caplog.text and "UNSCALED" in caplog.text


def test_every_column_has_a_declared_scaler(multi_tf):
    undeclared = [c for c in multi_tf.columns if indicators.scaler_for(c)[0] is None]
    assert undeclared == []


def test_unknown_column_falls_back_with_warning(caplog):
    s = pd.Series(np.random.default_rng(0).lognormal(size=500))
    with caplog.at_level("WARNING"):
        assert scale.scaler_name("my_new_feature_1H", s) == "RobustScaler"
    assert "not produced by any registered indicator" in caplog.text


def test_indicator_subset_is_scaled(bars15, tmp_path):
    df = build_features(bars15, start="2021-12-01", names=["rsi", "atr"])
    out = scale.scale_and_save(df, "EURUSD", "Majors", tmp_path, refit=True, cutoff=CUTOFF)
    saved = joblib.load(tmp_path / "Majors" / "EURUSD" / "all_scalers.pkl")
    assert set(saved["cols"]) == set(scale.feature_columns(df))
    assert "RSI14_4H" in saved["cols"] and "EMA20_1H" not in pd.read_parquet(out)


def test_scalers_of_an_older_schema_are_refit(multi_tf, tmp_path, caplog):
    scale.scale_and_save(multi_tf, "EURUSD", "Majors", tmp_path, refit=True, cutoff=CUTOFF)
    pkl = tmp_path / "Majors" / "EURUSD" / "all_scalers.pkl"
    saved = joblib.load(pkl)
    del saved["meta"]["schema"]  # as written by the old version
    joblib.dump(saved, pkl)
    with caplog.at_level("WARNING"):
        scale.scale_and_save(multi_tf, "EURUSD", "Majors", tmp_path, cutoff=CUTOFF)
    assert "refit" in caplog.text
    assert joblib.load(pkl)["meta"]["schema"] == storage.SCHEMA_VERSION


def test_main_reads_features_not_bars_and_refuses_old_files(bars15, tmp_path, monkeypatch):
    pair = tmp_path / "Raw" / "Majors" / "EURUSD"
    pair.mkdir(parents=True)
    feats = build_features(bars15, start="2021-12-01", names=["rsi"])
    storage.write_parquet(bars15, pair / "EURUSD_bars15min_a_to_b.parquet", {"label": "close"})
    storage.write_parquet(feats, pair / "EURUSD_multiTF_a_to_b.parquet", {"label": "close"})
    assert scale.latest_parquet(pair).name == "EURUSD_multiTF_a_to_b.parquet"

    monkeypatch.setattr(
        sys, "argv", ["scale", "--root", str(tmp_path), "--assets", "EURUSD", "--cutoff", CUTOFF]
    )
    scale.main()
    scaled = next((tmp_path / "Scaled" / "Majors" / "EURUSD").glob("*_scaled_*.parquet"))
    assert storage.read_meta(scaled)["label"] == "close"

    feats.to_parquet(pair / "EURUSD_multiTF_a_to_b.parquet")  # no metadata = old downloader
    with pytest.raises(SystemExit):
        scale.main()
