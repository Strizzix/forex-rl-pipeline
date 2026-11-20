import joblib
import numpy as np
import pandas as pd

from data_management import scale

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
