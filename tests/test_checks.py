"""
Quality checks and old/new comparison, on synthetic datasets.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from test_downloader import FakeDownloader

from data_management import checks, compare_versions, scale, storage

CUTOFF = "2022-02-15"


@pytest.fixture()
def dataset(tmp_path):
    """<root>/Raw + <root>/Scaled for EURUSD, built like the real pipeline."""
    FakeDownloader("2022-01-01", "2022-03-20", str(tmp_path / "Raw" / "Majors"), "EURUSD").run()
    pair = tmp_path / "Raw" / "Majors" / "EURUSD"
    raw = next(pair.glob("*_multiTF_*.parquet"))
    scale.scale_and_save(
        pd.read_parquet(raw),
        "EURUSD",
        "Majors",
        tmp_path / "Scaled",
        refit=True,
        cutoff=CUTOFF,
        source_meta=storage.read_meta(raw),
    )
    return tmp_path


def _run(root, only=None):
    return {r.kind: r for r in checks.run(root, assets=["EURUSD"], only=only)}


def test_clean_dataset_passes(dataset):
    reps = _run(dataset)
    assert reps["raw"].ok, reps["raw"].errors
    assert reps["scaled"].ok, reps["scaled"].errors
    assert reps["raw"].info["gap_weekend"] > 0 and reps["raw"].info["gap_anomalous"] == 0


def test_detects_open_labelled_rows(dataset):
    """Shifting the rows back to open labels breaks the bar-close checks."""
    path = next((dataset / "Raw" / "Majors" / "EURUSD").glob("*_multiTF_*.parquet"))
    df = pd.read_parquet(path)
    df.index = df.index - pd.Timedelta(minutes=15)
    storage.write_parquet(df, path, storage.read_meta(path))
    errors = " ".join(_run(dataset, "raw")["raw"].errors)
    assert "no bar in the cache" in errors and "bar-close rows" in errors


def test_detects_missing_indicator_column_and_old_schema(dataset):
    path = next((dataset / "Raw" / "Majors" / "EURUSD").glob("*_multiTF_*.parquet"))
    df = pd.read_parquet(path)
    storage.write_parquet(df.drop(columns=["RSI14_4H"]), path, storage.read_meta(path))
    assert "RSI14_4H" in " ".join(_run(dataset, "raw")["raw"].errors)
    df.to_parquet(path)  # no metadata = old downloader
    assert "rebuild with --mode full" in " ".join(_run(dataset, "raw")["raw"].errors)


def test_detects_scalers_fit_on_the_wrong_window(dataset):
    pkl = dataset / "Scaled" / "Majors" / "EURUSD" / "all_scalers.pkl"
    saved = joblib.load(pkl)
    saved["meta"]["fit_cutoff_date"] = "2022-03-10"  # not the window they were fit on
    joblib.dump(saved, pkl)
    assert "not centred on the fit window" in " ".join(_run(dataset, "scaled")["scaled"].errors)


def test_detects_tampered_raw_copy(dataset):
    path = next((dataset / "Scaled" / "Majors" / "EURUSD").glob("*_scaled_*.parquet"))
    df = pd.read_parquet(path)
    df.iloc[100, df.columns.get_loc("close_15min_raw")] += 0.01
    storage.write_parquet(df, path, storage.read_meta(path))
    assert "close_15min_raw" in " ".join(_run(dataset, "scaled")["scaled"].errors)


def test_gap_classification():
    idx = pd.date_range("2022-01-03", "2022-01-21 22:00", freq="15min", tz="Europe/Rome")
    idx = idx[idx.dayofweek < 5]  # weekends closed
    hole = (idx >= "2022-01-12 03:00") & (idx < "2022-01-12 05:00")  # 2h illiquid gap
    g = checks.analyze_gaps(idx[~hole])
    assert g["weekend"] == 2 and g["short"] == 1 and len(g["anomalies"]) == 0
    hole = (idx >= "2022-01-12") & (idx < "2022-01-20")  # 8 days missing midweek
    assert len(checks.analyze_gaps(idx[~hole])["anomalies"]) == 1


def test_compare_old_and_new_versions(dataset):
    path = next((dataset / "Raw" / "Majors" / "EURUSD").glob("*_multiTF_*.parquet"))
    new = pd.read_parquet(path)
    # simulate the old downloader: 15min columns at the open label, higher TFs unchanged
    c15 = [c for c in new.columns if c.endswith("_15min")]
    old15 = new[c15].copy()
    old15.index = old15.index - pd.Timedelta(minutes=15)
    old = new.drop(columns=c15).join(old15, how="outer")
    res = compare_versions.compare(old, new)
    assert (res["match"] == 1.0).all()
    assert compare_versions.report("EURUSD", old, new, {}, {"label": "close"})

    old["RSI14_1H"] *= 1.01  # an unexpected difference is reported
    assert not compare_versions.report("EURUSD", old, new, {}, {"label": "close"})
    old["spread_mean_15min"] = np.nan  # an expected one is not
    res = compare_versions.compare(old, new)
    assert res.loc["spread_mean_15min", "expected_diff"]
