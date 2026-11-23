"""
Encoder checks on synthetic data built with data_management (no download, CPU only).
"""

import glob
import json
import logging

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from conftest import synth_1min  # noqa: E402

from data_management.downloader import build_features, make_bars  # noqa: E402
from data_management.scale import scale_and_save  # noqa: E402
from encoder.components import (  # noqa: E402
    MarketStructureHead,
    PDHLBreakoutHead,
    RegimeContinuousHead,
    SessionBreakoutHead,
    TCNGRUEncoder,
    TrendMomentumHeadTF,
)
from encoder.config import DEFAULT_CONFIG, build_config  # noqa: E402
from encoder.dataset import (  # noqa: E402
    MarketPretrainConfig,
    MarketPretrainDataset,
    autodetect_feature_columns,
)

TFS = ["15min", "30min", "1h", "4h", "1d"]


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    """<root>/Scaled/Majors/<asset>/ with two synthetic assets, Dec 2021 -> Feb 2022."""
    root = tmp_path_factory.mktemp("fxdata")
    logging.disable(logging.WARNING)
    try:
        for i, asset in enumerate(["EURUSD", "GBPUSD"]):
            df1 = synth_1min(start="2021-01-01", days=420, seed=i)
            feats = build_features(make_bars(df1), start="2021-12-01")
            scale_and_save(
                feats,
                asset,
                "Majors",
                root / "Scaled",
                refit=True,
                cutoff="2022-01-15",
                source_meta={"label": "close"},  # as data_management.scale.main passes it
            )
    finally:
        logging.disable(logging.NOTSET)
    return root


def _dataset(data_root, **cfg):
    folder = data_root / "Scaled" / "Majors" / "EURUSD"
    df = pd.read_parquet(glob.glob(str(folder / "*_scaled_*.parquet"))[0])
    df.columns = [c.lower() for c in df.columns]  # as the trainer does
    config = MarketPretrainConfig(**{"seq_len": 64, "step_stride": 4, **cfg})
    cols = autodetect_feature_columns(df, TFS)
    return MarketPretrainDataset(
        df, config, cols, asset_name="EURUSD", scaler_path=str(folder / "all_scalers.pkl")
    )


def test_dataset_items(data_root):
    ds = _dataset(data_root, outcome_be_enabled=True)
    assert len(ds) > 100
    item = ds[len(ds) // 2]
    seq = item["seq"].numpy()
    assert seq.shape == (64, ds.n_input_features)
    assert np.isfinite(seq).all()
    assert item["masked_seq"].shape == item["seq"].shape
    for key in ("strat_side_sl", "strat_side_ss", "strat_side_pl", "strat_side_ps"):
        assert int(item[key]) in (-1, 0, 1)
    assert set(np.unique(item["mktstructure"].numpy())) <= {-1, 0, 1, 2}
    tmom = item["trend_momentum"].numpy()
    assert np.all((tmom == -999.0) | ((tmom >= -3) & (tmom <= 3)))


def test_score_offset_window_ends_before_the_signal(data_root):
    ds0 = _dataset(data_root, score_offset=0)
    ds1 = _dataset(data_root, score_offset=1)
    end = ds0.indices[50]
    assert np.array_equal(ds0._get_window(end), ds0.data[end - 63 : end + 1])
    assert np.array_equal(ds1._get_window(end), ds1.data[end - 64 : end])


def test_structural_features_add_five_inputs(data_root):
    base = _dataset(data_root)
    struct = _dataset(data_root, use_structural_features=True)
    assert struct.n_input_features == base.n_input_features + 5
    assert np.isfinite(struct.data[:, -5:]).all()


def test_model_shapes():
    torch.manual_seed(0)
    n_feat, n_tfs, hidden = 40, len(TFS), 32
    enc = TCNGRUEncoder(
        input_features=n_feat,
        seq_len=64,
        tcn_channels=[16, 16, 32],
        kernel_size=3,
        gru_hidden=hidden,
        gru_layers=1,
        dropout=0.1,
        n_assets=5,
        asset_embed_dim=8,
    )
    x = torch.randn(4, 64, n_feat)
    emb = enc(x, asset_ids=torch.tensor([0, 1, 2, 3]))
    assert emb.shape == (4, hidden)
    assert TrendMomentumHeadTF(hidden, n_tfs)(emb).shape == (4, n_tfs)
    assert RegimeContinuousHead(hidden, n_tfs, n_signals=4)(emb).shape == (4, n_tfs, 4)
    assert MarketStructureHead(hidden, n_tfs)(emb).shape == (4, n_tfs, 3)
    for head_cls in (SessionBreakoutHead, PDHLBreakoutHead):
        long_logit, short_logit = head_cls(hidden, ctx_dim=29)(emb, torch.zeros(4, 29))
        assert long_logit.shape[0] == 4 and short_logit.shape[0] == 4


def test_build_config_paths(tmp_path):
    conf = build_config({"class_name": "Minors"}, root=tmp_path / "d", models_root=tmp_path / "m")
    assert conf["data_dir"] == str(tmp_path / "d" / "Scaled")
    assert conf["results_dir"].endswith(
        str(tmp_path / "m" / "Encoder" / "Minors" / "Checkpoint" / "Bootstrap")
    )
    conf = build_config(
        {"training_mode": "retrain", "retrain_version_tag": "WF9"}, models_root=tmp_path
    )
    assert conf["results_dir"].endswith("Retrain/WF9") or conf["results_dir"].endswith(
        "Retrain\\WF9"
    )
    # nested dicts are merged, not replaced; the defaults are never modified
    conf = build_config({"augment": {"jitter": 0.5}})
    assert conf["augment"]["jitter"] == 0.5 and conf["augment"]["scaling"] == 0.05
    assert DEFAULT_CONFIG["augment"]["jitter"] == 0.02


@pytest.mark.parametrize("struct", [False, True])
def test_trainer_smoke(data_root, tmp_path, struct):
    """One tiny bootstrap epoch through the CLI: checkpoints written and reloadable."""
    from encoder.trainer import main, resume_checkpoint

    out = tmp_path / "run"
    sets = {
        "seq_len": 32,
        "batch_size": 16,
        "step_stride": 32,
        "bootstrap_epochs": 1,
        "num_workers": 0,
        "aug_workers": 0,
        "use_amp": False,
        "use_global_seed": True,
        "val_max_batches": 3,
        "bootstrap_train_start": "2021-12-08",
        "bootstrap_train_end": "2022-01-15",
        "bootstrap_val_start": "2022-01-16",
        "bootstrap_val_end": "2022-02-28",
        "drift_anchor_start": "2021-12-15",
        "drift_anchor_end": "2021-12-31",
        "keepalive": False,
        "tcn_channels": [16, 16, 32],
        "gru_hidden": 32,
        "proj_dim": 32,
        "memory_bank_size": 256,
        "emb_max_points": 128,
        "drift_max_batches": 2,
        "drift_max_points": 64,
        "augment": {"use_gpu_augment": False},
        "trainer": {"device": "cpu"},
        "use_structural_features": struct,
    }
    args = ["--class", "Majors", "--mode", "bootstrap", "--root", str(data_root)]
    args += ["--results-dir", str(out), "--set"] + [f"{k}={json.dumps(v)}" for k, v in sets.items()]
    history = main(args)
    assert history["val"], "no validation epoch recorded"
    # structural features: the training window's normalization is saved for live use
    assert (out / "structural_stats.json").exists() == struct

    ckpt = out / "best_full_checkpoint.pth"
    assert ckpt.exists() and (out / "best_encoder_only.pth").exists()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert state["extra"]["training_mode"] == "bootstrap"
    # what the trading agent needs to rebuild the exact input
    inputs = state["extra"]["inputs"]
    assert inputs["use_structural_features"] == struct and inputs["seq_len"] == 32
    assert set(inputs["structural_stats"]) == ({"EURUSD", "GBPUSD"} if struct else set())
    assert sorted(inputs["asset_map"]) == ["EURUSD", "GBPUSD"]
    # the checkpoint loads strictly into freshly built modules (state_dict names unchanged)
    n_in = (
        state["encoder_state"]["tcn.network.0.conv1.weight"].shape[1] - 8
    )  # minus asset embedding
    enc = TCNGRUEncoder(
        input_features=n_in,
        seq_len=32,
        tcn_channels=[16, 16, 32],
        kernel_size=3,
        gru_hidden=32,
        gru_layers=1,
        dropout=0.25,
        n_assets=5,
        asset_embed_dim=8,
    )
    enc.load_state_dict(state["encoder_state"])
    trend = TrendMomentumHeadTF(32, len(TFS))
    trend.load_state_dict(state["trend_head_state"])
    info = resume_checkpoint(str(ckpt), enc, trend)
    assert info["epoch"] == 1


# ── time convention (rows labelled at bar close) ─────────────────────────────
def test_session_features_use_the_bar_open_time():
    from encoder.dataset import _SESSION_FEATURE_NAMES

    idx = pd.DatetimeIndex(
        ["2022-01-10 08:00", "2022-01-10 08:15"], tz="Europe/Rome"
    )  # bars [06:45, 07:00) and [07:00, 07:15) UTC
    ds = MarketPretrainDataset.__new__(MarketPretrainDataset)
    feats = ds._compute_session_features(pd.DataFrame(index=idx))
    london = feats[:, _SESSION_FEATURE_NAMES.index("session_london")]
    assert list(london) == [0.0, 1.0]  # London opens at 07:00 UTC


def test_fx_trading_weekday_rolls_at_new_york_close():
    from encoder.dataset import bar_open_utc, fx_trading_weekday

    idx = pd.DatetimeIndex(
        ["2022-01-09 23:15", "2022-01-10 10:00", "2022-01-14 22:45", "2022-01-11 23:15"],
        tz="Europe/Rome",
    )  # Sun evening, Mon, Fri before the close, Tue evening after 17:00 NY
    assert list(fx_trading_weekday(bar_open_utc(idx))) == [0, 0, 4, 2]


def test_tf_close_follows_the_data_bins_in_summer_time():
    from encoder.dataset import is_tf_close

    rome = "Europe/Rome"
    # 4H bins are aligned in absolute time: 00/04/08 Rome in winter, 01/05/09 in summer
    assert is_tf_close(pd.Timestamp("2022-01-10 08:00", tz=rome), "4h")
    assert is_tf_close(pd.Timestamp("2022-07-05 09:00", tz=rome), "4h")
    assert not is_tf_close(pd.Timestamp("2022-07-05 08:00", tz=rome), "4h")
    assert is_tf_close(pd.Timestamp("2022-07-05 00:00", tz=rome), "1d")
    assert is_tf_close(pd.Timestamp("2022-07-05 09:30", tz=rome), "30min")


def test_tf_close_matches_the_rows_where_a_new_bar_appears(data_root):
    ds = _dataset(data_root)
    close4h = ds._df["close_4h_raw"] if "close_4h_raw" in ds._df else ds._df["close_4h"]
    changed = close4h.ne(close4h.shift()).to_numpy()
    flagged = np.array([ds._is_tf_close(t, "4h") for t in ds._df.index])
    # rows right after a market gap see a bar that closed during the gap: skip them
    contiguous = (ds._df.index.to_series().diff() == pd.Timedelta(minutes=15)).to_numpy()
    assert np.array_equal(changed[contiguous], flagged[contiguous])


# ── PDHL DI gate ─────────────────────────────────────────────────────────────
def test_pdhl_labels_respect_the_di_gate(data_root):
    gated = _dataset(data_root, step_stride=1, outcome_pdhl_di_thresh=0.15)
    open_ = _dataset(data_root, step_stride=1, outcome_pdhl_di_gate=False)
    di = np.tanh(
        (gated._get_unscaled_col("di_plus14_4h") - gated._get_unscaled_col("di_minus14_4h")) / 20
    )
    longs = np.where(gated._strat_hit_pdhl_long >= 0)[0]
    shorts = np.where(gated._strat_hit_pdhl_short >= 0)[0]
    assert (di[longs - 1] > 0.15).all() and (di[shorts - 1] < -0.15).all()
    n_gated = len(longs) + len(shorts)
    n_open = int((open_._strat_hit_pdhl_long >= 0).sum() + (open_._strat_hit_pdhl_short >= 0).sum())
    assert n_gated < n_open  # the gate really filters


# ── structural features ──────────────────────────────────────────────────────
def test_structural_features_are_live_and_use_training_stats(data_root):
    ds = _dataset(data_root, use_structural_features=True)
    struct = ds.data[:, -5:]
    assert (np.abs(struct).sum(axis=0) > 0).all()  # no feature stuck at 0
    assert ((struct == 0).mean(axis=0) < 0.05).all()
    stats = ds.structural_stats
    # another dataset normalized with these statistics gives the same values
    again = _dataset(data_root, use_structural_features=True)
    again_stats = {k: (v[0] + 1.0, v[1]) for k, v in stats.items()}  # shifted medians
    folder = data_root / "Scaled" / "Majors" / "EURUSD"
    df = pd.read_parquet(glob.glob(str(folder / "*_scaled_*.parquet"))[0])
    df.columns = [c.lower() for c in df.columns]
    shifted = MarketPretrainDataset(
        df,
        MarketPretrainConfig(seq_len=64, step_stride=4, use_structural_features=True),
        autodetect_feature_columns(df, TFS),
        asset_name="EURUSD",
        scaler_path=str(folder / "all_scalers.pkl"),
        structural_stats=again_stats,
    )
    assert np.allclose(again.data[:, -5:], struct)
    assert not np.allclose(shifted.data[:, -5:], struct)
    assert shifted.structural_stats == again_stats


# ── input discovery / schema ─────────────────────────────────────────────────
def test_trainer_refuses_open_labelled_or_raw_inputs(tmp_path):
    from data_management import storage
    from encoder.dataset import discover_asset_parquets
    from encoder.trainer import check_input_schema

    folder = tmp_path / "Majors" / "EURUSD"
    folder.mkdir(parents=True)
    df = pd.DataFrame({"close_15min": [1.0]}, index=pd.DatetimeIndex(["2022-01-03"], tz="UTC"))
    storage.write_parquet(df, folder / "EURUSD_bars15min_a.parquet", {"kind": "bars15min"})
    df.to_parquet(folder / "EURUSD_scaled_a.parquet")  # no metadata: older downloader
    found = discover_asset_parquets(str(tmp_path), "Majors")
    assert found["EURUSD"].endswith("EURUSD_scaled_a.parquet")  # never the bars cache
    with pytest.raises(RuntimeError, match="bar-close labels"):
        check_input_schema(found)
    check_input_schema(found, allow_open_labels=True)  # explicit opt-in: warning only
    with pytest.raises(RuntimeError, match="not scaled data"):
        check_input_schema({"EURUSD": str(folder / "EURUSD_bars15min_a.parquet")})
