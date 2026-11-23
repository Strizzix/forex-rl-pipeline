"""
Agent checks on synthetic data built with data_management (no download, CPU only).

The encoder checkpoint is a random "v26-style" one (encoder, trend / regime / market structure
heads and a StrategyOutcomeHead), the layout the environment reads.
"""

import copy
import glob
import json
import logging
import random

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from conftest import synth_1min  # noqa: E402

from agent.arbiter import TradeArbiter, TradeProposal  # noqa: E402
from agent.config import DEFAULT_CFG, build_cfg  # noqa: E402
from agent.env import (  # noqa: E402
    MarketEnv,
    add_session_features,
    build_encoder_feature_cols,
    ensure_datetime_index_utc,
    load_encoder_from_checkpoint,
)
from agent.pipeline import (  # noqa: E402
    find_encoder_checkpoints,
    latest_agent_checkpoint,
    main,
)
from agent.rainbow import RainbowAgent  # noqa: E402
from data_management.downloader import build_features, make_bars  # noqa: E402
from data_management.scale import scale_and_save  # noqa: E402
from encoder.components import (  # noqa: E402
    MarketStructureHead,
    RegimeContinuousHead,
    TCNGRUEncoder,
    TrendMomentumHeadTF,
)

HIDDEN = 32  # encoder gru_hidden of the test checkpoint
STATE_DIM = HIDDEN + 76  # embedding + head signals / market / position features (204 with 128)
TINY = {
    "seq_len": 32,
    "device": "cpu",
    "keepalive": False,
    "wf_mode": "manual",
    "wf_splits": [
        {
            "name": "Bootstrap",
            "train": ["2022-01-10", "2022-01-17"],
            "test": ["2022-01-18", "2022-01-25"],
            "encoder_wf_key": 0,
        }
    ],
    "env_cfg": {"seq_len": 32, "device": "cpu", "hold_reward_warmup_steps": 50},
    "runner_cfg": {"device": "cpu"},
    "rainbow_cfg": {
        "warmup_steps": 64,
        "batch_size": 32,
        "buffer_capacity": 5000,
        "hidden_dims": [64, 32],
        "per_beta_steps": 2000,
        "elite_buffer_capacity": 500,
    },
    "log_every_steps": 200,
}


class _StrategyOutcomeHead(nn.Module):
    """Same parameter names as the v26 StrategyOutcomeHead (trend/MR x long/short)."""

    def __init__(self, emb, e=8, hidden=64, ctx=29):
        super().__init__()
        self.strategy_emb = nn.Embedding(2, e)
        self.direction_emb = nn.Embedding(2, e)
        self.shared = nn.Sequential(
            nn.Linear(emb + 2 * e + ctx, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.hit_trunk = nn.Sequential(nn.Linear(hidden // 2, hidden // 2), nn.ReLU())
        self.side_trunk = nn.Sequential(nn.Linear(hidden // 2, hidden // 2), nn.ReLU())
        self.hit_head = nn.Linear(hidden // 2, 1)
        self.side_head = nn.Linear(hidden // 2, 1)


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
                source_meta={"label": "close"},
            )
    finally:
        logging.disable(logging.NOTSET)
    return root


@pytest.fixture(scope="module")
def scaled_df(data_root):
    f = glob.glob(str(data_root / "Scaled" / "Majors" / "EURUSD" / "*_scaled_*.parquet"))[0]
    return pd.read_parquet(f)


@pytest.fixture(scope="module")
def models_root(tmp_path_factory, scaled_df):
    """<models>/Encoder/Majors/Checkpoint/Bootstrap/best_full_checkpoint.pth (random weights)."""
    root = tmp_path_factory.mktemp("models")
    n_in = len(build_encoder_feature_cols(add_session_features(scaled_df)))
    torch.manual_seed(0)
    hidden, n_tfs = HIDDEN, 5
    enc = TCNGRUEncoder(
        input_features=n_in,
        seq_len=32,
        tcn_channels=[16, 16, 32],
        kernel_size=3,
        gru_hidden=hidden,
        gru_layers=1,
        dropout=0.1,
        n_assets=5,
        asset_embed_dim=8,
    )
    ckpt = {
        "encoder_state": enc.state_dict(),
        "trend_head_state": TrendMomentumHeadTF(hidden, n_tfs).state_dict(),
        "regime_cont_head_state": RegimeContinuousHead(hidden, n_tfs, n_signals=4).state_dict(),
        "mktstructure_head_state": MarketStructureHead(hidden, n_tfs).state_dict(),
        "strategy_outcome_head_state": _StrategyOutcomeHead(hidden).state_dict(),
        "cfg": {"tcn_channels": [16, 16, 32], "gru_hidden": hidden, "kernel_size": 3},
    }
    path = root / "Encoder" / "Majors" / "Checkpoint" / "Bootstrap" / "best_full_checkpoint.pth"
    path.parent.mkdir(parents=True)
    torch.save(ckpt, path)
    return root


def _seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _rainbow_cfg():
    cfg = copy.deepcopy(DEFAULT_CFG["rainbow_cfg"])
    cfg.update(TINY["rainbow_cfg"])
    return cfg


# ── Rainbow agent ─────────────────────────────────────────────────────────────


def test_rainbow_learns_and_round_trips(tmp_path):
    _seed()
    agent = RainbowAgent(STATE_DIM, _rainbow_cfg(), device="cpu")
    rng = np.random.default_rng(0)
    for asset in ("EURUSD", "GBPUSD"):
        for step in range(80):
            s, ns = rng.normal(size=(2, STATE_DIM)).astype(np.float32)
            done = step % 20 == 19
            agent.store(s, int(rng.integers(5)), float(rng.uniform(-1, 1)), ns, done, env_id=asset)
    assert set(agent._nstep_bufs) == {"EURUSD", "GBPUSD"}  # one N-step buffer per asset
    loss = agent.train_step()
    assert np.isfinite(loss)

    state = rng.normal(size=STATE_DIM).astype(np.float32)
    assert agent.act(state, mask=[True, False, False, False, False], deterministic=True) == 0
    assert agent.act(state) in range(5)

    agent.set_eval_mode()
    q = agent.q_values(torch.tensor(state))
    path = tmp_path / "agent.pth"
    agent.save(str(path))
    other = RainbowAgent(STATE_DIM, _rainbow_cfg(), device="cpu")
    other.load(str(path))
    other.set_eval_mode()
    assert torch.allclose(q, other.q_values(torch.tensor(state)), atol=1e-6)


# ── Arbiter ───────────────────────────────────────────────────────────────────


def _proposal(action, **kw):
    base = dict(
        asset="EURUSD",
        action=action,
        q_value=0.5,
        confidence=0.5,
        regime=0.5,
        trend_alignment=0.0,
        atr=1e-3,
        timestamp=pd.Timestamp("2022-01-12 10:00", tz="UTC"),
        state=None,
    )
    base.update(kw)
    return TradeProposal(**base)


def test_entry_filter_bool_matches_reason():
    arbiter = TradeArbiter(copy.deepcopy(DEFAULT_CFG["arbiter_cfg"]))
    hold = _proposal(0)
    assert arbiter.entry_filter_reason(hold) == "hold"
    assert not arbiter.passes_entry_filter(hold)
    for action in (1, 2, 3, 4):
        for mode in ("train", "eval"):
            p = _proposal(action, tmom_30m=1.0, di_dir_4h=0.5, adx_4h=0.1, brk_prob_4h=0.2)
            ok = arbiter.passes_entry_filter(p, mode)
            assert ok == (arbiter.entry_filter_reason(p, mode) == "pass")


# ── Environment ───────────────────────────────────────────────────────────────


def test_env_state_and_steps(models_root, scaled_df):
    _seed()
    ckpt = models_root / "Encoder/Majors/Checkpoint/Bootstrap/best_full_checkpoint.pth"
    enc, _, regime, trend, vol, outcome, tmom, rcont, mkt, soh = load_encoder_from_checkpoint(
        str(ckpt), device="cpu", seq_len=32
    )
    assert soh is not None and tmom is not None and rcont is not None and mkt is not None
    cfg = copy.deepcopy(DEFAULT_CFG["env_cfg"])
    cfg.update(TINY["env_cfg"])
    df = add_session_features(ensure_datetime_index_utc(scaled_df)).loc["2022-01-10":"2022-01-20"]
    cfg["encoder_feature_cols"] = build_encoder_feature_cols(df)
    env = MarketEnv(
        df,
        enc,
        regime,
        cfg,
        trend_head=trend,
        vol_head=vol,
        outcome_head=outcome,
        trend_mom_head=tmom,
        regime_cont_head=rcont,
        mktstr_head=mkt,
        strategy_outcome_head=soh,
    )
    state = env.reset()
    state = state[0] if isinstance(state, tuple) else state
    assert np.asarray(state).shape == (STATE_DIM,)
    for action in [1, 0, 0, 2, 0, 3, 0, 4] * 5:
        out = env.step(action)
        assert np.asarray(out[0]).shape == (STATE_DIM,)
        assert np.isfinite(np.asarray(out[0])).all() and np.isfinite(out[1])
        if out[2]:
            break


# ── Config and CLI ────────────────────────────────────────────────────────────


def test_build_cfg(tmp_path):
    cfg = build_cfg("run1_majors", {"rainbow_cfg": {"lr": 1e-4}, "seq_len": 64}, tmp_path)
    assert cfg["rainbow_cfg"]["lr"] == 1e-4
    assert cfg["rainbow_cfg"]["n_quant"] == DEFAULT_CFG["rainbow_cfg"]["n_quant"]  # one level deep
    assert cfg["seq_len"] == 64
    assert cfg["save_root"].startswith(str(tmp_path / "Rainbow" / "runs"))
    assert DEFAULT_CFG["rainbow_cfg"]["lr"] != 1e-4  # defaults untouched
    with pytest.raises(ValueError):
        build_cfg("run9")


def test_checkpoint_discovery(tmp_path):
    base = tmp_path / "Encoder" / "Minors" / "Checkpoint"
    for rel in [
        "Bootstrap/best_full_checkpoint.pth",
        "Retrain/WF2_2024H1/retrain_full_WF2_2024H1.pth",
        "Retrain/WF10_2028H1/retrain_full_WF10_2028H1.pth",
        "Retrain/WF3_2024H2/other.pth",  # not a full retrain checkpoint
    ]:
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).touch()
    found = find_encoder_checkpoints(tmp_path, "Minors")
    assert sorted(found) == [0, 2, 10]
    assert found[10].endswith("retrain_full_WF10_2028H1.pth")

    runs = tmp_path / "Rainbow" / "runs" / "Exploration" / "Minors"
    for wf in ("Minors_Bootstrap", "Minors_WF2", "Minors_WF10"):
        (runs / wf).mkdir(parents=True)
        (runs / wf / f"Minors_agent_{wf.split('_')[1]}.pth").touch()
    assert latest_agent_checkpoint(tmp_path, "Minors").endswith("Minors_agent_WF10.pth")
    assert latest_agent_checkpoint(tmp_path, "Majors") is None


def test_pipeline_cli_one_wf(data_root, models_root, tmp_path):
    """One tiny bootstrap WF through the CLI: trains, saves the agent, evaluates every asset."""
    _seed()
    torch.set_num_threads(1)
    config = tmp_path / "tiny.json"
    config.write_text(json.dumps(TINY))
    out = main(
        [
            "--run",
            "run1_majors",
            "--root",
            str(data_root),
            "--models-root",
            str(models_root),
            "--config",
            str(config),
        ]
    )
    assert "Majors" in out
    agent_ckpt = latest_agent_checkpoint(models_root, "Majors")  # where the joint runs look
    assert agent_ckpt and "Exploration" in agent_ckpt
    reloaded = RainbowAgent(STATE_DIM, _rainbow_cfg(), device="cpu")
    reloaded.load(agent_ckpt)
    summaries = glob.glob(str(models_root / "Rainbow" / "runs" / "**" / "*.json"), recursive=True)
    assert summaries
