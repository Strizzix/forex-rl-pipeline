"""
Agent checks on synthetic data built with data_management (no download, CPU only).

The encoder is a real one: one tiny bootstrap epoch of encoder.trainer, then
agent.strategy_head adds the StrategyOutcomeHead, as in a real run. So these tests also
check that the agent reads the encoder the way it was trained (inputs, heads, asset ids).
"""

import copy
import glob
import json
import logging
import os
import random

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from conftest import synth_1min  # noqa: E402

from agent.arbiter import TradeArbiter, TradeProposal  # noqa: E402
from agent.config import DEFAULT_CFG, build_cfg  # noqa: E402
from agent.env import (  # noqa: E402
    MarketEnv,
    encoder_inputs,
    ensure_datetime_index_utc,
    load_encoder_from_checkpoint,
)
from agent.pipeline import (  # noqa: E402
    find_encoder_checkpoints,
    latest_agent_checkpoint,
    main,
)
from agent.rainbow import RainbowAgent  # noqa: E402
from agent.strategy_head import (  # noqa: E402
    entry_lockout_mask,
    force_close_mask,
    trade_labels,
)
from agent.strategy_head import main as strategy_head_main  # noqa: E402
from data_management.downloader import build_features, make_bars  # noqa: E402
from data_management.scale import scale_and_save  # noqa: E402
from encoder.components import StrategyOutcomeHead  # noqa: E402
from encoder.dataset import (  # noqa: E402
    MarketPretrainConfig,
    MarketPretrainDataset,
    _load_all_scalers_file,
)

HIDDEN = 32  # gru_hidden of the test encoder
STATE_DIM = HIDDEN + 72  # embedding + head / market / position / SOH features (200 with 128)
TINY = {
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
    "env_cfg": {"device": "cpu", "hold_reward_warmup_steps": 50},
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


def _parquet(data_root, asset="EURUSD"):
    return glob.glob(str(data_root / "Scaled" / "Majors" / asset / "*_scaled_*.parquet"))[0]


@pytest.fixture(scope="module")
def models_root(tmp_path_factory, data_root):
    """Encoder bootstrap (1 epoch, structural features on) + the agent's strategy head."""
    from encoder.trainer import main as encoder_main

    root = tmp_path_factory.mktemp("models")
    boot = root / "Encoder" / "Majors" / "Checkpoint" / "Bootstrap"
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
        "bootstrap_val_end": "2022-02-20",
        "drift_anchor_start": "2021-12-15",
        "drift_anchor_end": "2021-12-31",
        "keepalive": False,
        "tcn_channels": [16, 16, 32],
        "gru_hidden": HIDDEN,
        "proj_dim": 32,
        "memory_bank_size": 256,
        "emb_max_points": 128,
        "drift_max_batches": 2,
        "drift_max_points": 64,
        "augment": {"use_gpu_augment": False},
        "trainer": {"device": "cpu"},
        "use_structural_features": True,
    }
    args = ["--class", "Majors", "--mode", "bootstrap", "--root", str(data_root)]
    args += ["--results-dir", str(boot), "--set"] + [
        f"{k}={json.dumps(v)}" for k, v in sets.items()
    ]
    encoder_main(args)
    done = strategy_head_main(
        [
            "--class",
            "Majors",
            "--root",
            str(data_root),
            "--models-root",
            str(root),
            "--set",
            "epochs=2",
            "stride=8",
            "device=cpu",
        ]
    )
    assert "Majors/Bootstrap" in done
    return root


def _agent_ckpt(models_root):
    return str(models_root / "Encoder" / "Majors" / "Agent" / "Bootstrap" / "encoder_agent.pth")


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


# ── Strategy outcome labels ───────────────────────────────────────────────────


def test_trade_labels_follow_the_env_rules():
    # entry at close=1.0, ATR 0.01, SL 1 ATR, TP 2 ATR, no spread
    close = np.ones(6)
    high = np.array([1.0, 1.005, 1.021, 1.0, 1.0, 1.0])
    low = np.array([1.0, 0.995, 0.999, 1.0, 1.0, 1.0])
    atr = np.full(6, 0.01)
    zero = np.zeros(6)
    no_fc = np.zeros(6, dtype=bool)
    hit, side = trade_labels(close, high, low, atr, zero, no_fc, 1.0, 2.0, +1)
    assert (hit[0], side[0]) == (1, 1)  # long: TP touched at bar 2, SL never
    hit, side = trade_labels(close, high, low, atr, zero, no_fc, 1.0, 2.0, -1)
    assert (hit[0], side[0]) == (1, 0)  # short: SL (1.01) touched at bar 2
    # both inside the same bar -> SL first
    h2, l2 = high.copy(), low.copy()
    h2[1], l2[1] = 1.03, 0.98
    hit, side = trade_labels(close, h2, l2, atr, zero, no_fc, 1.0, 2.0, +1)
    assert (hit[0], side[0]) == (1, 0)
    # force close at bar 1 before anything is touched -> timeout (hit 0)
    fc = no_fc.copy()
    fc[1] = True
    hit, side = trade_labels(close, high, low, atr, zero, fc, 1.0, 2.0, +1)
    assert (hit[0], side[0]) == (0, -1)
    # the data ends with the trade still open -> unknown
    assert hit[-1] == -1


def test_calendar_masks_match_the_env(data_root):
    df = ensure_datetime_index_utc(pd.read_parquet(_parquet(data_root))).iloc[:3000]
    env = MarketEnv.__new__(MarketEnv)
    env.timestamps = list(df.index.to_pydatetime())
    fc = [env._is_time_force_close(i) is not None for i in range(len(df))]
    lock = [env._is_entry_lockout(i) for i in range(len(df))]
    assert (force_close_mask(df.index) == np.array(fc)).all()
    assert (entry_lockout_mask(df.index) == np.array(lock)).all()


def test_strategy_head_loads_legacy_context():
    """v26 heads had a 29-dim context: they load and run with a zero context."""
    legacy = StrategyOutcomeHead(HIDDEN, ctx_dim=29)
    head = StrategyOutcomeHead.from_state_dict(legacy.state_dict(), HIDDEN)
    assert head.ctx_dim == 29
    head.eval()
    out = head.predict_all_combos(torch.zeros(3, HIDDEN))
    p = out["p_tp_trend_long"]
    assert p.shape == (3,) and ((p >= 0) & (p <= 1)).all()
    assert torch.allclose(out["p_tp_mr_short"] + out["p_sl_mr_short"], out["p_hit_mr_short"])


# ── Encoder alignment ─────────────────────────────────────────────────────────


def test_agent_checkpoint_has_the_heads(models_root):
    t = load_encoder_from_checkpoint(_agent_ckpt(models_root), device="cpu")
    enc, hidden, *_, tmom, rcont, mkt, soh = t
    assert hidden == HIDDEN
    assert all(m is not None for m in (tmom, rcont, mkt, soh))
    assert enc.input_meta and enc.input_meta["use_structural_features"]
    # the real heads: trend in [-3, 3] (tanh x 3), DI direction in [-1, 1], can be negative
    z = torch.randn(256, HIDDEN) * 5
    assert tmom(z).abs().max() <= 3.0
    di = rcont(z)[..., 3]
    assert di.abs().max() <= 1.0 and (di < 0).any()
    ckpt = torch.load(_agent_ckpt(models_root), map_location="cpu", weights_only=False)
    metrics = ckpt["extra"]["strategy_outcome"]["val_metrics"]
    assert set(metrics) == {"trend_long", "trend_short", "mr_long", "mr_short"}


def test_env_input_is_the_encoder_training_input(models_root, data_root):
    """Same columns, order, session and structural features as the encoder dataset."""
    enc = load_encoder_from_checkpoint(_agent_ckpt(models_root), device="cpu")[0]
    meta = enc.input_meta
    path = _parquet(data_root)
    scaler_path = os.path.join(os.path.dirname(path), "all_scalers.pkl")
    df = ensure_datetime_index_utc(pd.read_parquet(path))
    x_env = encoder_inputs(df, enc, "EURUSD", _load_all_scalers_file(scaler_path))

    raw = pd.read_parquet(path)
    raw.columns = [c.lower() for c in raw.columns]  # as the trainer does
    ds = MarketPretrainDataset(
        raw,
        MarketPretrainConfig(seq_len=32, step_stride=4, use_structural_features=True),
        meta["feature_cols"],
        asset_name="EURUSD",
        scaler_path=scaler_path,
        structural_stats=meta["structural_stats"]["EURUSD"],
    )
    assert x_env.shape == ds.data.shape
    np.testing.assert_allclose(x_env, ds.data, atol=1e-6)


def test_env_state_and_steps(models_root, data_root):
    _seed()
    t = load_encoder_from_checkpoint(_agent_ckpt(models_root), device="cpu")
    enc, _, regime, trend, vol, outcome, tmom, rcont, mkt, soh = t
    cfg = copy.deepcopy(DEFAULT_CFG["env_cfg"])
    cfg.update(TINY["env_cfg"])
    df = ensure_datetime_index_utc(pd.read_parquet(_parquet(data_root)))
    env = MarketEnv(
        df.loc["2022-01-10":"2022-01-20"],
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
        asset_name="EURUSD",
    )
    assert env.seq_len == 32  # the encoder's window, not the config default
    assert env.spread_col in env.df.columns  # the real spread, not the fixed fallback
    state = env.reset()
    state = state[0] if isinstance(state, tuple) else state
    assert np.asarray(state).shape == (STATE_DIM,)
    for action in [1, 0, 0, 2, 0, 3, 0, 4] * 5:
        out = env.step(action)
        assert np.asarray(out[0]).shape == (STATE_DIM,)
        assert np.isfinite(np.asarray(out[0])).all() and np.isfinite(out[1])
        if out[2]:
            break
    # the strategy head is live: P(TP) isn't the neutral 0.25 default
    assert env._strat_p_tp_tl != 0.25


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
    base = tmp_path / "Encoder" / "Minors" / "Agent"
    for rel in [
        "Bootstrap/encoder_agent.pth",
        "WF2_2024H1/encoder_agent.pth",
        "WF10_2028H1/encoder_agent.pth",
        "WF3_2024H2/strategy_outcome.json",  # not finished
    ]:
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).touch()
    found = find_encoder_checkpoints(tmp_path, "Minors")
    assert sorted(found) == [0, 2, 10]
    assert "WF10_2028H1" in found[10]

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
