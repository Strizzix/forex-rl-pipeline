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
from agent.market_calendar import swap_days  # noqa: E402
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


def test_arbiter_ranks_by_advantage_without_filters():
    arbiter = TradeArbiter(copy.deepcopy(DEFAULT_CFG["arbiter_cfg"]))
    # higher Q but a smaller advantage over HOLD ranks second; a trend trade against
    # the trend signals isn't filtered: the agent's state already has those signals
    a = _proposal(1, asset="EURUSD", q_value=0.9, q_hold=0.85, tmom_30m=-2.0, di_dir_4h=-0.8)
    b = _proposal(2, asset="GBPUSD", q_value=0.3, q_hold=0.10)
    approved = arbiter.select([a, b], remaining_trades=1)
    assert list(approved) == ["GBPUSD"] and b.rank == 0 and a.rank == 1
    assert a.reject_stage == "daily_quota"
    assert a.filter_check == "trend_long_wrong_sign"  # the verdict is still recorded
    assert arbiter.entry_filter_reason(a) == "pass"


def test_entry_filters_are_the_same_in_train_and_eval():
    cfg = copy.deepcopy(DEFAULT_CFG["arbiter_cfg"])
    cfg["entry_filters"] = True
    arbiter = TradeArbiter(cfg)
    hold = _proposal(0)
    assert arbiter.entry_filter_reason(hold) == "hold" and not arbiter.passes_entry_filter(hold)
    wrong = _proposal(1, tmom_30m=-1.0, adx_4h=0.5, di_dir_4h=0.5, regime_uncertainty=0.1)
    good = _proposal(1, tmom_30m=1.0, adx_4h=0.5, di_dir_4h=0.5, regime_uncertainty=0.1)
    for mode in ("train", "eval"):
        assert arbiter.entry_filter_reason(wrong, mode) == "trend_long_wrong_sign"
        assert arbiter.passes_entry_filter(good, mode)
    legacy = TradeArbiter({**cfg, "score_mode": "legacy"})
    assert legacy.score(good) != arbiter.score(good)


def test_portfolio_limits_become_action_masks():
    from agent.arbiter import MultiAssetRunner

    class _Env:
        position = None

    r = MultiAssetRunner.__new__(MultiAssetRunner)
    r.envs = {"EURUSD": _Env(), "GBPUSD": _Env()}
    r.arbiter = TradeArbiter(copy.deepcopy(DEFAULT_CFG["arbiter_cfg"]))
    r.block_new_trades, r.ks_enabled, r.train_daily_quota = False, False, False
    r.MAX_CONCURRENT, r.EVAL_MAX_CONCURRENT = 2, 1
    r.trades_opened_today = 99
    r.mode = "train"  # no portfolio quota in training by default
    assert r._portfolio_entry_state() == (True, "")
    r.mode = "eval"
    r.portfolio = type("P", (), {"daily_pnl_pct": 0.0, "daily_losses": 0})()
    r.DAILY_BLOCK_PCT, r.DAILY_REDUCE_PCT, r.MAX_DAILY_LOSSES = -0.02, -0.01, 3
    assert r._portfolio_entry_state() == (False, "daily_quota")
    r.trades_opened_today = 0
    r.envs["EURUSD"].position = {"dir": "LONG"}
    assert r._portfolio_entry_state() == (False, "position_slots")  # eval cap = 1


def test_replay_keeps_the_next_state_mask():
    _seed()
    cfg = _rainbow_cfg()
    cfg["n_step"] = 2
    agent = RainbowAgent(STATE_DIM, cfg, device="cpu")
    flat, in_pos = [True] * 5, [True, False, False, False, False]
    rng = np.random.default_rng(1)
    for k in range(70):
        s, ns = rng.normal(size=(2, STATE_DIM)).astype(np.float32)
        agent.store(
            s,
            1 if k % 2 else 0,
            0.1,
            ns,
            False,
            env_id="EURUSD",
            next_mask=in_pos if k % 2 else flat,
        )
    batch = agent.buffer.sample(32)
    next_masks = batch[5]
    assert next_masks.shape == (32, 5) and next_masks[:, 0].all()
    assert (~next_masks[:, 1:]).any() and next_masks.all(dim=1).any()
    assert np.isfinite(agent.train_step())
    left = agent._nstep_bufs["EURUSD"].flush()  # end of the data: what's still waiting
    assert all(len(x) == 7 for x in left)


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
    # the action mask is the same in training: no new trade with a position open
    env.training_mode = "train"
    env.position = env.position or {"dir": "LONG", "entry": 1.0, "sl": 0.9, "tp": 1.2}
    assert env.get_valid_action_mask() == [True, False, False, False, False]
    env.position = None
    lockout = [i for i, ts in enumerate(env.timestamps) if ts.weekday() == 2 and ts.hour == 21]
    env.current_step = lockout[0]  # Wednesday 21:00 UTC: before the force close
    assert env.get_valid_action_mask() == [True, False, False, False, False]


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
    summaries = glob.glob(
        str(models_root / "Rainbow" / "runs" / "**" / "eval_summary_*.json"), recursive=True
    )
    assert summaries
    report = json.load(open(summaries[0]))["counterfactual"]
    assert "decisions" in report  # the agent's eval decisions replayed


# ── Execution, P&L and calendar ───────────────────────────────────────────────


def _bare_env(index, close, atr=0.001, spread=0.0, **cfg_kw):
    """An env without encoder on a hand-made price path (only the trading mechanics)."""
    close = np.asarray(close, dtype=float)
    df = pd.DataFrame(
        {
            "close_15min_raw": close,
            "high_15min_raw": close,
            "low_15min_raw": close,
            "ATR14_15min_raw": np.full(len(close), atr),
            "spread_mean_15min_raw": np.full(len(close), spread),
        },
        index=index,
    )
    cfg = copy.deepcopy(DEFAULT_CFG["env_cfg"])
    cfg.update({"device": "cpu", "seq_len": 4, "slippage_noise": 0.0, **cfg_kw})
    env = MarketEnv(df, None, None, cfg)
    env.reset(start_idx=4)
    return env


@pytest.mark.parametrize(
    "day, close_utc, lockout_utc, rollover_utc",
    [("2024-01-10", "21:30", "20:30", "22:00"), ("2024-07-10", "20:30", "19:30", "21:00")],
)
def test_calendar_follows_the_new_york_rollover(day, close_utc, lockout_utc, rollover_utc):
    idx = pd.date_range(f"{day} 18:00", periods=24, freq="15min", tz="UTC")  # a Wednesday
    hhmm = idx.strftime("%H:%M")
    assert hhmm[force_close_mask(idx)][0] == close_utc
    assert hhmm[entry_lockout_mask(idx)][0] == lockout_utc
    swaps = swap_days(idx)
    assert hhmm[swaps > 0].tolist() == [rollover_utc] and swaps.max() == 3.0  # Wednesday x3
    # Friday in summer: the force close comes before the weekly close (21:00 UTC)
    fri = pd.date_range("2024-07-12 19:00", periods=8, freq="15min", tz="UTC")
    assert fri.strftime("%H:%M")[force_close_mask(fri)][0] == "20:30"


def test_force_close_books_the_real_result_net_of_costs():
    idx = pd.date_range("2024-07-10 18:00", periods=16, freq="15min", tz="UTC")
    env = _bare_env(idx, np.linspace(1.0, 1.0015, 16))
    env.step(1)  # long at 1.0004, SL 1 ATR = 0.0010 below
    pos = dict(env.position)
    for _ in range(10):
        _, reward, _, info = env.step(0)
        if env.position is None:
            break
    trade = env.episode_trades[-1]
    assert info["closed_trade_result"] == "wednesday_swap"
    assert trade["rr"] == pytest.approx(0.6, abs=1e-6)  # (1.0010 - 1.0004) / 0.0010
    # the P&L is net of the commission; the reward carries the trade's R
    assert trade["pnl"] == pytest.approx(0.6 * pos["risk_amount"] - pos["commission"])
    assert reward > 0.4
    assert env.balance == pytest.approx(env.initial_capital + trade["pnl"])


def test_open_trade_features_are_in_r():
    idx = pd.date_range("2024-01-08 10:00", periods=12, freq="15min", tz="UTC")
    close = np.r_[np.full(5, 1.0), 1.0005, 1.0005, 0.9995, np.full(4, 0.9995)]
    env = _bare_env(idx, close)
    env.step(1)
    env.step(0)  # +0.5 R
    assert env.position["max_favorable_excursion"] == pytest.approx(0.5)
    env.step(0)
    env.step(0)  # -0.5 R
    assert env.position["max_adverse_excursion"] == pytest.approx(0.5)
    assert env._open_r(1.0005) == pytest.approx(0.5)


def test_mtm_shaping_cancels_over_a_trade():
    idx = pd.date_range("2024-01-08 10:00", periods=20, freq="15min", tz="UTC")
    env = _bare_env(idx, np.ones(20), mtm_scale=1.0, gamma=1.0)
    env.position = {"dir": "LONG", "entry": 1.0, "sl": 0.999, "tp": 1.002, "risk_dist": 0.001}
    total = 0.0
    for k, price in enumerate([1.0003, 1.0008, 0.9996, 1.0001]):
        env.df.iloc[5 + k, env.df.columns.get_loc("close_15min_raw")] = price
        total += env._mtm_reward(5 + k)
    env.position = None
    total += env._mtm_reward(9)
    assert total == pytest.approx(0.0, abs=1e-12)


def test_swap_is_charged_at_the_rollover():
    idx = pd.date_range("2024-01-09 21:00", periods=8, freq="15min", tz="UTC")  # Tuesday
    env = _bare_env(idx, np.ones(8))
    env.reset(start_idx=1)
    env.step(1)
    balance = env.balance
    for _ in range(5):
        env.step(0)
    assert env.position["swap_usd"] > 0  # held through 22:00 UTC (17:00 New York)
    assert env.balance == pytest.approx(balance - env.position["swap_usd"])


def test_reward_clip_keeps_the_trade_economics():
    agent = RainbowAgent(STATE_DIM, _rainbow_cfg(), device="cpu")
    s = np.zeros(STATE_DIM, dtype=np.float32)
    agent.store(s, 1, 1.8, s, True, tag="tp", env_id="A")
    agent.store(s, 3, -1.0, s, True, tag="sl", env_id="B")
    assert sorted(agent.buffer._rewards[:2].tolist()) == pytest.approx([-1.0, 1.8])
