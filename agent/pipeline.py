"""
pipeline.py — walk-forward training / evaluation of the Rainbow DQN agent.

Rainbow DQN (QR-DQN + Dueling + PER + N-step + Noisy) on top of the frozen market encoder:
 one encoder per WF ({wf_idx: checkpoint})
 off-policy RainbowAgent: store -> PER -> train_step every few steps
 per-asset N-step returns (sequences never mixed)
 curriculum (max_trades 2 -> 4 -> 5) and progressive DD schedules
 dense reward: mark-to-market + opportunity cost
 deterministic eval: NoisyLinear -> mean weights, no noise
 separate runs per class (Run1) or a joint Majors + Minors portfolio (Run2 / Run3)

Run: python -m agent.pipeline --help
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from dateutil.relativedelta import relativedelta
from tqdm.auto import tqdm

from .arbiter import MultiAssetRunner, TradeArbiter
from .config import DATA_ROOT, DEFAULT_CFG, MODELS_ROOT, RUN_CONFIGS, build_cfg
from .env import (
    MarketEnv,
    add_session_features,
    build_encoder_feature_cols,
    ensure_datetime_index_utc,
    load_encoder_from_checkpoint,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _setup_file_logging(log_path: str):
    """Adds a FileHandler to the root logger, so every log also goes to a txt file."""
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info("Log file: %s", log_path)


# ============================================================
# KEEPALIVE THREAD
# ============================================================
class KeepaliveThread:
    def __init__(self, interval_seconds: int = 300):
        self.interval_seconds = int(interval_seconds)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.available = False
        try:
            import pyautogui

            self._pyautogui = pyautogui
            try:
                self._pyautogui.FAILSAFE = True
            except Exception:
                pass
            self.available = True
        except Exception as e:
            self._pyautogui = None
            logger.warning("Keepalive disabled: pyautogui not available (%s)", e)

    def start(self):
        if not self.available:
            return
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _press_shift_once(self):
        try:
            self._pyautogui.keyDown("shift")
            time.sleep(0.05)
            self._pyautogui.keyUp("shift")
        except Exception:
            pass

    def _run(self):
        try:
            time.sleep(min(5.0, float(self.interval_seconds) * 0.1))
        except Exception:
            pass
        while not self._stop.is_set():
            try:
                self._press_shift_once()
            except Exception:
                pass
            slept = 0
            while slept < self.interval_seconds and not self._stop.is_set():
                time.sleep(min(5, self.interval_seconds - slept))
                slept += min(5, self.interval_seconds - slept)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def _years_to_relativedelta(y: float):
    yf = float(y)
    if math.isclose(yf, round(yf)):
        return relativedelta(years=int(round(yf)))
    months = int(round(yf * 12))
    return relativedelta(months=months)


def generate_walk_forward_splits(
    start_date: str,
    end_date: str,
    train_years: float = 5,
    test_years: float = 1,
    step_years: float = 1,
):
    splits = []
    current_start = pd.to_datetime(start_date)
    if current_start.tzinfo is None:
        current_start = current_start.tz_localize("UTC")
    else:
        current_start = current_start.tz_convert("UTC")
    final_end = pd.to_datetime(end_date)
    if final_end.tzinfo is None:
        final_end = final_end.tz_localize("UTC")
    else:
        final_end = final_end.tz_convert("UTC")
    delta_train = _years_to_relativedelta(train_years)
    delta_test = _years_to_relativedelta(test_years)
    step = _years_to_relativedelta(step_years)
    while True:
        train_start = current_start
        train_end = train_start + delta_train - pd.Timedelta(minutes=15)
        test_start = train_end + pd.Timedelta(minutes=15)
        test_end = test_start + delta_test - pd.Timedelta(minutes=15)
        if test_end > final_end:
            break
        splits.append({"train": (train_start, train_end), "test": (test_start, test_end)})
        current_start = current_start + step
    return splits


def scan_directory_for_assets(data_dir: str) -> dict[str, str]:
    assets = {}
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    for entry in os.listdir(data_dir):
        p = os.path.join(data_dir, entry)
        if os.path.isfile(p) and (entry.endswith(".parquet") or entry.endswith(".csv")):
            asset = os.path.splitext(entry)[0]
            assets[asset] = p
    for entry in os.listdir(data_dir):
        folder = os.path.join(data_dir, entry)
        if os.path.isdir(folder):
            candidates = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith(".parquet") or f.endswith(".csv")
            ]
            if candidates:
                assets[entry] = sorted(candidates)[0]
    return assets


def aggregate_class_metrics(asset_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not asset_summaries:
        return {
            "assets": [],
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "pnl": 0.0,
            "max_drawdown": 0.0,
            "avg_dd": 0.0,
            "max_daily_dd": 0.0,
            "avg_daily_dd": 0.0,
            "avg_rr": 0.0,
            "avg_trades_per_asset": 0.0,
        }
    assets = list(asset_summaries.keys())
    trades = sum(int(a.get("trades", 0)) for a in asset_summaries.values())
    wins = sum(int(a.get("wins", 0)) for a in asset_summaries.values())
    profit = sum(float(a.get("sum_profit", 0.0)) for a in asset_summaries.values())
    loss = sum(float(a.get("sum_loss", 0.0)) for a in asset_summaries.values())
    pnl = float(profit - loss)
    win_rate = (wins / trades) if trades > 0 else 0.0
    profit_factor = (profit / (loss + 1e-12)) if loss > 0 else (float("inf") if profit > 0 else 0.0)
    max_drawdown = max(
        (float(a.get("drawdown", 0.0)) for a in asset_summaries.values()), default=0.0
    )
    avg_dd = float(np.mean([float(a.get("avg_dd", 0.0)) for a in asset_summaries.values()]))
    max_daily_dd = max(
        (float(a.get("max_daily_dd", 0.0)) for a in asset_summaries.values()), default=0.0
    )
    avg_daily_dd = float(
        np.mean([float(a.get("avg_daily_dd", 0.0)) for a in asset_summaries.values()])
    )
    avg_rr = float(
        np.mean([float(a.get("avg_rr_realized", 0.0)) for a in asset_summaries.values()])
    )
    avg_trades_per_asset = float(trades) / max(1, len(asset_summaries))
    return {
        "assets": assets,
        "trades": int(trades),
        "wins": int(wins),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "pnl": float(pnl),
        "max_drawdown": float(max_drawdown),
        "avg_dd": float(avg_dd),
        "max_daily_dd": float(max_daily_dd),
        "avg_daily_dd": float(avg_daily_dd),
        "avg_rr": float(avg_rr),
        "avg_trades_per_asset": float(avg_trades_per_asset),
    }


def compute_global_metrics(
    asset_summaries, portfolio, test_start, test_end, envs
) -> dict[str, Any]:
    total_trades = sum(int(v.get("trades", 0)) for v in asset_summaries.values())
    total_wins = sum(int(v.get("wins", 0)) for v in asset_summaries.values())
    total_losses = total_trades - total_wins
    total_profit = sum(float(v.get("sum_profit", 0.0)) for v in asset_summaries.values())
    total_loss = sum(float(v.get("sum_loss", 0.0)) for v in asset_summaries.values())
    pnl = float(total_profit - total_loss)
    win_rate = (total_wins / total_trades) if total_trades > 0 else 0.0
    profit_factor = (
        (total_profit / (total_loss + 1e-12))
        if total_loss > 0
        else (float("inf") if total_profit > 0 else 0.0)
    )
    initial_capital = (
        float(getattr(portfolio, "initial", getattr(portfolio, "initial_capital", 100000.0)))
        if portfolio
        else 100000.0
    )
    final_balance = (
        float(getattr(portfolio, "balance", initial_capital)) if portfolio else initial_capital
    )
    equity_curve = np.array(
        getattr(portfolio, "equity_curve", [initial_capital]) if portfolio else [initial_capital],
        dtype=np.float64,
    )
    running_max = np.maximum.accumulate(equity_curve)
    dd_trailing = np.maximum(0.0, (running_max - equity_curve) / np.maximum(1e-12, running_max))
    max_dd_trailing = float(np.max(dd_trailing)) if dd_trailing.size else 0.0
    avg_dd_trailing = float(np.mean(dd_trailing)) if dd_trailing.size else 0.0
    dd_static = np.maximum(0.0, (initial_capital - equity_curve) / max(1e-12, initial_capital))
    max_dd_static = float(np.max(dd_static)) if dd_static.size else 0.0
    asset_max_daily = [float(v.get("max_daily_dd", 0.0)) for v in asset_summaries.values()]
    asset_avg_daily = [float(v.get("avg_daily_dd", 0.0)) for v in asset_summaries.values()]
    max_daily_dd = float(max(asset_max_daily)) if asset_max_daily else 0.0
    avg_daily_dd = float(np.mean(asset_avg_daily)) if asset_avg_daily else 0.0
    expectancy = 0.0
    if total_trades > 0:
        avg_win = (total_profit / max(1, total_wins)) if total_wins > 0 else 0.0
        avg_loss = (total_loss / max(1, total_losses)) if total_losses > 0 else 0.0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    return_over_trailing_dd = (
        pnl / (initial_capital * max_dd_trailing) if max_dd_trailing > 0 else 0.0
    )
    return_over_max_dd = pnl / (initial_capital * max_dd_static) if max_dd_static > 0 else 0.0
    if equity_curve.size > 1:
        returns = np.diff(equity_curve) / np.maximum(1e-12, equity_curve[:-1])
        equity_volatility = float(np.std(returns))
        sharpe_like = float(np.mean(returns) / (np.std(returns) + 1e-12))
    else:
        equity_volatility = 0.0
        sharpe_like = 0.0
    days_with_0_trades = days_with_5_trades = 0
    try:
        start = pd.to_datetime(test_start).date()
        end = pd.to_datetime(test_end).date()
        all_dates = pd.date_range(start, end, freq="D").date
        trade_counts = {d: 0 for d in all_dates}
        for _, env in (envs or {}).items():
            for tr in getattr(env, "episode_trades", []):
                ex_idx = tr.get("exit_idx", None)
                if ex_idx is None:
                    continue
                try:
                    ts = env.timestamps[ex_idx]
                    d = pd.to_datetime(ts).date()
                    if d in trade_counts:
                        trade_counts[d] += 1
                except Exception:
                    continue
        days_with_0_trades = sum(1 for c in trade_counts.values() if c == 0)
        days_with_5_trades = sum(1 for c in trade_counts.values() if c >= 5)
    except Exception:
        pass

    # v26+: force-close statistics
    fc_count = fc_pnl = fc_rr_sum = swap_total_r = 0
    wed_count = fri_count = 0
    for _, env in (envs or {}).items():
        fc_count += getattr(env, "_force_close_count", 0)
        fc_pnl += getattr(env, "_force_close_pnl", 0.0)
        fc_rr_sum += getattr(env, "_force_close_rr_sum", 0.0)
        for tr in getattr(env, "episode_trades", []):
            if tr.get("is_force_close"):
                reason = tr.get("force_close_reason", "")
                if reason == "wednesday_swap":
                    wed_count += 1
                elif reason == "friday_weekend":
                    fri_count += 1
                swap_total_r += float(tr.get("swap_cost_r", 0.0))
    fc_avg_rr = fc_rr_sum / max(1, fc_count)

    return {
        "initial_capital": float(initial_capital),
        "final_balance": float(final_balance),
        "pnl": float(pnl),
        "profit_factor": float(profit_factor),
        "win_rate": float(win_rate),
        "expectancy_per_trade": float(expectancy),
        "max_dd_static": float(max_dd_static),
        "max_daily_dd": float(max_daily_dd),
        "avg_daily_dd": float(avg_daily_dd),
        "return_over_max_dd": float(return_over_max_dd),
        "max_dd_trailing": float(max_dd_trailing),
        "avg_dd_trailing": float(avg_dd_trailing),
        "return_over_trailing_dd": float(return_over_trailing_dd),
        "equity_volatility": float(equity_volatility),
        "sharpe_like": float(sharpe_like),
        "total_trades": int(total_trades),
        "days_with_0_trades": int(days_with_0_trades),
        "days_with_5_trades": int(days_with_5_trades),
        # v26+: force-close and swap tracking
        "force_close_count": int(fc_count),
        "force_close_pnl": float(fc_pnl),
        "force_close_avg_rr": float(fc_avg_rr),
        "force_close_wed": int(wed_count),
        "force_close_fri": int(fri_count),
        "swap_total_cost_r": float(swap_total_r),
    }


# ============================================================
# AGENT FACTORY
# ============================================================


def _make_agent(state_dim: int, cfg: dict[str, Any]) -> Any:
    """Creates a RainbowAgent with the parameters of cfg['rainbow_cfg']."""
    from .rainbow import RainbowAgent

    device = cfg.get("device", "cpu")
    agent_cfg = cfg.get("rainbow_cfg", {})
    agent = RainbowAgent(state_dim, agent_cfg, device=device)
    logger.info("Created RainbowAgent: state_dim=%d device=%s", state_dim, device)
    return agent


def _rainbow_train_loop(
    train_envs: dict[str, MarketEnv],
    agents: dict[str, any],
    asset_class_map: dict[str, str],
    arbiter: TradeArbiter,
    runner_cfg: dict[str, Any],
    cfg: dict[str, Any],
    max_steps: int,
    freeze_target_steps: int,
    pbar: any,
) -> MultiAssetRunner:
    """Rainbow DQN training loop: off-policy, store -> PER -> train_step every few steps."""

    logger.info("[TRAIN_LOOP] max_steps=%d freeze=%d", max_steps, freeze_target_steps)
    train_steps_per_update = 4
    _train_runner_cfg = runner_cfg.copy()
    _train_runner_cfg["min_q_gap"] = 0.0  # forced to 0 in training
    _train_runner_cfg["min_q_gap_train"] = 0.0
    runner = MultiAssetRunner(train_envs, asset_class_map, agents, arbiter, _train_runner_cfg)
    runner.max_steps = max_steps  # needed by _resolve_schedule (fraction schedules)
    global_step = 0

    # ── LR warmup scheduler ──────────────────────────────────────────
    warm_start_cfg = cfg.get("warm_start_cfg", {})
    _lr_warmup_steps = int(warm_start_cfg.get("lr_warmup_steps", 0))
    _lr_warmup_scale = float(warm_start_cfg.get("lr_warmup_scale", 1.0))
    _lr_target_scale = float(warm_start_cfg.get("lr_scale", 1.0))
    _base_lr = float(cfg.get("rainbow_cfg", {}).get("lr", 3e-4))
    _lr_warmup_active = _lr_warmup_steps > 0 and _lr_warmup_scale < _lr_target_scale

    def _update_lr(step: int):
        """Scales the LR linearly from lr_warmup_scale to lr_scale over lr_warmup_steps."""
        if not _lr_warmup_active or step > _lr_warmup_steps:
            return
        frac = step / max(1, _lr_warmup_steps)
        scale = _lr_warmup_scale + frac * (_lr_target_scale - _lr_warmup_scale)
        new_lr = _base_lr * scale
        for agent in agents.values():
            for pg in agent.optimizer.param_groups:
                pg["lr"] = new_lr

    # ── metric accumulators for the periodic log ──────────────────────
    _log_every = cfg.get("log_every_steps", 5000)
    _step_rewards = []
    _q_gaps = []
    _trades_tp = 0
    _trades_sl = 0
    _trades_hold = 0

    for asset, env in train_envs.items():
        logger.info(
            "[TRAIN_CHECK] %s: df_len=%d current_step=%d remaining=%d",
            asset,
            len(env.df),
            env.current_step,
            len(env.df) - 1 - env.current_step,
        )

    while global_step < max_steps:
        if not any(env.current_step < len(env.df) - 1 for env in train_envs.values()):
            logger.info("All training envs exhausted at step %d", global_step)
            break

        try:
            transitions, approved = runner.step_once()
        except Exception as e:
            logger.exception("step_once crashed at global_step=%d: %s", global_step, e)
            raise

        for asset, tr in transitions.items():
            s = tr["state"]
            a = tr["action"]
            r = tr["reward"]
            ns = tr["next_state"]
            done = tr["done"]
            info = tr.get("info", {}) or {}
            tag = "hold"
            if info.get("closed_trade_result") is not None:
                result = "tp" if info["closed_trade_result"] == "tp" else "sl"
                if result == "tp":
                    _trades_tp += 1
                else:
                    _trades_sl += 1
                # ── rank-aware tags for the elite buffer and the PER boost ──
                # prop.rank is 0-indexed (rank=0 -> rank1, rank=1 -> rank2, rank=2 -> rank3)
                # the "rank1_tp", "rank1_sl", ... tags turn on automatically:
                #   - PER priority boost via _TAG_PRIORITY_MULT in PrioritizedReplayBuffer
                #   - the elite buffer insert in agent.store()
                prop_r = tr.get("proposal")
                _rank = getattr(prop_r, "rank", None)
                if _rank is not None and _rank <= 1:  # rank 0 or 1 = rank1/rank2
                    tag = f"rank{_rank + 1}_{result}"  # "rank1_tp", "rank2_sl", ...
                else:
                    tag = result  # "tp" / "sl" base for rank3+
            else:
                _trades_hold += 1
            _step_rewards.append(float(r))
            agents[asset_class_map[asset]].store(
                torch.from_numpy(s.copy()),
                a,
                r,
                torch.from_numpy(ns.copy()),
                done,
                tag=tag,
                env_id=asset,
            )
            # Q-gap: difference between the best and the worst Q (conviction)
            prop = tr.get("proposal")
            if prop is not None and getattr(prop, "q_vals_masked", None) is not None:
                qv = np.array(prop.q_vals_masked, dtype=np.float32)
                if len(qv) >= 2:
                    _q_gaps.append(float(np.max(qv) - np.min(qv)))

        if global_step % train_steps_per_update == 0:
            _update_lr(global_step)
            for agent in agents.values():
                agent.train_step(freeze_target_steps=freeze_target_steps)

        # ── Periodic log ──────────────────────────────────────────────
        if global_step % _log_every == 0 and global_step > 0:
            n_closed = _trades_tp + _trades_sl
            win_rate = _trades_tp / n_closed if n_closed > 0 else 0.0
            avg_rew = float(np.mean(_step_rewards)) if _step_rewards else 0.0
            avg_qgap = float(np.mean(_q_gaps)) if _q_gaps else 0.0
            _agent0 = next(iter(agents.values()))
            buf_size = len(_agent0.buffer) if hasattr(_agent0, "buffer") else -1
            logger.info(
                "[TRAIN] step=%d | wr=%.1f%% (%dTP/%dSL) | "
                "avg_rew=%+.4f | avg_qgap=%.3f | buf=%d | active=%d",
                global_step,
                win_rate * 100,
                _trades_tp,
                _trades_sl,
                avg_rew,
                avg_qgap,
                buf_size,
                sum(env.current_step < len(env.df) - 1 for env in train_envs.values()),
            )
            pbar.set_postfix(
                {
                    "wr": f"{win_rate*100:.1f}%",
                    "rew": f"{avg_rew:+.4f}",
                    "qgap": f"{avg_qgap:.3f}",
                }
            )
            # reset the accumulators
            _step_rewards.clear()
            _q_gaps.clear()
            _trades_tp = _trades_sl = _trades_hold = 0

        global_step += 1
        pbar.update(1)

    # ── flush the leftover NStepBuffers of every asset (OUTSIDE the while) ──
    for asset in train_envs.keys():
        agent = agents[asset_class_map[asset]]
        nsbuf = agent._nstep_bufs.get(asset)
        if nsbuf and len(nsbuf._buf) > 0:
            buf_list = list(nsbuf._buf)
            for i in range(len(buf_list)):
                s_i, a_i, _, _, _, tag_i = buf_list[i]
                R = 0.0
                actual_ns = buf_list[-1][3]
                actual_done = True
                for j, (_, _, r_j, ns_j, d_j, _) in enumerate(buf_list[i:]):
                    R += (agent.gamma**j) * r_j
                    actual_ns = ns_j
                    actual_done = d_j
                    if d_j:
                        break
                agent.buffer.add(s_i, a_i, R, actual_ns, actual_done, tag_i)
            nsbuf.reset()
            logger.info(
                "[NSTEP] flushed %d leftover transitions for asset %s", len(buf_list), asset
            )

    return runner


# ============================================================
# EVAL LOOP
# ============================================================


def _eval_loop(
    test_envs: dict[str, MarketEnv],
    eval_agents: dict[str, any],
    asset_class_map: dict[str, str],
    arbiter_eval: TradeArbiter,
    eval_runner_cfg: dict[str, Any],
    pbar: any,
) -> MultiAssetRunner:
    """Eval loop, epsilon=0, deterministic."""
    eval_runner = MultiAssetRunner(
        test_envs, asset_class_map, eval_agents, arbiter_eval, eval_runner_cfg
    )
    max_eval_steps = max(len(env.df) for env in test_envs.values())
    eval_runner.max_steps = max_eval_steps  # per _resolve_schedule in eval
    global_eval_step = 0

    while global_eval_step < max_eval_steps:
        if not any(env.current_step < len(env.df) - 1 for env in test_envs.values()):
            break
        eval_runner.step_once()
        pbar.update(1)
        global_eval_step += 1

    return eval_runner


# ============================================================
# MAIN WALK-FORWARD
# ============================================================


def run_walk_forward_for_asset_map(
    class_name: str,
    asset_map: dict[str, str],
    encoder_ckpt: str | dict[int, str],
    cfg: dict[str, Any],
    do_train: bool = True,
) -> dict:
    """
    encoder_ckpt can be:
      - str  -> the same encoder for every WF (backward compatible)
      - dict -> {wf_idx: path}, a different encoder per WF

    cfg can hold "class_overrides": {"Majors": {...}, "Minors": {...}}
    for per-class overrides (e.g. different min_q_gap in joint mode).
    The overrides are applied before anything else.
    """
    # ── apply class_overrides if present ────────────────────────────────────
    class_overrides = cfg.get("class_overrides", {})
    if class_name in class_overrides:
        cfg = cfg.copy()
        for section_key, section_val in class_overrides[class_name].items():
            if isinstance(section_val, dict) and isinstance(cfg.get(section_key), dict):
                merged = cfg[section_key].copy()
                merged.update(section_val)
                cfg[section_key] = merged
            else:
                cfg[section_key] = section_val
        logger.info(
            "[%s] class_overrides applied: %s", class_name, list(class_overrides[class_name].keys())
        )
    logger.info(
        "Walk-forward [%s] | %d assets | agent_type=%s", class_name, len(asset_map), "rainbow"
    )

    # ── log: one file per WF, saved in the WF folder ──────────────────────
    # no global file outside any more — every WF has its own file.
    import datetime

    run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _current_wf_fh = None  # current handler, closed before moving to the next WF

    # ── Build the splits ─────────────────────────────────────────────────────
    wf_mode = cfg.get("wf_mode", "auto")
    if wf_mode == "manual" and cfg.get("wf_splits"):
        raw_splits = cfg["wf_splits"]
        splits = []
        for s in raw_splits:
            ts = pd.to_datetime(s["train"][0]).tz_localize("UTC")
            te = pd.to_datetime(s["train"][1]).tz_localize("UTC")
            vs = pd.to_datetime(s["test"][0]).tz_localize("UTC")
            ve = pd.to_datetime(s["test"][1]).tz_localize("UTC")
            splits.append(
                {
                    "train": (ts, te),
                    "test": (vs, ve),
                    "name": s.get("name", f"WF{len(splits)+1}"),
                    "encoder_wf_key": s.get("encoder_wf_key", len(splits)),
                    # val/OOS: taken from the config — saves val_summary_{wf_name}.json
                    "val": (
                        (
                            pd.to_datetime(s["val"][0]).tz_localize("UTC"),
                            pd.to_datetime(s["val"][1]).tz_localize("UTC"),
                        )
                        if "val" in s and s["val"] is not None
                        else None
                    ),
                }
            )
        logger.info("[WF] MANUAL mode: %d splits", len(splits))
    else:
        wf_cfg = cfg.get("wf", DEFAULT_CFG["wf"])
        splits = generate_walk_forward_splits(
            wf_cfg["start_date"],
            wf_cfg["end_date"],
            wf_cfg["train_window_years"],
            wf_cfg["test_window_years"],
            wf_cfg["step_years"],
        )
        # Adds encoder_wf_key = wf_idx (1-based) automatically
        for i, s in enumerate(splits):
            s["name"] = f"WF{i+1}"
            s["encoder_wf_key"] = i + 1
        logger.info("[WF] AUTO mode: %d splits", len(splits))
    os.makedirs(cfg.get("save_root", DEFAULT_CFG["save_root"]), exist_ok=True)

    arbiter_cfg = cfg.get("arbiter_cfg", DEFAULT_CFG["arbiter_cfg"]).copy()
    results_all = {}

    env_device = cfg.get("env_cfg", {}).get("device", "cpu")

    # encoder cache, no reload if the checkpoint doesn't change between WFs
    _encoder_cache: dict[str, tuple] = {}

    def _get_encoder_for_wf(wf_idx: int, encoder_wf_key: int = None):
        """Resolves the right checkpoint.
        encoder_wf_key: explicit key from the split (0=bootstrap, 1=WF1, ...)
        wf_idx: fallback if encoder_wf_key is not given
        """
        key = encoder_wf_key if encoder_wf_key is not None else wf_idx
        if isinstance(encoder_ckpt, str):
            ckpt_path = encoder_ckpt
        else:
            valid_keys = [k for k in encoder_ckpt if k <= key]
            if not valid_keys:
                raise ValueError(
                    f"No encoder for key={key} "
                    f"(split WF{wf_idx}). "
                    f"Available keys: {sorted(encoder_ckpt.keys())}"
                )
            ckpt_path = encoder_ckpt[max(valid_keys)]
        if ckpt_path not in _encoder_cache:
            logger.info("[ENCODER] loading the checkpoint for WF%d: %s", wf_idx, ckpt_path)
            # v24: extended tuple — (enc, emb_dim, regime_head, trend_head, vol_head,
            #                       outcome_head, trend_mom_head, regime_cont_head, mktstr_head)
            enc_tuple = load_encoder_from_checkpoint(
                ckpt_path,
                device=env_device,
                seq_len=cfg.get("env_cfg", {}).get("seq_len", 128),
                load_regime_head=cfg.get("use_regime_head", True),
                load_trend_head=cfg.get("use_trend_head", True),
                load_vol_head=cfg.get("use_vol_head", True),
                load_outcome_head=cfg.get("use_outcome_head", True),
            )
            enc, edim, rh, th, vh, oh = enc_tuple[:6]
            tmh = enc_tuple[6] if len(enc_tuple) > 6 else None  # trend_mom_head
            rch = enc_tuple[7] if len(enc_tuple) > 7 else None  # regime_cont_head
            msh = enc_tuple[8] if len(enc_tuple) > 8 else None  # mktstr_head
            soh = enc_tuple[9] if len(enc_tuple) > 9 else None  # strategy_outcome_head (v26)
            for m in (enc, rh, th, vh, oh, tmh, rch, msh, soh):
                if m is not None:
                    m.eval()
            _encoder_cache[ckpt_path] = enc_tuple
            logger.info(
                "[ENCODER] WF%d → %s (emb_dim=%d, tmh=%s, rch=%s, msh=%s, soh=%s)",
                wf_idx,
                os.path.basename(ckpt_path),
                edim,
                tmh is not None,
                rch is not None,
                msh is not None,
                soh is not None,
            )
        else:
            logger.info("[ENCODER] WF%d → cache hit: %s", wf_idx, os.path.basename(ckpt_path))
        return _encoder_cache[ckpt_path]

    _prev_agent_ckpt: str | None = None  # agent checkpoint of the previous WF

    for wf_idx, split in enumerate(splits, start=1):
        train_start, train_end = split["train"]
        test_start, test_end = split["test"]
        wf_name = split.get("name", f"WF{wf_idx}")
        enc_key = split.get("encoder_wf_key", wf_idx)

        logger.info(
            "WF %d [%s]: train %s → %s | test %s → %s | encoder_key=%d",
            wf_idx,
            wf_name,
            train_start.date(),
            train_end.date(),
            test_start.date(),
            test_end.date(),
            enc_key,
        )

        # ── per-WF log file: one file per WF, saved in its folder ────────
        # close the log of the previous WF before opening the new one.
        if _current_wf_fh is not None:
            try:
                logging.getLogger().removeHandler(_current_wf_fh)
                _current_wf_fh.close()
            except Exception:
                pass
        wf_log_dir = os.path.join(
            cfg.get("save_root", DEFAULT_CFG["save_root"]), f"{class_name}_{wf_name}"
        )
        os.makedirs(wf_log_dir, exist_ok=True)
        wf_log_file = os.path.join(wf_log_dir, f"train_{class_name}_{wf_name}_{run_tag}.txt")
        _current_wf_fh = logging.FileHandler(wf_log_file, encoding="utf-8")
        _current_wf_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(_current_wf_fh)
        _wf_fh = _current_wf_fh  # alias for the final closing block
        logger.info("WF log: %s", wf_log_file)

        # load the right encoder for this WF (Bootstrap uses key=0)
        _enc_tuple = _get_encoder_for_wf(wf_idx, enc_key)
        encoder, emb_dim, regime_head, trend_head, vol_head, outcome_head = _enc_tuple[:6]
        tmh = _enc_tuple[6] if len(_enc_tuple) > 6 else None
        rch = _enc_tuple[7] if len(_enc_tuple) > 7 else None
        msh = _enc_tuple[8] if len(_enc_tuple) > 8 else None
        soh = _enc_tuple[9] if len(_enc_tuple) > 9 else None  # strategy_outcome_head (v26)

        train_envs: dict[str, MarketEnv] = {}
        test_envs: dict[str, MarketEnv] = {}
        _asset_dfs: dict[str, Any] = {}  # df_full per asset — used by the secondary validation
        ENCODER_FEATURE_COLS = None

        for asset, path in asset_map.items():
            try:
                df_full = (
                    pd.read_parquet(path)
                    if path.endswith(".parquet")
                    else pd.read_csv(path, parse_dates=True)
                )
                df_full = ensure_datetime_index_utc(
                    df_full, tz_local=cfg.get("env_cfg", {}).get("tz_local", "Europe/Rome")
                )
                df_full = add_session_features(df_full)  # v22: add the 7 session cols (166→173)
                _asset_dfs[asset] = df_full  # kept for the secondary validation
                local_cols = build_encoder_feature_cols(df_full)
                if ENCODER_FEATURE_COLS is None:
                    ENCODER_FEATURE_COLS = local_cols
                    expected_ch = getattr(encoder, "input_features", None)
                    if expected_ch and len(ENCODER_FEATURE_COLS) != int(expected_ch):
                        # mismatch: MarketEnv handles it (zero padding / truncation)
                        logger.warning(
                            "Feature count mismatch: encoder=%d, parquet=%d "
                            "(handled by MarketEnv)",
                            expected_ch,
                            len(ENCODER_FEATURE_COLS),
                        )
                    logger.info(
                        "Encoder feature cols: %d (encoder expects: %s)",
                        len(ENCODER_FEATURE_COLS),
                        getattr(encoder, "input_features", "?"),
                    )
                elif local_cols != ENCODER_FEATURE_COLS:
                    raise AssertionError(f"Asset {asset} feature cols mismatch")

                train_df = df_full.loc[train_start:train_end].copy()
                test_df = df_full.loc[test_start:test_end].copy()
                logger.info(
                    "Asset %s: train_df=%d rows (%s → %s), test_df=%d rows",
                    asset,
                    len(train_df),
                    train_df.index[0] if len(train_df) > 0 else "empty",
                    train_df.index[-1] if len(train_df) > 0 else "empty",
                    len(test_df),
                )
                seq_len = cfg.get("env_cfg", {}).get("seq_len", 128)
                if len(train_df) < seq_len or len(test_df) < 1:
                    logger.warning("Asset %s skipped WF %d: insufficient data", asset, wf_idx)
                    continue

                env_cfg = cfg.get("env_cfg", DEFAULT_CFG["env_cfg"]).copy()
                env_cfg["encoder_feature_cols"] = ENCODER_FEATURE_COLS
                tm = cfg.get("training_mode", "full")
                env_cfg["training_mode"] = (
                    "train" if tm in ("full", "retrain", "train", "scratch") else tm
                )

                # v23: asset_id per AssetEmbedding (local to the class: 0..N-1)
                _asset_list = sorted(asset_map.keys())
                _asset_id = _asset_list.index(asset) if asset in _asset_list else 0
                train_envs[asset] = MarketEnv(
                    train_df,
                    encoder,
                    regime_head,
                    env_cfg,
                    trend_head=trend_head,
                    vol_head=vol_head,
                    outcome_head=outcome_head,
                    asset_id=_asset_id,
                    trend_mom_head=tmh,
                    regime_cont_head=rch,
                    mktstr_head=msh,
                    strategy_outcome_head=soh,
                )
                train_envs[asset].asset = asset
                test_envs[asset] = MarketEnv(
                    test_df,
                    encoder,
                    regime_head,
                    env_cfg,
                    trend_head=trend_head,
                    vol_head=vol_head,
                    outcome_head=outcome_head,
                    asset_id=_asset_id,
                    trend_mom_head=tmh,
                    regime_cont_head=rch,
                    mktstr_head=msh,
                    strategy_outcome_head=soh,
                )
                test_envs[asset].asset = asset
            except Exception:
                logger.exception("Failed to prepare asset %s", asset)

        if not train_envs:
            logger.warning("No train envs for WF %d, skipping", wf_idx)
            continue

        state_dim = next(iter(train_envs.values()))._get_state().shape[0]
        logger.info("State dimension: %d", state_dim)

        # Random seed
        seed = cfg.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ── create the agent, with the WF -> WF warm start ─────────────────────────
        training_mode = cfg.get("training_mode", "full")
        retrain_cfg = cfg.get("retrain_cfg", {})  # always defined before the branches
        warm_start_cfg = cfg.get("warm_start_cfg", {})
        warm_start_on = warm_start_cfg.get("enabled", True)
        is_bootstrap = enc_key == 0  # first WF = bootstrap, starts from scratch

        agent = _make_agent(state_dim, cfg)

        if training_mode == "retrain":
            # manual retrain mode — backward compatible
            retrain_cfg = cfg.get("retrain_cfg", {})
            load_path = cfg.get("retrain_load_paths", {}).get(class_name)
            if load_path:
                try:
                    agent.load(load_path)
                    logger.info("[AGENT] Retrain load: %s", load_path)
                except Exception:
                    logger.exception("[AGENT] Failed to load retrain weights")
                    agent.prepare_for_retrain(
                        reset_buffer=retrain_cfg.get("reset_replay_buffer", True),
                        lr_scale=retrain_cfg.get("lr_scale", 0.3),
                        reset_target=retrain_cfg.get("reset_target", False),
                        freeze_target_steps=int(retrain_cfg.get("freeze_target_steps", 3000)),
                        reset_elite_buffer=True,
                    )

        elif warm_start_on and not is_bootstrap and _prev_agent_ckpt is not None:
            # warm start from the previous WF — the elite buffer PERSISTS (accumulates rank1/2)
            try:
                agent.load(_prev_agent_ckpt, reset_elite=False)
                agent.prepare_for_retrain(
                    reset_buffer=warm_start_cfg.get("reset_replay_buffer", True),
                    lr_scale=warm_start_cfg.get("lr_warmup_scale", 0.1),
                    reset_target=warm_start_cfg.get("reset_target", False),
                    freeze_target_steps=int(warm_start_cfg.get("freeze_target_steps", 1000)),
                    reset_trunk=False,
                    reset_elite_buffer=True,
                    # reset_trunk=False: the trunk adapts
                    # to the new encoder through low-LR gradient updates.
                    # reset_trunk=True would throw away everything learnt in Bootstrap.
                    # The embedding space (state dims 0-127) changes, but the Q-value
                    # structure learnt on market patterns still transfers.
                )
                logger.info(
                    "[AGENT] WF%d [%s] warm start from the previous WF (lr_warmup=%.0fx)",
                    wf_idx,
                    wf_name,
                    warm_start_cfg.get("lr_warmup_scale", 0.1)
                    / cfg.get("rainbow_cfg", {}).get("lr", 3e-4),
                )
            except Exception:
                logger.exception("[AGENT] warm start failed — starting from scratch")

        elif is_bootstrap:
            agent.n_step = 1
            for _buf in agent._nstep_bufs.values():
                _buf.n = 1
            logger.info("[AGENT] Bootstrap WF — agent from scratch, n_step forced to 1")
        else:
            logger.info("[AGENT] WF%d — no previous checkpoint, starting from scratch", wf_idx)

        if training_mode == "eval":
            agent.set_eval_mode()

        # Runner config
        runner_cfg = cfg.get("runner_cfg", DEFAULT_CFG["runner_cfg"]).copy()
        runner_cfg["device"] = cfg.get("device", DEFAULT_CFG["device"])
        runner_cfg["initial_capital"] = float(
            cfg.get("initial_capital", DEFAULT_CFG["initial_capital"])
        )
        runner_cfg["safety"] = runner_cfg.get("safety", {})
        # n_future not used by Rainbow

        arbiter = TradeArbiter(arbiter_cfg)
        max_steps = max(len(env.df) for env in train_envs.values())
        arbiter.max_steps = max_steps  # used by _resolve_schedule for fraction schedules
        asset_class_map = {a: class_name for a in train_envs.keys()}

        # ============ TRAINING ============
        runner = None  # populated by _rainbow_train_loop

        if do_train and training_mode != "eval":
            pbar = tqdm(total=max_steps, desc=f"{class_name} WF{wf_idx} train", unit="step")
            try:
                freeze_target_steps = int(retrain_cfg.get("freeze_target_steps", 0))
                runner = _rainbow_train_loop(
                    train_envs,
                    {class_name: agent},
                    asset_class_map,
                    arbiter,
                    runner_cfg,
                    cfg,
                    max_steps,
                    freeze_target_steps,
                    pbar,
                )
            finally:
                pbar.close()

        # save the agent and update the warm-start pointer for the next WF
        wf_dir = os.path.join(
            cfg.get("save_root", DEFAULT_CFG["save_root"]), f"{class_name}_{wf_name}"
        )
        os.makedirs(wf_dir, exist_ok=True)
        agent_ckpt_path = os.path.join(wf_dir, f"{class_name}_agent_{wf_name}.pth")
        agent.save(agent_ckpt_path)
        _prev_agent_ckpt = agent_ckpt_path  # warm start of the next WF
        logger.info(
            "[AGENT] saved: %s → warm start ready for WF%d",
            os.path.basename(agent_ckpt_path),
            wf_idx + 1,
        )

        # ============ EVAL ============
        # copy the weights train -> eval through save/load
        eval_agent = _make_agent(state_dim, cfg)
        _tmp_ckpt = os.path.join(wf_dir, f"_tmp_eval_{class_name}_wf{wf_idx}.pth")
        try:
            agent.save(_tmp_ckpt)
            eval_agent.load(_tmp_ckpt)
        finally:
            if os.path.exists(_tmp_ckpt):
                os.remove(_tmp_ckpt)
        eval_agent.set_eval_mode()

        eval_runner_cfg = runner_cfg.copy()
        # daily SL limit per WF: Bootstrap=5, WF1=4, WF2+=3 (= live)
        _sl_map = runner_cfg.get("eval_max_daily_losses_by_wf", {})
        _eval_max_sl = int(_sl_map.get(enc_key, runner_cfg.get("max_daily_losses", 3)))
        eval_runner_cfg["max_daily_losses"] = _eval_max_sl
        logger.info("[EVAL] max_daily_losses=%d (enc_key=%d WF%d)", _eval_max_sl, enc_key, wf_idx)
        # v23: outcome_heads to the eval arbiter for the per-asset P(TP) filter
        # outcome_head is shared per class — the same for every asset
        if outcome_head is not None:
            eval_runner_cfg["outcome_heads"] = {a: outcome_head for a in asset_map}
            eval_runner_cfg["outcome_threshold"] = cfg.get("runner_cfg", {}).get(
                "outcome_threshold", 0.0
            )
        eval_runner_cfg["mode"] = "eval"
        eval_runner_cfg["baseline_max_dd"] = cfg.get("runner_cfg", {}).get("eval_max_dd", 0.045)
        eval_runner_cfg["baseline_daily_dd"] = cfg.get("runner_cfg", {}).get(
            "eval_daily_dd_limit", 0.025
        )

        for env in test_envs.values():
            env.max_dd = float(eval_runner_cfg["baseline_max_dd"])
            env.daily_dd_limit = float(eval_runner_cfg["baseline_daily_dd"])
            env.training_mode = "eval"

        arbiter_eval = TradeArbiter(arbiter_cfg)
        # pass max_rank_eval and eval_max_concurrent to the eval TradeArbiter
        arbiter_eval._max_rank_eval = int(runner_cfg.get("max_rank_eval", 99))
        pbar_eval = tqdm(
            total=max(len(env.df) for env in test_envs.values()),
            desc=f"{class_name} WF{wf_idx} eval",
            unit="step",
        )
        try:
            eval_runner = _eval_loop(
                test_envs,
                {class_name: eval_agent},
                asset_class_map,
                arbiter_eval,
                eval_runner_cfg,
                pbar_eval,
            )
        finally:
            pbar_eval.close()

        # Metrics
        asset_summaries = {}
        for asset, env in test_envs.items():
            s = env.render_last_trade_summary()
            s["pnl"] = float(s.get("sum_profit", 0.0) - s.get("sum_loss", 0.0))
            wins = int(s.get("wins", 0))
            losses = int(s.get("losses", 0))
            trades = int(s.get("trades", 0))
            avg_win = (s.get("sum_profit", 0.0) / wins) if wins > 0 else 0.0
            avg_loss = (s.get("sum_loss", 0.0) / losses) if losses > 0 else 0.0
            win_rate = s.get("win_rate", 0.0)
            loss_rate = (losses / trades) if trades > 0 else 0.0
            expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
            initial_capital = float(s.get("initial_capital", cfg.get("initial_capital", 100000.0)))
            max_dd = float(s.get("max_dd_static", s.get("drawdown", 0.0)))
            r_over_dd = s["pnl"] / (initial_capital * max_dd) if max_dd > 0 else 0.0
            s.update({"expectancy": float(expectancy), "return_over_max_dd": float(r_over_dd)})
            asset_summaries[asset] = s

        class_summary = aggregate_class_metrics(asset_summaries)
        global_summary = compute_global_metrics(
            asset_summaries, eval_runner.portfolio, test_start, test_end, test_envs
        )

        meta = {
            "wf_id": int(wf_idx),
            "wf_name": wf_name,
            "train_period": [str(train_start.date()), str(train_end.date())],
            "test_period": [str(test_start.date()), str(test_end.date())],
            "tz": "UTC",
            "training_mode": training_mode,
            "agent_type": "rainbow",
            # ── test stats summary (copy of global, quicker to read) ──
            "test_stats": {
                "total_trades": int(global_summary.get("total_trades", 0)),
                "win_rate": float(global_summary.get("win_rate", 0.0)),
                "profit_factor": float(global_summary.get("profit_factor", 0.0)),
                "pnl": float(global_summary.get("pnl", 0.0)),
                "pnl_pct": float(global_summary.get("pnl", 0.0))
                / float(cfg.get("initial_capital", 100000))
                * 100,
                "max_dd_static": float(global_summary.get("max_dd_static", 0.0)),
                "max_dd_trailing": float(global_summary.get("max_dd_trailing", 0.0)),
                "sharpe_like": float(global_summary.get("sharpe_like", 0.0)),
                "rank1_wr": (
                    (
                        float(eval_runner.arbiter_stats["rank1_tp"])
                        / max(
                            1,
                            eval_runner.arbiter_stats["rank1_tp"]
                            + eval_runner.arbiter_stats["rank1_sl"],
                        )
                    )
                    if eval_runner
                    and eval_runner.arbiter_stats.get("rank1_tp", 0)
                    + eval_runner.arbiter_stats.get("rank1_sl", 0)
                    > 0
                    else 0.0
                ),
                "trend_wr": (
                    (
                        float(eval_runner.arbiter_stats["trend_tp"])
                        / max(
                            1,
                            eval_runner.arbiter_stats["trend_tp"]
                            + eval_runner.arbiter_stats["trend_sl"],
                        )
                    )
                    if eval_runner
                    and eval_runner.arbiter_stats.get("trend_tp", 0)
                    + eval_runner.arbiter_stats.get("trend_sl", 0)
                    > 0
                    else 0.0
                ),
                "mr_wr": (
                    (
                        float(eval_runner.arbiter_stats["mr_tp"])
                        / max(
                            1,
                            eval_runner.arbiter_stats["mr_tp"] + eval_runner.arbiter_stats["mr_sl"],
                        )
                    )
                    if eval_runner
                    and eval_runner.arbiter_stats.get("mr_tp", 0)
                    + eval_runner.arbiter_stats.get("mr_sl", 0)
                    > 0
                    else 0.0
                ),
            },
        }

        eval_summary = {
            "assets": asset_summaries,
            "classes": {class_name: class_summary},
            "global": global_summary,
            "meta": meta,
            "arbiter_stats_train": runner.arbiter_stats if runner else {},
            "arbiter_stats_eval": eval_runner.arbiter_stats,
            "runner_diagnostics": {
                "qgap_filtered": int(eval_runner.arbiter_stats.get("qgap_filtered", 0)),
                "vote_blocked": int(eval_runner.arbiter_stats.get("vote_blocked", 0)),
                "positive_q_blocked": int(eval_runner.arbiter_stats.get("positive_q_blocked", 0)),
                "entry_filter_blocked": int(
                    eval_runner.arbiter_stats.get("entry_filter_blocked", 0)
                ),
                "decision_gate_blocked": int(
                    eval_runner.arbiter_stats.get("decision_gate_blocked", 0)
                ),
                "concurrent_blocked": int(eval_runner.arbiter_stats.get("concurrent_blocked", 0)),
                "daily_loss_blocked": int(eval_runner.arbiter_stats.get("daily_loss_blocked", 0)),
                "approved": int(eval_runner.arbiter_stats.get("approved", 0)),
                "rejected": int(eval_runner.arbiter_stats.get("rejected", 0)),
            },
        }

        eval_json_name = f"eval_summary_{wf_name}.json"
        eval_json_path = os.path.join(wf_dir, eval_json_name)
        with open(eval_json_path, "w") as f:
            json.dump(eval_summary, f, indent=2, default=str)
        logger.info("Saved %s → %s", eval_json_name, wf_dir)

        # ── secondary validation (optional) ─────────────────────────────
        # if the split has "val": (start, end) a second eval runs
        # on that period, to measure the test/val correlation
        # and anticipate the quality on the future live period.
        # "val": None -> skip. Missing -> skip.
        _val_period = split.get("val", None)
        if _val_period is not None:
            val_start_str, val_end_str = _val_period

            def _to_utc(v):
                if isinstance(v, str):
                    return pd.Timestamp(v, tz="UTC")
                ts = pd.Timestamp(v)
                return ts if ts.tzinfo is not None else ts.tz_localize("UTC")

            val_start = _to_utc(val_start_str)
            val_end = _to_utc(val_end_str)
            logger.info("[VAL] Secondary period %s → %s", val_start.date(), val_end.date())
            val_envs = {}
            _val_asset_list = sorted(asset_map.keys())
            for asset, df_full in _asset_dfs.items():
                val_df = df_full.loc[val_start:val_end].copy()
                if len(val_df) < 100:
                    logger.warning("[VAL] %s: val_df too short (%d), skip", asset, len(val_df))
                    continue
                # Bug fix: asset_map[asset] is a PATH (str), not an int
                _val_asset_id = _val_asset_list.index(asset) if asset in _val_asset_list else 0
                env_v = MarketEnv(
                    val_df,
                    encoder,
                    regime_head,
                    env_cfg,
                    trend_head=trend_head,
                    vol_head=vol_head,
                    outcome_head=outcome_head,
                    asset_id=_val_asset_id,
                    trend_mom_head=tmh,
                    regime_cont_head=rch,
                    mktstr_head=msh,
                    strategy_outcome_head=soh,
                )
                env_v.max_dd = float(eval_runner_cfg["baseline_max_dd"])
                env_v.daily_dd_limit = float(eval_runner_cfg["baseline_daily_dd"])
                env_v.training_mode = "eval"
                val_envs[asset] = env_v

            if val_envs:
                arbiter_val = TradeArbiter(arbiter_cfg)
                arbiter_val._max_rank_eval = int(runner_cfg.get("max_rank_eval", 99))
                pbar_val = tqdm(
                    total=max(len(e.df) for e in val_envs.values()),
                    desc=f"{class_name} WF{wf_idx} val",
                    unit="step",
                )
                try:
                    val_runner = _eval_loop(
                        val_envs,
                        {class_name: eval_agent},
                        asset_class_map,
                        arbiter_val,
                        eval_runner_cfg,
                        pbar_val,
                    )
                finally:
                    pbar_val.close()

                # Val metrics
                val_asset_summaries = {}
                for asset, env_v in val_envs.items():
                    sv = env_v.render_last_trade_summary()
                    sv["pnl"] = float(sv.get("sum_profit", 0.0) - sv.get("sum_loss", 0.0))
                    wins_v = int(sv.get("wins", 0))
                    losses_v = int(sv.get("losses", 0))
                    trades_v = int(sv.get("trades", 0))
                    avg_win_v = sv.get("sum_profit", 0.0) / wins_v if wins_v > 0 else 0.0
                    avg_loss_v = sv.get("sum_loss", 0.0) / losses_v if losses_v > 0 else 0.0
                    wr_v = sv.get("win_rate", 0.0)
                    lr_v = (losses_v / trades_v) if trades_v > 0 else 0.0
                    sv["expectancy"] = (wr_v * avg_win_v) - (lr_v * avg_loss_v)
                    val_asset_summaries[asset] = sv

                val_global = compute_global_metrics(
                    val_asset_summaries, val_runner.portfolio, val_start, val_end, val_envs
                )
                val_summary = {
                    "assets": val_asset_summaries,
                    "classes": {class_name: aggregate_class_metrics(val_asset_summaries)},
                    "global": val_global,
                    "meta": {
                        "wf_id": int(wf_idx),
                        "train_period": [str(train_start.date()), str(train_end.date())],
                        "test_period": [str(test_start.date()), str(test_end.date())],
                        "val_period": [str(val_start.date()), str(val_end.date())],
                        "agent_type": "rainbow",
                    },
                    "arbiter_stats_val": val_runner.arbiter_stats,
                }
                val_json = os.path.join(wf_dir, f"val_summary_{wf_name}.json")
                with open(val_json, "w") as f:
                    json.dump(val_summary, f, indent=2, default=str)
                logger.info("[VAL] Saved val_summary_%s.json → %s", wf_name, wf_dir)
                logger.info(
                    "[VAL] WR=%.1f%%  PF=%.3f  PnL=%+.0f  trades=%d",
                    val_global.get("win_rate", 0.0) * 100,
                    val_global.get("profit_factor", 0.0),
                    val_global.get("pnl", 0.0),
                    val_global.get("total_trades", 0),
                )
                results_all.setdefault(wf_name, {})["val_summary"] = val_json

        # ── the WF log is closed at the next WF (or after the loop) ────────
        # the handler is closed at the start of the next WF or after the loop.
        pass  # closed at the start of the next WF

        results_all[wf_name] = {
            "train_assets": list(train_envs.keys()),
            "eval_summary": eval_json_path,
        }

    # ── Close the last WF log handler ────────────────────────────────────
    if _current_wf_fh is not None:
        try:
            logging.getLogger().removeHandler(_current_wf_fh)
            _current_wf_fh.close()
        except Exception:
            pass

    return results_all


def _merge_agents(agent_M, agent_m, state_dim: int, cfg: dict[str, Any], alpha: float = 0.5) -> any:
    """Merges two agents pre-trained per class into a single agent.

    Weight averaging (model merging): it works because the two agents have
    the same architecture, the same state vector and the same action space.
    alpha=0.5 = equal weight; alpha>0.5 = more weight to Majors (more liquid).

    The result is an agent that:
      - knows the patterns of both classes
      - has a stable starting point for the joint fine-tuning
      - forgets neither policy (no catastrophic forgetting)
    """
    joint = _make_agent(state_dim, cfg)

    sd_M = agent_M.online.state_dict()  # ✅
    sd_m = agent_m.online.state_dict()  # ✅
    merged = {}
    for k in sd_M:
        if sd_M[k].dtype.is_floating_point:
            merged[k] = alpha * sd_M[k] + (1.0 - alpha) * sd_m[k]
        else:
            merged[k] = sd_M[k]

    joint.online.load_state_dict(merged, strict=True)  # ✅
    joint.target.load_state_dict(merged, strict=True)  # ✅
    logger.info(
        "[MERGE] Joint agent created: alpha=%.2f (Majors) + %.2f (Minors)", alpha, 1.0 - alpha
    )
    return joint


def run_walk_forward_joint(
    majors_map: dict[str, str],
    minors_map: dict[str, str],
    majors_encoder_ckpt: str | dict[int, str],
    minors_encoder_ckpt: str | dict[int, str],
    cfg: dict[str, Any],
    do_train: bool = True,
) -> dict:
    """Walk-forward with Majors + Minors on a SHARED portfolio.

    Live, the two asset classes run on the same account — the absolute DD,
    the correlation cap and the daily SL limit must be computed on the
    combined portfolio, so Majors and Minors can't break the DD
    each on its own without seeing each other.

    Architecture:
      - separate encoders for Majors and Minors (different distribution)
      - one agent, merged from the Majors and Minors agents of Run1
      - shared portfolio -> unified DD tracking
      - a single MultiAssetRunner -> cross-class correlation cap
    """
    logger.info(
        "Walk-forward JOINT (Majors+Minors) | %d+%d assets | shared portfolio",
        len(majors_map),
        len(minors_map),
    )

    import datetime

    log_dir = cfg.get("save_root", DEFAULT_CFG["save_root"])
    os.makedirs(log_dir, exist_ok=True)
    run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_Joint_{run_tag}.txt")
    _setup_file_logging(log_file)

    wf_mode = cfg.get("wf_mode", "auto")
    if wf_mode == "manual" and cfg.get("wf_splits"):
        raw_splits = cfg["wf_splits"]
        splits = []
        for s in raw_splits:
            ts = pd.to_datetime(s["train"][0]).tz_localize("UTC")
            te = pd.to_datetime(s["train"][1]).tz_localize("UTC")
            vs = pd.to_datetime(s["test"][0]).tz_localize("UTC")
            ve = pd.to_datetime(s["test"][1]).tz_localize("UTC")
            splits.append(
                {
                    "train": (ts, te),
                    "test": (vs, ve),
                    "name": s.get("name", f"WF{len(splits)+1}"),
                    "encoder_wf_key": s.get("encoder_wf_key", len(splits)),
                }
            )
    else:
        wf_cfg = cfg.get("wf", DEFAULT_CFG["wf"])
        splits = generate_walk_forward_splits(
            wf_cfg["start_date"],
            wf_cfg["end_date"],
            wf_cfg["train_window_years"],
            wf_cfg["test_window_years"],
            wf_cfg["step_years"],
        )
        for i, s in enumerate(splits):
            s["name"] = f"WF{i+1}"
            s["encoder_wf_key"] = i + 1

    env_device = cfg.get("env_cfg", {}).get("device", "cpu")
    _enc_cache: dict[str, tuple] = {}

    def _get_enc(ckpt_dict, wf_key):
        if isinstance(ckpt_dict, str):
            path = ckpt_dict
        else:
            valid = [k for k in ckpt_dict if k <= wf_key]
            if not valid:
                raise ValueError(f"No encoder for key={wf_key}")
            path = ckpt_dict[max(valid)]
        if path not in _enc_cache:
            t = load_encoder_from_checkpoint(
                path,
                device=env_device,
                seq_len=cfg.get("env_cfg", {}).get("seq_len", 128),
                load_regime_head=True,
                load_trend_head=True,
                load_vol_head=True,
                load_outcome_head=True,
            )
            for m in t:
                if m is not None and hasattr(m, "eval"):
                    m.eval()
            _enc_cache[path] = t
        return _enc_cache[path]

    arbiter_cfg = cfg.get("arbiter_cfg", DEFAULT_CFG["arbiter_cfg"]).copy()
    runner_cfg = cfg.get("runner_cfg", DEFAULT_CFG["runner_cfg"]).copy()
    runner_cfg["device"] = cfg.get("device", DEFAULT_CFG["device"])
    runner_cfg["initial_capital"] = float(
        cfg.get("initial_capital", DEFAULT_CFG["initial_capital"])
    )

    _prev_agent_ckpt_M: str | None = None
    _prev_agent_ckpt_m: str | None = None
    _prev_agent_ckpt_joint: str | None = None
    results_all = {}

    for wf_idx, split in enumerate(splits, start=1):
        train_start, train_end = split["train"]
        test_start, test_end = split["test"]
        wf_name = split.get("name", f"WF{wf_idx}")
        enc_key = split.get("encoder_wf_key", wf_idx)

        logger.info(
            "JOINT WF%d [%s]: train %s → %s | test %s → %s",
            wf_idx,
            wf_name,
            train_start.date(),
            train_end.date(),
            test_start.date(),
            test_end.date(),
        )

        # load the separate encoders
        enc_M = _get_enc(majors_encoder_ckpt, enc_key)
        enc_m = _get_enc(minors_encoder_ckpt, enc_key)
        encoder_M, _edim_M = enc_M[0], enc_M[1]
        encoder_m, _edim_m = enc_m[0], enc_m[1]
        rh_M, th_M, vh_M, oh_M = enc_M[2], enc_M[3], enc_M[4], enc_M[5]
        rh_m, th_m, vh_m, oh_m = enc_m[2], enc_m[3], enc_m[4], enc_m[5]
        tmh_M = enc_M[6] if len(enc_M) > 6 else None
        rch_M = enc_M[7] if len(enc_M) > 7 else None
        msh_M = enc_M[8] if len(enc_M) > 8 else None
        soh_M = enc_M[9] if len(enc_M) > 9 else None  # strategy_outcome_head (v26)
        tmh_m = enc_m[6] if len(enc_m) > 6 else None
        rch_m = enc_m[7] if len(enc_m) > 7 else None
        msh_m = enc_m[8] if len(enc_m) > 8 else None
        soh_m = enc_m[9] if len(enc_m) > 9 else None  # strategy_outcome_head (v26)

        # build the envs of every asset
        all_train_envs: dict[str, MarketEnv] = {}
        all_test_envs: dict[str, MarketEnv] = {}
        asset_class_map: dict[str, str] = {}

        for cls_name, a_map, enc, rh, th, vh, oh, tmh, rch, msh, soh in [
            ("Majors", majors_map, encoder_M, rh_M, th_M, vh_M, oh_M, tmh_M, rch_M, msh_M, soh_M),
            ("Minors", minors_map, encoder_m, rh_m, th_m, vh_m, oh_m, tmh_m, rch_m, msh_m, soh_m),
        ]:
            COLS = None
            for asset, path in a_map.items():
                try:
                    df_full = (
                        pd.read_parquet(path)
                        if path.endswith(".parquet")
                        else pd.read_csv(path, parse_dates=True)
                    )
                    df_full = ensure_datetime_index_utc(
                        df_full, tz_local=cfg.get("env_cfg", {}).get("tz_local", "Europe/Rome")
                    )
                    df_full = add_session_features(df_full)
                    if COLS is None:
                        COLS = build_encoder_feature_cols(df_full)

                    train_df = df_full.loc[train_start:train_end].copy()
                    test_df = df_full.loc[test_start:test_end].copy()
                    seq_len = cfg.get("env_cfg", {}).get("seq_len", 128)
                    if len(train_df) < seq_len or len(test_df) < 1:
                        logger.warning("Joint: asset %s skip (insufficient data)", asset)
                        continue

                    env_cfg = cfg.get("env_cfg", DEFAULT_CFG["env_cfg"]).copy()
                    env_cfg["encoder_feature_cols"] = COLS
                    tm = cfg.get("training_mode", "full")
                    env_cfg["training_mode"] = (
                        "train" if tm in ("full", "retrain", "train", "scratch") else tm
                    )
                    _asset_list = sorted(a_map.keys())
                    _asset_id = _asset_list.index(asset) if asset in _asset_list else 0

                    all_train_envs[asset] = MarketEnv(
                        train_df,
                        enc,
                        rh,
                        env_cfg,
                        trend_head=th,
                        vol_head=vh,
                        outcome_head=oh,
                        asset_id=_asset_id,
                        trend_mom_head=tmh,
                        regime_cont_head=rch,
                        mktstr_head=msh,
                        strategy_outcome_head=soh,
                    )
                    all_train_envs[asset].asset = asset
                    all_test_envs[asset] = MarketEnv(
                        test_df,
                        enc,
                        rh,
                        env_cfg,
                        trend_head=th,
                        vol_head=vh,
                        outcome_head=oh,
                        asset_id=_asset_id,
                        trend_mom_head=tmh,
                        regime_cont_head=rch,
                        mktstr_head=msh,
                        strategy_outcome_head=soh,
                    )
                    all_test_envs[asset].asset = asset
                    asset_class_map[asset] = cls_name
                except Exception:
                    logger.exception("Joint: failed to prepare asset %s", asset)

        if not all_train_envs:
            logger.warning("Joint WF%d: no envs, skipping", wf_idx)
            continue

        state_dim = next(iter(all_train_envs.values()))._get_state().shape[0]
        logger.info(
            "Joint state_dim=%d | assets=%d (%s)",
            state_dim,
            len(all_train_envs),
            list(all_train_envs.keys()),
        )

        # ── one agent for the shared portfolio ─────────────────────────────
        # initialization per WF:
        #   WF1 Joint Bootstrap: merged weights of the final Majors and Minors agents
        #                        (given through cfg["joint_init_ckpt_M/m"])
        #   WF2+ Joint:          warm start from the joint agent of the previous WF
        training_mode = cfg.get("training_mode", "full")
        warm_start_cfg = cfg.get("warm_start_cfg", {})
        is_bootstrap = enc_key == 0
        merge_alpha = float(cfg.get("joint_merge_alpha", 0.5))

        agent = _make_agent(state_dim, cfg)

        if is_bootstrap:
            # first joint run: merge of the two agents trained separately
            init_M = cfg.get("joint_init_ckpt_M")  # final Majors agent
            init_m = cfg.get("joint_init_ckpt_m")  # final Minors agent
            if init_M and init_m and os.path.exists(init_M) and os.path.exists(init_m):
                _tmp_M = _make_agent(state_dim, cfg)
                _tmp_M.load(init_M)
                _tmp_m = _make_agent(state_dim, cfg)
                _tmp_m.load(init_m)
                agent = _merge_agents(_tmp_M, _tmp_m, state_dim, cfg, alpha=merge_alpha)
                agent.prepare_for_retrain(
                    reset_buffer=True,
                    lr_scale=warm_start_cfg.get("lr_warmup_scale", 0.1),
                    reset_target=False,
                    freeze_target_steps=int(warm_start_cfg.get("freeze_target_steps", 3000)),
                )
                logger.info("[JOINT] Bootstrap: merge Majors+Minors (alpha=%.2f)", merge_alpha)
            else:
                logger.warning(
                    "[JOINT] Bootstrap: joint_init_ckpt_M/m not set "
                    "— agent initialized from scratch (suboptimal)"
                )

        elif _prev_agent_ckpt_joint is not None:
            # next WFs: warm start from the previous joint agent
            try:
                agent.load(_prev_agent_ckpt_joint, reset_elite=False)
                agent.prepare_for_retrain(
                    reset_buffer=warm_start_cfg.get("reset_replay_buffer", True),
                    lr_scale=warm_start_cfg.get("lr_warmup_scale", 0.1),
                    reset_target=warm_start_cfg.get("reset_target", False),
                    freeze_target_steps=int(warm_start_cfg.get("freeze_target_steps", 3000)),
                )
                logger.info("[JOINT] WF%d warm start from the previous joint agent", wf_idx)
            except Exception:
                logger.exception("[JOINT] warm start failed — from scratch")

        agents = {"Majors": agent, "Minors": agent}  # the same agent for both classes

        if training_mode == "eval":
            agent.set_eval_mode()

        # training on the shared portfolio
        arbiter = TradeArbiter(arbiter_cfg)
        max_steps = max(len(env.df) for env in all_train_envs.values())
        arbiter.max_steps = max_steps  # used by _resolve_schedule for fraction schedules

        if do_train and training_mode != "eval":
            pbar = tqdm(total=max_steps, desc=f"Joint WF{wf_idx} train", unit="step")
            try:
                _rainbow_train_loop(
                    all_train_envs,
                    agents,
                    asset_class_map,
                    arbiter,
                    runner_cfg,
                    cfg,
                    max_steps,
                    int(cfg.get("retrain_cfg", {}).get("freeze_target_steps", 0)),
                    pbar,
                )
            finally:
                pbar.close()

        # save the agents per class
        wf_dir = os.path.join(cfg.get("save_root", DEFAULT_CFG["save_root"]), f"Joint_{wf_name}")
        os.makedirs(wf_dir, exist_ok=True)
        # save the single agent (agents["Majors"] and agents["Minors"] point to the same one)
        joint_ckpt_path = os.path.join(wf_dir, f"Joint_agent_{wf_name}.pth")
        agent.save(joint_ckpt_path)
        _prev_agent_ckpt_joint = joint_ckpt_path
        logger.info(
            "[JOINT] single agent saved → %s → warm start ready for WF%d",
            os.path.basename(joint_ckpt_path),
            wf_idx + 1,
        )

        results_all[wf_name] = {"wf_idx": wf_idx, "assets": list(all_train_envs.keys())}

    return results_all


def run_pipeline(
    majors_data_dir: str | None = None,
    minors_data_dir: str | None = None,
    majors_encoder_ckpt: str | dict[int, str] | None = None,
    minors_encoder_ckpt: str | dict[int, str] | None = None,
    cfg_override: dict[str, Any] | None = None,
    do_train: bool | None = None,
):
    """
    majors_encoder_ckpt / minors_encoder_ckpt:
      str  -> the same encoder for every WF
      dict -> {wf_idx: path}, e.g.:
             {1: bootstrap.pth, 2: retrain_2021_Q4.pth,
              3: retrain_2022_Q4.pth, 4: retrain_2022_Q4.pth}
    """
    cfg = copy.deepcopy(DEFAULT_CFG)  # deep: the nested dicts are updated below
    if cfg_override:
        for k, v in cfg_override.items():
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k] = v

    if do_train is None:
        do_train_effective = bool(cfg.get("do_train", True))
    else:
        do_train_effective = bool(do_train)

    results = {}
    run_classes = cfg.get("run_classes", ["Majors", "Minors"])
    joint_mode = cfg.get("joint_mode", False)

    if (
        joint_mode
        and majors_data_dir
        and minors_data_dir
        and majors_encoder_ckpt
        and minors_encoder_ckpt
    ):
        # Shared Majors + Minors portfolio — simulates real live trading
        majors_map = scan_directory_for_assets(majors_data_dir)
        minors_map = scan_directory_for_assets(minors_data_dir)
        if majors_map and minors_map:
            results["Joint"] = run_walk_forward_joint(
                majors_map,
                minors_map,
                majors_encoder_ckpt,
                minors_encoder_ckpt,
                cfg,
                do_train=do_train_effective,
            )
    else:
        # separate mode: every class runs on its own
        if "Majors" in run_classes and majors_data_dir and majors_encoder_ckpt:
            majors_map = scan_directory_for_assets(majors_data_dir)
            if majors_map:
                results["Majors"] = run_walk_forward_for_asset_map(
                    "Majors", majors_map, majors_encoder_ckpt, cfg, do_train=do_train_effective
                )

        if "Minors" in run_classes and minors_data_dir and minors_encoder_ckpt:
            minors_map = scan_directory_for_assets(minors_data_dir)
            if minors_map:
                results["Minors"] = run_walk_forward_for_asset_map(
                    "Minors", minors_map, minors_encoder_ckpt, cfg, do_train=do_train_effective
                )

    return results


# ============================================================
# CLI
# ============================================================
def _wf_number(path: Path) -> int:
    m = re.search(r"WF(\d+)", path.name)
    return int(m.group(1)) if m else -1


def find_encoder_checkpoints(models_root, class_name: str) -> dict[int, str]:
    """{0: bootstrap, 1: WF1 retrain, ...} from <models>/Encoder/<class>/Checkpoint."""
    base = Path(models_root) / "Encoder" / class_name / "Checkpoint"
    found = {}
    boot = base / "Bootstrap" / "best_full_checkpoint.pth"
    if boot.exists():
        found[0] = str(boot)
    for wf_dir in sorted((base / "Retrain").glob("WF*"), key=_wf_number):
        ckpt = wf_dir / f"retrain_full_{wf_dir.name}.pth"
        if _wf_number(wf_dir) > 0 and ckpt.exists():
            found[_wf_number(wf_dir)] = str(ckpt)
    return found


def latest_agent_checkpoint(models_root, class_name: str) -> str | None:
    """Agent of the last WF (Bootstrap if it's the only one) of the Run1 exploration of a class."""
    base = Path(models_root) / "Rainbow" / "runs" / "Exploration" / class_name
    wf_dirs = sorted(base.glob(f"{class_name}_*"), key=_wf_number)  # Bootstrap = -1, first
    for wf_dir in reversed(wf_dirs):
        agents = sorted(wf_dir.glob(f"{class_name}_agent_*.pth"))
        if agents:
            return str(agents[-1])
    return None


def _parse_value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Walk-forward training / evaluation of the Rainbow DQN agent.",
        epilog="Every other setting is in agent/config.py: change it with --set key=value "
        "(nested dicts as JSON) or with a JSON file passed to --config.",
    )
    p.add_argument("--run", choices=list(RUN_CONFIGS), default="run1_majors")
    p.add_argument("--root", default=str(DATA_ROOT), help="data root (reads <root>/Scaled)")
    p.add_argument(
        "--models-root",
        default=str(MODELS_ROOT),
        help="encoders in <models>/Encoder, runs in <models>/Rainbow",
    )
    p.add_argument("--eval-only", action="store_true", help="no training, evaluation only")
    p.add_argument("--config", help="JSON file with config overrides")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument(
        "--init-majors", help="joint runs: Majors agent to start from (default: last Run1 WF)"
    )
    p.add_argument(
        "--init-minors", help="joint runs: Minors agent to start from (default: last Run1 WF)"
    )
    args = p.parse_args(argv)

    overrides = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            overrides.update(json.load(f))
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            p.error(f"--set expects KEY=VALUE, got {item!r}")
        overrides[key] = _parse_value(value)
    cfg = build_cfg(args.run, overrides, models_root=args.models_root)

    if cfg.get("joint_mode"):
        cfg["joint_init_ckpt_M"] = (
            args.init_majors
            or cfg.get("joint_init_ckpt_M")
            or latest_agent_checkpoint(args.models_root, "Majors")
        )
        cfg["joint_init_ckpt_m"] = (
            args.init_minors
            or cfg.get("joint_init_ckpt_m")
            or latest_agent_checkpoint(args.models_root, "Minors")
        )

    scaled = Path(args.root) / "Scaled"
    encoders = {c: find_encoder_checkpoints(args.models_root, c) for c in ("Majors", "Minors")}
    for c, found in encoders.items():
        logger.info(
            "%s encoders: %s", c, {k: Path(v).parent.name for k, v in found.items()} or "none found"
        )

    keepalive_thread = None
    try:
        if cfg.get("keepalive", False):
            keepalive_thread = KeepaliveThread(interval_seconds=cfg.get("keepalive_interval", 300))
            keepalive_thread.start()
        out = run_pipeline(
            majors_data_dir=str(scaled / "Majors"),
            minors_data_dir=str(scaled / "Minors"),
            majors_encoder_ckpt=encoders["Majors"] or None,
            minors_encoder_ckpt=encoders["Minors"] or None,
            cfg_override=cfg,
            do_train=False if args.eval_only else None,
        )
        logger.info("[%s] pipeline done. Output keys: %s", args.run, list(out.keys()))
        return out
    finally:
        if keepalive_thread:
            try:
                keepalive_thread.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
