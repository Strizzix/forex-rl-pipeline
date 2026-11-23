"""
config.py — configuration of the Rainbow DQN agent and of the walk-forward runs.

DEFAULT_CFG holds every setting; RUN_CONFIGS holds only what each training phase
changes (run1_majors / run1_minors -> run2_joint -> run3_joint). build_cfg() merges
them and turns the relative save_root into <models>/Rainbow/runs/<...>.

Folders:
    <root>/Scaled/<Majors|Minors>/<ASSET>/...                         input (scaled parquets)
    <models>/Encoder/<class>/Checkpoint/Bootstrap|Retrain/...         encoder checkpoints
    <models>/Rainbow/runs/<Exploration|Consolidation|Production>/...  agent runs
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

DATA_ROOT = Path(os.environ.get("FX_DATA_ROOT", "data"))
MODELS_ROOT = Path(os.environ.get("FX_MODELS_ROOT", "models"))

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_CFG: dict[str, Any] = {
    "seq_len": 128,
    "device": "cuda",
    "initial_capital": 100000.0,
    # ── Encoder heads ──────────────────────────────────────────────────────
    # use_vol_head=True: vol_pred_15m + vol_pred_1h are in the state vector
    # (dims 184-185). If False those dims are 0 — incomplete state.
    "use_regime_head": True,
    "use_trend_head": True,
    "use_vol_head": True,  # FIX: was False -> incomplete state
    # v23: outcome head for the arbiter filter (0.0 = off)
    "use_outcome_head": True,
    # v25: encoder trained with n_assets=5 per class — asset embedding on
    "n_assets_majors": 5,
    "n_assets_minors": 5,
    # ── Walk-forward ───────────────────────────────────────────────────────
    # mode: "auto" (splits generated automatically) | "manual" (uses wf_splits)
    "wf_mode": "manual",
    # manual splits — one entry per WF, in order.
    # used if wf_mode="manual".
    # train = agent data period (last 1.5 years of the encoder period)
    # test  = validation period (next half-year, OOS)
    # encoder_wf_key = key in the encoder checkpoint map (see find_encoder_checkpoints)
    "wf_splits": [
        # Bootstrap: bootstrap encoder, the agent learns a base policy on a wide period
        {
            "name": "Bootstrap",
            "train": ("2019-01-01", "2022-12-31"),
            "test": ("2023-01-01", "2023-06-30"),
            "val": ("2023-07-01", "2023-12-31"),  # simulated live: 2023H2
            "encoder_wf_key": 0,
        },
        # WF1: encoder WF1 (val on 2023H1) → deployed on 2023H2
        {
            "name": "WF1_2023H1",
            "train": ("2021-01-01", "2022-12-31"),
            "test": ("2023-01-01", "2023-06-30"),
            "val": ("2023-07-01", "2023-12-31"),  # simulated live: 2023H2
            "encoder_wf_key": 1,
        },
        # WF2: encoder WF2 (val on 2023H2) → deployed on 2024H1
        {
            "name": "WF2_2023H2",
            "train": ("2021-07-01", "2023-06-30"),
            "test": ("2023-07-01", "2023-12-31"),
            "val": ("2024-01-01", "2024-06-30"),  # simulated live: 2024H1
            "encoder_wf_key": 2,
        },
        # WF3: encoder WF3 (val on 2024H1) → deployed on 2024H2
        {
            "name": "WF3_2024H1",
            "train": ("2022-01-01", "2023-12-31"),
            "test": ("2024-01-01", "2024-06-30"),
            "val": ("2024-07-01", "2024-12-31"),  # simulated live: 2024H2
            "encoder_wf_key": 3,
        },
        # WF4: encoder WF4 (val on 2024H2) → deployed on 2025H1
        {
            "name": "WF4_2024H2",
            "train": ("2022-07-01", "2024-06-30"),
            "test": ("2024-07-01", "2024-12-31"),
            "val": ("2025-01-01", "2025-06-30"),  # simulated live: 2025H1
            "encoder_wf_key": 4,
        },
        # WF5: encoder WF5 (val on 2025H1) → deployed on 2025H2
        {
            "name": "WF5_2025H1",
            "train": ("2023-01-01", "2024-12-31"),
            "test": ("2025-01-01", "2025-06-30"),
            "val": ("2025-07-01", "2025-12-31"),  # simulated live: 2025H2
            "encoder_wf_key": 5,
        },
    ],
    # config of the auto mode (used only if wf_mode="auto")
    "wf": {
        "start_date": "2015-01-01",
        "end_date": "2023-12-31",
        "train_window_years": 5,
        "test_window_years": 1,
        "step_years": 1,
    },
    "agent_type": "rainbow",
    # ══════════════════════════════════════════════════════════════════════
    # ENV CFG
    # ══════════════════════════════════════════════════════════════════════
    "env_cfg": {
        "seq_len": 128,
        "rr": 1.8,  # TREND RR — breakeven WR = 35.7%
        "sl_atr_mult": 1.0,  # wider SL: fewer false intrabar SLs on 15min
        # ── MR strategy ──────────────────────────────────────────────────
        "rr_mr": 1.0,  # MR RR (breakeven WR = 50%)
        "sl_atr_mult_mr": 1.0,
        "max_bars_mr": 20,  # max 5h for MR
        "adx_trend_min": 0.25,
        "adx_ranging_max": 0.35,
        "commission_per_lot": 5.0,
        "pip_size": 0.0001,
        "pip_value": 10.0,  # USD/pip/lot (majors XxxUSD)
        "min_lot": 0.01,
        "trade_risk_pct": 0.004,  # 0.4% per trade
        "max_trades_per_day": 5,
        "max_dd": 0.30,
        "daily_dd_limit": 0.08,
        "spread_col": "spread_mean_15min_raw",
        "tz_local": "Europe/Rome",
        "device": "cuda",
        "slippage_noise": 0.1,
        "mask_mode": "explore",
        "hold_reward_warmup_steps": 5000,
        "model_tf_order": ["15min", "30min", "1h", "4h", "1d"],
        # ── Dense reward ─────────────────────────────────────────────────
        "mtm_reward_enabled": True,
        "mtm_scale": 0.05,
        "opportunity_cost_enabled": True,
        "opportunity_cost_scale": 0.001,
        # ── v26+: swap and force close ──────────────────────────────────────
        # swap_hold_penalty_per_day_r: penalty for every day the position
        # goes through the swap. It adds up: after 2 days = 2× penalty.
        # Teaches the agent to close before the 2nd and 3rd rollover.
        "hold_uncertainty_bonus": 0,
        "hold_uncertainty_thr": 0.85,
        "swap_hold_penalty_per_day_r": 0.03,  # 3% of 1R per day after the first
        # force_close_penalty_scale: penalty for trades closed by the force close,
        # proportional to the RR missed vs the TP. Soft (0.10 = 10% of the gap).
        "force_close_penalty_scale": 0.05,
    },
    # ══════════════════════════════════════════════════════════════════════
    # ARBITER CFG
    # ══════════════════════════════════════════════════════════════════════
    "arbiter_cfg": {
        "max_trades_per_day": 5,
        # w_trend > w_conf: trend acc (62-65%) > regime acc (55-65%)
        "w_q": 1.0,
        "w_conf": 0.30,  # v26 Run1: less weight on raw confidence
        "w_trend": 0.55,  # v26 Run1: multi-TF trend/context more informative than the Qs
        "w_atr": 0.55,  # v26 Run1: penalizes entries in ATR spikes
        "min_score_threshold": -1e9,
        "max_regime_uncertainty": 0.65,
        # allowed_regimes=[]: DON'T filter by regime — regime acc 55-60% on
        # OOD periods isn't reliable enough for a hard gate. The agent learns by itself
        # when to trade in each regime through the state vector.
        "allowed_regimes": [],
    },
    # ══════════════════════════════════════════════════════════════════════
    # RUNNER CFG
    # ══════════════════════════════════════════════════════════════════════
    "runner_cfg": {
        "device": "cuda",
        "initial_capital": 100000.0,
        "mode": "train",
        "arbiter_bonus": 0,
        "arbiter_sl_penalty": 0,
        "rejection_penalty_weak": 0,
        "decay_steps": 200000,
        "min_shaping_scale": 0.15,
        "arbiter_rolling_window": 50,
        "hold_bias_penalty": 0.01,
        "soft_reset_every_steps": 15000,
        "curriculum": {"steps": [0.00, 0.35, 0.75], "max_trades": [2, 4, 5]},
        # DD: baseline refs used in eval (fixed). Training uses the schedules.
        "baseline_max_dd": 0.20,
        "baseline_daily_dd": 0.08,
        "baseline_avg_dd": 0.10,
        "baseline_dd_env_ratio": 0.60,
        # progressive schedule: wide while exploring, tighter as training goes on
        # fraction-based schedule [0.0-1.0] -> turned into absolute steps by
        # _resolve_schedule() with max_steps of the current WF.
        # e.g. WF of 40k steps: (0.20, 0.40) -> step 8000 -> max_dd=0.40
        # e.g. Bootstrap of 80k: (0.20, 0.40) -> step 16000 -> max_dd=0.40
        "max_dd_schedule": [(0.00, 0.50), (0.20, 0.40), (0.55, 0.32), (0.85, 0.26)],
        "daily_dd_schedule": [(0.00, 0.20), (0.20, 0.15), (0.55, 0.11), (0.85, 0.09)],
        # eval DD: tighter than training (live-like conditions)
        "eval_max_dd": 0.12,
        "eval_daily_dd_limit": 0.06,
        # Conviction filter
        "min_q_gap": 0.015,
        "min_q_gap_train": 0.0,
        "eval_vote_n": 1,
        "eval_vote_threshold": 1,
        "eval_vote_strong_bypass": 1.5,
        "eval_require_positive_q": False,
        "outcome_threshold": 0.0,
        # daily SLs and concurrent positions
        "max_daily_losses": 8,
        "eval_max_daily_losses_by_wf": {0: 8, 1: 7, 2: 6, 3: 5, 4: 5},
        "max_concurrent_positions": 4,
        "concurrent_schedule": [(0.00, 5), (0.60, 3)],
        "daily_reduce_pct": -0.03,
        "daily_block_pct": -0.06,
        # Health filter
        "health_filter_enabled": True,
        "health_window_size": 50,
        "health_min_trades": 30,
        "health_min_expectancy": -0.05,
        "health_min_pf": 1.00,
        "health_penalty_dist_all": 0.25,
        "health_max_penalty": 0.60,
        # directional filters (curriculum: 0.0=OFF, 0.4=Run2, 1.0=Run3)
        "train_filter_scale": 0.0,  # Run1: OFF — pure exploration, no filter in training
        "adx_ranging_max": 0.32,
        "brk_min_mr": 0.30,
        # TREND bonus in training — balances the MR drift of the Q-network.
        # Run1: 0.0 (zeroed — the bonus caused forced TREND trades with WR 4% that
        #           polluted the buffer. The agent learns the difference by itself).
        # Run2/3: 0.0 (train==eval, Q-network already calibrated).
        "trend_train_bonus": 0.0,
        # Rank: max_rank_eval=2 (only rank1+rank2 in eval). Rank curriculum in training.
        "max_rank_eval": 1,
        "eval_max_concurrent": 2,  # max 2 open positions in eval (vs 4-5 in training)
        "rank_qgap_train_r2": 0.000,
        "rank_qgap_train_r3": 0.000,
        "rank_qgap_train_r4": 0.000,
        # joint mode and per-class overrides
        "joint_mode": False,
        "class_overrides": {
            "Majors": {"runner_cfg": {"min_q_gap": 0.04}},
            "Minors": {"runner_cfg": {"min_q_gap": 0.008}},
        },
        # safety gates (OFF in Run1, ON in Run2/3)
        "decision_gate": {
            "enabled": False,
            "regime_mismatch_block": True,
            "atr_spike_mult": 2.5,
            "atr_window": 50,
            "action_pingpong_n": 3,
            "min_conf_product": 0.05,
        },
        "kill_switch": {
            "enabled": False,
            "dd_velocity_pct": -0.01,
            "dd_velocity_window_min": 30,
            "loss_streak_n": 5,
            "pnl_anomaly_mult": 3.0,
            "stuck_positions_n": 3,
            "stuck_min_bars_pct": 0.80,
            "persistence_min": 240,
        },
        "safety": {
            "train_mode": True,
            "decision_gate": {"enabled": False},
            "kill_switch": {"enabled": False},
        },
    },
    # ══════════════════════════════════════════════════════════════════════
    # TRAINING
    # ══════════════════════════════════════════════════════════════════════
    "save_root": "Exploration/Majors",  # relative to <models>/Rainbow/runs
    # ─────────────────────────────────────────────────────────────────────
    # RAINBOW CONFIG  (used when agent_type = "rainbow")
    # ─────────────────────────────────────────────────────────────────────
    "rainbow_cfg": {
        "n_actions": 5,  # v26: HOLD, LONG_TREND, SHORT_TREND, LONG_MR, SHORT_MR
        "hidden_dims": [256, 256, 128],
        "n_quant": 32,  # QR-DQN quantiles
        "noisy_sigma": 0.5,  # NoisyLinear noise
        "lr": 3e-4,
        "batch_size": 256,
        "buffer_capacity": 200_000,
        "per_alpha": 0.6,  # how much the PER priorities weigh
        "per_beta_start": 0.4,  # initial IS correction
        "per_beta_end": 1.0,  # final IS correction
        "per_beta_steps": 300_000,  # steps of the beta annealing
        "gamma": 0.99,
        "n_step": 5,  # n-step returns
        "target_update_tau": 0.005,  # target soft update
        "grad_clip": 1.0,
        "reward_clip": 1.0,
        "warmup_steps": 1000,  # steps before training starts
    },
    "log_every_steps": 5000,  # log the training metrics every N steps
    "seed": 42,
    "training_mode": "full",  # "full" | "retrain" | "eval"
    "joint_mode": False,
    # ["Minors"] or ["Majors", "Minors"] to run the other classes
    "run_classes": ["Majors"],
    # WF -> WF warm start
    # The bootstrap agent starts from scratch. Every next WF warm-starts from the previous one.
    # The encoder changes between WFs (embedding space shift) — the LR warmup compensates.
    "warm_start_cfg": {
        "enabled": True,  # False = every WF starts from scratch
        "reset_replay_buffer": True,  # empties PER + NStepBuffer (old data = noise)
        "reset_target": False,  # keep the target network (more stable)
        "freeze_target_steps": 1000,
        # LR warmup: the first N steps of the WF use lr_warmup_scale,
        # then it rises linearly to lr_scale within lr_warmup_steps
        "lr_warmup_scale": 0.2,  # initial LR of a WF after bootstrap = 3e-4 × 0.2
        "lr_scale": 0.5,  # target LR after the warmup = 3e-4 × 0.5
        "lr_warmup_steps": 1000,  # steps to go from lr_warmup to lr_scale
    },
    # kept for backward compatibility (manual retrain)
    "retrain_cfg": {
        "reset_replay_buffer": True,
        "reset_target": False,
        "freeze_target_steps": 3000,
        "lr_scale": 0.3,
    },
    "retrain_load_paths": {"Majors": None, "Minors": None},
    # ── Joint mode: final agent checkpoints of the single-class runs ──────
    # set them BEFORE starting the joint training:
    #   joint_init_ckpt_M: final Majors agent
    #   joint_init_ckpt_m: final Minors agent
    "joint_init_ckpt_M": None,
    "joint_init_ckpt_m": None,
    # alpha=0.5: equal weight; >0.5: more weight to Majors (more liquid, more data)
    "joint_merge_alpha": 0.5,
    "keepalive": True,
    "keepalive_interval": 300,
}


# ══════════════════════════════════════════════════════════════════════════════
# RUN CONFIGS — what changes from DEFAULT_CFG in each training phase
# ══════════════════════════════════════════════════════════════════════════════
# Only the values that CHANGE from DEFAULT_CFG.
# build_cfg(run_mode) merges them.

RUN_CONFIGS: dict[str, dict[str, Any]] = {
    # ──────────────────────────────────────────────────────────────────────
    # RUN 1 — EXPLORATION MAJORS (separate)
    # goal: free exploration, data collection, warm start for Run2
    # ──────────────────────────────────────────────────────────────────────
    "run1_majors": {
        "save_root": "Exploration/Majors",
        "run_classes": ["Majors"],
        "joint_mode": False,
        "rainbow_cfg": {
            "elite_sample_frac": 0.25,
            "elite_buffer_capacity": 20_000,
        },
        "runner_cfg": {
            # Run1: light filters, wide DD, explore everything
            "max_rank_eval": 2,
            "rank_qgap_train_r3": 0.000,
            "rank_qgap_train_r4": 0.000,
        },
    },
    # ──────────────────────────────────────────────────────────────────────
    # RUN 1 — EXPLORATION MINORS (separate)
    # same as Majors, on the Minors assets
    # ──────────────────────────────────────────────────────────────────────
    "run1_minors": {
        "save_root": "Exploration/Minors",
        "run_classes": ["Minors"],
        "joint_mode": False,
        "rainbow_cfg": {
            "elite_sample_frac": 0.25,
            "elite_buffer_capacity": 20_000,
        },
        "runner_cfg": {
            "max_rank_eval": 2,
            "rank_qgap_train_r3": 0.000,
            "rank_qgap_train_r4": 0.000,
        },
    },
    # ──────────────────────────────────────────────────────────────────────
    # RUN 2 — CONSOLIDATION JOINT (Majors + Minors together)
    # goal: consolidate the policy, filters at 40%, joint portfolio
    # needs: Run1 done for both classes
    # ──────────────────────────────────────────────────────────────────────
    "run2_joint": {
        "save_root": "Consolidation/Joint",
        "run_classes": ["Majors", "Minors"],
        "joint_mode": True,
        "rainbow_cfg": {
            "elite_sample_frac": 0.25,
            "elite_buffer_capacity": 20_000,
        },
        "runner_cfg": {
            # curriculum filters at 40%
            "train_filter_scale": 0.40,
            "adx_ranging_max": 0.30,
            "brk_min_mr": 0.28,
            "trend_train_bonus": 0.0,  # Run2: off — Q-network already calibrated
            # rank curriculum: rank3+ need conviction in training
            "max_rank_eval": 2,
            "rank_qgap_train_r3": 0.025,
            "rank_qgap_train_r4": 0.040,
            # tighter DD (more mature agent)
            "max_dd_schedule": [(0.00, 0.32), (0.20, 0.26), (0.55, 0.22), (0.85, 0.18)],
            "daily_dd_schedule": [(0.00, 0.12), (0.20, 0.09), (0.55, 0.07), (0.85, 0.06)],
            "eval_max_dd": 0.11,
            "eval_daily_dd_limit": 0.05,
            # daily SLs and concurrent positions
            "max_daily_losses": 5,
            "eval_max_daily_losses_by_wf": {0: 7, 1: 6, 2: 5, 3: 5, 4: 5},
            "max_concurrent_positions": 3,
            "concurrent_schedule": [(0.00, 3), (0.60, 2)],
            "daily_reduce_pct": -0.015,
            "daily_block_pct": -0.02,
            # stricter conviction filter
            "min_q_gap": 0.012,
            "eval_vote_n": 5,
            "eval_vote_threshold": 3,
            "eval_vote_strong_bypass": 1.75,
            "eval_require_positive_q": True,
            # stricter health filter
            "health_min_trades": 20,
            "health_min_expectancy": 0.0,
            "health_min_pf": 1.03,
            "health_max_penalty": 0.55,
            # Safety gates ON
            "decision_gate": {
                "enabled": True,
                "regime_mismatch_block": True,
                "atr_spike_mult": 2.5,
                "atr_window": 50,
                "action_pingpong_n": 3,
                "min_conf_product": 0.05,
            },
            "kill_switch": {"enabled": False},  # still OFF in Run2
            "joint_mode": True,
        },
        "arbiter_cfg": {
            "w_q": 1.0,
            "w_conf": 0.25,
            "w_trend": 0.65,
            "w_atr": 0.60,
            "max_trades_per_day": 5,
            "allowed_regimes": [],
        },
    },
    # ──────────────────────────────────────────────────────────────────────
    # RUN 3 — PRODUCTION JOINT (Majors + Minors together)
    # goal: final policy, train==eval, prop-firm ready
    # needs: Run2 done
    # ──────────────────────────────────────────────────────────────────────
    "run3_joint": {
        "save_root": "Production/Joint",
        "run_classes": ["Majors", "Minors"],
        "joint_mode": True,
        "rainbow_cfg": {
            "elite_sample_frac": 0.25,
            "elite_buffer_capacity": 20_000,
        },
        "runner_cfg": {
            # filters at 100% (train==eval)
            "train_filter_scale": 1.00,
            "adx_ranging_max": 0.28,
            "brk_min_mr": 0.25,
            "trend_train_bonus": 0.0,  # Run3: off — train==eval
            # full rank curriculum
            "max_rank_eval": 2,
            "rank_qgap_train_r3": 0.040,
            "rank_qgap_train_r4": 0.060,
            # live-like DD
            "max_dd_schedule": [(0.00, 0.18), (0.25, 0.14), (0.60, 0.10), (0.85, 0.08)],
            "daily_dd_schedule": [(0.00, 0.07), (0.25, 0.05), (0.60, 0.04), (0.85, 0.035)],
            "eval_max_dd": 0.10,
            "eval_daily_dd_limit": 0.04,
            # daily SLs and concurrent positions (prop firm)
            "max_daily_losses": 3,
            "eval_max_daily_losses_by_wf": {0: 4, 1: 4, 2: 3, 3: 3, 4: 3},
            "max_concurrent_positions": 2,
            "concurrent_schedule": [(0.00, 2)],
            "daily_reduce_pct": -0.010,
            "daily_block_pct": -0.020,
            # maximum conviction
            "min_q_gap": 0.015,
            "eval_vote_n": 5,
            "eval_vote_threshold": 4,
            "eval_vote_strong_bypass": 2.0,
            "eval_require_positive_q": True,
            # production health filter
            "health_min_trades": 20,
            "health_min_expectancy": 0.0,
            "health_min_pf": 1.05,
            "health_max_penalty": 0.60,
            # every safety gate ON
            "decision_gate": {
                "enabled": True,
                "regime_mismatch_block": True,
                "atr_spike_mult": 2.5,
                "atr_window": 50,
                "action_pingpong_n": 3,
                "min_conf_product": 0.05,
            },
            "kill_switch": {
                "enabled": True,
                "dd_velocity_pct": -0.01,
                "dd_velocity_window_min": 30,
                "loss_streak_n": 5,
                "pnl_anomaly_mult": 3.0,
                "stuck_positions_n": 3,
                "stuck_min_bars_pct": 0.80,
                "persistence_min": 240,
            },
            "joint_mode": True,
        },
        "arbiter_cfg": {
            "w_q": 1.0,
            "w_conf": 0.20,
            "w_trend": 0.70,
            "w_atr": 0.65,
            "max_trades_per_day": 5,
            "allowed_regimes": [],
        },
    },
}


def _deep_update(base: dict, overrides: dict) -> dict:
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            base[key].update(val)
        else:
            base[key] = val
    return base


def build_cfg(run_mode: str, overrides: dict | None = None, models_root=None) -> dict[str, Any]:
    """DEFAULT_CFG + RUN_CONFIGS[run_mode] + overrides (one level deep), absolute save_root."""
    if run_mode not in RUN_CONFIGS:
        raise ValueError(f"invalid run_mode '{run_mode}'. Pick one of: {list(RUN_CONFIGS)}")
    cfg = _deep_update(copy.deepcopy(DEFAULT_CFG), copy.deepcopy(RUN_CONFIGS[run_mode]))
    _deep_update(cfg, copy.deepcopy(overrides or {}))
    save_root = Path(cfg["save_root"])
    if not save_root.is_absolute():
        cfg["save_root"] = str(Path(models_root or MODELS_ROOT) / "Rainbow" / "runs" / save_root)
    return cfg
