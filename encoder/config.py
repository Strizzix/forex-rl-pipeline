"""
config.py — default training configuration of the encoder.

DEFAULT_CONFIG is the full set of knobs used by trainer.train_pretrainer_with_contrastive.
Change a value for one run with `--set key=value` or a JSON file (`--config`),
see `python -m encoder.trainer --help`.

Folders (same root as data_management):
    <root>/Scaled/<Majors|Minors>/<ASSET>/...           input (scaled parquets + scalers)
    <models>/Encoder/<class>/Checkpoint/Bootstrap        bootstrap run
    <models>/Encoder/<class>/Checkpoint/Retrain/<tag>    walk-forward retrain <tag>
"""

import copy
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("FX_DATA_ROOT", "data"))
MODELS_ROOT = Path(os.environ.get("FX_MODELS_ROOT", "models"))


def default_results_dir(models_root, class_name, mode, tag=None):
    base = Path(models_root) / "Encoder" / class_name / "Checkpoint"
    if mode == "retrain":
        return str(base / "Retrain" / (tag or "retrain"))
    return str(base / "Bootstrap")


DEFAULT_CONFIG = {
    "class_name": "Majors",
    "data_dir": None,  # default <root>/Scaled
    "results_dir": None,  # default from default_results_dir()
    "symbols": [],  # [] = every asset found under data_dir/<class>
    "tf_list": ["15min", "30min", "1h", "4h", "1d"],
    # ── model ───────────────────────────────────────────────────────────────
    "seq_len": 256,
    "batch_size": 64,
    "step_stride": 8,
    "tcn_channels": [64, 64, 128, 128, 256, 256],
    "kernel_size": 3,
    "gru_hidden": 128,
    "gru_layers": 1,
    "dropout": 0.25,
    "strategy_context_dim": 29,
    "n_assets": 5,
    "asset_embed_dim": 8,
    "use_session_features": True,
    # v30: 5 structural features validated OOS (2025 holdout confirmed) as continuous
    # encoder inputs. False = dataset identical to before (instant rollback).
    # Turn on at WF8 to train with the new set; compare val AUC +
    # encoder_filter_sweep against the baseline with the same WF/seed.
    "use_structural_features": False,
    # the input must be scaled data with bar-close labels (data_management >= 0.2);
    # True only to train on files of the older downloader anyway
    "allow_open_labels": False,
    # ── global seed — ONLY for A/B tests (feature on/off) ───────────────────
    # use_global_seed=True makes training reproducible (same initial weights,
    # shuffle, dropout), so a baseline-vs-feature comparison isolates the effect of
    # the features alone. global_seed_deterministic=True sets cudnn.deterministic +
    # benchmark=False: on this TCN (dilated conv1d) that's VERY slow (up to 10×),
    # and the seed alone is enough for the comparison -> leave it False.
    # Tests done: set use_global_seed=False again for production.
    "use_global_seed": False,
    "global_seed": 42,
    "global_seed_deterministic": False,  # keep False (True is very slow on the TCN)
    "global_seed_full_determinism": False,  # always False (even slower)
    # AB-COMPARE: two explicit ROOTS to keep OFF and ON physically apart.
    # Under each root the trainer writes <class>/Checkpoint, so the production
    # encoders are never touched. None = production root. Set
    # ab_separate_outputs=True for the tests.
    "ab_separate_outputs": False,
    "ab_root_features_off": None,  # e.g. ".../Encoder_AB_base"
    "ab_root_features_on": None,  # e.g. ".../Encoder_AB_struct"
    # ── market structure head ───────────────────────────────────────────────
    "w_mktstructure": 0.15,
    "trend_momentum_huber_delta": 1.0,
    # thresholds per asset class:
    # Majors: tighter ATR/price -> higher dist threshold to separate RANGING from TRENDING
    # Minors: wider ATR/price (JPY crosses, GBP crosses) -> lower threshold
    # dist_thresh: fraction of 1 ATR from the EMA below which the bar is RANGING
    # bo_window: breakout window in candles (shorter = more BREAKOUT events)
    "mktstructure_dist_thresh_majors": {
        "15min": 0.30,
        "30min": 0.30,
        "1h": 0.50,
        "4h": 0.70,
        "1d": 0.70,
    },
    "mktstructure_dist_thresh_minors": {
        "15min": 0.25,
        "30min": 0.25,
        "1h": 0.35,
        "4h": 0.50,
        "1d": 0.50,
    },
    "mktstructure_bo_window_majors": {"15min": 20, "30min": 20, "1h": 20, "4h": 10, "1d": 10},
    "mktstructure_bo_window_minors": {"15min": 20, "30min": 20, "1h": 15, "4h": 8, "1d": 8},
    # v25: per-TF class weights = inverse frequency (w_brk = ranging_pct / brk_pct)
    # 0=RANGING, 1=LONG_BRK, 2=SHORT_BRK
    "mktstructure_class_weights": [1.0, 6.0, 6.0],  # fallback for TFs not listed
    "mktstructure_class_weights_by_tf": {
        "15min": [1.0, 3.5, 3.8],  # RANGING≈64% LONG≈18% SHORT≈17%
        "30min": [1.0, 6.5, 7.7],  # RANGING≈77% LONG≈12% SHORT≈10%
        "1h": [1.0, 11.0, 13.0],  # RANGING≈86% LONG≈8%  SHORT≈6%
        "4h": [1.0, 19.0, 22.0],  # RANGING≈91% LONG≈5%  SHORT≈4%
        "1d": [1.0, 80.0, 98.0],  # RANGING≈98% LONG≈1%  SHORT≈1%
    },
    "strategy_hit_focal_gamma": 2.0,  # also used by the session/pdhl heads
    "strategy_side_margin_atr": 0.15,
    "strategy_side_min_hit_bars": 2,
    # ── v28: SessionBreakoutHead + PDHLBreakoutHead + RegimeQualityHead ─────
    # SESSION: different RR per universe (best from the TP sweep)
    "outcome_session_tp_r_majors": 2.5,
    "outcome_session_tp_r_minors": 1.8,
    "outcome_session_sl_r": 1.0,
    "strategy_session_horizon_bars": 40,
    # PDHL: RR=1.8 best for both universes
    "outcome_pdhl_tp_r": 1.8,
    "outcome_pdhl_sl_r": 1.0,
    "strategy_pdhl_horizon_bars": 64,
    # PDHL DI gate (as the simulator): long only if tanh((DI+ - DI-)/20) on 4H > thresh,
    # short only if < -thresh. It never worked before 0.2 (column case), set False to get
    # the labels of the earlier runs back.
    "outcome_pdhl_di_gate": True,
    "outcome_pdhl_di_thresh": 0.15,
    # executable fills, entry bar in the scan, gap-aware horizon (see dataset)
    "outcome_strict_execution": True,
    "regime_quality_horizon_bars": 32,
    # loss weights of the strategy heads
    "w_session_head": 0.5,
    "w_pdhl_head": 0.5,
    "w_regime_quality_head": 0.15,  # low: the label comes from side -> redundant gradient
    "v28_min_n_for_full_score": 150.0,  # coverage penalty: min n for a full score
    "qual_context_dim": 29,  # = strategy_context_dim
    # pos_weight of the side loss, from the best RRs of the TP sweep:
    # SESSION Majors RR=2.5: WR~37% -> pos_weight=(1-0.37)/0.37 ≈ 1.70
    # SESSION Minors RR=1.8: WR~43% -> pos_weight=(1-0.43)/0.43 ≈ 1.33
    # weighted average (~2 universes): ≈ 1.5
    # PDHL RR=1.8 both: WR~52% -> pos_weight=(1-0.52)/0.52 ≈ 0.92 -> 1.0
    "session_side_pos_weight": 1.5,
    "pdhl_side_pos_weight": 1.0,
    "score_w_v28": 0.60,  # the heads the live system decides with
    # ── BREAKEVEN in the labels ─────────────────────────────────────────────
    # Needed by the sample weights: finds the trades BE closes at break-even.
    # Without it be_reachable is always 0 and the weight is a constant.
    # Must match BE_ENABLED / BE_TRIGGER_R / BE_OFFSET_R of the live system.
    "outcome_be_enabled": True,
    "outcome_be_trigger_r": 0.25,
    "outcome_be_offset_r": 0.07,
    # ── anti-phase penalty ──────────────────────────────────────────────────
    # 0.0 = MEASURE only (metrics antiphase_session / antiphase_pdhl).
    # Start from 0: look at the long/short correlation first. If it's strongly
    # negative the anti-phase is real, then raise it to 0.05-0.10.
    "w_antiphase": 0.0,
    # which bar the model sees: 1 = closed bars only. Must match
    # ENCODER_SCORE_OFFSET of the live system and of the backtest.
    "score_offset": 1,
    # ── optimization ────────────────────────────────────────────────────────
    "lr": 1.4e-4,
    "weight_decay": 1e-6,
    "trainer": {
        "gradient_clip": 1.0,
        "device": None,  # None = cuda if available, else cpu
    },
    "num_workers": 4,
    "prefetch_factor": 2,
    "accumulation_steps": 1,
    "validate_every_epochs": 1,
    "val_max_batches": 200,
    "keepalive": True,
    "keepalive_interval_seconds": 300,
    "debug_mode": False,
    "mask_prob": 0.25,
    # ── trend labels ────────────────────────────────────────────────────────
    # "atr_adjusted": threshold = atr_mult × TF_multiplier × ATR%  -> ~25/50/25
    # "absolute": fixed threshold on the price return             -> ~10/80/10 (legacy)
    "trend_label_mode": "atr_adjusted",
    "trend_atr_mult": 0.4,  # base multiplier in ATR units; 0.3 if still FLAT-heavy
    "trend_atr_mult_tf": None,  # None = {15min:1.0, 30min:1.4, 1h:2.0, 4h:4.0, 1d:8.0}
    # legacy absolute threshold — only with trend_label_mode="absolute" or no ATR
    "trend_quantile_tail": 0.20,
    "force_recompute_trend_thresholds": False,
    "use_quantile_trend_thresholds": False,
    "trend_threshold_scaled": 1e-3,
    "trend_threshold_scaled_tf": {},
    # ── trend loss ──────────────────────────────────────────────────────────
    "trend_focal_gamma": 2.0,
    "emb_var_reg_weight": 0.1,  # penalizes embedding dimensional collapse (intra_var -> 0)
    "trend_dynamic_weights_alpha": 0.2,  # blend: 0=dynamic only, 1=static only
    "log_trend_train_debug": True,
    "log_trend_train_every": 500,
    "atr_lookback": 14,
    "force_recompute_thresholds": True,
    # ── contrastive ─────────────────────────────────────────────────────────
    "proj_dim": 128,
    "temperature": 0.1,
    "loss_weights": {"cross_pos": 1.0, "cross_neg": 1.0, "simclr_batch": 1.0},
    "lambda_contrast": 0.05,
    # directional contrastive loss (v25): pushes bull embeddings away from bear ones
    # using the sign of tmom_30m; margin = cosine threshold
    "lambda_dir_contrast": 0.15,
    "dir_contrast_margin": 0.3,
    "dir_contrast_tmom_thr": 0.3,
    "dir_contrast_tmom_tf_idx": 1,  # 1 = 30min in the standard tf_list
    "dir_contrast_max_pairs": 2048,
    "contrast_k_neg": 64,
    "memory_bank_size": 16384,
    "use_memory_bank_for_negatives": True,
    "simclr_every": 2,
    "aug_workers": 4,
    "augment": {
        "jitter": 0.02,
        "scaling": 0.05,
        "dropout": 0.1,
        "time_mask": 10,
        "cutout": 16,
        "permute_prob": 0.1,
        "use_gpu_augment": True,
        "deterministic": False,
        "seed": 42,
    },
    # ── loss balancing ──────────────────────────────────────────────────────
    "use_ema_balancing": True,
    "ema_alpha": 0.99,
    "ema_alpha_warmup": 0.95,
    "ema_warmup_epochs": 2,
    "variance_loss_weight": 0.1,
    "dummy_reg": 1e-6,
    "use_amp": True,
    "trend_loss_exclude_tfs": ["15min"],  # too many samples per bar, overwhelms the gradient
    "label_exclude_tfs": ["15min"],
    # ── score weights (sum 1.0) ─────────────────────────────────────────────
    # The old weights (0.2 trend + 0.4 regime + 0.5 v28 = 1.1) gave 40% to a
    # SATURATED metric: regime_corr sits at 0.92-0.94 on every run. It didn't change
    # the epoch choice but raised the score LEVEL by a fixed ~0.37. Now the weight
    # goes to what really discriminates; v28 dominates (it's what the live system
    # uses), regime stays as a guard on the backbone health.
    # Note: The score level changed: the 0.72-0.75 envelope and the 0.68 re-bootstrap
    # threshold must be rebuilt on the new runs.
    "score_w_trend": 0.25,
    "score_w_outcome": 0.0,  # v27: classic OutcomeHead off
    "score_w_regime": 0.15,
    "emb_max_points": 2048,
    "emb_seed": 0,
    "resume_from": None,
    # ── training mode ───────────────────────────────────────────────────────
    "training_mode": "bootstrap",  # "bootstrap" | "retrain"
    # walk-forward schedule (half-yearly), e.g.:
    #   WF1: train 2019-01 -> 2022-12  val 2023-01 -> 2023-06  tag="WF1_2023H1"
    #   WF2: train 2019-07 -> 2023-06  val 2023-07 -> 2023-12  tag="WF2_2023H2"
    #   ...
    # retrain: set these fields (or --wf-tag/--train-end/--val-start/--val-end/--start-from)
    "retrain_train_end": "2025-12-31",
    "retrain_val_start": "2026-01-01",
    "retrain_val_end": "2026-06-30",
    "retrain_version_tag": "WF7_2026H1",
    "retrain_epochs": 30,
    "rolling_years": 4,
    "retrain_schedule_custom": None,
    "retrain_start_from": None,  # checkpoint of the previous WF
    "retrain_save_mode": "best",  # best: phase3 LR very low, peak often at ep7-8 (end of phase2)
    # pre-retrain baseline: evaluates the anchor checkpoint on the new val period
    # BEFORE any training (pre_retrain_baseline_<tag>.json). ~1 min.
    "pre_retrain_baseline_eval": True,
    # baseline score < (1 - threshold) -> "DEGRADED" in the log: the anchor
    # checkpoint may be wrong or too old
    "baseline_degradation_threshold": 0.05,
    "bootstrap_train_start": "2014-01-01",
    "bootstrap_train_end": "2022-12-31",
    "bootstrap_val_start": "2023-01-01",
    "bootstrap_val_end": "2023-12-31",
    "bootstrap_epochs": 30,
    "freeze_tcn": "none",
    "freeze_tcn_keep_last_n": 1,
    "w_regime_bootstrap": 0.8,  # regime continuous Huber loss weight
    "regime_cont_mae_threshold": 0.8,  # retrain_safe: max mean_mae
    "w_regime_retrain": 1.2,
    # trend_loss_weight interacts with EMA balancing: _ema() normalizes every loss
    # to its own EMA (each ≈ 1.0 in the total), so tmom really gets this × the others.
    "trend_loss_weight": 2.0,
    "tmom_flat_threshold": 0.15,  # v25: flat samples left out of the loss
    "tmom_sign_penalty_weight": 0.3,  # hinge on the sign, doesn't distort the magnitude
    "trend_ema_bypass_epochs": 3,  # no EMA balancing for the trend in epochs 1-3
    "use_kl_anchor": True,
    "kl_anchor_weight": 0.015,
    "kl_anchor_ckpt": None,
    # embedding drift anchor: stable pre-COVID period, inside every training window
    "drift_anchor_start": "2019-01-01",
    "drift_anchor_end": "2019-12-31",
    "drift_threshold": 0.20,
    "drift_max_batches": 30,
    "drift_max_points": 1024,
    "regime_drift_threshold": 0.12,
    "live_mode": False,
    "live_val_days": 21,
    # 1d left out of the per-batch monitor (too few samples), still in loss and val metrics
    "monitor_tfs": ["30min", "1h", "4h"],
    "monitor_ema_alpha": 0.9,
    "regime_conf_low_threshold": 0.00,  # disabled: regime is continuous now
}


def build_config(overrides=None, root=None, models_root=None):
    """DEFAULT_CONFIG + overrides, with data_dir / results_dir filled in."""
    conf = copy.deepcopy(DEFAULT_CONFIG)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(conf.get(k), dict):
            conf[k].update(v)
        else:
            conf[k] = v
    if not conf.get("data_dir"):
        conf["data_dir"] = str(Path(root or DATA_ROOT) / "Scaled")
    if not conf.get("results_dir") and not conf.get("ab_separate_outputs"):
        conf["results_dir"] = default_results_dir(
            models_root or MODELS_ROOT,
            conf["class_name"],
            conf["training_mode"],
            conf.get("retrain_version_tag"),
        )
    return conf
