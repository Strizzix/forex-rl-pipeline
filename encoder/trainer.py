"""
trainer.py — multi-head pretraining of the TCN-GRU market encoder.

Model
─────
• TCNGRUEncoder: TCN (dilated causal convolutions) + GRU, with asset embedding.
• Heads:
    - TrendMomentumHeadTF  : continuous trend momentum regression, one scalar per TF
    - RegimeContinuousHead : [vol_regime, atr_norm, adx, di_direction] per TF
    - MarketStructureHead  : RANGING / LONG_BREAKOUT / SHORT_BREAKOUT per TF
    - SessionBreakoutHead  : P(TP first) of the London session breakout, long/short
    - PDHLBreakoutHead     : P(TP first) of the PDH/PDL breakout, long/short
    - RegimeQualityHead    : is the bar tradable for the deployed strategies
    - ProjectionHead       : SimCLR / cross-asset contrastive + directional contrastive

Training modes
──────────────
• bootstrap: train from scratch on a fixed train/val split.
• retrain:   walk-forward retrain from the previous checkpoint (rolling window,
             phase schedule per asset class, KL anchor, drift checks).

Run: python -m encoder.trainer --help
"""

import gc
import glob
import json
import logging
import math
import multiprocessing as mp
import os
import threading
import time
import warnings
from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as tmp
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from .components import (
    ContrastivePretrainer,
    DirectionalContrastiveLoss,
    MarketStructureHead,
    PDHLBreakoutHead,
    ProjectionHead,
    RegimeContinuousHead,
    RegimeQualityHead,
    SessionBreakoutHead,
    TCNGRUEncoder,
    TrendMomentumHeadTF,
    _validate_batch,
    variance_loss,
)
from .config import DATA_ROOT, MODELS_ROOT, build_config, default_results_dir
from .dataset import (
    MarketPretrainConfig,
    MarketPretrainDataset,
    analyze_threshold_impact,
    autodetect_feature_columns,
    compute_trend_thresholds_for_asset,
    discover_asset_parquets,
    discover_scaler_files_auto,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("encoder_trainer")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
except Exception:
    pass
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def set_global_seed(seed: int = 42, deterministic: bool = True, full_determinism: bool = False):
    """Makes training REPRODUCIBLE for the A/B comparison (feature on/off).
    Fixes the sources of randomness that matter for the comparison: initial
    weights, shuffle, dropout, augmentation, numpy — all free in speed.

    deterministic=True: turns off cudnn.benchmark and turns on cudnn.deterministic.
      Moderate cost (~10-20%). Enough for a reliable A/B comparison: same initial
      weights and same data order between baseline and new.

    full_determinism=True: also torch.use_deterministic_algorithms + CUBLAS
      workspace. ⚠️ VERY SLOW (can be 10× or more: some deterministic kernels
      fall back to slow implementations or the CPU). NOT needed for the A/B
      comparison — it only removes ~1e-6 floating-point differences, irrelevant
      when comparing AUCs. Leave it False unless you need bit-for-bit
      reproducibility.
    """
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if full_determinism:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        # IMPORTANT: use_deterministic_algorithms is a GLOBAL process flag.
        # If a previous run in the same Python session turned it on, it stays
        # on (and very slow) until we turn it off EXPLICITLY here.
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
    logger.info(
        "[SEED] global seed=%d | cudnn.deterministic=%s | full_determinism=%s "
        "(full=True is very slow and NOT needed for the A/B comparison)",
        seed,
        deterministic,
        full_determinism,
    )


# =============================================================
# Paths
# =============================================================
def resolve_paths(conf: dict[str, Any]) -> dict[str, Any]:
    """Fills data_dir / results_dir (defaults from config.py) and applies the AB roots."""
    class_name = conf.get("class_name", "Majors")
    if not conf.get("data_dir"):
        conf["data_dir"] = str(DATA_ROOT / "Scaled")

    # ── AB-COMPARE: this root has PRIORITY (even if results_dir is already set).
    # With ab_separate_outputs on, the OFF/ON root is picked from
    # use_structural_features and OVERWRITES results_dir, so the two experiments
    # never end up in the same folder, nor in production.
    if conf.get("ab_separate_outputs", False):
        _struct_on = bool(conf.get("use_structural_features", False))
        _off_root = conf.get("ab_root_features_off")
        _on_root = conf.get("ab_root_features_on")
        _root = _on_root if _struct_on else _off_root
        # GUARD: if the root of this mode is missing, DON'T fall back to the
        # production folder — stop here.
        if not _root:
            raise RuntimeError(
                f"[AB-COMPARE] ab_separate_outputs=True but "
                f"'ab_root_features_{'on' if _struct_on else 'off'}' is missing. "
                f"Set the root for the "
                f"{'ON (struct)' if _struct_on else 'OFF (base)'} mode "
                f"or turn ab_separate_outputs off. "
                f"The production folder is not used, for safety."
            )
        conf["results_dir"] = os.path.join(_root, class_name, "Checkpoint")
        return conf

    # normal behaviour (production): only if not set already
    if not conf.get("results_dir"):
        conf["results_dir"] = default_results_dir(
            MODELS_ROOT,
            class_name,
            str(conf.get("training_mode", "bootstrap")).lower(),
            conf.get("retrain_version_tag"),
        )
    return conf


# =============================================================
# Schedules
# =============================================================
# ── Retrain schedules — 10 epochs ─────────────────────────────────────────
# Rationale:
#   Phase 1 (ep 1-3): backbone frozen, only heads adapt to new WF window.
#     regime_cont_head FROZEN — it needs the GRU to be stable first.
#     mktstr+trend+vol free @ moderate LR to absorb distributional shift.
#   Phase 2 (ep 4-8): GRU + regime_cont_head unfrozen at low LR.
#     TCN-last-2 blocks also unfrozen at very low LR for subtle adaptation.
#     regime_cont_head needs more epochs because adx is the slowest signal.
#   Phase 3 (ep 9-10): refreeze GRU+regime_cont, finalize heads only.
#     Prevents late-stage overfitting on small WF windows.
# Majors: tighter LRs — less distributional shift between WF windows.
# Minors: slightly higher LRs — more variation in vol/spread between windows.

MAJORS_SCHEDULE = [
    {
        "name": "phase1_freeze_backbone",
        "epochs": (1, 3),
        "freeze_tcn": True,
        "freeze_gru": True,
        "freeze_backbone_other": False,
        "freeze_regime_cont_head": False,
        "lr_multipliers": {
            "tcn": 0.0,
            "gru": 0.0,
            "backbone_other": 0.05,
            "heads": 0.25,
            "regime_cont_head": 0.01,
        },
        "description": "Freeze TCN+GRU | heads+regime_cont(micro) adapt to WF window",
    },
    {
        "name": "phase2_unfreeze_gru_regime",
        "epochs": (4, 25),
        "freeze_tcn": True,
        "freeze_gru": False,
        "freeze_backbone_other": False,
        "freeze_regime_cont_head": False,
        "unfreeze_tcn_last_n": 2,
        "lr_multipliers": {
            "tcn": 0.0,
            "tcn_last": 0.008,
            "gru": 0.04,
            "backbone_other": 0.04,
            "heads": 0.12,
            "regime_cont_head": 0.07,
        },
        "description": "GRU+regime_cont+TCN-last2 trainable | fine-tune embedding",
    },
    {
        "name": "phase3_finalize",
        "epochs": (26, 30),
        "freeze_tcn": True,
        "freeze_gru": True,
        "freeze_backbone_other": False,
        "freeze_regime_cont_head": True,
        "lr_multipliers": {
            "tcn": 0.0,
            "gru": 0.0,
            "backbone_other": 0.025,
            "heads": 0.07,
            "regime_cont_head": 0.0,
        },
        "description": "Refreeze backbone | finalize heads only",
    },
]

MINORS_SCHEDULE = [
    {
        "name": "phase1_freeze_backbone",
        "epochs": (1, 3),
        "freeze_tcn": True,
        "freeze_gru": True,
        "freeze_backbone_other": False,
        "freeze_regime_cont_head": False,
        "lr_multipliers": {
            "tcn": 0.0,
            "gru": 0.0,
            "backbone_other": 0.05,
            "heads": 0.30,
            "regime_cont_head": 0.015,
        },
        "description": "Freeze TCN+GRU | heads+regime_cont(micro) adapt to WF window",
    },
    {
        "name": "phase2_unfreeze_gru_regime",
        "epochs": (4, 25),
        "freeze_tcn": True,
        "freeze_gru": False,
        "freeze_backbone_other": False,
        "freeze_regime_cont_head": False,
        "unfreeze_tcn_last_n": 2,
        "lr_multipliers": {
            "tcn": 0.0,
            "tcn_last": 0.008,
            "gru": 0.07,
            "backbone_other": 0.05,
            "heads": 0.15,
            "regime_cont_head": 0.11,
        },
        "description": "GRU+regime_cont+TCN-last2 trainable | higher LR for cross-class shift",
    },
    {
        "name": "phase3_finalize",
        "epochs": (26, 30),
        "freeze_tcn": True,
        "freeze_gru": True,
        "freeze_backbone_other": False,
        "freeze_regime_cont_head": True,
        "lr_multipliers": {
            "tcn": 0.0,
            "gru": 0.0,
            "backbone_other": 0.025,
            "heads": 0.08,
            "regime_cont_head": 0.0,
        },
        "description": "Refreeze backbone | finalize heads only",
    },
]

_SCHEDULES = {"majors": MAJORS_SCHEDULE, "minors": MINORS_SCHEDULE}


def auto_select_schedule(class_name: str) -> tuple[str, list[dict]]:
    key = class_name.lower().strip()
    if key in _SCHEDULES:
        return key, _SCHEDULES[key]
    return ("minors", MINORS_SCHEDULE) if "minor" in key else ("majors", MAJORS_SCHEDULE)


def get_phase_for_epoch(schedule: list[dict], epoch_in_retrain: int) -> Optional[dict]:
    for phase in schedule:
        s, e = phase["epochs"]
        if s <= epoch_in_retrain <= e:
            return phase
    return None


# =============================================================
# Module helpers
# =============================================================
def get_tcn_module(encoder) -> Optional[nn.Module]:
    for attr in ("tcn", "tcn_net", "tcn_blocks", "temporal_conv", "conv_net"):
        if hasattr(encoder, attr):
            return getattr(encoder, attr)
    for _, mod in encoder.named_children():
        children = list(mod.children())
        if children and any(
            isinstance(c, (nn.Conv1d, nn.Sequential, nn.ModuleList)) for c in children
        ):
            return mod
    return None


def get_gru_module(encoder) -> Optional[nn.Module]:
    for attr in ("gru", "rnn", "lstm", "temporal_rnn", "gru_net"):
        if hasattr(encoder, attr):
            return getattr(encoder, attr)
    for _, mod in encoder.named_children():
        if isinstance(mod, (nn.GRU, nn.LSTM, nn.RNN)):
            return mod
    return None


def set_requires_grad(module: Optional[nn.Module], flag: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = flag


# =============================================================
# Live-Safe Trend Monitor
# =============================================================
def compute_trend_monitor_stats(
    trend_scalar_batch: torch.Tensor,
    is_close_batch: torch.Tensor,
    tf_list: list[str],
    monitor_tfs: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Per-batch scalar trend monitor stats.
    trend_scalar_batch: [B, n_tfs] — ATR-normalised forward return in [-3, 3].
    Stats: mean, std, pct_bullish (>0.1), pct_bearish (<-0.1), pct_neutral.
    """
    stats = {}
    monitor_tfs = monitor_tfs or tf_list
    with torch.no_grad():
        scalars = trend_scalar_batch.float()
    for i, tf in enumerate(tf_list):
        if tf not in monitor_tfs:
            continue
        mask = is_close_batch[:, i].bool()
        n = int(mask.sum().item())
        if n == 0:
            for k in (
                f"tmom_mean_{tf}",
                f"tmom_std_{tf}",
                f"pct_bullish_{tf}",
                f"pct_bearish_{tf}",
                f"pct_neutral_{tf}",
            ):
                stats[k] = float("nan")
            stats[f"n_samples_{tf}"] = 0
            continue
        vals = scalars[mask, i]
        nf = float(n)
        stats[f"tmom_mean_{tf}"] = round(float(vals.mean().item()), 4)
        stats[f"tmom_std_{tf}"] = round(float(vals.std(correction=0).item()), 4)
        stats[f"pct_bullish_{tf}"] = round(float((vals > 0.1).sum().item()) / nf * 100, 2)
        stats[f"pct_bearish_{tf}"] = round(float((vals < -0.1).sum().item()) / nf * 100, 2)
        stats[f"pct_neutral_{tf}"] = round(float((vals.abs() <= 0.1).sum().item()) / nf * 100, 2)
        stats[f"n_samples_{tf}"] = n
    return stats


def compute_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 3,
) -> dict[str, Any]:
    """Per-class accuracy + row-normalised confusion matrix.
    Returns {per_class_acc, per_class_n, confusion_pct, macro_acc}."""
    result: dict[str, Any] = {
        "per_class_acc": {},
        "per_class_n": {},
        "confusion_pct": {},
        "macro_acc": float("nan"),
    }
    if len(y_true) == 0:
        return result
    accs, weights = [], []
    for cls in range(n_classes):
        mask = y_true == cls
        n_cls = int(mask.sum())
        result["per_class_n"][cls] = n_cls
        if n_cls == 0:
            result["per_class_acc"][cls] = float("nan")
            result["confusion_pct"][cls] = {}
            continue
        acc = float((y_pred[mask] == cls).sum()) / n_cls
        result["per_class_acc"][cls] = round(acc, 4)
        accs.append(acc)
        weights.append(n_cls)
        result["confusion_pct"][cls] = {
            int(pc): round(float((y_pred[mask] == pc).sum()) / n_cls, 4)
            for pc in range(n_classes)
            if (y_pred[mask] == pc).any()
        }
    if accs:
        result["macro_acc"] = round(float(np.mean(accs)), 4)
    return result


def check_trend_monitor_alerts(
    current: dict[str, Any],
    baseline: Optional[dict[str, Any]],
    tf_list: list[str],
    monitor_tfs: Optional[list[str]] = None,
    **kwargs,
) -> tuple[list[str], dict[str, Any]]:
    """Scalar-based trend alerts (TrendMomentumHeadTF).
    Replaces 3-class CLASS_COLLAPSE/CONF_COLLAPSE with regression-based alerts.
      TMOM_BIAS_UP_{tf}   : mean prediction > +0.5 (systematic bullish bias)
      TMOM_BIAS_DOWN_{tf} : mean prediction < -0.5 (systematic bearish bias)
      TMOM_COLLAPSED_{tf} : std < 0.05, all predictions near zero (no signal)
      TMOM_DIR_POOR_{tf}  : direction accuracy < 0.48 (worse than random)
    """
    alerts = []
    report = {}
    monitor_tfs = monitor_tfs or tf_list
    for tf in monitor_tfs:
        tf_report = {}
        mean_val = current.get(f"tmom_mean_{tf}")
        std_val = current.get(f"tmom_std_{tf}")
        dir_acc = current.get(f"tmomentum_{tf}_dir_acc")
        if mean_val is not None and not math.isnan(mean_val):
            tf_report["tmom_mean"] = mean_val
            if mean_val > 0.5:
                alerts.append(f"TMOM_BIAS_UP_{tf}")
                tf_report["bias_up"] = True
            elif mean_val < -0.5:
                alerts.append(f"TMOM_BIAS_DOWN_{tf}")
                tf_report["bias_down"] = True
        if std_val is not None and not math.isnan(std_val):
            tf_report["tmom_std"] = std_val
            if std_val < 0.05:
                alerts.append(f"TMOM_COLLAPSED_{tf}")
                tf_report["collapsed"] = True
        if dir_acc is not None and not math.isnan(dir_acc) and dir_acc < 0.48:
            alerts.append(f"TMOM_DIR_POOR_{tf}")
            tf_report["dir_poor"] = True
        report[tf] = tf_report
    return alerts, report


def compute_trend_head_grad_norm(trend_head: nn.Module) -> float:
    total = 0.0
    n = 0
    for p in trend_head.parameters():
        if p.grad is not None:
            total += float(p.grad.data.norm(2).item() ** 2)
            n += 1
    return math.sqrt(total) if n > 0 else 0.0


# =============================================================
# Live-Safe Regime Monitor
# =============================================================
def load_trend_monitor_baseline(ckpt_path: str) -> Optional[dict]:
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        stats = (ckpt.get("extra", {}) or {}).get("trend_monitor", None)
        if stats:
            logger.info(f"[TREND MONITOR] Baseline: {os.path.basename(ckpt_path)}")
        return stats
    except Exception as e:
        logger.warning(f"[TREND MONITOR] Baseline load: {e}")
        return None


def load_anchor_model(ckpt_path, encoder, proj_head, device) -> Optional[dict[str, nn.Module]]:
    if not ckpt_path or not os.path.exists(ckpt_path):
        logger.warning(f"[ANCHOR] Not found: {ckpt_path}.")
        return None
    try:
        import copy

        anchor_encoder = copy.deepcopy(encoder)
        anchor_proj_head = copy.deepcopy(proj_head)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        def _try_load(module, key):
            if key not in ckpt or ckpt[key] is None:
                return
            try:
                module.load_state_dict(ckpt[key])
            except Exception as e:
                logger.warning("%s: strict load failed (%s), loading with strict=False", key, e)
                module.load_state_dict(ckpt[key], strict=False)

        _try_load(anchor_encoder, "encoder_state")
        _try_load(anchor_proj_head, "proj_head_state")
        for m in (anchor_encoder, anchor_proj_head):
            m.to(device).float().eval()
            set_requires_grad(m, False)
        logger.info(f"[ANCHOR] Loaded: {os.path.basename(ckpt_path)}")
        return {"encoder": anchor_encoder, "proj_head": anchor_proj_head}
    except Exception as e:
        logger.warning(f"[ANCHOR] Failed: {e}")
        return None


# =============================================================
# Phase apply
# =============================================================
def apply_phase(
    encoder,
    trend_head,
    proj_head,
    phase,
    base_lr,
    weight_decay,
    regime_cont_head=None,
    mktstructure_head=None,
    session_head=None,
    pdhl_head=None,
    regime_quality_head=None,
):
    freeze_tcn = bool(phase.get("freeze_tcn", True))
    freeze_gru = bool(phase.get("freeze_gru", True))
    freeze_backbone_other = bool(phase.get("freeze_backbone_other", False))
    freeze_regime_cont_head = bool(
        phase.get("freeze_regime_cont_head", False)
    )  # continuous (real loss)
    unfreeze_tcn_last_n = int(phase.get("unfreeze_tcn_last_n", 0))
    lr_mults = phase.get(
        "lr_multipliers",
        {"tcn": 0.0, "gru": 0.0, "backbone_other": 0.05, "heads": 0.2, "regime_cont_head": 0.0},
    )

    tcn_module = get_tcn_module(encoder)
    gru_module = get_gru_module(encoder)
    if tcn_module:
        set_requires_grad(tcn_module, not freeze_tcn)
    if gru_module:
        set_requires_grad(gru_module, not freeze_gru)

    tcn_last_params = []
    if freeze_tcn and unfreeze_tcn_last_n > 0 and tcn_module is not None:
        try:
            all_blocks = list(tcn_module.network)
            last_blocks = all_blocks[-unfreeze_tcn_last_n:]
            for blk in last_blocks:
                for p in blk.parameters():
                    p.requires_grad = True
                    tcn_last_params.append(p)
            logger.info("[PHASE] TCN last-%d blocks UNFROZEN @ tcn_last lr", unfreeze_tcn_last_n)
        except Exception as e:
            logger.warning("[PHASE] TCN last-N unfreeze failed: %s", e)

    logger.info(
        "[PHASE] TCN=%s GRU=%s regime_cont=%s",
        "FROZEN" if freeze_tcn else "train",
        "FROZEN" if freeze_gru else "train",
        "FROZEN" if freeze_regime_cont_head else "train",
    )

    tcn_param_ids = {id(p) for p in tcn_module.parameters()} if tcn_module else set()
    gru_param_ids = {id(p) for p in gru_module.parameters()} if gru_module else set()
    encoder_other = [
        p for p in encoder.parameters() if id(p) not in tcn_param_ids and id(p) not in gru_param_ids
    ]
    for p in encoder_other:
        p.requires_grad = not freeze_backbone_other

    # 3-class regime_head: no loss, freeze always — kept only for alignment monitor
    # continuous regime_cont_head: carries real loss, controlled separately
    if regime_cont_head is not None:
        set_requires_grad(regime_cont_head, not freeze_regime_cont_head)

    # All other heads always trainable in retrain
    other_head_params = (
        list(trend_head.parameters())
        + list(proj_head.parameters())
        + (list(mktstructure_head.parameters()) if mktstructure_head else [])
        + (list(session_head.parameters()) if session_head else [])
        + (list(pdhl_head.parameters()) if pdhl_head else [])
        + (list(regime_quality_head.parameters()) if regime_quality_head else [])
    )
    for p in other_head_params:
        p.requires_grad = True

    param_groups = []

    def _add(params, name, lr_key, default_mult):
        lr = base_lr * float(lr_mults.get(lr_key, default_mult))
        active = [p for p in params if p.requires_grad]
        if active and lr > 0:
            param_groups.append(
                {"params": active, "lr": lr, "weight_decay": weight_decay, "name": name}
            )
            logger.info(f"[PHASE] {name:<20}: {len(active):4d} params @ lr={lr:.2e}")

    _add(other_head_params, "heads", "heads", 0.2)
    _add(encoder_other, "backbone_other", "backbone_other", 0.05)
    if regime_cont_head is not None and not freeze_regime_cont_head:
        _add(list(regime_cont_head.parameters()), "regime_cont", "regime_cont_head", 0.0)
    # 3-class regime_head: never in optimizer (no loss) — intentionally excluded
    if gru_module and not freeze_gru:
        _add(list(gru_module.parameters()), "gru", "gru", 0.0)
    if tcn_module and not freeze_tcn:
        _add(list(tcn_module.parameters()), "tcn", "tcn", 0.0)
    if tcn_last_params:
        lr_tcn_last = base_lr * float(lr_mults.get("tcn_last", 0.008))
        if lr_tcn_last > 0:
            param_groups.append(
                {
                    "params": tcn_last_params,
                    "lr": lr_tcn_last,
                    "weight_decay": weight_decay,
                    "name": "tcn_last",
                }
            )
            logger.info(
                f"[PHASE] {'tcn_last':<20}: {len(tcn_last_params):4d} params @ lr={lr_tcn_last:.2e}"
            )

    if not param_groups:
        logger.warning("[PHASE] No trainable groups — fallback to heads only")
        for p in other_head_params:
            p.requires_grad = True
        param_groups.append(
            {
                "params": other_head_params,
                "lr": base_lr * 0.1,
                "weight_decay": weight_decay,
                "name": "heads_fallback",
            }
        )

    optimizer = torch.optim.AdamW(param_groups)
    total = sum(p.numel() for g in param_groups for p in g["params"])
    summary = ", ".join(f"{g['name']}@{g['lr']:.1e}" for g in param_groups)
    logger.info(f"[PHASE] [{summary}] total_trainable={total:,}")
    return optimizer


# =============================================================
# Drift utilities
# =============================================================
def compute_emb_drift_stats(
    encoder, dataloader, device, max_batches=30, max_points=1024, seed=0, asset_map: dict = None
):
    """Embedding statistics for drift detection.
    Passes asset_ids like training does (the encoder uses the asset embedding)."""
    encoder.eval()
    zs = []
    asset_map = asset_map or {}
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            seq = batch.get("seq") if isinstance(batch, dict) else None
            if seq is None:
                continue
            try:
                _assets = batch.get("asset", [None] * seq.size(0))
                _aids = torch.tensor(
                    [asset_map.get(a, 0) for a in _assets], dtype=torch.long, device=device
                )
                emb = encoder(seq.to(device, non_blocking=True).float(), asset_ids=_aids)
                zs.append(emb.detach().cpu())
            except Exception:
                continue
    if not zs:
        return {"emb_norm_mean": float("nan"), "emb_var_mean": float("nan"), "emb_n": 0}
    z_all = torch.cat(zs, dim=0).float()
    N = int(z_all.size(0))
    if N > max_points:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(N, generator=g)[:max_points]
        z_all = z_all[idx]
    norms = torch.norm(z_all, dim=-1)
    return {
        "emb_norm_mean": float(norms.mean().item()),
        "emb_var_mean": float(z_all.var(dim=0).mean().item()) if z_all.size(0) > 1 else 0.0,
        "emb_n": int(z_all.size(0)),
    }


def check_embedding_drift(current, baseline, threshold=0.20):
    report = {}
    flags = []
    for key in ("emb_norm_mean", "emb_var_mean"):
        bv = baseline.get(key)
        cv = current.get(key)
        if bv is None or cv is None or math.isnan(bv) or math.isnan(cv):
            report[key] = {"base": bv, "curr": cv, "rel_change": None, "drift": False}
            continue
        rc = 0.0 if abs(bv) < 1e-8 else abs(cv - bv) / abs(bv)
        d = rc > threshold
        report[key] = {
            "base": round(bv, 6),
            "curr": round(cv, 6),
            "rel_change": round(rc, 4),
            "drift": d,
        }
        flags.append(d)
    return bool(flags) and all(flags), report


def check_regime_drift(current, previous, threshold=0.12):
    report = {}
    flags = []
    for key in ("emb_norm_mean", "emb_var_mean"):
        pv = previous.get(key)
        cv = current.get(key)
        if pv is None or cv is None or math.isnan(pv) or math.isnan(cv):
            report[key] = {"prev": pv, "curr": cv, "rel_change": None, "drift": False}
            continue
        rc = 0.0 if abs(pv) < 1e-8 else abs(cv - pv) / abs(pv)
        d = rc > threshold
        report[key] = {
            "prev": round(pv, 6),
            "curr": round(cv, 6),
            "rel_change": round(rc, 4),
            "drift": d,
        }
        flags.append(d)
    return any(flags), report


def load_baseline_emb_stats(ckpt_path: str) -> Optional[dict]:
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        extra = ckpt.get("extra", {}) or {}
        stats = extra.get("emb_drift_stats", {})
        if not stats:
            stats = {k: extra[k] for k in ("emb_norm_mean", "emb_var_mean") if k in extra}
        if "emb_norm_mean" in stats and "emb_var_mean" in stats:
            logger.info(
                f"[DRIFT] Baseline: norm={stats['emb_norm_mean']:.4f} "
                f"var={stats['emb_var_mean']:.4f}"
            )
            return stats
    except Exception as e:
        logger.warning(f"[DRIFT] Baseline load: {e}")
    return None


# =============================================================
# Utilities
# =============================================================
def keepalive_loop(interval_seconds: int):
    try:
        import pyautogui
    except Exception:
        logger.warning("pyautogui not available.")
        return
    logger.info("Keepalive every %d s.", interval_seconds)
    while True:
        try:
            pyautogui.press("shift")
        except Exception as e:
            logger.warning("Keepalive: %s", e)
        time.sleep(interval_seconds)


def save_checkpoint(
    path,
    encoder,
    trend_head,
    proj_head,
    optimizer,
    epoch,
    mktstructure_head=None,
    regime_cont_head=None,
    session_head=None,
    pdhl_head=None,
    regime_quality_head=None,
    extra=None,
):
    """Saves the encoder + the active v28 heads."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "encoder_state": encoder.state_dict(),
            "trend_head_state": trend_head.state_dict(),
            "proj_head_state": proj_head.state_dict() if proj_head else None,
            "mktstructure_head_state": (
                mktstructure_head.state_dict() if mktstructure_head else None
            ),
            "regime_cont_head_state": regime_cont_head.state_dict() if regime_cont_head else None,
            "session_head_state": session_head.state_dict() if session_head else None,
            "pdhl_head_state": pdhl_head.state_dict() if pdhl_head else None,
            "regime_quality_head_state": (
                regime_quality_head.state_dict() if regime_quality_head else None
            ),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "extra": extra or {},
        },
        path,
    )


def _safe_corr(a, b):
    """Pearson correlation with zero-std guard. Returns nan instead of RuntimeWarning."""
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(a, b)
    v = float(c[0, 1])
    return float("nan") if np.isnan(v) else v


def _sf(x, d=0.0):
    """Safe float: returns default if x is NaN / invalid."""
    try:
        v = float(x)
        return d if math.isnan(v) else v
    except Exception:
        return d


def _masked_bce_with_logits(logits, targets, valid_mask, pos_weight=None):
    """
    logits: [B] or [B,1]
    targets: [B] in {0,1}
    valid_mask: [B] bool
    """
    if logits.dim() > 1:
        logits = logits.squeeze(-1)
    targets = targets.float()
    valid_mask = valid_mask.bool()
    if valid_mask.sum().item() == 0:
        return logits.sum() * 0.0
    lv = logits[valid_mask]
    tv = targets[valid_mask]
    return F.binary_cross_entropy_with_logits(
        lv, tv, pos_weight=pos_weight.to(lv.device) if pos_weight is not None else None
    )


def _to_dev_w(w, device):
    """Moves the sample weights to the device, whatever their shape.

    The collate can return a float, a list or a CPU tensor: without this
    normalization, indexing with a CUDA mask fails.
    """
    if w is None:
        return None
    if not isinstance(w, torch.Tensor):
        w = torch.tensor(w)
    return w.to(device).float()


def _masked_focal_bce_with_logits(
    logits, targets, valid_mask, pos_weight=None, gamma=2.0, sample_w=None
):
    """Masked focal BCE, with an optional per-sample weight.

    sample_w says HOW CLEAN the outcome of that trade is (see _conf in the
    dataset). The binary label "TP before SL" puts together a trade that went
    to target without suffering and one that got there after dropping to
    -0.9R: asking to predict them the same way means asking the model to
    separate what the label mixed up, and the AUC pays for it. With weighted
    samples the model learns where the signal is clear instead of spending
    capacity on photo-finish cases, which are mostly noise.


    logits: [B] or [B,1] · targets: [B] in {0,1} · valid_mask: [B] bool
    """
    if logits.dim() > 1:
        logits = logits.squeeze(-1)
    targets = targets.float()
    valid_mask = valid_mask.bool()
    if valid_mask.sum().item() == 0:
        return logits.sum() * 0.0

    lv = logits[valid_mask]
    tv = targets[valid_mask]

    bce = F.binary_cross_entropy_with_logits(
        lv,
        tv,
        pos_weight=pos_weight.to(lv.device) if pos_weight is not None else None,
        reduction="none",
    )
    pt = torch.exp(-bce)
    fl = ((1.0 - pt) ** gamma) * bce
    if sample_w is not None:
        w = sample_w
        if w.dim() > 1:
            w = w.squeeze(-1)
        w = w[valid_mask].to(fl.device).float().clamp(0.0, 1.0)
        den = w.sum()
        # if every weight is zero the batch teaches nothing: avoid the
        # division by zero and fall back to the plain mean
        if den > 1e-6:
            return (fl * w).sum() / den
    return fl.mean()


# feature -> slice of the context vector (29 dims in total)
# used for debugging and to check the expected size
STRATEGY_CONTEXT_LAYOUT = {
    "trend_momentum": (0, 3),  # 30min, 1h, 4h
    "regime_continuous": (3, 15),  # 3 TFs × 4 signals (vol_regime, atr_norm, adx, di_dir)
    "mktstructure_probs": (15, 24),  # 3 TF × 3 classi (ranging,long_brk,short_brk)
    "timing_15m": (24, 29),  # 5 features temporali 15min
}


def build_strategy_context(
    batch: dict[str, Any],
    trend_head: Optional[nn.Module],
    mktstructure_head: Optional[nn.Module],
    emb: torch.Tensor,
    tf_list: list[str],
) -> torch.Tensor:
    """
    Builds the 29-dim strategy context vector for v28 heads.
    Layout: see STRATEGY_CONTEXT_LAYOUT.

    Included features:
      - trend proxy (trend_head pred) 30min,1h,4h   -> 3  (NO future leakage)
      - regime_continuous 30min,1h,4h x 4           -> 12 (observed current signals, no future leakage)
      - mktstructure probs 30min,1h,4h x 3          -> 9  (snapshot frozen)
      - timing_15m                                   -> 5

    NOTE: uses trend_head.forward() detached, NOT batch["trend_momentum"]
    (which is future-derived -> leakage). mktstructure_head must be the
    frozen snapshot (mktstructure_ctx_snap), not the live head.
    Total = 29 dims
    """
    device = emb.device
    B = emb.size(0)

    tf_pick = ["30min", "1h", "4h"]
    tf_idx = [tf_list.index(tf) for tf in tf_pick if tf in tf_list]

    parts = []

    # trend proxy: trend_head prediction on the current embedding (detached)
    # NOT batch["trend_momentum"], which is future-derived -> leakage
    if trend_head is not None and tf_idx:
        with torch.no_grad():
            tmom_pred = trend_head(emb).float()  # [B, n_tfs]
        tmom_sel = tmom_pred[:, tf_idx].detach()  # [B, 3]
        parts.append(tmom_sel.to(emb.dtype))
    else:
        parts.append(torch.zeros((B, len(tf_pick)), device=device, dtype=emb.dtype))

    # regime_continuous [B, n_tfs, 4]
    regc = batch.get("regime_continuous", None)
    if regc is not None:
        regc_t = (
            regc.to(device).float()
            if isinstance(regc, torch.Tensor)
            else torch.tensor(regc, device=device, dtype=torch.float32)
        )
        if regc_t.dim() == 3 and tf_idx:
            regc_sel = regc_t[:, tf_idx, :].reshape(B, -1)
            regc_sel = torch.where(regc_sel < -900.0, torch.zeros_like(regc_sel), regc_sel)
            parts.append(regc_sel)
        else:
            parts.append(torch.zeros((B, len(tf_pick) * 4), device=device, dtype=emb.dtype))
    else:
        parts.append(torch.zeros((B, len(tf_pick) * 4), device=device, dtype=emb.dtype))

    # mktstructure probs — frozen snapshot (no within-batch co-adaptation)
    # mktstructure_head must be mktstructure_ctx_snap, not the live head
    if mktstructure_head is not None and tf_idx:
        with torch.no_grad():
            ms_logits = mktstructure_head(emb)
            ms_probs = torch.softmax(ms_logits.float(), dim=-1)
        ms_sel = ms_probs[:, tf_idx, :].reshape(B, -1).detach()
        parts.append(ms_sel.to(emb.dtype))
    else:
        parts.append(torch.zeros((B, len(tf_pick) * 3), device=device, dtype=emb.dtype))

    # timing_15m [B, 5]
    timing = batch.get("timing_15m", None)
    if timing is not None:
        timing_t = (
            timing.to(device).float()
            if isinstance(timing, torch.Tensor)
            else torch.tensor(timing, device=device, dtype=torch.float32)
        )
        if timing_t.dim() == 2:
            parts.append(timing_t)
        else:
            parts.append(torch.zeros((B, 5), device=device, dtype=emb.dtype))
    else:
        parts.append(torch.zeros((B, 5), device=device, dtype=emb.dtype))

    ctx = torch.cat(parts, dim=-1)
    _expected = STRATEGY_CONTEXT_LAYOUT["timing_15m"][1]  # 29
    if ctx.size(1) != _expected:
        raise RuntimeError(
            f"[build_strategy_context] dim mismatch: got {ctx.size(1)}, expected {_expected}. "
            f"Parts: {[p.shape for p in parts]}"
        )
    return ctx


def save_encoder_only(encoder, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(encoder.state_dict(), path)


def resume_checkpoint(
    path,
    encoder,
    trend_head,
    proj_head=None,
    optimizer=None,
    mktstructure_head=None,
    regime_cont_head=None,
    session_head=None,
    pdhl_head=None,
    regime_quality_head=None,
    map_location=None,
) -> dict:
    """Loads a checkpoint — only the active v28 heads."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    def _try_load(module, key):
        if module is None or key not in ckpt or ckpt[key] is None:
            return
        try:
            module.load_state_dict(ckpt[key])
        except Exception as e:
            logger.warning("%s: strict load failed (%s), loading with strict=False", key, e)
            module.load_state_dict(ckpt[key], strict=False)

    _try_load(encoder, "encoder_state")
    _try_load(trend_head, "trend_head_state")
    _try_load(proj_head, "proj_head_state")
    _try_load(mktstructure_head, "mktstructure_head_state")
    _try_load(regime_cont_head, "regime_cont_head_state")
    _try_load(session_head, "session_head_state")
    _try_load(pdhl_head, "pdhl_head_state")
    _try_load(regime_quality_head, "regime_quality_head_state")
    if optimizer and ckpt.get("optimizer_state"):
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception as e:
            logger.warning("Optimizer load: %s", e)
    return {"epoch": int(ckpt.get("epoch", -1)), "extra": ckpt.get("extra", {})}


def find_best_checkpoint(results_dir, name_patterns=None):
    if not results_dir or not os.path.exists(results_dir):
        return None
    if name_patterns is None:
        name_patterns = ["best_full_checkpoint.pth", "best*.pth", "latest_checkpoint.pth"]
    for nm in name_patterns:
        if "*" in nm or "?" in nm:
            matches = glob.glob(os.path.join(results_dir, nm))
            if matches:
                return max(matches, key=os.path.getmtime)
        else:
            cand = os.path.join(results_dir, nm)
            if os.path.exists(cand):
                return cand
    return None


def save_best_score(
    results_dir: str, score: float, epoch: int, mode: str, extra: Optional[dict] = None
):
    """Saves the best score to its own JSON file — append-safe."""
    path = os.path.join(results_dir, "best_score.json")
    record = {
        "score": float(score),
        "epoch": int(epoch),
        "mode": mode,
        "ts": pd.Timestamp.now().isoformat(),
        **(extra or {}),
    }
    # read the previous history
    history_scores = []
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                _prev = json.load(f)
            if isinstance(_prev, dict):
                history_scores = _prev.get("history", [_prev])
            elif isinstance(_prev, list):
                history_scores = _prev
        except Exception:
            history_scores = []
    history_scores.append(record)
    out = {"best": record, "history": history_scores}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    except Exception as e:
        logger.warning("[BEST SCORE] save failed: %s", e)


def load_best_score(results_dir: str, mode: str) -> Optional[float]:
    """Loads the best score from its own file — survives crashes."""
    path = os.path.join(results_dir, "best_score.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # look for the best score of this mode
        if isinstance(data, dict):
            history_scores = data.get("history", [])
            candidates = [
                h["score"]
                for h in history_scores
                if h.get("mode") == mode and isinstance(h.get("score"), (int, float))
            ]
            if candidates:
                best = max(candidates)
                logger.info("[BEST SCORE] Recovered best=%.6f mode=%s from %s", best, mode, path)
                return best
        return None
    except Exception as e:
        logger.warning("[BEST SCORE] load failed: %s", e)
        return None


def compute_embedding_metrics(
    encoder, proj, dataloader, device, max_batches=50, max_points=2048, seed=0
):
    encoder.eval()
    proj.eval()
    zs = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            seq = batch.get("seq") if isinstance(batch, dict) else None
            if seq is None:
                continue
            try:
                _aids = batch.get("asset_id", None)
                if _aids is not None:
                    _aids = _aids.to(device, non_blocking=True)
                emb = encoder(seq.to(device, non_blocking=True).float(), asset_ids=_aids)
                zs.append(proj(emb).detach().cpu())
            except Exception as e:
                logger.warning("[EMB METRICS] batch %d failed: %s", i, e)
                continue
    if not zs:
        return {
            "uniformity": float("nan"),
            "intra_var_mean": float("nan"),
            "emb_batches": 0,
            "emb_points": 0,
        }
    z_all = torch.cat(zs, dim=0).float()
    N = int(z_all.size(0))
    intra_var_mean = float(z_all.var(dim=0).mean().item()) if N > 1 else 0.0
    if N > max_points:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(N, generator=g)[:max_points]
        z_u = z_all[idx]
        N_u = max_points
    else:
        z_u = z_all
        N_u = N
    try:
        if N_u <= 1:
            uniformity = 0.0
        else:
            chunk = 256
            total = 0.0
            count = 0
            jj = torch.arange(N_u)
            for i0 in range(0, N_u, chunk):
                i1 = min(N_u, i0 + chunk)
                zi = z_u[i0:i1]
                diff = zi[:, None, :] - z_u[None, :, :]
                sq = (diff * diff).sum(dim=-1)
                ii = torch.arange(i0, i1)[:, None]
                em = ii != jj[None, :]
                vals = torch.exp(-2.0 * sq[em])
                total += float(vals.sum().item())
                count += int(em.sum().item())
            uniformity = float(np.log((total / max(1, count)) + 1e-12))
    except Exception as e:
        logger.warning(f"[EMB] uniformity: {e}")
        uniformity = float("nan")
    return {
        "uniformity": uniformity,
        "intra_var_mean": intra_var_mean,
        "emb_batches": int(len(zs)),
        "emb_points": int(N_u),
    }


def evaluate_v28_heads(
    session_head,
    pdhl_head,
    regime_quality_head,
    encoder,
    trend_head,
    mktstructure_head,
    val_loader,
    device,
    tf_list,
    max_batches: int = 9999,
    asset_map: dict = None,
) -> dict:
    """
    Evaluates SessionBreakoutHead, PDHLBreakoutHead, RegimeQualityHead on the val set.
    Metrics for every combo (session_long/short, pdhl_long/short):
      v28_{name}_side_auc   -> AUC of P(TP first)  [main metric]
      v28_{name}_side_acc   -> accuracy at threshold 0.5
      v28_{name}_tp_rate    -> % positive labels (WR proxy)
      v28_{name}_hit_auc    -> alias of side_auc (kept for the score)
      v28_{name}_side_n     -> number of valid samples
    Aggregate: v28_session_hit_auc, v28_pdhl_hit_auc,
               v28_session_side_auc, v28_pdhl_side_auc
    RegimeQuality: v28_regime_quality_auc, v28_regime_quality_acc,
                   v28_regime_quality_rate, v28_regime_quality_n
    """
    from sklearn.metrics import roc_auc_score

    asset_map = asset_map or {}
    metrics = {}

    heads_cfg = [
        ("session_long", session_head, "strat_hit_sl", "strat_side_sl", 0),
        ("session_short", session_head, "strat_hit_ss", "strat_side_ss", 1),
        ("pdhl_long", pdhl_head, "strat_hit_pl", "strat_side_pl", 0),
        ("pdhl_short", pdhl_head, "strat_hit_ps", "strat_side_ps", 1),
    ]
    storage = {name: {"side_prob": [], "side_y": []} for name, *_ in heads_cfg}
    qual_prob, qual_y = [], []

    session_head.eval()
    pdhl_head.eval()
    regime_quality_head.eval()
    with torch.no_grad():
        for bc, vb in enumerate(val_loader):
            if bc >= max_batches:
                break
            vseq = vb["seq"].to(device).float()
            vmasked = vb.get("masked_seq", vseq).to(device).float()
            _vaid = vb.get("asset", [None] * vseq.size(0))
            _vaid_t = torch.tensor(
                [asset_map.get(a, 0) for a in _vaid], dtype=torch.long, device=device
            )
            vemb = encoder(vmasked, asset_ids=_vaid_t)
            vctx = build_strategy_context(
                batch=vb,
                trend_head=trend_head,
                mktstructure_head=mktstructure_head,
                emb=vemb,
                tf_list=tf_list,
            )

            for name, head, _hk, sk, direction in heads_cfg:
                ls_l, ls_s = head(vemb, extra_context=vctx)
                ls = ls_l if direction == 0 else ls_s
                sp = torch.sigmoid(ls.squeeze(-1))
                _sr = vb.get(sk)
                if _sr is None:
                    continue
                _sy = (
                    _sr.to(device)
                    if isinstance(_sr, torch.Tensor)
                    else torch.tensor(_sr, device=device)
                ).float()
                _vs = _sy >= 0
                if _vs.any():
                    storage[name]["side_prob"].append(sp[_vs].cpu().numpy())
                    storage[name]["side_y"].append(_sy[_vs].cpu().numpy())

            _rq_raw = vb.get("regime_quality")
            if _rq_raw is not None:
                _rq_lbl = (
                    _rq_raw.to(device)
                    if isinstance(_rq_raw, torch.Tensor)
                    else torch.tensor(_rq_raw, device=device)
                ).float()
                _vq = _rq_lbl >= 0
                if _vq.any():
                    rq_p = regime_quality_head.predict_score(vemb, vctx)
                    qual_prob.append(rq_p[_vq].cpu().numpy())
                    qual_y.append(_rq_lbl[_vq].cpu().numpy())

    def _auc(prob_list, y_list):
        if not prob_list:
            return float("nan"), float("nan"), float("nan"), 0
        p = np.concatenate(prob_list)
        y = np.concatenate(y_list)
        n = len(y)
        if len(np.unique(y)) < 2:
            return float("nan"), float("nan"), float("nan"), n
        try:
            auc = float(roc_auc_score(y, p))
        except Exception:
            auc = float("nan")
        acc = float(((p >= 0.5).astype(float) == y).mean())
        rate = float(y.mean())
        return auc, acc, rate, n

    agg_side = {"session": [], "pdhl": []}
    for name, _head, _hk, _sk, _direction in heads_cfg:
        st = storage[name]
        # side AUC only — hit removed (useless, always ~90% positive)
        s_auc, s_acc, s_rate, s_n = _auc(st["side_prob"], st["side_y"])
        metrics[f"v28_{name}_side_auc"] = s_auc  # P(TP first) AUC
        metrics[f"v28_{name}_side_acc"] = s_acc
        metrics[f"v28_{name}_tp_rate"] = s_rate  # % label positivi (WR proxy)
        metrics[f"v28_{name}_side_n"] = s_n
        # alias: v28_{name}_hit_auc is really the side_auc (P(TP first)),
        # the "hit" name is kept for the score computation and the logs
        metrics[f"v28_{name}_hit_auc"] = s_auc  # alias → side_auc
        grp = "session" if "session" in name else "pdhl"
        if not math.isnan(s_auc):
            agg_side[grp].append(s_auc)

    for grp in ("session", "pdhl"):
        metrics[f"v28_{grp}_hit_auc"] = (
            float(np.mean(agg_side[grp])) if agg_side[grp] else float("nan")
        )
        metrics[f"v28_{grp}_side_auc"] = metrics[f"v28_{grp}_hit_auc"]

    rq_auc, rq_acc, rq_rate, rq_n = _auc(qual_prob, qual_y)
    metrics["v28_regime_quality_auc"] = rq_auc
    metrics["v28_regime_quality_acc"] = rq_acc
    metrics["v28_regime_quality_rate"] = rq_rate
    metrics["v28_regime_quality_n"] = rq_n
    return metrics


def train_pretrainer_with_contrastive(conf: dict[str, Any]):

    # ── global seed for a reproducible A/B comparison (default OFF in production) ─
    if conf.get("use_global_seed", False):
        set_global_seed(
            int(conf.get("global_seed", 42)),
            deterministic=bool(conf.get("global_seed_deterministic", True)),
            full_determinism=bool(conf.get("global_seed_full_determinism", False)),
        )

    # resolve the paths BEFORE logging / AB tag: so results_dir already uses the
    # right AB root (ab_root_features_off/on) based on use_structural_features.
    conf = resolve_paths(conf)

    # ── AB-COMPARE: baseline/new are separated at the ROOT ────────────────────
    # (see resolve_paths: ab_root_features_off / ab_root_features_on). Here we only
    # tag ON/OFF in the [AB-COMPARE] line and in the file names, without
    # creating extra subfolders.
    if conf.get("ab_separate_outputs", False):
        _ab_suffix = "struct" if conf.get("use_structural_features", False) else "base"
        _base_tag = conf.get("retrain_version_tag") or "retrain"
        if not str(_base_tag).endswith(f"_{_ab_suffix}"):
            conf["retrain_version_tag"] = f"{_base_tag}_{_ab_suffix}"
        _root_used = os.path.dirname(os.path.dirname(conf.get("results_dir", "")))
        logger.info(
            "[AB-COMPARE] mode %s — root: %s", _ab_suffix.upper(), _root_used or "(default)"
        )

    # ── File logging — every console log -> .txt file, append mode ─────
    _log_dir = conf["results_dir"]
    os.makedirs(_log_dir, exist_ok=True)
    _log_tag = conf.get("retrain_version_tag") or "bootstrap"
    _log_path = os.path.join(_log_dir, f"training_log_{_log_tag}.txt")

    # remove duplicated FileHandlers on the root logger AND the local one
    # (avoids duplicates if train_pretrainer_with_contrastive is called
    #  more than once in the same Python session)
    _log_path_abs = os.path.abspath(_log_path)
    for _target_logger in (logging.getLogger(), logger):
        for _h in _target_logger.handlers[:]:
            if isinstance(_h, logging.FileHandler):
                try:
                    if os.path.abspath(getattr(_h, "baseFilename", "")) == _log_path_abs:
                        _target_logger.removeHandler(_h)
                        _h.close()
                except Exception:
                    pass

    _file_handler = logging.FileHandler(_log_path, mode="a", encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)

    # same format as the basicConfig console handler
    _file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s:%(name)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )

    # added to the ROOT logger so it catches EVERYTHING (tqdm excluded)
    # — not only the trainer but also the dataset etc.
    _root_logger = logging.getLogger()
    _root_logger.addHandler(_file_handler)

    logger.info("=" * 70)
    logger.info("[LOG FILE] %s", _log_path)
    logger.info(
        "[LOG] mode=%s class=%s device=%s",
        conf.get("training_mode", "?"),
        conf.get("class_name", "?"),
        conf.get("trainer", {}).get("device", "?"),
    )
    logger.info("=" * 70)
    logger.info("=" * 70)
    logger.info("[LOG] Training log: %s", _log_path)
    logger.info(
        "[LOG] Run started — mode=%s class=%s",
        conf.get("training_mode", "?"),
        conf.get("class_name", "?"),
    )
    logger.info("=" * 70)
    base_dir = conf["data_dir"]
    class_name = conf.get("class_name", "Majors")
    symbols = conf.get("symbols", [])
    results_dir = conf["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    _CLASS_THRESHOLDS = {
        "majors": {"15min": 0.0005, "30min": 0.0008, "1h": 0.0012, "4h": 0.0018, "1d": 0.0035},
        "minors": {"15min": 0.0008, "30min": 0.0013, "1h": 0.0018, "4h": 0.0035, "1d": 0.0055},
    }
    _cn_key = class_name.lower().strip()
    if _cn_key in _CLASS_THRESHOLDS and not conf.get("trend_threshold_scaled_tf"):
        conf["trend_threshold_scaled_tf"] = _CLASS_THRESHOLDS[_cn_key]
        logger.info(
            "[TREND THRESH] Auto-set for class=%s: %s",
            class_name,
            conf["trend_threshold_scaled_tf"],
        )

    training_mode = str(conf.get("training_mode", "bootstrap")).lower().strip()
    live_mode = bool(conf.get("live_mode", False))
    rolling_years = int(conf.get("rolling_years", 4))
    live_val_days = int(conf.get("live_val_days", 21))

    drift_threshold = float(conf.get("drift_threshold", 0.20))
    drift_max_batches = int(conf.get("drift_max_batches", 30))
    drift_max_points = int(conf.get("drift_max_points", 1024))
    drift_anchor_start = conf.get("drift_anchor_start", "2020-01-01")
    drift_anchor_end = conf.get("drift_anchor_end", "2020-12-31")
    regime_drift_threshold = float(conf.get("regime_drift_threshold", 0.12))

    use_kl_anchor = bool(conf.get("use_kl_anchor", True))
    kl_anchor_weight = float(conf.get("kl_anchor_weight", 0.015))
    kl_anchor_ckpt = conf.get("kl_anchor_ckpt", None)

    w_regime_bootstrap = float(conf.get("w_regime_bootstrap", 1.0))
    w_regime_retrain = float(conf.get("w_regime_retrain", 1.2))
    retrain_save_mode = str(conf.get("retrain_save_mode", "last"))

    monitor_tfs = conf.get("monitor_tfs", ["30min", "1h"])
    monitor_ema_alpha = float(conf.get("monitor_ema_alpha", 0.9))

    _schedule_key, active_schedule_template = auto_select_schedule(class_name)
    if conf.get("retrain_schedule_custom"):
        active_schedule_template = conf["retrain_schedule_custom"]
        _schedule_key = "custom"
    logger.info(f"[SCHEDULE] {class_name} -> {_schedule_key}")

    if training_mode == "retrain":
        train_end = conf.get("retrain_train_end")
        val_start = conf.get("retrain_val_start")
        val_end = conf.get("retrain_val_end")
        version_tag = conf.get("retrain_version_tag")
        if not train_end or not version_tag:
            raise RuntimeError("[RETRAIN] Missing retrain_train_end / version_tag")
        end_dt = pd.to_datetime(train_end)
        train_start = (end_dt - pd.DateOffset(years=rolling_years)).strftime("%Y-%m-%d")
        if live_mode:
            val_end = end_dt.strftime("%Y-%m-%d")
            val_start = (end_dt - pd.Timedelta(days=live_val_days)).strftime("%Y-%m-%d")
            logger.info(f"[LIVE VAL] last {live_val_days} days: " f"{val_start} -> {val_end}")
        else:
            val_start = val_start or train_start
            val_end = val_end or train_end
        logger.info(
            f"[RETRAIN] {train_start}->{train_end} | "
            f"val:{val_start}->{val_end} | "
            f"w_regime={w_regime_retrain} kl={kl_anchor_weight}"
        )
    else:
        train_start = conf.get("bootstrap_train_start")
        train_end = conf.get("bootstrap_train_end")
        val_start = conf.get("bootstrap_val_start", train_start)
        val_end = conf.get("bootstrap_val_end", train_end)
        version_tag = None
        if not train_start or not train_end:
            raise RuntimeError("[BOOTSTRAP] Missing train_start/end")
        logger.info(f"[BOOTSTRAP] {train_start}->{train_end}")

    device_str = (conf.get("trainer", {}) or {}).get("device", None) or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_str)
    seq_len = int(conf["seq_len"])
    batch_size = int(conf["batch_size"])
    base_lr = float(conf["lr"])
    weight_decay = float(conf.get("weight_decay", 1e-6))
    debug_mode = bool(conf.get("debug_mode", False))

    use_amp = bool(conf.get("use_amp", True)) and (device.type == "cuda") and not debug_mode
    if use_amp:
        try:
            amp_scaler = torch.amp.GradScaler("cuda", init_scale=256.0, growth_interval=2000)

            def autocast_ctx_factory():
                return torch.amp.autocast(device_type="cuda")

            logger.info("[AMP] Mixed precision FP16 ENABLED")
        except Exception:
            try:
                from torch.cuda.amp import GradScaler as _GS
                from torch.cuda.amp import autocast as _ac

                amp_scaler = _GS(init_scale=256.0, growth_interval=2000)

                def autocast_ctx_factory():
                    return _ac()

                logger.info("[AMP] Mixed precision FP16 ENABLED (legacy)")
            except Exception:
                use_amp = False
                amp_scaler = None

                def autocast_ctx_factory():
                    return nullcontext()

    else:
        use_amp = False
        amp_scaler = None

        def autocast_ctx_factory():
            return nullcontext()

        logger.info("[AMP] Mixed precision DISABLED")

    asset_parquets = discover_asset_parquets(base_dir, class_name, symbols)
    assets = list(asset_parquets.keys())
    logger.info(f"Assets: {assets}")

    TF_LIST = conf.get("tf_list", ["15min", "30min", "1h", "4h", "1d"])
    collected_feature_cols = None
    for _asset, p in asset_parquets.items():
        try:
            df_head = (
                pd.read_parquet(p).head(1) if p.endswith(".parquet") else pd.read_csv(p, nrows=1)
            )
        except Exception:
            df_head = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p, nrows=1)
        df_head.columns = [c.lower() for c in df_head.columns]
        fc = autodetect_feature_columns(df_head, TF_LIST)
        if collected_feature_cols is None:
            collected_feature_cols = fc
        else:
            collected_feature_cols = [c for c in collected_feature_cols if c in fc]
            if not collected_feature_cols:
                raise RuntimeError("No common feature columns")
    if not collected_feature_cols:
        raise RuntimeError("No feature columns detected.")

    per_asset_scaler_paths: dict[str, str] = dict(conf.get("scaler_paths", {}) or {})
    if len(per_asset_scaler_paths) < len(assets):
        for a, path in discover_scaler_files_auto(base_dir, class_name).items():
            if a not in per_asset_scaler_paths:
                per_asset_scaler_paths[a] = path

    _label_excl = conf.get("label_exclude_tfs", ["15min"])
    mpconf_train = MarketPretrainConfig(
        # Which bar the model sees. 1 = closed bars only, so it learns to
        # decide with the information it will really have live.
        # Must match ENCODER_SCORE_OFFSET in live_config and in the backtest:
        # training on one input and evaluating on another gives no error, it
        # just gives results that don't mean what they seem to.
        score_offset=int(conf.get("score_offset", 1)),
        # BE in the labels: finds the trades the breakeven closes at break-even.
        # Needed by the sample weights; without it be_reachable is always 0.
        outcome_be_enabled=bool(conf.get("outcome_be_enabled", False)),
        outcome_be_trigger_r=float(conf.get("outcome_be_trigger_r", 0.25)),
        outcome_be_offset_r=float(conf.get("outcome_be_offset_r", 0.07)),
        seq_len=seq_len,
        step_stride=int(conf.get("step_stride", 1)),
        mask_prob=conf.get("mask_prob", 0.25),
        tf_list=tuple(TF_LIST),
        atr_lookback=conf.get("atr_lookback", 14),
        trend_threshold_scaled=float(conf.get("trend_threshold_scaled", 1e-3)),
        label_exclude_tfs=_label_excl,
        trend_threshold_scaled_tf=conf.get("trend_threshold_scaled_tf", None),
        trend_quantile_tail=float(conf.get("trend_quantile_tail", 0.20)),
        trend_label_mode=conf.get("trend_label_mode", "atr_adjusted"),
        trend_atr_mult=float(conf.get("trend_atr_mult", 0.4)),
        trend_atr_mult_tf=conf.get("trend_atr_mult_tf", None),
        use_session_features=bool(conf.get("use_session_features", True)),
        use_structural_features=bool(conf.get("use_structural_features", False)),
        # ── v28: session labels differ per universe ───────────────────────────────
        outcome_session_tp_r_majors=float(conf.get("outcome_session_tp_r_majors", 2.5)),
        outcome_session_tp_r_minors=float(conf.get("outcome_session_tp_r_minors", 1.8)),
        outcome_session_sl_r=float(conf.get("outcome_session_sl_r", 1.0)),
        strategy_session_horizon_bars=int(conf.get("strategy_session_horizon_bars", 32)),
        # ── v28: pdhl labels ──────────────────────────────────────────────────────
        outcome_pdhl_tp_r=float(conf.get("outcome_pdhl_tp_r", 1.8)),
        outcome_pdhl_sl_r=float(conf.get("outcome_pdhl_sl_r", 1.0)),
        strategy_pdhl_horizon_bars=int(conf.get("strategy_pdhl_horizon_bars", 64)),
        # ── v28: regime quality labels ────────────────────────────────────────────
        regime_quality_horizon_bars=int(conf.get("regime_quality_horizon_bars", 32)),
        # ── side label denoising (invariato) ─────────────────────────────────────
        strategy_side_margin_atr=float(conf.get("strategy_side_margin_atr", 0.15)),
        strategy_side_min_hit_bars=int(conf.get("strategy_side_min_hit_bars", 2)),
    )
    # Inject per-class mktstructure thresholds into dataset config
    _is_minors = "minor" in class_name.lower()
    _thresh_key = (
        "mktstructure_dist_thresh_minors" if _is_minors else "mktstructure_dist_thresh_majors"
    )
    _bo_key = "mktstructure_bo_window_minors" if _is_minors else "mktstructure_bo_window_majors"
    mpconf_train.mktstructure_dist_thresh_by_tf = conf.get(_thresh_key)
    mpconf_train.mktstructure_bo_window_by_tf = conf.get(_bo_key)
    logger.info(
        "[MKTSTR THRESH] class=%s dist_thresh=%s bo_window=%s",
        class_name,
        mpconf_train.mktstructure_dist_thresh_by_tf,
        mpconf_train.mktstructure_bo_window_by_tf,
    )
    mpconf_val = MarketPretrainConfig(**mpconf_train.__dict__)

    if conf.get("analyze_thresholds", False):
        logger.info("[THRESHOLD ANALYSIS] Running analyze_threshold_impact...")
        for _an_asset, _an_path in list(asset_parquets.items())[:1]:
            _an_spa = per_asset_scaler_paths.get(_an_asset)
            analyze_threshold_impact(_an_path, TF_LIST, scaler_path=_an_spa)
        logger.info("[THRESHOLD ANALYSIS] Done — set analyze_thresholds=False to train.")
        import sys

        sys.exit(0)

    trend_thresholds_path = conf.get("trend_thresholds_path") or os.path.join(
        results_dir, "trend_thresholds.json"
    )
    per_asset_trend_thresholds: dict = {}
    per_asset_regime_thresholds: dict = {}
    trend_config = {"trend_quantile_tail": float(conf.get("trend_quantile_tail", 0.20))}
    use_quantile_th = bool(conf.get("use_quantile_trend_thresholds", True))
    if not use_quantile_th:
        logger.info("[TREND THRESH] Using fixed economic thresholds per TF (quantile disabled)")
    elif (
        training_mode == "retrain"
        and os.path.exists(trend_thresholds_path)
        and not conf.get("force_recompute_trend_thresholds", False)
    ):
        with open(trend_thresholds_path) as f:
            raw = json.load(f)
        for asset, tf_dict in raw.items():
            per_asset_trend_thresholds[str(asset).upper().strip()] = {
                tf: tuple(v) for tf, v in tf_dict.items()
            }
    else:
        for asset, p in asset_parquets.items():
            asset_key = str(asset).upper().strip()
            spa = per_asset_scaler_paths.get(asset)
            try:
                per_asset_trend_thresholds[asset_key] = compute_trend_thresholds_for_asset(
                    p, train_start, train_end, TF_LIST, trend_config, scaler_path=spa
                )
            except Exception as e:
                logger.warning(f"Trend thresh {asset_key}: {e}")
                per_asset_trend_thresholds[asset_key] = {}
        with open(trend_thresholds_path, "w") as f:
            json.dump(
                {
                    a: {tf: list(v) for tf, v in d.items()}
                    for a, d in per_asset_trend_thresholds.items()
                },
                f,
                indent=2,
            )

    train_datasets = []
    val_datasets = []
    drift_datasets = []
    for asset, p in asset_parquets.items():
        asset_key = str(asset).upper().strip()
        tr_th = per_asset_trend_thresholds.get(asset_key, {})
        spa = per_asset_scaler_paths.get(asset)
        train_datasets.append(
            MarketPretrainDataset(
                df_or_path=p,
                config=mpconf_train,
                feature_cols=collected_feature_cols,
                asset_name=asset,
                scaler_path=spa,
                slice_period=(train_start, train_end),
                regime_thresholds={},
                trend_thresholds=tr_th,
            )
        )
        val_datasets.append(
            MarketPretrainDataset(
                df_or_path=p,
                config=mpconf_val,
                feature_cols=collected_feature_cols,
                asset_name=asset,
                scaler_path=spa,
                slice_period=(val_start, val_end),
                regime_thresholds={},
                trend_thresholds=tr_th,
            )
        )
        drift_datasets.append(
            MarketPretrainDataset(
                df_or_path=p,
                config=mpconf_val,
                feature_cols=collected_feature_cols,
                asset_name=asset,
                scaler_path=spa,
                slice_period=(drift_anchor_start, drift_anchor_end),
                regime_thresholds={},
                trend_thresholds=tr_th,
            )
        )

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    drift_dataset = ConcatDataset(drift_datasets)

    def _make_loader(dataset, batch_size, shuffle, drop_last, nw, prefetch=2):
        nw = int(max(0, nw or 0))
        cands = []
        cur = nw
        seen = set()
        while cur >= 0:
            if cur not in seen:
                cands.append(cur)
                seen.add(cur)
            if cur <= 1:
                break
            cur = max(1, cur // 2)
        if 0 not in seen:
            cands.append(0)
        last_exc = None
        for n in cands:
            try:
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    num_workers=n,
                    pin_memory=False,
                    prefetch_factor=(prefetch if n > 0 else None),
                    persistent_workers=(n > 0),
                )
                it = iter(loader)
                del it
                logger.info(f"DataLoader num_workers={n}")
                return loader, n
            except Exception as e:
                logger.warning(f"DataLoader nw={n}: {e}")
                last_exc = e
                gc.collect()
                time.sleep(0.1)
        raise RuntimeError(f"Cannot create DataLoader: {last_exc}")

    try:
        tmp.set_sharing_strategy("file_system")
    except Exception:
        pass
    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    num_workers = 0 if debug_mode else int(conf.get("num_workers", 6) or 0)
    prefetch_factor = int(conf.get("prefetch_factor", 2))

    train_loader, _ = _make_loader(
        train_dataset, batch_size, True, True, num_workers, prefetch_factor
    )
    val_loader, _ = _make_loader(
        val_dataset, batch_size, False, False, max(0, num_workers // 2), prefetch_factor
    )
    drift_loader, _ = _make_loader(drift_dataset, batch_size, False, False, 0, 2)
    regime_drift_loader, _ = _make_loader(train_dataset, batch_size, False, False, 0, 2)

    _sample_ds = train_datasets[0] if train_datasets else None
    if _sample_ds is not None:
        try:
            _sample_ds._ensure_loaded()
            n_features_total = _sample_ds.n_input_features
        except Exception:
            n_features_total = (
                len(collected_feature_cols)
                + (7 if conf.get("use_session_features", True) else 0)
                + (5 if conf.get("use_structural_features", False) else 0)
            )
    else:
        n_features_total = (
            len(collected_feature_cols)
            + (7 if conf.get("use_session_features", True) else 0)
            + (5 if conf.get("use_structural_features", False) else 0)
        )
    logger.info(
        "[FEATURES] total=%d (base=%d + session=%d + struct=%d)",
        n_features_total,
        len(collected_feature_cols),
        (7 if conf.get("use_session_features", True) else 0),
        (5 if conf.get("use_structural_features", False) else 0),
    )

    # ── DI direction data availability check (v25) ──────────────────────────
    # di_direction requires DI_plus14 / DI_minus14 columns in the parquet.
    # If missing, the 4th signal of RegimeContinuousHead trains on all-NaN
    # targets → wasted capacity and misleading regime_corr score.
    _di_avail_by_tf: dict[str, bool] = {}
    for _tf in TF_LIST:
        _di_ok = any(
            getattr(_ds, "_regime_cont_by_tf", {}).get(_tf, {}).get("di_direction") is not None
            for _ds in train_datasets
        )
        _di_avail_by_tf[_tf] = _di_ok
    _di_missing_tfs = [tf for tf, ok in _di_avail_by_tf.items() if not ok]
    _di_avail_tfs = [tf for tf, ok in _di_avail_by_tf.items() if ok]
    if _di_missing_tfs:
        logger.warning(
            "[DI DIR] ⚠️  di_direction NOT available for TFs: %s  "
            "(DI_plus14/DI_minus14 missing from parquet — re-download with updated downloader). "
            "RegimeContinuousHead slot[3] will train on NaN targets for these TFs.",
            _di_missing_tfs,
        )
    if _di_avail_tfs:
        logger.info("[DI DIR] ✅ di_direction available for TFs: %s", _di_avail_tfs)
    # ────────────────────────────────────────────────────────────────────────

    n_tfs = len(TF_LIST)
    gru_hidden = int(conf.get("gru_hidden", 128))
    proj_dim = int(conf.get("proj_dim", 128))
    n_assets = int(conf.get("n_assets", 0))
    asset_embed_dim = int(conf.get("asset_embed_dim", 8))

    encoder = TCNGRUEncoder(
        input_features=n_features_total,
        seq_len=seq_len,
        tcn_channels=conf.get("tcn_channels", [64, 64, 128]),
        kernel_size=conf.get("kernel_size", 3),
        gru_hidden=gru_hidden,
        gru_layers=conf.get("gru_layers", 1),
        dropout=conf.get("dropout", 0.2),
        n_assets=n_assets,
        asset_embed_dim=asset_embed_dim,
    )
    trend_head = TrendMomentumHeadTF(emb_dim=gru_hidden, n_tfs=n_tfs)
    regime_cont_head = RegimeContinuousHead(emb_dim=gru_hidden, n_tfs=n_tfs, n_signals=4)
    proj_head = ProjectionHead(emb_dim=gru_hidden, proj_dim=proj_dim)
    mktstructure_head = MarketStructureHead(emb_dim=gru_hidden, n_tfs=len(TF_LIST))
    # Frozen snapshot of mktstructure_head for the strategy context.
    # Refreshed at the start of every epoch -> no within-batch co-adaptation
    # between mktstructure_head and the strategy heads, same information.
    import copy as _copy

    mktstructure_ctx_snap = _copy.deepcopy(mktstructure_head)

    # v28: heads of the final strategies
    strategy_ctx_dim = int(conf.get("strategy_context_dim", 29))
    session_ctx_dim = int(conf.get("session_context_dim", strategy_ctx_dim))
    pdhl_ctx_dim = int(conf.get("pdhl_context_dim", strategy_ctx_dim))
    qual_ctx_dim = int(conf.get("qual_context_dim", strategy_ctx_dim))
    session_head = SessionBreakoutHead(emb_dim=gru_hidden, ctx_dim=session_ctx_dim)
    pdhl_head = PDHLBreakoutHead(emb_dim=gru_hidden, ctx_dim=pdhl_ctx_dim)
    regime_quality_head = RegimeQualityHead(emb_dim=gru_hidden, ctx_dim=qual_ctx_dim)

    for m in (
        encoder,
        trend_head,
        regime_cont_head,
        proj_head,
        mktstructure_head,
        mktstructure_ctx_snap,
        session_head,
        pdhl_head,
        regime_quality_head,
    ):
        m.to(device)
        m.float()

    asset_map = {a: i for i, a in enumerate(assets)}

    def _build_full_optimizer():
        params = [
            p
            for p in (
                list(encoder.parameters())
                + list(trend_head.parameters())
                + list(regime_cont_head.parameters())
                + list(proj_head.parameters())
                + list(mktstructure_head.parameters())
                + list(session_head.parameters())
                + list(pdhl_head.parameters())
                + list(regime_quality_head.parameters())
            )
            if p.requires_grad
        ]
        return torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)

    optimizer = _build_full_optimizer()

    cp_config = {
        "temperature": float(conf.get("temperature", 0.1)),
        "memory_bank": {
            "enabled": (
                False if debug_mode else bool(conf.get("use_memory_bank_for_negatives", True))
            ),
            "capacity": int(conf.get("memory_bank_size", 16384)),
        },
        "proj_dim": proj_dim,
        "loss_weights": conf.get(
            "loss_weights", {"cross_pos": 1.0, "cross_neg": 1.0, "simclr_batch": 1.0}
        ),
        "aug_workers": int(conf.get("aug_workers", 4)),
    }
    cp = ContrastivePretrainer(
        encoder=encoder, proj_head=proj_head, device=device, config=cp_config
    )

    if bool(conf.get("keepalive", False)):
        threading.Thread(
            target=keepalive_loop,
            args=(int(conf.get("keepalive_interval_seconds", 300)),),
            daemon=True,
        ).start()

    if training_mode == "retrain" and not conf.get("resume_from"):
        custom = conf.get("retrain_start_from")
        if custom:
            conf["resume_from"] = custom
        else:
            qc = os.path.join(results_dir, f"retrain_full_{version_tag}.pth")
            conf["resume_from"] = qc if os.path.exists(qc) else find_best_checkpoint(results_dir)
    elif not conf.get("resume_from"):
        latest = os.path.join(results_dir, "latest_checkpoint.pth")
        if os.path.exists(latest):
            conf["resume_from"] = latest

    start_epoch = 1
    best_score_bootstrap = None
    best_score_retrain = None

    if conf.get("resume_from"):
        try:
            res = resume_checkpoint(
                conf["resume_from"],
                encoder,
                trend_head,
                proj_head,
                optimizer,
                regime_cont_head=regime_cont_head,
                mktstructure_head=mktstructure_head,
                session_head=session_head,
                pdhl_head=pdhl_head,
                regime_quality_head=regime_quality_head,
                map_location=device,
            )
            start_epoch = int(res.get("epoch", 0)) + 1
            logger.info(f"Resumed from epoch {start_epoch-1}")
        except Exception as e:
            logger.warning(f"Resume: {e}")
        logger.info("[RESUME] Forcing dataset re-init for all datasets...")
        for _ds_list in [train_datasets, val_datasets, drift_datasets]:
            for _ds in _ds_list:
                if hasattr(_ds, "_scalers_loaded"):
                    _ds._scalers_loaded = False
                    _ds._next_tf_close_by_tf = {}
                    _ds._regime_continuous_by_tf = {}
                    _ds._true_vol_by_tf = {}
                    _ds._regime_score_by_tf = {}
                    _ds._regime_thresholds_by_tf = {}

        # ── Best score — first from its own file, then from the checkpoint ──
    if training_mode == "bootstrap":
        best_score_bootstrap = load_best_score(results_dir, "bootstrap")
        if best_score_bootstrap is None:
            # fallback: read it from the checkpoint if there is one
            _best_ckpt_path = os.path.join(results_dir, "best_full_checkpoint.pth")
            if os.path.exists(_best_ckpt_path):
                try:
                    _bc = torch.load(_best_ckpt_path, map_location="cpu", weights_only=False)
                    _prev = float(_bc.get("extra", {}).get("score", float("-inf")))
                    if _prev > float("-inf"):
                        best_score_bootstrap = _prev
                        logger.info("[RESUME] Best score from checkpoint: %.6f", _prev)
                        save_best_score(
                            results_dir,
                            _prev,
                            0,
                            "bootstrap",
                            {"note": "recovered_from_checkpoint"},
                        )
                except Exception as e:
                    logger.warning("[RESUME] checkpoint score read: %s", e)
    else:
        best_score_retrain = load_best_score(results_dir, "retrain")
        if best_score_retrain is None:
            _best_ckpt_path = os.path.join(results_dir, "best_full_checkpoint.pth")
            if os.path.exists(_best_ckpt_path):
                try:
                    _bc = torch.load(_best_ckpt_path, map_location="cpu", weights_only=False)
                    _prev = float(_bc.get("extra", {}).get("score", float("-inf")))
                    if _prev > float("-inf"):
                        best_score_retrain = _prev
                        logger.info("[RESUME] Best score from checkpoint: %.6f", _prev)
                        save_best_score(
                            results_dir, _prev, 0, "retrain", {"note": "recovered_from_checkpoint"}
                        )
                except Exception as e:
                    logger.warning("[RESUME] checkpoint score read: %s", e)

    if training_mode == "bootstrap":
        _ftcn = str(conf.get("freeze_tcn", "none"))
        if _ftcn != "none":
            tcn_m = get_tcn_module(encoder)
            if tcn_m:
                _keep = int(conf.get("freeze_tcn_keep_last_n", 1))
                # .network to reach the real TemporalBlocks
                # .children() only returns the container (TCNEncoderBlock has 1 child)
                blocks = (
                    list(tcn_m.network) if hasattr(tcn_m, "network") else list(tcn_m.children())
                )
                freeze_up_to = len(blocks) if _ftcn == "full" else max(0, len(blocks) - _keep)
                for i, blk in enumerate(blocks):
                    set_requires_grad(blk, i >= freeze_up_to)
            optimizer = _build_full_optimizer()

    if training_mode == "retrain":
        n_retrain_epochs = int(conf.get("retrain_epochs", 18))
        end_epoch = start_epoch + n_retrain_epochs - 1
        logger.info(f"[RETRAIN] epochs {start_epoch}->{end_epoch}")
    else:
        end_epoch = int(conf.get("bootstrap_epochs", 25))

    retrain_encoder_path = None
    retrain_full_ckpt_path = None
    drift_baseline_path = os.path.join(results_dir, "drift_anchor_baseline.pth")
    regime_thresholds_prev_path = os.path.join(results_dir, "regime_thresholds_prev.json")
    if training_mode == "retrain":
        retrain_encoder_path = os.path.join(results_dir, f"encoder_{version_tag}.pth")
        retrain_full_ckpt_path = os.path.join(results_dir, f"retrain_full_{version_tag}.pth")

    baseline_emb_stats: Optional[dict] = None
    if training_mode == "retrain":
        baseline_emb_stats = load_baseline_emb_stats(drift_baseline_path)
        if baseline_emb_stats is None:
            # fallback: load from the resume_from checkpoint (emb_drift_stats is in extra)
            _fb_ckpt = conf.get("resume_from") or find_best_checkpoint(results_dir)
            if _fb_ckpt and os.path.exists(str(_fb_ckpt)):
                baseline_emb_stats = load_baseline_emb_stats(_fb_ckpt)
                if baseline_emb_stats:
                    logger.info("[DRIFT] Baseline loaded from resume_from: %s", _fb_ckpt)
        if baseline_emb_stats is None:
            logger.warning("[DRIFT] No baseline. Structural drift DISABLED.")

    trend_monitor_baseline: Optional[dict] = None
    if training_mode == "retrain":
        prev_ckpt = conf.get("resume_from") or find_best_checkpoint(results_dir)
        trend_monitor_baseline = load_trend_monitor_baseline(prev_ckpt)

    previous_regime_stats: Optional[dict] = None
    last_clean_live_epoch: int = start_epoch - 1

    anchor_model: Optional[dict[str, nn.Module]] = None
    if training_mode == "retrain" and use_kl_anchor:
        anchor_ckpt = kl_anchor_ckpt or conf.get("resume_from") or find_best_checkpoint(results_dir)
        if anchor_ckpt:
            anchor_model = load_anchor_model(anchor_ckpt, encoder, proj_head, device)

    gradient_clip = float((conf.get("trainer", {}) or {}).get("gradient_clip", 1.0))
    accum_steps = int(conf.get("accumulation_steps", 1))
    validate_every = int(conf.get("validate_every_epochs", 1))
    val_max_batches = int(conf.get("val_max_batches", 200))
    simclr_every = int(conf.get("simclr_every", 1))
    lambda_contrast = float(conf.get("lambda_contrast", 0.05))
    lambda_dir_contrast = float(conf.get("lambda_dir_contrast", 0.3))
    _dir_contrast_fn = DirectionalContrastiveLoss(
        margin=float(conf.get("dir_contrast_margin", 0.3)),
        tmom_threshold=float(conf.get("dir_contrast_tmom_thr", 0.3)),
        max_pairs=int(conf.get("dir_contrast_max_pairs", 2048)),
        tmom_tf_idx=int(conf.get("dir_contrast_tmom_tf_idx", 1)),  # 1=30min
    ).to(device)
    variance_loss_weight = float(conf.get("variance_loss_weight", 0.01))
    dummy_reg = float(conf.get("dummy_reg", 1e-6))

    use_ema_balancing = bool(conf.get("use_ema_balancing", True))
    ema_alpha = float(conf.get("ema_alpha", 0.99))
    ema_alpha_warmup = float(conf.get("ema_alpha_warmup", 0.95))
    ema_warmup_epochs = int(conf.get("ema_warmup_epochs", 2))
    loss_ema = {"trend": 1.0, "regime": 1.0, "contrast": 1.0}  # v28: vol/recon removed

    trend_loss_weight = float(conf.get("trend_loss_weight", 1.0))
    log_trend_train_debug = bool(conf.get("log_trend_train_debug", True))
    log_trend_train_every = int(conf.get("log_trend_train_every", 500))

    trend_loss_exclude_tfs = set(conf.get("trend_loss_exclude_tfs", ["15min"]))
    trend_loss_exclude_idx = {i for i, tf in enumerate(TF_LIST) if tf in trend_loss_exclude_tfs}
    if trend_loss_exclude_idx:
        logger.info(
            "[TREND LOSS] Excluding TFs from loss: %s (idx=%s)",
            trend_loss_exclude_tfs,
            trend_loss_exclude_idx,
        )

    score_w_trend = float(conf.get("score_w_trend", 0.5))
    score_w_regime = float(conf.get("score_w_regime", 1.0))
    emb_max_points = int(conf.get("emb_max_points", 2048))
    emb_seed = int(conf.get("emb_seed", 0))

    history = {"train": [], "val": []}
    current_phase_name = None

    _hist_tag = version_tag or "bootstrap"
    _history_path = os.path.join(results_dir, f"pretrain_history_{_hist_tag}.json")
    os.makedirs(results_dir, exist_ok=True)

    def _save_history():
        try:
            existing: dict[str, list] = {"train": [], "val": []}
            if os.path.exists(_history_path):
                try:
                    with open(_history_path, encoding="utf-8") as _f:
                        _prev = json.load(_f)
                    for _k in ("train", "val"):
                        _prev_list = _prev.get(_k, [])
                        _curr_list = history.get(_k, [])
                        # In retrain mode never carry over epochs from previous runs
                        # (Bootstrap epochs would accumulate in the WF JSON).
                        # In bootstrap mode with resume, keep epochs before start_epoch
                        # so interrupted runs can be resumed cleanly.
                        if training_mode == "retrain":
                            _keep = []
                        else:
                            _keep = _prev_list[: max(0, start_epoch - 1)]
                        existing[_k] = _keep + _curr_list
                except Exception as _je:
                    logger.warning("[HISTORY] merge failed (%s), overwrite", _je)
                    existing = history
            else:
                existing = history
            _serializable = json.loads(
                json.dumps(
                    existing,
                    default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x,
                )
            )
            with open(_history_path, "w", encoding="utf-8") as _f:
                json.dump(_serializable, _f, indent=2)
        except Exception as _e:
            logger.warning("[HISTORY] full save failed: %s", _e)

    trend_monitor_ema: dict[str, float] = {}

    # =============================================================
    # Pre-retrain baseline evaluation
    # Evaluates the anchor checkpoint on the new val period BEFORE
    # any training. Saves pre_retrain_baseline_{version_tag}.json
    # and logs full metrics for reference / OOS attendibility check.
    # =============================================================
    if training_mode == "retrain" and conf.get("pre_retrain_baseline_eval", True):
        logger.info(
            "[PRE-RETRAIN BASELINE] Running pre-retrain eval on val period %s → %s ...",
            val_start,
            val_end,
        )
        logger.info("[PRE-RETRAIN BASELINE] Starting eval | version=%s", version_tag)
        logger.info("[PRE-RETRAIN BASELINE] val_loader batches (max=%d)", val_max_batches)

        _bl_metrics: dict[str, float] = {}
        _bl_score: Optional[float] = None

        try:
            # ── Eval mode ────────────────────────────────────────────────────────
            encoder.eval()
            trend_head.eval()
            regime_cont_head.eval()
            proj_head.eval()
            mktstructure_head.eval()
            session_head.eval()
            pdhl_head.eval()
            regime_quality_head.eval()

            _bl_tmom_pred: dict[str, list] = {tf: [] for tf in TF_LIST}
            _bl_tmom_true: dict[str, list] = {tf: [] for tf in TF_LIST}
            _bl_rc_pred: dict[str, list] = {tf: [] for tf in TF_LIST}
            _bl_rc_true: dict[str, list] = {tf: [] for tf in TF_LIST}
            _bl_mktstr_logits: dict[str, list] = {tf: [] for tf in TF_LIST}
            _bl_mktstr_true: dict[str, list] = {tf: [] for tf in TF_LIST}

            with torch.no_grad():
                for _bl_cnt, _bl_vb in enumerate(val_loader):
                    if _bl_cnt >= val_max_batches:
                        break
                    _bl_seq = _bl_vb["seq"].to(device, non_blocking=True).float()
                    _bl_masked = (
                        _bl_vb.get("masked_seq", _bl_seq).to(device, non_blocking=True).float()
                    )
                    _bl_vassets = _bl_vb.get("asset", None) or ([None] * _bl_seq.size(0))
                    _bl_vaid = torch.tensor(
                        [asset_map.get(a, 0) for a in _bl_vassets], dtype=torch.long, device=device
                    )
                    _bl_emb = encoder(_bl_masked, asset_ids=_bl_vaid)

                    # ── TrendMomentum ──────────────────────────────────────────
                    _bl_tmom_out = trend_head(_bl_emb)  # [B, n_tfs]
                    _bl_tmom_tgt = _bl_vb.get("trend_momentum", None)
                    if _bl_tmom_tgt is not None:
                        _bl_tt = (
                            _bl_tmom_tgt.to(device).float()
                            if isinstance(_bl_tmom_tgt, torch.Tensor)
                            else torch.tensor(_bl_tmom_tgt, device=device, dtype=torch.float32)
                        )
                        for _tfi, _tf in enumerate(TF_LIST):
                            _valid = (_bl_tt[:, _tfi] > -900.0) & torch.isfinite(_bl_tt[:, _tfi])
                            if _valid.any():
                                _bl_tmom_pred[_tf].append(
                                    _bl_tmom_out[:, _tfi][_valid].cpu().numpy()
                                )
                                _bl_tmom_true[_tf].append(_bl_tt[:, _tfi][_valid].cpu().numpy())

                    # ── RegimeContinuous ──────────────────────────────────────
                    _bl_rc_tgt = _bl_vb.get("regime_continuous", None)
                    if _bl_rc_tgt is not None:
                        _bl_rt = (
                            _bl_rc_tgt.to(device).float()
                            if isinstance(_bl_rc_tgt, torch.Tensor)
                            else torch.tensor(_bl_rc_tgt, device=device, dtype=torch.float32)
                        )
                        _bl_rc_out = regime_cont_head(_bl_emb)  # [B, n_tfs, 4]
                        for _tfi, _tf in enumerate(TF_LIST):
                            _t = _bl_rt[:, _tfi, :]  # [B, 4]
                            _p = _bl_rc_out[:, _tfi, :]  # [B, 4]
                            _valid = (_t > -900.0).all(dim=-1)
                            if _valid.any():
                                _bl_rc_pred[_tf].append(_p[_valid].cpu().numpy())
                                _bl_rc_true[_tf].append(_t[_valid].cpu().numpy())

                    # ── MktStructure ──────────────────────────────────────────
                    _bl_ms_all = _bl_vb.get("mktstructure", None)
                    if _bl_ms_all is not None:
                        _bl_ms_t = (
                            _bl_ms_all.to(device).long()
                            if isinstance(_bl_ms_all, torch.Tensor)
                            else torch.tensor(_bl_ms_all, device=device, dtype=torch.long)
                        )
                        _bl_ms_logits = mktstructure_head(_bl_emb)  # [B, n_tfs, 3]
                        for _tfi, _tf in enumerate(TF_LIST):
                            if _tfi >= _bl_ms_t.size(1):
                                break
                            _bl_ms_tgt_i = _bl_ms_t[:, _tfi]
                            _bl_ms_prob_i = torch.softmax(_bl_ms_logits[:, _tfi, :].float(), dim=-1)
                            _valid = _bl_ms_tgt_i >= 0
                            if _valid.any():
                                _bl_mktstr_logits[_tf].append(_bl_ms_prob_i[_valid].cpu().numpy())
                                _bl_mktstr_true[_tf].append(_bl_ms_tgt_i[_valid].cpu().numpy())

            # ── Compute baseline metrics ──────────────────────────────────────
            _bl_trend_accs: list = []
            _bl_rc_corrs: list = []
            _bl_n_total = 0

            _bl_sig_names = ["vol_regime", "atr_norm", "adx", "di_direction"]

            for _tf in TF_LIST:
                # TrendMomentum
                if _bl_tmom_pred[_tf] and _bl_tmom_true[_tf]:
                    _p = np.concatenate(_bl_tmom_pred[_tf])
                    _t = np.concatenate(_bl_tmom_true[_tf])
                    _n = len(_p)
                    _bl_n_total += _n
                    _dir_acc = float(np.mean(np.sign(_p) == np.sign(_t)))
                    _mae = float(np.mean(np.abs(_p - _t)))
                    try:
                        _corr = _safe_corr(_p, _t)
                    except Exception:
                        _corr = float("nan")
                    _bl_metrics[f"tmomentum_{_tf}_dir_acc"] = _dir_acc
                    _bl_metrics[f"tmomentum_{_tf}_mae"] = _mae
                    _bl_metrics[f"tmomentum_{_tf}_corr"] = _corr
                    _bl_metrics[f"tmomentum_{_tf}_n"] = _n
                    if _tf not in trend_loss_exclude_tfs:
                        _bl_trend_accs.append(_dir_acc)
                    if _tf in (monitor_tfs or TF_LIST):
                        _rc_mae = _bl_metrics.get(f"regime_{_tf}_mean_mae", float("nan"))
                        logger.info(
                            "[PRE-RETRAIN BASELINE] %s | tmom_dir=%.3f mae=%.4f | regime_mean_mae=%.4f",
                            _tf,
                            _dir_acc,
                            _mae,
                            _rc_mae if not math.isnan(_rc_mae) else 0.0,
                        )

                # RegimeContinuous
                if _bl_rc_pred[_tf] and _bl_rc_true[_tf]:
                    _rcp = np.concatenate(_bl_rc_pred[_tf])  # [N, 4]
                    _rct = np.concatenate(_bl_rc_true[_tf])  # [N, 4]
                    _n_sig = min(_rcp.shape[1], _rct.shape[1], len(_bl_sig_names))
                    _sig_maes: list = []
                    _sig_corrs: list = []
                    for _si in range(_n_sig):
                        _sn = _bl_sig_names[_si]
                        _sm = float(np.mean(np.abs(_rcp[:, _si] - _rct[:, _si])))
                        _sig_maes.append(_sm)
                        _bl_metrics[f"regime_{_tf}_{_sn}_mae"] = _sm
                        try:
                            _sc = _safe_corr(_rcp[:, _si], _rct[:, _si])
                        except Exception:
                            _sc = float("nan")
                        if not math.isnan(_sc):
                            _bl_metrics[f"regime_{_tf}_{_sn}_corr"] = _sc
                            if _tf in (monitor_tfs or TF_LIST):
                                _bl_rc_corrs.append(max(0.0, _sc))
                                _sig_corrs.append(_sc)
                    _mean_mae = float(np.mean(_sig_maes)) if _sig_maes else float("nan")
                    _bl_metrics[f"regime_{_tf}_mean_mae"] = _mean_mae
                    # Log AFTER tmom line for same TF — need to patch tmom log above
                    # (already logged tmom above; log regime only for non-monitor TFs here)
                    if _tf not in (monitor_tfs or TF_LIST):
                        logger.info(
                            "[PRE-RETRAIN BASELINE] %s | regime_mean_mae=%.4f", _tf, _mean_mae
                        )

                # MktStructure brk_auc
                if _bl_mktstr_logits[_tf] and _bl_mktstr_true[_tf]:
                    _msp = np.concatenate(_bl_mktstr_logits[_tf])  # [N, 3]
                    _mst = np.concatenate(_bl_mktstr_true[_tf])  # [N]
                    try:
                        _ms_pred = np.argmax(_msp, axis=1)
                        _ms_acc = float(np.mean(_ms_pred == _mst))
                        _is_brk = (_mst > 0).astype(int)
                        _brk_prob = _msp[:, 1] + _msp[:, 2]
                        _ms_auc = (
                            float(roc_auc_score(_is_brk, _brk_prob))
                            if len(np.unique(_is_brk)) > 1
                            else float("nan")
                        )
                        _bl_metrics[f"mktstr_{_tf}_acc"] = _ms_acc
                        _bl_metrics[f"mktstr_{_tf}_brk_auc"] = _ms_auc
                    except Exception:
                        pass

            # Re-log monitor TFs with regime once available
            # (regime computed after tmom loop — log combined here for any missed TFs)
            for _tf in monitor_tfs or []:
                _td = _bl_metrics.get(f"tmomentum_{_tf}_dir_acc", float("nan"))
                _tm = _bl_metrics.get(f"tmomentum_{_tf}_mae", float("nan"))
                _rma = _bl_metrics.get(f"regime_{_tf}_mean_mae", float("nan"))
                if not math.isnan(_td):
                    logger.info(
                        "[PRE-RETRAIN BASELINE] %s | tmom_dir=%.3f mae=%.4f | regime_mean_mae=%.4f",
                        _tf,
                        _td,
                        _tm,
                        _rma if not math.isnan(_rma) else 0.0,
                    )

            # ── v28 / Strategy head baseline eval ────────────────────────────
            # Evaluates session/pdhl/regime_quality on the anchor BEFORE the retrain,
            # with the SAME score+coverage logic as the per-epoch loop, so the
            # baseline is directly comparable with the validation score.
            _bl_v28_score = 0.0
            try:
                _bl_v28_metrics = evaluate_v28_heads(
                    session_head,
                    pdhl_head,
                    regime_quality_head,
                    encoder,
                    trend_head,
                    mktstructure_head,
                    val_loader,
                    device,
                    TF_LIST,
                    max_batches=val_max_batches,
                    asset_map=asset_map,
                )
                _bl_metrics.update(_bl_v28_metrics)

                # v28_score = mean of (AUC-0.5)*2 over sess/pdhl side+hit
                _bl_v28_aucs = [
                    _bl_v28_metrics.get(k, float("nan"))
                    for k in (
                        "v28_session_hit_auc",
                        "v28_session_side_auc",
                        "v28_pdhl_hit_auc",
                        "v28_pdhl_side_auc",
                    )
                ]
                # Coverage PER HEAD, same as the per-epoch computation: baseline and
                # retrain must use the same formula, or the "does the anchor
                # generalize?" check compares two different measures.
                _bl_v28_min_n = float(conf.get("v28_min_n_for_full_score", 150.0))
                # same pairs as the per-epoch computation: baseline and retrain must
                # use the same formula, or the "does the anchor generalize?" check
                # compares two different measures
                _bl_pairs = [
                    ("v28_session_long_side_auc", "v28_session_long_side_n"),
                    ("v28_session_short_side_auc", "v28_session_short_side_n"),
                    ("v28_pdhl_long_side_auc", "v28_pdhl_long_side_n"),
                    ("v28_pdhl_short_side_auc", "v28_pdhl_short_side_n"),
                ]
                _bn, _bd, _bc, _bns = 0.0, 0.0, [], []
                for _ak, _nk in _bl_pairs:
                    _a = _bl_v28_metrics.get(_ak, float("nan"))
                    if math.isnan(_a):
                        continue
                    _n = float(_bl_v28_metrics.get(_nk, 0) or 0)
                    _c = min(1.0, _n / max(1.0, _bl_v28_min_n))
                    _bn += (_a - 0.5) * 2.0 * _c
                    _bd += _c
                    _bc.append(_c)
                    _bns.append(_n)
                _bl_v28_raw = (_bn / _bd) if _bd > 0 else 0.0
                _bl_v28_score = _bl_v28_raw
                _bl_v28_cov = float(np.mean(_bc)) if _bc else 0.0
                _bl_v28_n_mean = float(np.mean(_bns)) if _bns else 0.0

                _bl_metrics["v28_score"] = _bl_v28_score
                _bl_metrics["v28_n_mean"] = _bl_v28_n_mean
                _bl_metrics["v28_cov_factor"] = _bl_v28_cov

                logger.info(
                    "[PRE-RETRAIN BASELINE V28] sess_hit=%.3f sess_side=%.3f | "
                    "pdhl_hit=%.3f pdhl_side=%.3f | qual_auc=%.3f | "
                    "n_mean=%.0f cov=%.2f v28_score=%.3f",
                    _bl_v28_metrics.get("v28_session_hit_auc", float("nan")),
                    _bl_v28_metrics.get("v28_session_side_auc", float("nan")),
                    _bl_v28_metrics.get("v28_pdhl_hit_auc", float("nan")),
                    _bl_v28_metrics.get("v28_pdhl_side_auc", float("nan")),
                    _bl_v28_metrics.get("v28_regime_quality_auc", float("nan")),
                    _bl_v28_n_mean,
                    _bl_v28_cov,
                    _bl_v28_score,
                )
                logger.info(
                    "[PRE-RETRAIN BASELINE V28 DETAIL] "
                    "sess_L side=%.3f (n=%.0f) | sess_S side=%.3f (n=%.0f) | "
                    "pdhl_L side=%.3f (n=%.0f) | pdhl_S side=%.3f (n=%.0f)",
                    _bl_v28_metrics.get("v28_session_long_side_auc", float("nan")),
                    _bl_v28_metrics.get("v28_session_long_side_n", 0),
                    _bl_v28_metrics.get("v28_session_short_side_auc", float("nan")),
                    _bl_v28_metrics.get("v28_session_short_side_n", 0),
                    _bl_v28_metrics.get("v28_pdhl_long_side_auc", float("nan")),
                    _bl_v28_metrics.get("v28_pdhl_long_side_n", 0),
                    _bl_v28_metrics.get("v28_pdhl_short_side_auc", float("nan")),
                    _bl_v28_metrics.get("v28_pdhl_short_side_n", 0),
                )
            except Exception as _bl_v28e:
                logger.warning("[PRE-RETRAIN BASELINE V28] eval failed: %s", _bl_v28e)
                # fallback: score without v28 (legacy behaviour), v28_score=0

            # Overall baseline score
            _bl_score_w_v28 = float(conf.get("score_w_v28", 0.40))
            _bl_trend_acc = float(np.mean(_bl_trend_accs)) if _bl_trend_accs else float("nan")
            _bl_trend_score = 0.0 if math.isnan(_bl_trend_acc) else (_bl_trend_acc - 0.5) * 2.0
            _bl_regime_score = float(np.mean(_bl_rc_corrs)) if _bl_rc_corrs else 0.0
            _bl_score = (
                score_w_trend * _bl_trend_score
                + score_w_regime * _bl_regime_score
                + _bl_score_w_v28 * _bl_v28_score
            )
            _bl_metrics["_v28_score"] = _bl_v28_score
            _bl_metrics["_score"] = _bl_score
            _bl_metrics["_trend_score"] = _bl_trend_score
            _bl_metrics["_regime_score"] = _bl_regime_score
            _bl_metrics["_trend_acc"] = _bl_trend_acc

            # ── Primary summary log ───────────────────────────────────────────
            logger.info(
                "[PRE-RETRAIN BASELINE] score=%.4f | tmom_dir=%.3f(sc=%.3f×%.2f) "
                "regime_corr=%.3f(×%.2f) v28=%.3f(×%.2f) | version=%s",
                _bl_score,
                _bl_trend_acc,
                _bl_trend_score,
                score_w_trend,
                _bl_regime_score,
                score_w_regime,
                _bl_v28_score,
                _bl_score_w_v28,
                version_tag,
            )

            # ── Per-TF detail log (same format as normal val) ────────────────
            for _tf in monitor_tfs or TF_LIST:
                _td = _bl_metrics.get(f"tmomentum_{_tf}_dir_acc", float("nan"))
                _tm = _bl_metrics.get(f"tmomentum_{_tf}_mae", float("nan"))
                _rma = _bl_metrics.get(f"regime_{_tf}_mean_mae", float("nan"))
                if not math.isnan(_td):
                    logger.info(
                        "[PRE-RETRAIN BASELINE] %s | tmom_dir=%.3f mae=%.4f | regime_mean_mae=%.4f",
                        _tf,
                        _td,
                        _tm,
                        _rma if not math.isnan(_rma) else 0.0,
                    )

            # ── Save JSON ─────────────────────────────────────────────────────
            _bl_json_path = os.path.join(results_dir, f"pre_retrain_baseline_{version_tag}.json")
            _bl_save = {
                "version_tag": version_tag,
                "timestamp": pd.Timestamp.now().isoformat(),
                "score": _bl_score,
                "score_components": {
                    "trend": {"score": _bl_trend_score, "weight": score_w_trend},
                    "regime": {"score": _bl_regime_score, "weight": score_w_regime},
                    "v28": {"score": _bl_v28_score, "weight": _bl_score_w_v28},
                },
                # v28 at top level: walk-forward reliability check — how the model of
                # the PREVIOUS retrain (anchor) does on the downstream heads
                # in the new OOS period (the one where it would have been deployed).
                "v28": {
                    "session_hit_auc": _bl_metrics.get("v28_session_hit_auc", float("nan")),
                    "session_side_auc": _bl_metrics.get("v28_session_side_auc", float("nan")),
                    "pdhl_hit_auc": _bl_metrics.get("v28_pdhl_hit_auc", float("nan")),
                    "pdhl_side_auc": _bl_metrics.get("v28_pdhl_side_auc", float("nan")),
                    "regime_quality_auc": _bl_metrics.get("v28_regime_quality_auc", float("nan")),
                    "regime_quality_n": _bl_metrics.get("v28_regime_quality_n", 0),
                    "n_mean": _bl_metrics.get("v28_n_mean", 0.0),
                    "cov_factor": _bl_metrics.get("v28_cov_factor", 0.0),
                    "v28_score": _bl_v28_score,
                    "detail": {
                        _n: {
                            "side_auc": _bl_metrics.get(f"v28_{_n}_side_auc", float("nan")),
                            "side_acc": _bl_metrics.get(f"v28_{_n}_side_acc", float("nan")),
                            "side_n": _bl_metrics.get(f"v28_{_n}_side_n", 0),
                            "tp_rate": _bl_metrics.get(f"v28_{_n}_tp_rate", float("nan")),
                        }
                        for _n in ("session_long", "session_short", "pdhl_long", "pdhl_short")
                    },
                },
                "metrics": _bl_metrics,
                "val_start": val_start,
                "val_end": val_end,
                "train_end": train_end,
                "anchor_ckpt": str(conf.get("resume_from", "")),
                "emb": {},
            }
            try:
                _bl_serializable = json.loads(
                    json.dumps(
                        _bl_save,
                        default=lambda x: (
                            float(x) if isinstance(x, (np.floating, np.integer)) else x
                        ),
                    )
                )
                with open(_bl_json_path, "w", encoding="utf-8") as _blf:
                    json.dump(_bl_serializable, _blf, indent=2)
                logger.info("[PRE-RETRAIN BASELINE] Saved → %s", _bl_json_path)
            except Exception as _bl_je:
                logger.warning("[PRE-RETRAIN BASELINE] JSON save failed: %s", _bl_je)

            # ── Optional degradation alert ────────────────────────────────────
            _bl_deg = float(conf.get("baseline_degradation_threshold", 0.05))
            if _bl_deg > 0 and _bl_score < (1.0 - _bl_deg):
                logger.warning(
                    "[PRE-RETRAIN BASELINE] ⚠️  DEGRADED score=%.4f < %.2f — "
                    "check anchor checkpoint quality.",
                    _bl_score,
                    1.0 - _bl_deg,
                )

        except Exception as _bl_e:
            logger.warning("[PRE-RETRAIN BASELINE] eval failed: %s", _bl_e)
        finally:
            # Restore train mode for the training loop
            encoder.train()
            trend_head.train()
            regime_cont_head.train()
            proj_head.train()
            mktstructure_head.train()
            session_head.train()
            pdhl_head.train()
            regime_quality_head.train()

    # =========================
    # Epoch loop
    # =========================
    for epoch in range(start_epoch, end_epoch + 1):

        # long/short correlation of the last batch, per head. It's the MEASURE
        # of the anti-phase: worth watching even with w_antiphase=0, because it
        # tells whether the problem exists before trying to fix it.
        #   ~0   the two legs are independent (goal)
        #   <0   anti-phase: one is the mirror image of the other
        #   >0   collapse: same score whatever the direction
        _af_acc = {"session_sum": 0.0, "session_n": 0, "pdhl_sum": 0.0, "pdhl_n": 0}

        # refresh the frozen snapshot for the context — once per epoch,
        # never inside the batch loop -> no within-epoch co-adaptation.
        with torch.no_grad():
            mktstructure_ctx_snap.load_state_dict(mktstructure_head.state_dict())
        for _p in mktstructure_ctx_snap.parameters():
            _p.requires_grad_(False)
        mktstructure_ctx_snap.eval()

        if training_mode == "retrain":
            epoch_in_retrain = epoch - start_epoch + 1
            phase = (
                get_phase_for_epoch(active_schedule_template, epoch_in_retrain)
                or active_schedule_template[-1]
            )
            if phase["name"] != current_phase_name:
                current_phase_name = phase["name"]
                logger.info(f"[PHASE] ===== {phase['name']} | {phase['description']} =====")
                optimizer = apply_phase(
                    encoder,
                    trend_head,
                    proj_head,
                    phase,
                    base_lr,
                    weight_decay,
                    regime_cont_head=regime_cont_head,
                    mktstructure_head=mktstructure_head,
                    session_head=session_head,
                    pdhl_head=pdhl_head,
                    regime_quality_head=regime_quality_head,
                )
        model_params = [p for pg in optimizer.param_groups for p in pg["params"] if p.requires_grad]

        encoder.train()
        trend_head.train()
        regime_cont_head.train()
        proj_head.train()
        mktstructure_head.train()
        session_head.train()
        pdhl_head.train()
        regime_quality_head.train()

        for _ds in train_datasets:
            _ds._asset_id = asset_map.get(getattr(_ds, "asset_name", None), 0)

        w_regime = w_regime_retrain if training_mode == "retrain" else w_regime_bootstrap
        _cur_ema_alpha = ema_alpha_warmup if epoch <= ema_warmup_epochs else ema_alpha
        running = {
            "task_total": 0.0,
            "contrast_total": 0.0,
            "kl_total": 0.0,
            "dir_contrast_total": 0.0,
        }
        _grad_norm_accum: list[float] = []
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{end_epoch}", leave=False)
        batch_idx = -1
        optimizer.zero_grad()

        epoch_trend_monitor_accum: dict[str, list[float]] = {}
        for tf in monitor_tfs:
            for key in (
                f"tmom_mean_{tf}",
                f"tmom_std_{tf}",
                f"pct_bullish_{tf}",
                f"pct_bearish_{tf}",
                f"pct_neutral_{tf}",
            ):
                epoch_trend_monitor_accum[key] = []

        train_iter = iter(train_loader)
        while True:
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            except Exception as e:
                logger.warning("DataLoader: %s", e)
                continue

            batch_idx += 1
            bad, reasons = _validate_batch(batch, seq_len)
            if bad:
                logger.warning("Corrupt batch ep%d idx%d: %s", epoch, batch_idx, reasons)
                pbar.update(1)
                continue

            seq = batch["seq"]
            if seq.ndim != 3:
                pbar.update(1)
                continue
            if seq.size(1) != seq_len:
                if seq.size(1) > seq_len:
                    seq = seq[:, :seq_len, :].contiguous()
                else:
                    pad = torch.zeros(
                        (seq.size(0), seq_len - seq.size(1), seq.size(2)), dtype=seq.dtype
                    )
                    seq = torch.cat([seq, pad], dim=1)

            seq = seq.to(device, non_blocking=True).float()
            masked_seq = batch.get("masked_seq", seq).to(device, non_blocking=True).float()

            mask = batch.get("mask", None)
            if mask is None:
                mask = torch.zeros((seq.size(0), seq_len, 1), dtype=torch.uint8)
            if isinstance(mask, torch.Tensor):
                if mask.ndim == 2:
                    mask = mask.unsqueeze(-1)
                elif mask.ndim == 3 and mask.size(2) != 1:
                    mask = mask[:, :, :1]
                elif mask.ndim != 3:
                    mask = torch.zeros((seq.size(0), seq_len, 1), dtype=torch.uint8)
            else:
                mask = torch.zeros((seq.size(0), seq_len, 1), dtype=torch.uint8)
            mask = mask.to(device, non_blocking=True).bool()

            regime = batch.get("regime").to(device, non_blocking=True)
            is_close = (
                batch.get("is_close", torch.zeros((seq.size(0), n_tfs), dtype=torch.uint8))
                .to(device, non_blocking=True)
                .bool()
            )

            timestamps_raw = batch.get("timestamp", None)
            timestamps_list = (
                [str(t) if t is not None else None for t in timestamps_raw]
                if isinstance(timestamps_raw, (list, tuple))
                else (
                    [str(timestamps_raw)] * seq.size(0)
                    if timestamps_raw is not None
                    else [None] * seq.size(0)
                )
            )
            assets_batch = batch.get("asset", None) or ([None] * seq.size(0))
            asset_ids = [
                asset_map.get(a, asset_map.get(next(iter(asset_map), None), 0))
                for a in assets_batch
            ]
            _unknown = [a for a in assets_batch if a not in asset_map]
            if _unknown and batch_idx == 0:
                logger.warning(
                    "[ASSET IDS] Unknown assets in batch: %s (mapped to first known)", set(_unknown)
                )
            _asset_ids_t = torch.tensor(asset_ids, dtype=torch.long, device=device)

            with autocast_ctx_factory():
                emb_masked = encoder(masked_seq, asset_ids=_asset_ids_t)
                z_proj_local = proj_head(emb_masked)
                # vol/recon: weight=0 -> no forward (avoids a shape mismatch from the stub)
                strategy_context = build_strategy_context(
                    batch=batch,
                    trend_head=trend_head,
                    mktstructure_head=mktstructure_ctx_snap,  # frozen snapshot, no co-adaptation
                    emb=emb_masked,
                    tf_list=TF_LIST,
                )

            with torch.no_grad():
                _t_sc_mon = trend_head(emb_masked).detach()
                batch_tm = compute_trend_monitor_stats(
                    _t_sc_mon, is_close.detach(), TF_LIST, monitor_tfs
                )
            for tf in monitor_tfs:
                for key in (
                    f"tmom_mean_{tf}",
                    f"tmom_std_{tf}",
                    f"pct_bullish_{tf}",
                    f"pct_bearish_{tf}",
                    f"pct_neutral_{tf}",
                ):
                    v = batch_tm.get(key)
                    if v is not None and not math.isnan(v):
                        epoch_trend_monitor_accum[key].append(v)

            # ── Trend Momentum loss — Huber + flat filter + sign penalty ─────────
            # Three anti-collapse-to-0 mechanisms:
            #   1. flat filter: drops |tmom|<0.15, moves the minimum from pred=0 to pred≈±0.6
            #   2. sign penalty: hinge loss on the sign, punishes the wrong direction
            #   3. Huber fallback for completely flat TFs (no gradient lost)
            loss_trend_raw = torch.tensor(0.0, device=device)
            trend_terms = 0
            _tmom_tgt_train = batch.get("trend_momentum", None)
            if _tmom_tgt_train is not None:
                _tmom_tgt_t = (
                    _tmom_tgt_train.to(device).float()
                    if isinstance(_tmom_tgt_train, torch.Tensor)
                    else torch.tensor(_tmom_tgt_train, device=device, dtype=torch.float32)
                )  # [B, n_tfs]
                t_scalars = trend_head(emb_masked)  # [B, n_tfs]
                _huber_d = float(conf.get("trend_momentum_huber_delta", 1.0))
                _flat_thr = float(conf.get("tmom_flat_threshold", 0.15))
                _sign_w = float(conf.get("tmom_sign_penalty_weight", 0.3))
                for tf_i in range(t_scalars.size(1)):
                    if tf_i in trend_loss_exclude_idx:
                        continue
                    _tgt_i = _tmom_tgt_t[:, tf_i]
                    _pred_i = t_scalars[:, tf_i]
                    # Mask 1: skip sentinel + non-finite
                    _valid = (_tgt_i > -900.0) & torch.isfinite(_tgt_i)
                    if not _valid.any():
                        continue
                    # Mask 2: flat filter — esclude |tmom| < threshold
                    _dir_mask = _valid & (_tgt_i.abs() >= _flat_thr)
                    if _dir_mask.any():
                        _p = _pred_i[_dir_mask]
                        _t = _tgt_i[_dir_mask]
                        _huber = F.huber_loss(_p, _t, delta=_huber_d, reduction="mean")
                        _sign_pen = F.relu(-_p * torch.sign(_t)).mean()
                        loss_trend_raw = loss_trend_raw + _huber + _sign_w * _sign_pen
                        trend_terms += 1
                    elif _valid.any():
                        # completely flat TF: plain Huber so the gradient isn't lost
                        loss_trend_raw = loss_trend_raw + F.huber_loss(
                            _pred_i[_valid], _tgt_i[_valid], delta=_huber_d, reduction="mean"
                        )
                        trend_terms += 1
            if trend_terms > 0:
                loss_trend_raw = loss_trend_raw / float(trend_terms)
            # Train debug log: scalar stats per TF
            if log_trend_train_debug and (batch_idx % log_trend_train_every == 0):
                with torch.no_grad():
                    _td_scalars = trend_head(emb_masked).detach()
                    parts = []
                    for tf_i in range(n_tfs):
                        m = is_close[:, tf_i]
                        if not bool(m.any()):
                            continue
                        _sc = _td_scalars[m, tf_i]
                        _n = int(m.sum())
                        _mn = float(_sc.mean())
                        _std = float(_sc.std(correction=0))
                        _bull = int((_sc > 0.1).sum())
                        _bear = int((_sc < -0.1).sum())
                        parts.append(
                            f"{TF_LIST[tf_i]} n={_n} mean={_mn:+.2f} std={_std:.2f} bull={_bull} bear={_bear}"
                        )
                    if parts:
                        logger.info(
                            "[TRAIN TMOM] ep=%d step=%d | %s", epoch, batch_idx, " | ".join(parts)
                        )

            # ── Regime Continuous loss — Huber regression per TF per signal ────────
            # target batch["regime_continuous"]: [B, n_tfs, 3]
            #   [:,:,0] = vol_regime  (vol/vol_mean, inverse-scaled, ~[0,3])
            #   [:,:,1] = atr_norm×100 (~[0.05,2])
            #   [:,:,2] = adx/100      (~[0,1])
            # skip = -999.0. Huber delta per signal: [0.3, 0.1, 0.1]
            loss_regime_raw = torch.tensor(0.0, device=device)
            _rc_tgt = batch.get("regime_continuous", None)
            if _rc_tgt is not None:
                _rc_tgt_t = (
                    _rc_tgt.to(device).float()
                    if isinstance(_rc_tgt, torch.Tensor)
                    else torch.tensor(_rc_tgt, device=device, dtype=torch.float32)
                )
                _rc_pred = regime_cont_head(emb_masked)  # [B, n_tfs, 4] in v25
                # Huber delta per signal: vol_regime=0.3, atr_norm=0.1, adx=0.1, di_dir=0.05
                # di_direction in [-1,1] -> smaller delta for a precise gradient
                _rc_deltas = [0.3, 0.1, 0.1, 0.05]  # v25: di_direction added
                _rc_terms = 0
                for _tf_i in range(_rc_tgt_t.size(1)):
                    for _sig_i in range(_rc_tgt_t.size(2)):
                        _tgt_s = _rc_tgt_t[:, _tf_i, _sig_i]
                        _pred_s = _rc_pred[:, _tf_i, _sig_i]
                        _valid = (_tgt_s > -900.0) & torch.isfinite(_tgt_s)
                        if not _valid.any():
                            continue
                        _delta = _rc_deltas[_sig_i] if _sig_i < len(_rc_deltas) else 0.1
                        loss_regime_raw = loss_regime_raw + F.huber_loss(
                            _pred_s[_valid], _tgt_s[_valid], delta=_delta, reduction="mean"
                        )
                        _rc_terms += 1
                if _rc_terms > 0:
                    loss_regime_raw = loss_regime_raw / float(_rc_terms)

            # ── v28: SessionBreakoutHead loss ─────────────────────────────
            loss_session = torch.tensor(0.0, device=device)
            _w_sess = float(conf.get("w_session_head", 0.5))
            if _w_sess > 0:
                try:
                    _focal_g = float(conf.get("strategy_hit_focal_gamma", 2.0))
                    # pos_weight: ~22% positive -> weight ≈ 3.5 to balance
                    _pw_sess = float(conf.get("session_side_pos_weight", 3.5))
                    _pw_t = torch.tensor([_pw_sess], device=device)
                    ls_l, ls_s = session_head(emb_masked, extra_context=strategy_context)
                    # side loss only (TP_first vs SL_first) — hit loss removed:
                    # hit_rate≈90% on random bars -> unbalanced label, no signal.
                    # side_rate≈22% on random bars vs 32% on real entries -> useful signal.
                    for _sk, _sl, _ck in [
                        ("strat_side_sl", ls_l.squeeze(-1), "strat_cnf_sl"),
                        ("strat_side_ss", ls_s.squeeze(-1), "strat_cnf_ss"),
                    ]:
                        _sr = batch.get(_sk)
                        if _sr is None:
                            continue
                        _sy = (
                            _sr.to(device)
                            if isinstance(_sr, torch.Tensor)
                            else torch.tensor(_sr, device=device)
                        ).float()
                        _vs = _sy >= 0
                        # ⚠️ strat_cnf_* is a Python float: the collate makes it
                        # a CPU tensor. Without .to(device), indexing with a
                        # CUDA mask raised an error that the surrounding try/
                        # except swallowed, zeroing the loss: the head got
                        # no gradient and the AUC stayed at 0.500.
                        _cw = _to_dev_w(batch.get(_ck), device)
                        if _vs.any():
                            loss_session = loss_session + _masked_focal_bce_with_logits(
                                _sl, _sy, _vs, gamma=_focal_g, pos_weight=_pw_t, sample_w=_cw
                            )
                    # ── ANTI-PHASE between long and short ─────────────────
                    # Over seven runs sess_L and sess_S move against each other:
                    # when one goes up the other goes down. A head judging the
                    # QUALITY of the setup shouldn't do that — it looks like it's
                    # catching the directional bias of the period instead of
                    # how good the signal is.
                    # This penalizes the correlation between the two scores on the
                    # SAME sample. The legs aren't merged (that's the way to make
                    # the embeddings collapse, already seen): it only stops one
                    # from being the mirror image of the other.
                    # The term is squared, so it punishes both positive and
                    # negative correlation.
                    _w_af = float(conf.get("w_antiphase", 0.0))
                    # ALWAYS measure, CORRECT only if the weight is > 0.
                    # ⚠️ The computation used to be nested inside `if _w_af > 0`,
                    # so with w_antiphase=0 (the default) it never ran and the
                    # metrics stayed nan — the opposite of the intent: at zero
                    # weight the measure is PURE, and that's exactly when you need
                    # it, to know if the problem exists before correcting it.
                    _a = torch.sigmoid(ls_l.squeeze(-1))
                    _b = torch.sigmoid(ls_s.squeeze(-1))
                    if _a.numel() > 8:
                        _ac = _a - _a.mean()
                        _bc = _b - _b.mean()
                        _den = (_ac.norm() * _bc.norm()).clamp_min(1e-6)
                        _corr = (_ac * _bc).sum() / _den
                        # accumulated over ALL batches: a single batch is too
                        # noisy a measure to decide whether to turn on the
                        # penalty. The mean is computed at the end of the epoch.
                        _af_acc["session_sum"] += float(_corr.detach())
                        _af_acc["session_n"] += 1
                        if _w_af > 0:
                            loss_session = loss_session + _w_af * _corr.pow(2)
                    loss_session = loss_session * _w_sess
                except Exception as _e:
                    logger.warning("SessionHead loss FAILED (head not trained): %s", _e)

            # ── v28: PDHLBreakoutHead loss ────────────────────────────────────
            loss_pdhl = torch.tensor(0.0, device=device)
            _w_pdhl = float(conf.get("w_pdhl_head", 0.5))
            if _w_pdhl > 0:
                try:
                    _pw_pdhl = float(conf.get("pdhl_side_pos_weight", 4.0))
                    _pw_tp = torch.tensor([_pw_pdhl], device=device)
                    ls_l, ls_s = pdhl_head(emb_masked, extra_context=strategy_context)
                    # side loss only — hit_rate≈98% on random bars -> useless.
                    # pdhl side_rate≈23% -> pos_weight=4.0 to balance.
                    for _sk, _sl, _ck in [
                        ("strat_side_pl", ls_l.squeeze(-1), "strat_cnf_pl"),
                        ("strat_side_ps", ls_s.squeeze(-1), "strat_cnf_ps"),
                    ]:
                        _sr = batch.get(_sk)
                        if _sr is None:
                            continue
                        _sy = (
                            _sr.to(device)
                            if isinstance(_sr, torch.Tensor)
                            else torch.tensor(_sr, device=device)
                        ).float()
                        _vs = _sy >= 0
                        _cw = _to_dev_w(batch.get(_ck), device)
                        if _vs.any():
                            loss_pdhl = loss_pdhl + _masked_focal_bce_with_logits(
                                _sl, _sy, _vs, gamma=_focal_g, pos_weight=_pw_tp, sample_w=_cw
                            )
                    _w_af = float(conf.get("w_antiphase", 0.0))
                    _a = torch.sigmoid(ls_l.squeeze(-1))
                    _b = torch.sigmoid(ls_s.squeeze(-1))
                    if _a.numel() > 8:
                        _ac = _a - _a.mean()
                        _bc = _b - _b.mean()
                        _den = (_ac.norm() * _bc.norm()).clamp_min(1e-6)
                        _corr = (_ac * _bc).sum() / _den
                        _af_acc["pdhl_sum"] += float(_corr.detach())
                        _af_acc["pdhl_n"] += 1
                        if _w_af > 0:
                            loss_pdhl = loss_pdhl + _w_af * _corr.pow(2)
                    loss_pdhl = loss_pdhl * _w_pdhl
                except Exception as _e:
                    logger.warning("PDHLHead loss FAILED (head not trained): %s", _e)

            # ── v28: RegimeQualityHead loss ───────────────────────────────────
            loss_regime_quality = torch.tensor(0.0, device=device)
            _w_qual = float(conf.get("w_regime_quality_head", 0.15))
            if _w_qual > 0:
                try:
                    _rq_raw = batch.get("regime_quality")
                    if _rq_raw is not None:
                        _rq_logit = regime_quality_head(
                            emb_masked, extra_context=strategy_context
                        ).squeeze(-1)
                        _rq_lbl = (
                            _rq_raw.to(device)
                            if isinstance(_rq_raw, torch.Tensor)
                            else torch.tensor(_rq_raw, device=device)
                        ).float()
                        _vq = _rq_lbl >= 0
                        if _vq.any():
                            loss_regime_quality = (
                                _masked_bce_with_logits(_rq_logit, _rq_lbl, _vq) * _w_qual
                            )
                except Exception as _e:
                    logger.warning("RegimeQualityHead loss FAILED (head not trained): %s", _e)

            kl_loss = torch.tensor(0.0, device=device)
            if training_mode == "retrain" and anchor_model is not None:
                try:
                    with torch.no_grad():
                        anchor_emb = anchor_model["encoder"](masked_seq, asset_ids=_asset_ids_t)
                        anchor_proj = F.normalize(
                            anchor_model["proj_head"](anchor_emb).float(), dim=-1
                        )
                    curr_proj = F.normalize(proj_head(emb_masked).float(), dim=-1)
                    # Cosine-based KL: penalise directional drift of embeddings.
                    # More meaningful than regime_head KL (3-class head no longer trained).
                    _sim = (curr_proj * anchor_proj.detach()).sum(dim=-1)  # [B]
                    kl_loss = (1.0 - _sim).mean()
                except Exception as e:
                    logger.warning(f"[KL] {e}")
                    kl_loss = torch.tensor(0.0, device=device)

            regimes_list = []
            try:
                r_np = regime.detach().cpu().numpy()
                ic_np = is_close.detach().cpu().numpy()
                for i in range(r_np.shape[0]):
                    vals = r_np[i]
                    mv = (ic_np[i].astype(bool)) & (vals >= 0)
                    regimes_list.append(int(np.round(np.mean(vals[mv]))) if np.any(mv) else None)
            except Exception:
                regimes_list = [None] * seq.size(0)

            do_simclr = ((batch_idx % simclr_every) == 0) and not debug_mode
            contrast_loss = torch.tensor(0.0, device=device)
            if is_close.sum().item() >= 2:
                try:
                    contrast_loss, _s, _z = cp.compute_total_contrastive_loss_partial(
                        seq_batch=seq,
                        asset_ids=asset_ids,
                        timestamps=timestamps_list,
                        loss_weights=conf.get(
                            "loss_weights",
                            {"cross_pos": 1.0, "cross_neg": 1.0, "simclr_batch": 1.0},
                        ),
                        simclr_aug_config=conf.get("augment", {}),
                        do_simclr=do_simclr,
                        k_neg=int(conf.get("contrast_k_neg", 64)),
                        regimes=regimes_list,
                    )
                except Exception as e:
                    logger.warning("Contrastive: %s", e)
                    contrast_loss = torch.tensor(0.0, device=device)

            if use_ema_balancing:

                def _ema(key, t, _alpha=_cur_ema_alpha):
                    v = float(t.detach().cpu().item())
                    loss_ema[key] = max(
                        0.01,
                        float(_alpha * loss_ema[key] + (1 - _alpha) * v),
                    )
                    # Cap amplification: never divide by less than 20% of current raw value.
                    # Prevents 10×+ amplification when a loss collapses faster than EMA adapts.
                    _denom = max(loss_ema[key], v * 0.2, 0.01)
                    return t / torch.tensor(_denom, device=t.device, dtype=torch.float32)

                # EMA bypass for the trend in the first N epochs (default 3).
                # Problem: in the very first steps loss_ema["trend"]≈1.0 (init);
                # when the loss collapses the EMA follows slowly -> the divisor stays
                # high -> amplified gradient on an already collapsed model.
                # With the bypass the trend contributes raw × trend_loss_weight,
                # which is exactly what the parameter is meant for.
                _bypass_trend_ema = epoch <= int(conf.get("trend_ema_bypass_epochs", 3))
                loss_trend = loss_trend_raw if _bypass_trend_ema else _ema("trend", loss_trend_raw)
                loss_regime = _ema("regime", loss_regime_raw)
                loss_contrast = _ema("contrast", contrast_loss)
            else:
                loss_trend = loss_trend_raw
                loss_regime = loss_regime_raw
                loss_contrast = contrast_loss

            try:
                var_loss_val = variance_loss(z_proj_local) * variance_loss_weight
            except Exception:
                var_loss_val = torch.tensor(0.0, device=device)

            # ── Market Structure loss — CrossEntropy 3-class per TF ─────────────────
            # batch["mktstructure"] = [B, n_tfs] int64  (0=RANGING,1=LONG_BRK,2=SHORT_BRK,-1=skip)  v25
            # We sum loss over all TFs that have valid labels, normalise by n_terms.
            loss_mktstr_raw = torch.tensor(0.0, device=device)
            _w_mktstr = float(conf.get("w_mktstructure", 0.15))
            if _w_mktstr > 0:
                try:
                    _mktstr_tgt_all = batch.get("mktstructure", None)
                    if _mktstr_tgt_all is not None:
                        _mktstr_tgt_all = (
                            _mktstr_tgt_all.to(device).long()
                            if isinstance(_mktstr_tgt_all, torch.Tensor)
                            else torch.tensor(_mktstr_tgt_all, device=device, dtype=torch.long)
                        )
                        # _mktstr_tgt_all: [B, n_tfs]
                        _mktstr_logits_all = mktstructure_head(emb_masked)  # [B, n_tfs, 3]
                        # Per-TF class weights — upweight RANGING+BREAKOUT on longer TFs
                        # where TRENDING dominates (1H 68%, 4H 77%)
                        _mktstr_cw_by_tf = conf.get(
                            "mktstructure_class_weights_by_tf",
                            {
                                "15min": [1.0, 1.0, 2.0],
                                "30min": [1.0, 1.0, 2.0],
                                "1h": [2.0, 0.6, 3.0],
                                "4h": [3.5, 0.4, 3.5],
                                "1d": [3.5, 0.4, 3.5],
                            },
                        )
                        # v25: classes 0=RANGING, 1=LONG_BRK, 2=SHORT_BRK
                        # LONG/SHORT_BRK are rare -> higher weight to balance
                        _mktstr_default_cw = conf.get("mktstructure_class_weights", [1.0, 6.0, 6.0])
                        _mktstr_terms = 0
                        for _tf_i in range(_mktstr_tgt_all.size(1)):
                            _tgt_i = _mktstr_tgt_all[:, _tf_i]  # [B]
                            _lgits_i = _mktstr_logits_all[:, _tf_i, :]  # [B, 3]
                            if not (_tgt_i >= 0).any():
                                continue
                            _tf_name = TF_LIST[_tf_i] if _tf_i < len(TF_LIST) else ""
                            _cw = torch.tensor(
                                _mktstr_cw_by_tf.get(_tf_name.lower(), _mktstr_default_cw),
                                device=device,
                                dtype=_lgits_i.dtype,
                            )
                            loss_mktstr_raw = loss_mktstr_raw + F.cross_entropy(
                                _lgits_i.float(),
                                _tgt_i,
                                weight=_cw,
                                ignore_index=-1,
                                reduction="mean",
                            )
                            _mktstr_terms += 1
                        if _mktstr_terms > 0:
                            loss_mktstr_raw = loss_mktstr_raw / float(_mktstr_terms)
                except Exception as _ms:
                    logger.warning("MktStructure loss FAILED (head not trained): %s", _ms)

            # loss_trend_raw already computed above — no duplicate block needed.
            # ── Directional Contrastive Loss ──────────────────────────
            # pushes emb_bull away from emb_bear using tmom as the label

            # ── Directional Contrastive Loss ──────────────────────────
            # pushes emb_bull away from emb_bear using tmom as the label
            loss_dir_contrast = torch.tensor(0.0, device=device)
            if lambda_dir_contrast > 0 and _tmom_tgt_train is not None:
                try:
                    # uses the TARGET tmom (not the predicted one) to split bull/bear:
                    # the prediction collapses to 0 in the first epochs -> no signal,
                    # the target is the ground truth -> stable split from epoch 1
                    _tmom_tgt_for_dc = (
                        _tmom_tgt_train.to(device).float()
                        if isinstance(_tmom_tgt_train, torch.Tensor)
                        else torch.tensor(_tmom_tgt_train, device=device, dtype=torch.float32)
                    )
                    # replace -999 (skip) with 0 so they aren't classified
                    _tmom_tgt_for_dc = torch.where(
                        _tmom_tgt_for_dc < -900,
                        torch.zeros_like(_tmom_tgt_for_dc),
                        _tmom_tgt_for_dc,
                    )
                    loss_dir_contrast = _dir_contrast_fn(emb_masked, _tmom_tgt_for_dc)
                except Exception:
                    loss_dir_contrast = torch.tensor(0.0, device=device)

            task_loss = (
                trend_loss_weight * loss_trend
                + w_regime * loss_regime
                + _w_mktstr * loss_mktstr_raw
                + loss_session  # v28
                + loss_pdhl  # v28
                + loss_regime_quality  # v28
                + lambda_dir_contrast * loss_dir_contrast
            )
            # Variance regularization: penalise dimensional collapse
            # intra_var≈0.007 → embeddings collapsed into low-dim subspace
            _var_reg_w = float(conf.get("emb_var_reg_weight", 0.1))
            _emb_var_reg = torch.tensor(0.0, device=device)
            if _var_reg_w > 0:
                _emb_var_per_dim = emb_masked.float().var(dim=0)  # [D]
                _emb_var_reg = (
                    _var_reg_w * F.relu(1.0 - _emb_var_per_dim).mean()
                )  # penalises var < 1
            total_loss = (
                task_loss
                + lambda_contrast * loss_contrast
                + var_loss_val
                + kl_anchor_weight * kl_loss
                + _emb_var_reg
            )
            if dummy_reg > 0:
                total_loss = total_loss + dummy_reg * emb_masked.pow(2).mean()

            if not torch.isfinite(total_loss):
                optimizer.zero_grad()
                pbar.update(1)
                continue

            if use_amp and amp_scaler is not None:
                amp_scaler.scale(total_loss / max(1, accum_steps)).backward()
                if (batch_idx + 1) % accum_steps == 0:
                    amp_scaler.unscale_(optimizer)
                    # Check for FP16 overflow in trend head before stepping
                    _tgn = compute_trend_head_grad_norm(trend_head)
                    if not math.isfinite(_tgn):
                        logger.warning(
                            "[AMP] trend head grad_norm=nan ep=%d step=%d — skipping step",
                            epoch,
                            batch_idx,
                        )
                        amp_scaler.update()
                        optimizer.zero_grad()
                        pbar.update(1)
                        continue
                    # Separate clip for trend head (tighter: prevents FP16 overflow cascade)
                    torch.nn.utils.clip_grad_norm_(list(trend_head.parameters()), max_norm=0.5)
                    if gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model_params, gradient_clip)
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                    _grad_norm_accum.append(_tgn)
                    optimizer.zero_grad()
            else:
                (total_loss / max(1, accum_steps)).backward()
                if (batch_idx + 1) % accum_steps == 0:
                    if gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model_params, gradient_clip)
                    _grad_norm_accum.append(compute_trend_head_grad_norm(trend_head))
                    optimizer.step()
                    optimizer.zero_grad()

            running["task_total"] += float((loss_trend_raw + loss_regime_raw).detach().cpu().item())
            running["contrast_total"] += float(contrast_loss.detach().cpu().item())
            running["kl_total"] += float(kl_loss.detach().cpu().item())
            running["dir_contrast_total"] += float(loss_dir_contrast.detach().cpu().item())
            pbar.update(1)

        pbar.close()
        logger.info(
            "[TRAIN STATS] ep=%d batches=%d task=%.4f contrast=%.4f kl=%.4f " "dir_contrast=%.4f",
            epoch,
            batch_idx + 1,
            running["task_total"],
            running["contrast_total"],
            running["kl_total"],
            running["dir_contrast_total"],
        )
        if batch_idx < 0:
            logger.error("[TRAIN] ep=%d — DataLoader returned 0 batches!", epoch)
        if (batch_idx + 1) % accum_steps != 0:
            if use_amp and amp_scaler is not None:
                amp_scaler.unscale_(optimizer)
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model_params, gradient_clip)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                optimizer.zero_grad()
            else:
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model_params, gradient_clip)
                optimizer.step()
                optimizer.zero_grad()

        epoch_trend_monitor: dict[str, Any] = {}
        for key, vals in epoch_trend_monitor_accum.items():
            mean_val = float(np.mean(vals)) if vals else float("nan")
            if math.isnan(mean_val):
                epoch_trend_monitor[key] = mean_val
            else:
                if key not in trend_monitor_ema:
                    trend_monitor_ema[key] = mean_val
                else:
                    trend_monitor_ema[key] = (
                        monitor_ema_alpha * trend_monitor_ema[key]
                        + (1 - monitor_ema_alpha) * mean_val
                    )
                epoch_trend_monitor[key] = trend_monitor_ema[key]
        _grad_norm_mean = float(np.mean(_grad_norm_accum)) if _grad_norm_accum else 0.0
        epoch_trend_monitor["grad_norm_trend_head"] = _grad_norm_mean

        monitor_alerts, monitor_report = check_trend_monitor_alerts(
            current=epoch_trend_monitor,
            baseline=trend_monitor_baseline,
            tf_list=TF_LIST,
            monitor_tfs=monitor_tfs,
        )

        if monitor_alerts:
            logger.warning("[TREND MONITOR] ⚠️  ep=%d alerts=%s", epoch, monitor_alerts)
        else:
            logger.info(
                "[TMOM MONITOR] ✅ ep=%d | %s",
                epoch,
                " | ".join(
                    f"{tf}: mean={epoch_trend_monitor.get(f'tmom_mean_{tf}', float('nan')):+.3f} "
                    f"std={epoch_trend_monitor.get(f'tmom_std_{tf}', float('nan')):.3f} "
                    f"bull={epoch_trend_monitor.get(f'pct_bullish_{tf}', 0):.0f}% "
                    f"bear={epoch_trend_monitor.get(f'pct_bearish_{tf}', 0):.0f}% "
                    f"neut={epoch_trend_monitor.get(f'pct_neutral_{tf}', 0):.0f}%"
                    for tf in monitor_tfs
                ),
            )
        logger.info(
            "[TMOM MONITOR] grad_norm=%.4f", epoch_trend_monitor.get("grad_norm_trend_head", 0.0)
        )

        epoch_regime_monitor: dict[str, Any] = {}  # v28: 3-class metrics removed

        # cluster_var every N epochs — heavy forward pass, not needed every time

        current_regime_thresholds_for_monitor: dict[str, tuple] = {}
        for asset_th in per_asset_regime_thresholds.values():
            for tf, th in asset_th.items():
                if tf not in current_regime_thresholds_for_monitor:
                    current_regime_thresholds_for_monitor[tf] = th
            break

        # ── per-epoch regime_cont train summary (mean across batches) ──────────
        # epoch_regime_monitor has tmom_align stats; val metrics have per-signal corr
        # This gives a lightweight per-epoch view without waiting for val loop
        if monitor_tfs and epoch_regime_monitor:
            _rc_align_parts = []
            for _tf in monitor_tfs:
                _ah = epoch_regime_monitor.get(f"tmom_align_high_{_tf}", float("nan"))
                _al = epoch_regime_monitor.get(f"tmom_align_low_{_tf}", float("nan"))
                if not (math.isnan(_ah) and math.isnan(_al)):
                    _rc_align_parts.append(f"{_tf}: tmom|H|={_ah:.3f} tmom|L|={_al:.3f}")
            if _rc_align_parts:
                logger.info(
                    "[REGIME CONT] ep=%d alignment | %s", epoch, " | ".join(_rc_align_parts)
                )

        retrain_safe = True
        retrain_safe_reasons = []
        # Pre-val retrain_safe: tmom monitor alerts only (TMOM_COLLAPSED, TMOM_BIAS)
        critical = [
            a
            for a in monitor_alerts
            if any(k in a for k in ("TMOM_COLLAPSED", "TMOM_BIAS", "ENTROPY_SPIKE", "ENTROPY_HIGH"))
        ]
        if critical:
            retrain_safe = False
            retrain_safe_reasons.extend(critical)
        logger.info(
            "[RETRAIN SAFE] ep=%d -> %s%s",
            epoch,
            "✅ YES" if retrain_safe else "⚠️  NO",
            f" ({', '.join(retrain_safe_reasons)})" if retrain_safe_reasons else "",
        )

        drift_detected = False
        drift_report = {}
        current_emb_stats = {}
        if training_mode == "retrain" and baseline_emb_stats is not None:
            current_emb_stats = compute_emb_drift_stats(
                encoder,
                drift_loader,
                device,
                max_batches=drift_max_batches,
                max_points=drift_max_points,
                seed=0,
                asset_map=asset_map,
            )
            drift_detected, drift_report = check_embedding_drift(
                current_emb_stats, baseline_emb_stats, drift_threshold
            )
            if drift_detected:
                logger.warning("[STRUCTURAL DRIFT] ⚠️  ep=%d", epoch)
            else:
                logger.info("[STRUCTURAL DRIFT] ✅ ep=%d", epoch)

        regime_drift_detected = False
        regime_drift_report = {}
        if training_mode == "retrain":
            rolling_stats = compute_emb_drift_stats(
                encoder,
                regime_drift_loader,
                device,
                max_batches=drift_max_batches,
                max_points=drift_max_points,
                seed=0,
                asset_map=asset_map,
            )
            if previous_regime_stats is not None:
                regime_drift_detected, regime_drift_report = check_regime_drift(
                    rolling_stats, previous_regime_stats, regime_drift_threshold
                )
                if regime_drift_detected:
                    logger.warning("[REGIME DRIFT] ⚠  ep=%d (save NOT blocked)", epoch)
                else:
                    logger.info("[REGIME DRIFT] ✅ ep=%d", epoch)
            previous_regime_stats = rolling_stats

        skip_save = live_mode and drift_detected

        _extra_base = {
            "training_mode": training_mode,
            "live_mode": live_mode,
            "version_tag": version_tag,
            "class_name": class_name,
            "schedule": _schedule_key,
            "phase": current_phase_name,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "w_regime": w_regime,
            "kl_anchor_weight": kl_anchor_weight,
            "retrain_save_mode": retrain_save_mode,
            "drift_detected": drift_detected,
            "drift_report": drift_report,
            "emb_drift_stats": current_emb_stats,
            "drift_anchor_start": drift_anchor_start,
            "drift_anchor_end": drift_anchor_end,
            "regime_drift_detected": regime_drift_detected,
            "regime_drift_report": regime_drift_report,
            "trend_monitor": epoch_trend_monitor,
            "trend_monitor_alerts": monitor_alerts,
            "trend_monitor_report": monitor_report,
            "regime_monitor": epoch_regime_monitor,
            "retrain_safe": retrain_safe,
            "retrain_safe_reasons": retrain_safe_reasons,
        }

        do_validation = epoch % validate_every == 0

        if do_validation:
            # Set ALL heads to eval before any validation step — this is critical.
            # Dropout is disabled and BatchNorm uses running stats only in eval mode.
            encoder.eval()
            proj_head.eval()
            trend_head.eval()
            regime_cont_head.eval()
            mktstructure_head.eval()
            session_head.eval()
            pdhl_head.eval()
            regime_quality_head.eval()
            with torch.no_grad():
                try:
                    _diag_b = next(iter(val_loader))
                    _diag_seq = _diag_b["seq"][:16].to(device).float()
                    _diag_aid = _diag_b.get("asset_id")
                    if _diag_aid is not None:
                        _diag_aid = _diag_aid[:16].to(device)
                    _diag_emb = encoder(_diag_seq, asset_ids=_diag_aid)
                    _emb_std = _diag_emb.std().item()
                    logger.info(
                        "[EMB DIAG] ep=%d | mean=%.4f std=%.4f " "norm_mean=%.4f norm_std=%.4f",
                        epoch,
                        _diag_emb.mean().item(),
                        _emb_std,
                        _diag_emb.norm(dim=-1).mean().item(),
                        _diag_emb.norm(dim=-1).std().item(),
                    )
                    if _emb_std < 0.01:
                        logger.warning(
                            "[EMB DIAG] ⚠️ EMBEDDING COLLAPSE ep=%d std=%.6f", epoch, _emb_std
                        )
                except Exception as _ediag_e:
                    logger.warning("[EMB DIAG] failed: %s", _ediag_e)

            val_regime_true = []
            val_rc_pred_by_tf = {tf: [] for tf in TF_LIST}
            val_rc_true_by_tf = {tf: [] for tf in TF_LIST}
            val_masks = []
            val_is_close = []
            metrics = {}
            val_mktstr_logits_by_tf = {tf: [] for tf in TF_LIST}
            val_tmom_pred_by_tf = {tf: [] for tf in TF_LIST}
            val_tmom_true_by_tf = {tf: [] for tf in TF_LIST}
            val_mktstr_true_by_tf = {tf: [] for tf in TF_LIST}
            # ── StrategyOutcomeHead: computed by evaluate_strategy_outcome_head ──

            with torch.no_grad():
                for vb_count, vb in enumerate(val_loader):
                    if vb_count >= val_max_batches:
                        break
                    vseq = vb["seq"].to(device, non_blocking=True).float()
                    vmasked = vb.get("masked_seq", vseq).to(device, non_blocking=True).float()
                    vmask = (
                        vb.get("mask", torch.zeros_like(vseq[..., 0]).to(torch.uint8))
                        .to(device, non_blocking=True)
                        .bool()
                    )
                    vregime = vb.get("regime").to(device, non_blocking=True)
                    visc = (
                        vb.get("is_close", torch.zeros((vseq.size(0), n_tfs), dtype=torch.uint8))
                        .to(device, non_blocking=True)
                        .bool()
                    )
                    _vassets = vb.get("asset", None) or ([None] * vseq.size(0))
                    _vaid_t = torch.tensor(
                        [asset_map.get(a, 0) for a in _vassets], dtype=torch.long, device=device
                    )
                    vemb = encoder(vmasked, asset_ids=_vaid_t)
                    # scalar trend momentum collection
                    _vt_scalars = trend_head(vemb)  # [B, n_tfs]
                    _vt_tgt = vb.get("trend_momentum", None)
                    if _vt_tgt is not None:
                        _vt_tgt_t = (
                            _vt_tgt.to(device).float()
                            if isinstance(_vt_tgt, torch.Tensor)
                            else torch.tensor(_vt_tgt, device=device, dtype=torch.float32)
                        )
                        for _tf_i, _tf in enumerate(TF_LIST):
                            _tgt_i = _vt_tgt_t[:, _tf_i]
                            _pred_i = _vt_scalars[:, _tf_i]
                            _valid = (_tgt_i > -900.0) & torch.isfinite(_tgt_i)
                            if _valid.any():
                                val_tmom_pred_by_tf[_tf].append(
                                    _pred_i[_valid].detach().cpu().numpy()
                                )
                                val_tmom_true_by_tf[_tf].append(_tgt_i[_valid].cpu().numpy())
                    val_regime_true.append(vregime.cpu().numpy())
                    # regime_continuous collection
                    _vrc_tgt = vb.get("regime_continuous", None)
                    if _vrc_tgt is not None:
                        _vrc_tgt_t = (
                            _vrc_tgt.to(device).float()
                            if isinstance(_vrc_tgt, torch.Tensor)
                            else torch.tensor(_vrc_tgt, device=device, dtype=torch.float32)
                        )
                        _vrc_pred = regime_cont_head(vemb)  # [B, n_tfs, 3]
                        for _tf_i, _tf in enumerate(TF_LIST):
                            _t = _vrc_tgt_t[:, _tf_i, :]  # [B, 3]
                            _p = _vrc_pred[:, _tf_i, :]  # [B, 3]
                            _valid = (_t > -900.0).all(dim=-1)  # all signals present
                            if _valid.any():
                                val_rc_pred_by_tf[_tf].append(_p[_valid].detach().cpu().numpy())
                                val_rc_true_by_tf[_tf].append(_t[_valid].cpu().numpy())
                    val_masks.append(vmask.cpu().numpy())
                    val_is_close.append(visc.cpu().numpy())
                    # ── StrategyOutcomeHead: collection removed (v27) ────────────
                    # the strategy metrics come from evaluate_strategy_outcome_head()
                    # after the main loop, no duplicated code.

                    # ── mktstructure val collection ─────────────────────────────
                    _vms_all = vb.get("mktstructure", None)
                    if _vms_all is not None:
                        _vms_all_t = (
                            _vms_all.to(device).long()
                            if isinstance(_vms_all, torch.Tensor)
                            else torch.tensor(_vms_all, device=device, dtype=torch.long)
                        )  # [B, n_tfs]
                        _vms_logits_all = mktstructure_head(vemb)  # [B, n_tfs, 3]
                        for _tf_i, _tf in enumerate(TF_LIST):
                            if _tf_i >= _vms_all_t.size(1):
                                break
                            _vms_tgt_i = _vms_all_t[:, _tf_i]  # [B]
                            _vms_lgit_i = _vms_logits_all[:, _tf_i, :]  # [B, 3]
                            _vms_prob_i = torch.softmax(_vms_lgit_i.float(), dim=-1)
                            _vms_valid = _vms_tgt_i >= 0
                            if _vms_valid.any():
                                val_mktstr_logits_by_tf[_tf].append(
                                    _vms_prob_i[_vms_valid].detach().cpu().numpy()
                                )
                                val_mktstr_true_by_tf[_tf].append(
                                    _vms_tgt_i[_vms_valid].cpu().numpy()
                                )

            def _cat(parts):
                return np.concatenate(parts, axis=0) if parts else np.array([])

            mask_all = _cat(val_masks)
            is_close_all = _cat(val_is_close)

            # ── trend_momentum metrics — per TF ─────────────────────────────────
            for _tf in TF_LIST:
                _tp = val_tmom_pred_by_tf.get(_tf, [])
                _tt = val_tmom_true_by_tf.get(_tf, [])
                if not _tp or not _tt:
                    continue
                try:
                    _p = np.concatenate(_tp)  # [N] predicted scalar
                    _t = np.concatenate(_tt)  # [N] true scalar
                    _mae = float(np.mean(np.abs(_p - _t)))
                    _rmse = float(np.sqrt(np.mean((_p - _t) ** 2)))
                    # Direction accuracy: does sign(pred) == sign(true)?
                    _dir_acc = float(np.mean(np.sign(_p) == np.sign(_t)))
                    # Correlation
                    _corr = _safe_corr(_p, _t) if len(_p) > 2 else float("nan")
                    # Bias: is prediction systematically bullish/bearish?
                    _bias = float(np.mean(_p))
                    metrics[f"tmomentum_{_tf}_mae"] = _mae
                    metrics[f"tmomentum_{_tf}_rmse"] = _rmse
                    metrics[f"tmomentum_{_tf}_dir_acc"] = _dir_acc
                    metrics[f"tmomentum_{_tf}_corr"] = _corr
                    metrics[f"tmomentum_{_tf}_bias"] = _bias
                    metrics[f"tmomentum_{_tf}_n"] = len(_p)
                    if _tf in (monitor_tfs or TF_LIST):
                        logger.info(
                            "[TMOM] ep=%d %s | mae=%.4f rmse=%.4f dir_acc=%.3f corr=%.3f bias=%+.3f n=%d",
                            epoch,
                            _tf,
                            _mae,
                            _rmse,
                            _dir_acc,
                            _corr,
                            _bias,
                            len(_p),
                        )
                except Exception as _tme2:
                    logger.warning("[TMOM] metrics failed %s: %s", _tf, _tme2)

            # ── mktstructure metrics — per TF ───────────────────────────────
            _MKTSTR_NAMES = {0: "RANGING", 1: "LONG_BRK", 2: "SHORT_BRK"}  # v25
            for _tf in TF_LIST:
                _ms_probs = val_mktstr_logits_by_tf.get(_tf, [])
                _ms_true = val_mktstr_true_by_tf.get(_tf, [])
                if not _ms_probs or not _ms_true:
                    continue
                try:
                    _ms_p = np.concatenate(_ms_probs)  # [N, 3]
                    _ms_t = np.concatenate(_ms_true)  # [N]
                    _ms_pred = _ms_p.argmax(axis=-1)
                    _ms_acc = float(accuracy_score(_ms_t, _ms_pred))
                    _pca_ms = compute_per_class_accuracy(_ms_t, _ms_pred, n_classes=3)
                    _ms_macro = _pca_ms["macro_acc"]

                    # ── AUC of every class separately (v25 fix) ─────────────
                    # brk_auc_long:  P(LONG_BRK)  vs {RANGING, SHORT_BRK}
                    # brk_auc_short: P(SHORT_BRK) vs {RANGING, LONG_BRK}
                    # brk_auc_any:   max(P(L),P(S)) = 1-P(RANGING) vs {RANGING}
                    #
                    # BUG v25.0: used only _ms_p[:,2]=P(SHORT_BRK) for brk_auc
                    # -> measured SHORT_BRK vs everything, ignoring LONG_BRK
                    # FIX: three separate AUCs + combined AUC
                    def _safe_auc(y_true_bin, y_score):
                        return (
                            float(roc_auc_score(y_true_bin, y_score))
                            if len(np.unique(y_true_bin)) > 1
                            else float("nan")
                        )

                    _auc_long = _safe_auc((_ms_t == 1).astype(int), _ms_p[:, 1])
                    _auc_short = _safe_auc((_ms_t == 2).astype(int), _ms_p[:, 2])
                    # AUC combinato: P(any breakout) = 1-P(RANGING) vs is_ranging
                    _auc_any = _safe_auc((_ms_t != 0).astype(int), 1.0 - _ms_p[:, 0])
                    # backward-compat alias (was P(SHORT_BRK), now = brk_auc_any)
                    _ms_brk_auc = _auc_any

                    # ── Directional confusion (the metric that really matters) ──
                    # confusion_pct[cls] = {pred: frac} for the samples with true=cls
                    # we extract the LONG<->SHORT confusion and save it in the JSON
                    _conf_L = _pca_ms["confusion_pct"].get(1, {})  # true=LONG_BRK
                    _conf_S = _pca_ms["confusion_pct"].get(2, {})  # true=SHORT_BRK
                    # P(predict SHORT | true LONG) — directional confusion
                    _dir_conf_L2S = float(_conf_L.get(2, 0.0))
                    # P(predico LONG  | vero SHORT) — confusione direzionale
                    _dir_conf_S2L = float(_conf_S.get(1, 0.0))
                    # directional accuracy = mean of the two correct accs (no ranging)
                    _acc_L = _pca_ms["per_class_acc"].get(1, float("nan"))
                    _acc_S = _pca_ms["per_class_acc"].get(2, float("nan"))
                    _dir_acc = (
                        float(np.nanmean([_acc_L, _acc_S]))
                        if not (math.isnan(_acc_L) and math.isnan(_acc_S))
                        else float("nan")
                    )

                    # ── Save everything in the metrics dict (JSON) ───────────
                    metrics[f"mktstr_{_tf}_acc"] = _ms_acc
                    metrics[f"mktstr_{_tf}_macro_acc"] = _ms_macro
                    # backward-compat (was P(SHORT_BRK), now P(any breakout))
                    metrics[f"mktstr_{_tf}_brk_auc"] = _ms_brk_auc
                    # new: AUC per class
                    metrics[f"mktstr_{_tf}_auc_long_brk"] = _auc_long
                    metrics[f"mktstr_{_tf}_auc_short_brk"] = _auc_short
                    metrics[f"mktstr_{_tf}_auc_any_brk"] = _auc_any
                    # per-class accuracy (v25 labels)
                    metrics[f"mktstr_{_tf}_acc_ranging"] = _pca_ms["per_class_acc"].get(
                        0, float("nan")
                    )
                    metrics[f"mktstr_{_tf}_acc_long_brk"] = _acc_L
                    metrics[f"mktstr_{_tf}_acc_short_brk"] = _acc_S
                    # backward-compat alias (v24 naming → v25 meaning)
                    metrics[f"mktstr_{_tf}_acc_trending"] = _acc_L  # was TRENDING, now LONG_BRK
                    metrics[f"mktstr_{_tf}_acc_breakout"] = _acc_S  # was BREAKOUT, now SHORT_BRK
                    # directional confusion — the critical metric for the DDQN
                    metrics[f"mktstr_{_tf}_dir_conf_long2short"] = _dir_conf_L2S
                    metrics[f"mktstr_{_tf}_dir_conf_short2long"] = _dir_conf_S2L
                    metrics[f"mktstr_{_tf}_dir_acc"] = _dir_acc
                    metrics[f"mktstr_{_tf}_n"] = len(_ms_t)

                    logger.info(
                        "[MKTSTR] ep=%d %s | acc=%.3f macro=%.3f "
                        "auc(L=%.3f S=%.3f any=%.3f) | "
                        "R=%.1f%%(n=%d) L=%.1f%%(n=%d) S=%.1f%%(n=%d) | "
                        "dir_acc=%.3f conf(L→S=%.2f S→L=%.2f)",
                        epoch,
                        _tf,
                        _ms_acc,
                        _sf(_ms_macro),
                        _sf(_auc_long),
                        _sf(_auc_short),
                        _sf(_auc_any),
                        _pca_ms["per_class_acc"].get(0, float("nan")) * 100,
                        _pca_ms["per_class_n"].get(0, 0),
                        _acc_L * 100 if not math.isnan(_acc_L) else float("nan"),
                        _pca_ms["per_class_n"].get(1, 0),
                        _acc_S * 100 if not math.isnan(_acc_S) else float("nan"),
                        _pca_ms["per_class_n"].get(2, 0),
                        _sf(_dir_acc),
                        _dir_conf_L2S,
                        _dir_conf_S2L,
                    )

                    # confusion detail for LONG and SHORT (already in the log, now
                    # also saved implicitly through dir_conf_L2S/S2L in the metrics)
                    for _cls_idx, _cls_name in [(1, "LONG_BRK"), (2, "SHORT_BRK")]:
                        _brk_conf = _pca_ms["confusion_pct"].get(_cls_idx, {})
                        if _brk_conf:
                            _cs = " ".join(
                                f"→{_MKTSTR_NAMES.get(p,'?')}:{v*100:.0f}%"
                                for p, v in sorted(_brk_conf.items())
                            )
                            logger.info(
                                "[MKTSTR CONF] ep=%d %s true=%s: %s", epoch, _tf, _cls_name, _cs
                            )
                except Exception as _mse:
                    logger.warning("[MKTSTR] metrics failed %s: %s", _tf, _mse)

            for i, tf in enumerate(TF_LIST):
                mtf = (
                    is_close_all[:, i].astype(bool)
                    if is_close_all.size
                    else np.array([], dtype=bool)
                )
                # ── trend_momentum scalar metrics per TF ────────────────────────────
                _tmom_p = val_tmom_pred_by_tf.get(tf, [])
                _tmom_t = val_tmom_true_by_tf.get(tf, [])
                if _tmom_p and _tmom_t:
                    _p = np.concatenate(_tmom_p)
                    _t = np.concatenate(_tmom_t)
                    _mae = float(np.mean(np.abs(_p - _t)))
                    _rmse = float(np.sqrt(np.mean((_p - _t) ** 2)))
                    _dir_acc = float(np.mean(np.sign(_p) == np.sign(_t)))
                    _corr = _safe_corr(_p, _t) if len(_p) > 2 else float("nan")
                    _bias = float(np.mean(_p))
                    _bull = float(np.mean(_p > 0.1)) * 100
                    _bear = float(np.mean(_p < -0.1)) * 100
                    _neut = float(np.mean(np.abs(_p) <= 0.1)) * 100
                    metrics[f"tmomentum_{tf}_mae"] = _mae
                    metrics[f"tmomentum_{tf}_rmse"] = _rmse
                    metrics[f"tmomentum_{tf}_dir_acc"] = _dir_acc
                    metrics[f"tmomentum_{tf}_corr"] = _corr
                    metrics[f"tmomentum_{tf}_bias"] = _bias
                    metrics[f"tmomentum_{tf}_pct_bull"] = _bull
                    metrics[f"tmomentum_{tf}_pct_bear"] = _bear
                    metrics[f"tmomentum_{tf}_pct_neut"] = _neut
                    metrics[f"tmomentum_{tf}_n"] = len(_p)
                    if tf in (monitor_tfs or TF_LIST):
                        logger.info(
                            "[TMOM VAL] ep=%d %s | mae=%.4f rmse=%.4f "
                            "dir_acc=%.3f corr=%+.3f bias=%+.3f | "
                            "bull=%.1f%% neut=%.1f%% bear=%.1f%% n=%d",
                            epoch,
                            tf,
                            _mae,
                            _rmse,
                            _dir_acc,
                            _corr if not math.isnan(_corr) else 0.0,
                            _bias,
                            _bull,
                            _neut,
                            _bear,
                            len(_p),
                        )
                else:
                    for _sfx in (
                        "mae",
                        "rmse",
                        "dir_acc",
                        "corr",
                        "bias",
                        "pct_bull",
                        "pct_bear",
                        "pct_neut",
                    ):
                        metrics[f"tmomentum_{tf}_{_sfx}"] = float("nan")
                    metrics[f"tmomentum_{tf}_n"] = 0
                # ── Regime continuous val metrics ──────────────────────────────────
                _sig_names = ["vol_regime", "atr_norm", "adx", "di_direction"]
                _rcp = val_rc_pred_by_tf.get(tf, [])
                _rct = val_rc_true_by_tf.get(tf, [])
                if _rcp and _rct:
                    _p3 = np.concatenate(_rcp)  # [N, 4]  (v25: +di_direction)
                    _t3 = np.concatenate(_rct)  # [N, 4]
                    _sig_log = []
                    for _si, _sn in enumerate(_sig_names):
                        _p_s = _p3[:, _si]
                        _t_s = _t3[:, _si]
                        _valid_s = (_t_s > -900.0) & np.isfinite(_t_s) & np.isfinite(_p_s)
                        if not _valid_s.any():
                            metrics[f"regime_{tf}_{_sn}_mae"] = float("nan")
                            metrics[f"regime_{tf}_{_sn}_corr"] = float("nan")
                            continue
                        _pv = _p_s[_valid_s]
                        _tv = _t_s[_valid_s]
                        _mae = float(np.mean(np.abs(_pv - _tv)))
                        _corr = _safe_corr(_pv, _tv) if len(_pv) > 2 else float("nan")
                        _bias = float(np.mean(_pv - _tv))
                        metrics[f"regime_{tf}_{_sn}_mae"] = _mae
                        metrics[f"regime_{tf}_{_sn}_corr"] = _corr
                        metrics[f"regime_{tf}_{_sn}_bias"] = _bias
                        metrics[f"regime_{tf}_{_sn}_n"] = int(_valid_s.sum())
                        _sig_log.append(
                            f"{_sn}: mae={_mae:.3f} corr={_corr:+.3f} bias={_bias:+.3f}"
                        )
                    # composite: mean MAE across signals (lower = better)
                    _all_mae = [
                        metrics.get(f"regime_{tf}_{_sn}_mae", float("nan")) for _sn in _sig_names
                    ]
                    _all_mae = [v for v in _all_mae if not math.isnan(v)]
                    metrics[f"regime_{tf}_mean_mae"] = (
                        float(np.mean(_all_mae)) if _all_mae else float("nan")
                    )
                    if tf in (monitor_tfs or TF_LIST) and _sig_log:
                        logger.info(
                            "[REGIME CONT] ep=%d %s | %s | n=%d",
                            epoch,
                            tf,
                            " | ".join(_sig_log),
                            metrics.get(f"regime_{tf}_{_sig_names[0]}_n", 0),
                        )
                else:
                    for _sn in _sig_names:
                        for _sfx in ("mae", "corr", "bias"):
                            metrics[f"regime_{tf}_{_sn}_{_sfx}"] = float("nan")
                    metrics[f"regime_{tf}_mean_mae"] = float("nan")

            # ── regime_cont MAE retrain_safe check (inside val block) ─────────
            _rc_mae_thresh_val = float(conf.get("regime_cont_mae_threshold", 0.8))
            for _tf_val in monitor_tfs:
                _r_mae_val = metrics.get(f"regime_{_tf_val}_mean_mae", float("nan"))
                if not math.isnan(_r_mae_val) and _r_mae_val > _rc_mae_thresh_val:
                    retrain_safe = False
                    retrain_safe_reasons.append(
                        f"regime_{_tf_val}_mean_mae={_r_mae_val:.3f}>{_rc_mae_thresh_val:.2f}"
                    )

            # ── di_direction dedicated monitor (v25) ──────────────────────────
            # Separate from generic [REGIME CONT] so it's immediately visible.
            # Shows corr per monitor TF + mean; warns if data is missing.
            _di_corrs_mon = {}
            for _tf_di in monitor_tfs or TF_LIST:
                _c_di = metrics.get(f"regime_{_tf_di}_di_direction_corr", float("nan"))
                _di_corrs_mon[_tf_di] = _c_di
            _di_valid_corrs = [c for c in _di_corrs_mon.values() if not math.isnan(c)]
            _di_mean_corr = float(np.mean(_di_valid_corrs)) if _di_valid_corrs else float("nan")
            # Store composite metric in history
            metrics["di_direction_mean_corr"] = _di_mean_corr
            if _di_valid_corrs:
                _di_parts = " | ".join(
                    f"{tf}={_di_corrs_mon[tf]:+.3f}"
                    for tf in (monitor_tfs or TF_LIST)
                    if not math.isnan(_di_corrs_mon.get(tf, float("nan")))
                )
                logger.info(
                    "[DI DIR] ep=%d corr | %s | mean=%+.3f", epoch, _di_parts, _di_mean_corr
                )
            else:
                logger.warning(
                    "[DI DIR] ep=%d ⚠️  NO di_direction data — "
                    "all NaN (DI_plus14/DI_minus14 missing from parquet). "
                    "Re-download with updated downloader and restart.",
                    epoch,
                )

            # retrain_safe: di_direction corr check (only when data is available)
            # Threshold: after epoch 5, corr should be > 0 (better than random).
            # After epoch 10, corr should be >= 0.2 (meaningful learning).
            _di_corr_thresh_ep5 = float(conf.get("di_direction_corr_min_ep5", 0.0))
            _di_corr_thresh_ep10 = float(conf.get("di_direction_corr_min_ep10", 0.2))
            if _di_valid_corrs:  # only check if data is present
                if epoch >= 10 and _di_mean_corr < _di_corr_thresh_ep10:
                    retrain_safe = False
                    retrain_safe_reasons.append(
                        f"di_direction_mean_corr={_di_mean_corr:.3f}<{_di_corr_thresh_ep10:.2f}@ep{epoch}"
                    )
                elif epoch >= 5 and _di_mean_corr < _di_corr_thresh_ep5:
                    retrain_safe = False
                    retrain_safe_reasons.append(
                        f"di_direction_mean_corr={_di_mean_corr:.3f}<{_di_corr_thresh_ep5:.2f}@ep{epoch}"
                    )
            # ────────────────────────────────────────────────────────────────────

            emb_stats = compute_embedding_metrics(
                encoder,
                proj_head,
                val_loader,
                device,
                max_batches=val_max_batches,
                max_points=emb_max_points,
                seed=emb_seed,
            )

            # ── v28: evaluate the strategy heads ─────────────────────────────
            try:
                _v28_metrics = evaluate_v28_heads(
                    session_head,
                    pdhl_head,
                    regime_quality_head,
                    encoder,
                    trend_head,
                    mktstructure_head,
                    val_loader,
                    device,
                    TF_LIST,
                    max_batches=val_max_batches,
                    asset_map=asset_map,
                )
                metrics.update(_v28_metrics)
                logger.info(
                    "[V28] ep=%d | sess_hit_auc=%.3f sess_side_auc=%.3f | "
                    "pdhl_hit_auc=%.3f pdhl_side_auc=%.3f | qual_auc=%.3f",
                    epoch,
                    _sf(metrics.get("v28_session_hit_auc", float("nan"))),
                    _sf(metrics.get("v28_session_side_auc", float("nan"))),
                    _sf(metrics.get("v28_pdhl_hit_auc", float("nan"))),
                    _sf(metrics.get("v28_pdhl_side_auc", float("nan"))),
                    _sf(metrics.get("v28_regime_quality_auc", float("nan"))),
                )
                logger.info(
                    "[V28 DETAIL] ep=%d | "
                    "sess_L hit=%.3f side=%.3f | sess_S hit=%.3f side=%.3f | "
                    "pdhl_L hit=%.3f side=%.3f | pdhl_S hit=%.3f side=%.3f",
                    epoch,
                    _sf(metrics.get("v28_session_long_hit_auc", float("nan"))),
                    _sf(metrics.get("v28_session_long_side_auc", float("nan"))),
                    _sf(metrics.get("v28_session_short_hit_auc", float("nan"))),
                    _sf(metrics.get("v28_session_short_side_auc", float("nan"))),
                    _sf(metrics.get("v28_pdhl_long_hit_auc", float("nan"))),
                    _sf(metrics.get("v28_pdhl_long_side_auc", float("nan"))),
                    _sf(metrics.get("v28_pdhl_short_hit_auc", float("nan"))),
                    _sf(metrics.get("v28_pdhl_short_side_auc", float("nan"))),
                )
            except Exception as _v28e:
                logger.warning("[V28] evaluate failed: %s", _v28e)
                for _n in ("session_long", "session_short", "pdhl_long", "pdhl_short"):
                    for _k in ("hit_auc", "hit_acc", "hit_rate", "side_auc", "side_acc", "tp_rate"):
                        metrics.setdefault(f"v28_{_n}_{_k}", float("nan"))
                for _k in (
                    "session_hit_auc",
                    "session_side_auc",
                    "pdhl_hit_auc",
                    "pdhl_side_auc",
                    "regime_quality_auc",
                    "regime_quality_acc",
                ):
                    metrics.setdefault(f"v28_{_k}", float("nan"))

            _trend_score_tfs = [tf for tf in TF_LIST if tf not in trend_loss_exclude_tfs]

            # ── Trend: (dir_acc - 0.5) × 2  →  [0, 1], random baseline = 0 ─────
            _tmom_accs = [
                metrics.get(f"tmomentum_{tf}_dir_acc", float("nan")) for tf in _trend_score_tfs
            ]
            _tmom_valid = [v for v in _tmom_accs if not math.isnan(v)]
            trend_acc = float(np.mean(_tmom_valid)) if _tmom_valid else float("nan")
            _trend_score = 0.0 if math.isnan(trend_acc) else (trend_acc - 0.5) * 2.0

            # ── Regime: mean correlation of continuous head ──────────────────────
            # corr per signal (vol_regime, atr_norm, adx, di_direction) over monitor_tfs.
            # corr = 0 → random, corr = 1 → perfect. Direct use in [0, 1].
            _rc_corrs = []
            for _tf in monitor_tfs or TF_LIST:
                for _sig in ("vol_regime", "atr_norm", "adx", "di_direction"):
                    _c = metrics.get(f"regime_{_tf}_{_sig}_corr", float("nan"))
                    if not math.isnan(_c):
                        _rc_corrs.append(max(0.0, _c))  # clip negative (random noise)
            regime_acc = float(np.mean(_rc_corrs)) if _rc_corrs else float("nan")
            _regime_score = 0.0 if math.isnan(regime_acc) else regime_acc

            # Score: bounded [0,1] components for numerical stability.
            # StrategyOutcomeHead score: mean of hit_auc and side_auc centered on 0.5.
            _strat_hit_auc = _sf(
                metrics.get("strategy_outcome_hit_auc", float("nan")), float("nan")
            )
            _strat_side_auc = _sf(
                metrics.get("strategy_outcome_side_auc", float("nan")), float("nan")
            )
            _strat_parts = []
            if not math.isnan(_strat_hit_auc):
                _strat_parts.append((_strat_hit_auc - 0.5) * 2.0)
            if not math.isnan(_strat_side_auc):
                _strat_parts.append((_strat_side_auc - 0.5) * 2.0)
            # v28: bonus from the strategy heads
            _w_v28 = float(conf.get("score_w_v28", 0.5))
            _v28_aucs = [
                metrics.get(k, float("nan"))
                for k in (
                    "v28_session_hit_auc",
                    "v28_session_side_auc",
                    "v28_pdhl_hit_auc",
                    "v28_pdhl_side_auc",
                )
            ]
            # ── COVERAGE PER LEG, with the AUC OF THE LEG ─────────────────
            # ⚠️ Fix of a subtle flaw: the GROUP AUC used to be paired
            # (v28_pdhl_hit_auc = equal-weight mean of long and short) with
            # the count of a SINGLE leg. The comment said "each head weighted
            # by its own coverage", but the AUC going in was already the mean:
            # the noise of pdhl_short (n=102-121) was inside that mean BEFORE
            # the coverage acted, so the penalty only worked between groups
            # (session vs pdhl), never between the legs of one group. The AUCs
            # per leg were computed and logged, but not used.
            # Now the AUC of the leg goes in, weighted by the sample of THAT
            # leg: the only way for a thin leg to count less.
            # Before: cov = min(1, mean(n)/150), applied to the whole v28.
            # With n = [228, 212, 147, 112] the mean is 175 -> ZERO penalty,
            # while pdhl_short (112 samples) alone would be worth 0.75.
            # Thin heads entered the score at full weight, their noise hidden
            # by the populated ones: and those are exactly the ones where a
            # high AUC is most easily luck.
            # Now every head is weighted by its OWN coverage, so it counts in
            # proportion to how reliable it is.
            _v28_min_n = float(conf.get("v28_min_n_for_full_score", 150.0))
            _v28_pairs = [
                ("v28_session_long_side_auc", "v28_session_long_side_n"),
                ("v28_session_short_side_auc", "v28_session_short_side_n"),
                ("v28_pdhl_long_side_auc", "v28_pdhl_long_side_n"),
                ("v28_pdhl_short_side_auc", "v28_pdhl_short_side_n"),
            ]
            _num, _den, _covs, _ns = 0.0, 0.0, [], []
            for _ak, _nk in _v28_pairs:
                _a = metrics.get(_ak, float("nan"))
                if math.isnan(_a):
                    continue
                _n = float(metrics.get(_nk, 0) or 0)
                _c = min(1.0, _n / max(1.0, _v28_min_n))
                _num += (_a - 0.5) * 2.0 * _c
                _den += _c
                _covs.append(_c)
                _ns.append(_n)
            _v28_score = (_num / _den) if _den > 0 else 0.0
            # actual mean coverage, for the log and the diagnostics
            _v28_cov = float(np.mean(_covs)) if _covs else 0.0
            _v28_n_mean = float(np.mean(_ns)) if _ns else 0.0
            # anti-phase: always measured, independent of w_antiphase
            # mean over all the batches of the epoch, not the last one
            metrics["antiphase_session"] = (
                _af_acc["session_sum"] / _af_acc["session_n"]
                if _af_acc["session_n"]
                else float("nan")
            )
            metrics["antiphase_pdhl"] = (
                _af_acc["pdhl_sum"] / _af_acc["pdhl_n"] if _af_acc["pdhl_n"] else float("nan")
            )
            metrics["antiphase_n_batch"] = _af_acc["session_n"]
            metrics["v28_score"] = _v28_score
            metrics["v28_n_mean"] = _v28_n_mean
            metrics["v28_cov_factor"] = _v28_cov
            _v28_bonus = _w_v28 * _v28_score

            # Score ribilanciato: 30% trend + 30% regime + 40% v28 (downstream)
            _w_trend_sc = float(conf.get("score_w_trend", 0.30))
            _w_regime_sc = float(conf.get("score_w_regime", 0.30))
            _w_v28_sc = float(conf.get("score_w_v28", 0.40))
            score = (
                _w_trend_sc * _trend_score + _w_regime_sc * _regime_score + _w_v28_sc * _v28_score
            )
            logger.info(
                "[SCORE] ep=%d | trend=%.3f×%.2f regime=%.3f×%.2f "
                "v28=%.3f×%.2f cov=%.2f → score=%.4f",
                epoch,
                _trend_score,
                _w_trend_sc,
                _regime_score,
                _w_regime_sc,
                _v28_score,
                _w_v28_sc,
                _v28_cov,
                score,
            )
            logger.info(
                "[VAL] ep=%d score=%.4f | tmom_dir=%.3f(sc=%.3f) regime_corr=%.3f "
                "di_dir_corr=%s "
                "v28_sess_hit=%.3f v28_pdhl_hit=%.3f v28_qual=%.3f "
                "| antiphase sess=%s pdhl=%s "
                "| live=%s val=%s->%s | safe=%s",
                epoch,
                score,
                0.0 if math.isnan(trend_acc) else trend_acc,
                _trend_score,
                _regime_score,
                f"{_di_mean_corr:+.3f}" if not math.isnan(_di_mean_corr) else "N/A",
                _sf(metrics.get("v28_session_hit_auc", float("nan"))),
                _sf(metrics.get("v28_pdhl_hit_auc", float("nan"))),
                _sf(metrics.get("v28_regime_quality_auc", float("nan"))),
                (
                    f"{metrics['antiphase_session']:+.3f}"
                    if not math.isnan(metrics.get("antiphase_session", float("nan")))
                    else "n/d"
                ),
                (
                    f"{metrics['antiphase_pdhl']:+.3f}"
                    if not math.isnan(metrics.get("antiphase_pdhl", float("nan")))
                    else "n/d"
                ),
                live_mode,
                val_start,
                val_end,
                "✅" if retrain_safe else "⚠️",
            )

            _extra_val = {
                "score": score,
                "metrics": metrics,
                "emb": emb_stats,
                "retrain_safe": retrain_safe,
                "retrain_safe_reasons": list(retrain_safe_reasons),
                **_extra_base,
            }

            if not skip_save:
                if training_mode == "bootstrap":
                    if best_score_bootstrap is None or score > best_score_bootstrap:
                        best_score_bootstrap = score
                        save_best_score(
                            results_dir,
                            score,
                            epoch,
                            "bootstrap",
                            {
                                "tmom_dir_acc": _sf(trend_acc),
                                "regime_corr": _sf(_regime_score),
                                "tmom_dir_centered": _sf(_trend_score),
                            },
                        )
                        save_checkpoint(
                            os.path.join(results_dir, "best_full_checkpoint.pth"),
                            encoder,
                            trend_head,
                            proj_head,
                            optimizer,
                            epoch,
                            regime_cont_head=regime_cont_head,
                            mktstructure_head=mktstructure_head,
                            session_head=session_head,
                            pdhl_head=pdhl_head,
                            regime_quality_head=regime_quality_head,
                            extra=_extra_val,
                        )
                        save_encoder_only(
                            encoder, os.path.join(results_dir, "best_encoder_only.pth")
                        )
                        logger.warning(
                            "[BOOTSTRAP] BEST ep=%d score=%.6f retrain_safe=%s",
                            epoch,
                            score,
                            "✅" if retrain_safe else "⚠️",
                        )
                        # In bootstrap current_emb_stats={} — compute drift stats now
                        if current_emb_stats:
                            drift_bs = current_emb_stats
                        else:
                            try:
                                drift_bs = compute_emb_drift_stats(
                                    encoder,
                                    drift_loader,
                                    device,
                                    max_batches=drift_max_batches,
                                    max_points=drift_max_points,
                                    seed=0,
                                    asset_map=asset_map,
                                )
                            except Exception as _dbe:
                                logger.warning("[DRIFT] compute failed: %s", _dbe)
                                drift_bs = {}
                        save_checkpoint(
                            drift_baseline_path,
                            encoder,
                            trend_head,
                            proj_head,
                            optimizer,
                            epoch,
                            mktstructure_head=mktstructure_head,
                            regime_cont_head=regime_cont_head,
                            session_head=session_head,
                            pdhl_head=pdhl_head,
                            regime_quality_head=regime_quality_head,
                            extra={
                                "emb_drift_stats": drift_bs,
                                "drift_anchor_start": drift_anchor_start,
                                "drift_anchor_end": drift_anchor_end,
                                "class_name": class_name,
                                "bootstrap_best_epoch": epoch,
                                "bootstrap_best_score": score,
                                "trend_monitor": epoch_trend_monitor,
                                "regime_monitor": epoch_regime_monitor,
                                "note": "drift_anchor_baseline_from_best",
                            },
                        )
                        logger.info(
                            "[DRIFT] Baseline saved ep=%d norm=%.4f var=%.4f",
                            epoch,
                            drift_bs.get("emb_norm_mean", float("nan")),
                            drift_bs.get("emb_var_mean", float("nan")),
                        )
                else:
                    is_last = epoch == end_epoch
                    if retrain_save_mode == "best":
                        do_save = best_score_retrain is None or score > best_score_retrain
                    elif retrain_save_mode == "last":
                        do_save = is_last
                    else:
                        do_save = True

                    # retrain_safe gating — don't save if the model degraded
                    if do_save and not retrain_safe:
                        logger.warning(
                            "[RETRAIN] Skipping save ep=%d: retrain_safe=False (%s)",
                            epoch,
                            ", ".join(retrain_safe_reasons[:3]),
                        )
                        do_save = False

                    if do_save:
                        if retrain_save_mode == "best":
                            best_score_retrain = score
                        # AB-COMPARE: keep the key AUCs of the best epoch
                        _ab_best_metrics = {
                            "epoch": epoch,
                            "score": score,
                            "session_side_auc": _sf(
                                metrics.get("v28_session_side_auc", float("nan"))
                            ),
                            "pdhl_side_auc": _sf(metrics.get("v28_pdhl_side_auc", float("nan"))),
                            "session_hit_auc": _sf(
                                metrics.get("v28_session_hit_auc", float("nan"))
                            ),
                            "pdhl_hit_auc": _sf(metrics.get("v28_pdhl_hit_auc", float("nan"))),
                            "regime_quality_auc": _sf(
                                metrics.get("v28_regime_quality_auc", float("nan"))
                            ),
                        }
                        save_best_score(
                            results_dir,
                            score,
                            epoch,
                            "retrain",
                            {
                                "version_tag": version_tag,
                                "tmom_dir_acc": _sf(trend_acc),
                                "regime_corr": _sf(_regime_score),
                                "tmom_dir_centered": _sf(_trend_score),
                            },
                        )
                        if retrain_encoder_path:
                            save_encoder_only(encoder, retrain_encoder_path)
                        if retrain_full_ckpt_path:
                            save_checkpoint(
                                retrain_full_ckpt_path,
                                encoder,
                                trend_head,
                                proj_head,
                                optimizer,
                                epoch,
                                mktstructure_head=mktstructure_head,
                                regime_cont_head=regime_cont_head,
                                session_head=session_head,
                                pdhl_head=pdhl_head,
                                regime_quality_head=regime_quality_head,
                                extra=_extra_val,
                            )
                        logger.warning(
                            "[RETRAIN] Saved (%s) ep=%d score=%.6f", retrain_save_mode, epoch, score
                        )
                        # save the drift baseline for the next WF
                        if current_emb_stats:
                            save_checkpoint(
                                drift_baseline_path,
                                encoder,
                                trend_head,
                                proj_head,
                                optimizer,
                                epoch,
                                mktstructure_head=mktstructure_head,
                                regime_cont_head=regime_cont_head,
                                session_head=session_head,
                                pdhl_head=pdhl_head,
                                regime_quality_head=regime_quality_head,
                                extra={
                                    "emb_drift_stats": current_emb_stats,
                                    "drift_anchor_start": drift_anchor_start,
                                    "drift_anchor_end": drift_anchor_end,
                                    "class_name": class_name,
                                    "note": "drift_anchor_baseline_from_retrain_best",
                                },
                            )
                            logger.info(
                                "[DRIFT] Retrain baseline saved ep=%d norm=%.4f var=%.4f",
                                epoch,
                                current_emb_stats.get("emb_norm_mean", float("nan")),
                                current_emb_stats.get("emb_var_mean", float("nan")),
                            )
                        try:
                            with open(regime_thresholds_prev_path, "w") as f:
                                json.dump(
                                    {
                                        a: {tf: list(v) for tf, v in d.items()}
                                        for a, d in per_asset_regime_thresholds.items()
                                    },
                                    f,
                                    indent=2,
                                )
                            logger.info("[REGIME MONITOR] regime_thresholds_prev.json saved.")
                        except Exception as e:
                            logger.warning("[REGIME MONITOR] prev save: %s", e)

                save_checkpoint(
                    os.path.join(results_dir, "latest_checkpoint.pth"),
                    encoder,
                    trend_head,
                    proj_head,
                    optimizer,
                    epoch,
                    regime_cont_head=regime_cont_head,
                    mktstructure_head=mktstructure_head,
                    session_head=session_head,
                    pdhl_head=pdhl_head,
                    regime_quality_head=regime_quality_head,
                    extra=_extra_val,
                )
            else:
                logger.warning("[STRUCTURAL DRIFT] Checkpoint SKIPPED ep=%d", epoch)

            history["train"].append(running)
            history["val"].append(
                {
                    "metrics": metrics,
                    "emb": emb_stats,
                    "score": score,
                    "structural_drift": drift_report,
                    "regime_drift": regime_drift_report,
                    "trend_monitor": epoch_trend_monitor,
                    "trend_monitor_alerts": monitor_alerts,
                    "regime_monitor": epoch_regime_monitor,
                    "retrain_safe": retrain_safe,
                }
            )
            _save_history()

        else:
            if not skip_save:
                save_checkpoint(
                    os.path.join(results_dir, "latest_checkpoint.pth"),
                    encoder,
                    trend_head,
                    proj_head,
                    optimizer,
                    epoch,
                    regime_cont_head=regime_cont_head,
                    mktstructure_head=mktstructure_head,
                    session_head=session_head,
                    pdhl_head=pdhl_head,
                    regime_quality_head=regime_quality_head,
                    extra={"note": "validation_skipped", **_extra_base},
                )
                if training_mode == "retrain":
                    tagged_enc = os.path.join(results_dir, f"encoder_{version_tag}_ep{epoch}.pth")
                    save_encoder_only(encoder, tagged_enc)
                    if retrain_encoder_path:
                        save_encoder_only(encoder, retrain_encoder_path)
                    tagged_full = os.path.join(
                        results_dir, f"retrain_full_{version_tag}_ep{epoch}.pth"
                    )
                    save_checkpoint(
                        tagged_full,
                        encoder,
                        trend_head,
                        proj_head,
                        optimizer,
                        epoch,
                        regime_cont_head=regime_cont_head,
                        mktstructure_head=mktstructure_head,
                        session_head=session_head,
                        pdhl_head=pdhl_head,
                        regime_quality_head=regime_quality_head,
                        extra={"note": "live_no_validation", "tagged_epoch": epoch, **_extra_base},
                    )
                    logger.info(
                        "[LIVE] ep=%d structural=%s regime=%s safe=%s",
                        epoch,
                        "✅" if not drift_detected else "⚠️",
                        "⚠️" if regime_drift_detected else "✅",
                        "✅" if retrain_safe else "⚠️",
                    )
                last_clean_live_epoch = epoch
            else:
                logger.warning(
                    "[STRUCTURAL DRIFT] live guard: ep=%d NOT saved. " "Last clean: %d",
                    epoch,
                    last_clean_live_epoch,
                )
            history["train"].append(running)
            _save_history()
    _save_history()

    # ── AB-COMPARE: one parsable summary line for the feature on/off comparison ─
    # One line per training with the best epoch and the key AUCs. A script can
    # aggregate the logs of every WF/universe into a verdict table.
    try:
        _ab = locals().get("_ab_best_metrics", None)
        _struct_on = bool(conf.get("use_structural_features", False))
        _wf_tag = conf.get("retrain_version_tag") or conf.get("class_name") or "?"
        _cn = conf.get("class_name", "?")
        if _ab:
            logger.info(
                "[AB-COMPARE] wf=%s universe=%s struct=%s seed=%s best_ep=%d "
                "score=%.4f session_side_auc=%.4f pdhl_side_auc=%.4f "
                "session_hit_auc=%.4f pdhl_hit_auc=%.4f regime_quality_auc=%.4f",
                _wf_tag,
                _cn,
                "ON" if _struct_on else "OFF",
                conf.get("global_seed", "n/a"),
                _ab["epoch"],
                _ab["score"],
                _ab["session_side_auc"],
                _ab["pdhl_side_auc"],
                _ab["session_hit_auc"],
                _ab["pdhl_hit_auc"],
                _ab["regime_quality_auc"],
            )
        else:
            logger.info(
                "[AB-COMPARE] wf=%s universe=%s struct=%s — no best saved",
                _wf_tag,
                _cn,
                "ON" if _struct_on else "OFF",
            )
    except Exception as _abe:
        logger.warning("[AB-COMPARE] summary log failed: %s", _abe)

    # ── close the file handler at the end ────────────────────────────────
    _save_history()
    try:
        _root_logger.removeHandler(_file_handler)
        _file_handler.flush()
        _file_handler.close()
        logger.info("[LOG FILE] Closed: %s", _log_path)
    except Exception as _fhe:
        logger.warning("[LOG FILE] close failed: %s", _fhe)
    return history


# =============================================================
# CLI
# =============================================================
def _parse_value(text):
    """'0.5' -> 0.5, 'true' -> True, '[1,2]' -> [1, 2], anything else stays a string."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Multi-head pretraining of the market encoder (bootstrap or walk-forward retrain).",
        epilog="Every other setting is in encoder/config.py: change it with --set key=value "
        "or put several in a JSON file and pass it with --config.",
    )
    p.add_argument("--class", dest="class_name", choices=["Majors", "Minors"], default=None)
    p.add_argument("--mode", choices=["bootstrap", "retrain"], default=None)
    p.add_argument("--root", default=None, help="data root (reads <root>/Scaled)")
    p.add_argument(
        "--models-root", default=None, help="where the checkpoints go (<models>/Encoder/...)"
    )
    p.add_argument(
        "--results-dir", default=None, help="exact output folder (overrides the default)"
    )
    p.add_argument("--config", default=None, help="JSON file with config overrides")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="single overrides")
    g = p.add_argument_group("walk-forward retrain")
    g.add_argument("--wf-tag", help="e.g. WF7_2026H1")
    g.add_argument("--train-end", help="last training day, e.g. 2025-12-31")
    g.add_argument("--val-start", help="first validation day")
    g.add_argument("--val-end", help="last validation day")
    g.add_argument("--start-from", help="checkpoint of the previous WF")
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
    cli = {
        "class_name": args.class_name,
        "training_mode": args.mode,
        "results_dir": args.results_dir,
        "retrain_version_tag": args.wf_tag,
        "retrain_train_end": args.train_end,
        "retrain_val_start": args.val_start,
        "retrain_val_end": args.val_end,
        "retrain_start_from": args.start_from,
    }
    overrides.update({k: v for k, v in cli.items() if v is not None})
    conf = build_config(overrides, root=args.root, models_root=args.models_root)

    warnings.filterwarnings("ignore", category=FutureWarning)
    _cn = conf.get("class_name", "Majors")
    _sk, _ = auto_select_schedule(_cn)
    logger.info(
        "Starting v28 | mode=%s | class=%s | live=%s | "
        "schedule=%s | w_session=%.2f w_pdhl=%.2f w_qual=%.2f | ema_bal=%s",
        conf["training_mode"],
        _cn,
        conf["live_mode"],
        _sk,
        conf.get("w_session_head", 0.5),
        conf.get("w_pdhl_head", 0.5),
        conf.get("w_regime_quality_head", 0.15),
        "ON" if conf["use_ema_balancing"] else "OFF",
    )
    return train_pretrainer_with_contrastive(conf)


if __name__ == "__main__":
    main()
