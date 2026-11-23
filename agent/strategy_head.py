"""
strategy_head.py — the agent's StrategyOutcomeHead, fitted on a frozen encoder.

MarketEnv has four trades: TREND (SL = sl_atr_mult x ATR, TP = rr x SL) and MEAN
REVERSION (SL = sl_atr_mult_mr x ATR, TP = rr_mr x SL), long and short, opened at the
close of a bar and closed at SL, TP or the Wednesday / Friday force close (there is no
other timeout). For each of them the head predicts, from the embedding the agent sees:

    hit  = P(SL or TP touched before the force close)
    side = P(TP first | touched)          P(TP) = hit x side

P(TP) of the four trades goes in the agent state and in the arbiter score. The labels
follow the env rules: entry at the close plus the spread, SL/TP on the real 15min
high / low, SL first when both are inside the same bar.

For every encoder checkpoint of a class (Bootstrap, WF1, ...) the head is trained on the
checkpoint's own training window, with early stopping on its validation window (no data
the encoder hasn't seen), and a copy of the checkpoint with the head is saved:

    <models>/Encoder/<class>/Agent/<Bootstrap | WFn_...>/encoder_agent.pth

    python -m agent.strategy_head --class Majors --root D:/Data --models-root D:/Models
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from encoder.components import StrategyOutcomeHead
from encoder.dataset import DATA_TZ, _load_all_scalers_file

from .config import DATA_ROOT, DEFAULT_CFG, MODELS_ROOT
from .env import (
    encoder_asset_id,
    encoder_inputs,
    ensure_datetime_index_utc,
    load_encoder_from_checkpoint,
)
from .market_calendar import entry_lockout_mask, force_close_mask

logger = logging.getLogger(__name__)

AGENT_CKPT_NAME = "encoder_agent.pth"
DEFAULTS = {
    "stride": 2,  # one training sample every N bars
    "epochs": 30,
    "patience": 4,
    "batch_size": 1024,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "hidden": 64,
    "embed_dim": 8,
    "max_horizon": 600,  # bars; the force close comes first (Mon entry -> Wed 21:30 ~ 200)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
}


# ── labels ────────────────────────────────────────────────────────────────────


def trade_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    spread: np.ndarray,
    force_close: np.ndarray,
    sl_mult: float,
    rr: float,
    direction: int,
    max_horizon: int = 600,
    return_r: bool = False,
):
    """
    Outcome of a trade opened at the close of every bar (direction +1 long, -1 short).
    Returns (hit, side), int8: hit 1 = SL or TP touched before the force close, 0 = closed
    by the force close, -1 = unknown (no ATR, data ends first); side 1 = TP first,
    0 = SL first, -1 = not touched. With return_r also the result in R: rr at the TP,
    -1 at the SL, the move to the force-close bar's close otherwise, NaN if unknown.
    """
    n = len(close)
    entry = close + direction * np.nan_to_num(spread, nan=0.0)
    risk = atr * sl_mult
    sl = entry - direction * risk
    tp = entry + direction * risk * rr
    valid = np.isfinite(entry) & np.isfinite(risk) & (risk > 0)
    hit = np.zeros(n, dtype=np.int8)
    side = np.full(n, -1, dtype=np.int8)
    r_mult = np.full(n, np.nan)
    open_ = valid.copy()
    for k in range(1, min(max_horizon, n - 1) + 1):
        if not open_.any():
            break
        rows = np.flatnonzero(open_[: n - k])
        hk, lk = high[rows + k], low[rows + k]
        if direction > 0:
            sl_hit, tp_hit = lk <= sl[rows], hk >= tp[rows]
        else:
            sl_hit, tp_hit = hk >= sl[rows], lk <= tp[rows]
        touched = sl_hit | tp_hit
        r = rows[touched]
        hit[r] = 1
        side[r] = np.where(sl_hit[touched], 0, 1)  # SL first on a same-bar collision
        r_mult[r] = np.where(sl_hit[touched], -1.0, rr)
        open_[r] = False
        # force close at the end of this bar (after its SL/TP check)
        fc = rows[~touched & force_close[rows + k]]
        r_mult[fc] = direction * (close[fc + k] - entry[fc]) / risk[fc]
        open_[fc] = False
    unknown = ~valid | open_  # still open: the data (or max_horizon) ended first
    hit[unknown] = -1
    side[unknown] = -1
    r_mult[unknown] = np.nan
    return (hit, side, r_mult) if return_r else (hit, side)


def strategy_labels(
    df: pd.DataFrame, env_cfg: dict, max_horizon: int = 600, return_r: bool = False
):
    """hit, side (and R) [n, 4] in StrategyOutcomeHead.COMBOS order, from the env's raw columns.
    Action a (1..4) of the env is combo a - 1."""
    cols = {str(c).lower(): c for c in df.columns}

    def raw(name, default=None):
        c = cols.get(f"{name}_raw")
        if c is None:
            if default is None:
                raise KeyError(f"{name}_raw missing from the data")
            return np.full(len(df), default, dtype=np.float64)
        return pd.to_numeric(df[c], errors="coerce").values.astype(np.float64)

    close, high, low = raw("close_15min"), raw("high_15min"), raw("low_15min")
    atr = raw("atr14_15min")
    spread = raw(
        str(env_cfg.get("spread_col", "spread_mean_15min_raw")).lower().removesuffix("_raw"),
        default=float(env_cfg.get("default_spread", 0.0001)),
    )
    fc = force_close_mask(df.index)
    params = {
        "trend": (float(env_cfg.get("sl_atr_mult", 1.0)), float(env_cfg.get("rr", 1.8))),
        "mr": (float(env_cfg.get("sl_atr_mult_mr", 1.0)), float(env_cfg.get("rr_mr", 1.0))),
    }
    hits, sides, rs = [], [], []
    for si, di, _ in StrategyOutcomeHead.COMBOS:
        sl_mult, rr = params[StrategyOutcomeHead.STRATEGIES[si]]
        h, s, r = trade_labels(
            close, high, low, atr, spread, fc, sl_mult, rr, 1 if di == 0 else -1, max_horizon, True
        )
        hits.append(h)
        sides.append(s)
        rs.append(r)
    out = (np.stack(hits, axis=1), np.stack(sides, axis=1))
    return out + (np.stack(rs, axis=1),) if return_r else out


# ── embeddings and training ───────────────────────────────────────────────────


@torch.no_grad()
def embed_bars(encoder, x: np.ndarray, ends: np.ndarray, seq_len: int, asset_id: int, device: str):
    """Embedding of the window ending at every bar in `ends` (ends >= seq_len - 1)."""
    windows = np.lib.stride_tricks.sliding_window_view(x, seq_len, axis=0)  # [n-L+1, F, L]
    out = []
    use_ids = getattr(encoder, "asset_embedding", None) is not None
    for i in range(0, len(ends), 512):
        w = windows[ends[i : i + 512] - seq_len + 1].transpose(0, 2, 1)
        xb = torch.from_numpy(np.ascontiguousarray(w)).to(device)
        ids = torch.full((len(w),), asset_id, dtype=torch.long, device=device) if use_ids else None
        out.append(encoder(xb, asset_ids=ids).float().cpu())
    return torch.cat(out) if out else torch.zeros(0, getattr(encoder, "gru_hidden", 1))


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = pd.Series(p).rank().values
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def _combo_batch(z, hit, side):
    """Repeats every embedding for the 4 combos: (z, strategy, direction, hit, side)."""
    n = z.shape[0]
    s = torch.tensor([c[0] for c in StrategyOutcomeHead.COMBOS]).repeat(n)
    d = torch.tensor([c[1] for c in StrategyOutcomeHead.COMBOS]).repeat(n)
    return z.repeat_interleave(4, dim=0), s, d, hit.reshape(-1), side.reshape(-1)


def _loss(head, z, hit, side, device):
    zz, s, d, h, sd = (t.to(device) for t in _combo_batch(z, hit, side))
    hit_logit, side_logit = head(zz, s, d)
    hit_logit, side_logit = hit_logit.squeeze(-1), side_logit.squeeze(-1)
    mh, ms = h >= 0, sd >= 0
    loss = F.binary_cross_entropy_with_logits(hit_logit[mh], h[mh].float())
    if ms.any():
        loss = loss + F.binary_cross_entropy_with_logits(side_logit[ms], sd[ms].float())
    return loss


@torch.no_grad()
def evaluate_head(head, z, hit, side, device) -> dict:
    head.eval()
    probs = head.predict_all_combos(z.to(device))
    out = {}
    for i, (_, _, name) in enumerate(StrategyOutcomeHead.COMBOS):
        h, s = hit[:, i].numpy(), side[:, i].numpy()
        p_hit = probs[f"p_hit_{name}"].cpu().numpy()
        p_side = probs[f"p_tp_given_hit_{name}"].cpu().numpy()
        p_tp = probs[f"p_tp_{name}"].cpu().numpy()
        known = h >= 0
        tp_real = (s == 1)[known]
        out[name] = {
            "n": int(known.sum()),
            "hit_rate": float((h[known] == 1).mean()) if known.any() else float("nan"),
            "tp_rate": float(tp_real.mean()) if known.any() else float("nan"),
            "p_tp_mean": float(p_tp[known].mean()) if known.any() else float("nan"),
            "auc_hit": _auc(h[known], p_hit[known]),
            "auc_side": _auc(s[s >= 0], p_side[s >= 0]),
            "auc_tp": _auc(tp_real.astype(int), p_tp[known]),
        }
    return out


def fit_strategy_head(z_tr, hit_tr, side_tr, z_va, hit_va, side_va, emb_dim, params):
    """Trains a StrategyOutcomeHead with early stopping on the validation loss."""
    device = params["device"]
    torch.manual_seed(params["seed"])
    head = StrategyOutcomeHead(emb_dim, hidden=params["hidden"], embed_dim=params["embed_dim"])
    head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    best, best_state, bad = float("inf"), copy.deepcopy(head.state_dict()), 0
    gen = torch.Generator().manual_seed(params["seed"])
    for epoch in range(params["epochs"]):
        head.train()
        perm = torch.randperm(z_tr.shape[0], generator=gen)
        for i in range(0, len(perm), params["batch_size"]):
            b = perm[i : i + params["batch_size"]]
            loss = _loss(head, z_tr[b], hit_tr[b], side_tr[b], device)
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            val = float(_loss(head, z_va, hit_va, side_va, device)) if len(z_va) else 0.0
        logger.info("[SOH] epoch %d | val loss %.4f", epoch + 1, val)
        if val < best - 1e-4:
            best, best_state, bad = val, copy.deepcopy(head.state_dict()), 0
        else:
            bad += 1
            if bad >= params["patience"]:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head.cpu(), best


def _ts(value, end: bool = False) -> pd.Timestamp:
    """Encoder window bound -> UTC, as encoder.dataset.slice_df_by_period reads it."""
    ts = pd.Timestamp(value)
    if end:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    ts = ts.tz_localize(DATA_TZ) if ts.tzinfo is None else ts
    return ts.tz_convert("UTC")


def _period_samples(df, x_len, seq_len, start, end, stride, hit):
    """Bars of [start, end] with a full window, entries allowed and at least one label."""
    ts = df.index
    sel = (ts >= start) & (ts <= end) & ~entry_lockout_mask(ts) & (hit >= 0).any(axis=1)
    sel[: seq_len - 1] = False
    sel &= np.arange(len(df)) < x_len
    idx = np.flatnonzero(sel)
    return idx[::stride] if stride > 1 else idx


def build_agent_checkpoint(
    ckpt_path: str, asset_paths: dict[str, str], out_path: str, env_cfg: dict, params: dict
) -> dict:
    """Fits the head for one encoder checkpoint and saves the agent copy. Returns its metrics."""
    device = params["device"]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    extra = ckpt.get("extra") or {}
    if not all(k in extra for k in ("train_start", "train_end", "val_start", "val_end")):
        raise ValueError(f"{ckpt_path}: no train/val window in the checkpoint")
    meta = extra.get("inputs") or {}
    seq_len = int(meta.get("seq_len", env_cfg.get("seq_len", 128)))
    t = load_encoder_from_checkpoint(ckpt_path, device=device, seq_len=seq_len)
    encoder, emb_dim = t[0], t[1]
    tr = (_ts(extra["train_start"]), _ts(extra["train_end"], end=True))
    va = (_ts(extra["val_start"]), _ts(extra["val_end"], end=True))

    parts = {"tr": ([], [], []), "va": ([], [], [])}
    for asset, path in sorted(asset_paths.items()):
        df = ensure_datetime_index_utc(pd.read_parquet(path))
        df = df.loc[: va[1] + pd.Timedelta(days=7)]  # labels need a few bars after val_end
        scaler_map = None
        if meta.get("use_structural_features"):
            scaler_map = _load_all_scalers_file(
                os.path.join(os.path.dirname(path), "all_scalers.pkl")
            )
        x = encoder_inputs(df, encoder, asset, scaler_map)
        hit, side = strategy_labels(df, env_cfg, params["max_horizon"])
        aid = encoder_asset_id(encoder, asset, list(asset_paths))
        for key, (a, b), stride in (("tr", tr, params["stride"]), ("va", va, params["stride"])):
            rows = _period_samples(df, len(x), seq_len, a, b, stride, hit)
            if len(rows) == 0:
                continue
            parts[key][0].append(embed_bars(encoder, x, rows, seq_len, aid, device))
            parts[key][1].append(torch.from_numpy(hit[rows].astype(np.int64)))
            parts[key][2].append(torch.from_numpy(side[rows].astype(np.int64)))
        logger.info("[SOH] %s: %d bars", asset, len(df))
    if not parts["tr"][0] or not parts["va"][0]:
        raise ValueError(f"{ckpt_path}: no samples in the train / val window")
    z_tr, h_tr, s_tr = (torch.cat(p) for p in parts["tr"])
    z_va, h_va, s_va = (torch.cat(p) for p in parts["va"])
    logger.info("[SOH] samples: train %d | val %d", len(z_tr), len(z_va))

    head, val_loss = fit_strategy_head(z_tr, h_tr, s_tr, z_va, h_va, s_va, emb_dim, params)
    metrics = evaluate_head(head, z_va, h_va, s_va, "cpu")
    for name, m in metrics.items():
        logger.info(
            "[SOH] val %-11s tp_rate=%.3f p_tp=%.3f auc_hit=%.3f auc_side=%.3f auc_tp=%.3f",
            name,
            m["tp_rate"],
            m["p_tp_mean"],
            m["auc_hit"],
            m["auc_side"],
            m["auc_tp"],
        )
    out = dict(ckpt)
    out["strategy_outcome_head_state"] = head.state_dict()
    out["extra"] = dict(extra)
    out["extra"]["strategy_outcome"] = {
        "source": os.path.abspath(ckpt_path),
        "trade_params": {
            k: env_cfg.get(k)
            for k in ("rr", "sl_atr_mult", "rr_mr", "sl_atr_mult_mr", "spread_col")
        },
        "train": [str(tr[0]), str(tr[1])],
        "val": [str(va[0]), str(va[1])],
        "val_loss": val_loss,
        "val_metrics": metrics,
        "params": {k: v for k, v in params.items() if k != "device"},
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(out, out_path)
    with open(os.path.join(os.path.dirname(out_path), "strategy_outcome.json"), "w") as f:
        json.dump(out["extra"]["strategy_outcome"], f, indent=2, default=str)
    logger.info("[SOH] saved %s", out_path)
    return metrics


def encoder_checkpoints(models_root, class_name: str) -> dict[str, str]:
    """{run folder name: encoder checkpoint}: Bootstrap and the WF retrains of a class."""
    base = Path(models_root) / "Encoder" / class_name / "Checkpoint"
    found = {}
    boot = base / "Bootstrap" / "best_full_checkpoint.pth"
    if boot.exists():
        found["Bootstrap"] = str(boot)
    for wf_dir in sorted((base / "Retrain").glob("WF*")):
        ckpt = wf_dir / f"retrain_full_{wf_dir.name}.pth"
        if ckpt.exists():
            found[wf_dir.name] = str(ckpt)
    return found


def _asset_paths(data_dir: str) -> dict[str, str]:
    paths = {}
    for folder in sorted(glob.glob(os.path.join(data_dir, "*"))):
        files = sorted(glob.glob(os.path.join(folder, "*_scaled_*.parquet")))
        if files:
            paths[os.path.basename(folder)] = files[-1]
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Fit the agent's StrategyOutcomeHead on frozen encoders"
    )
    ap.add_argument("--class", dest="classes", nargs="+", default=["Majors", "Minors"])
    ap.add_argument("--root", default=str(DATA_ROOT), help="data root (with Scaled/<class>)")
    ap.add_argument("--models-root", default=str(MODELS_ROOT))
    ap.add_argument("--only", nargs="*", help="run folders to do (e.g. Bootstrap WF3_2024H2)")
    ap.add_argument("--force", action="store_true", help="redo the checkpoints already done")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="see DEFAULTS")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    params = dict(DEFAULTS)
    for kv in args.set:
        k, v = kv.split("=", 1)
        params[k] = type(DEFAULTS[k])(v) if k in DEFAULTS and DEFAULTS[k] is not None else v
    env_cfg = DEFAULT_CFG["env_cfg"]
    done = {}
    for cls in args.classes:
        assets = _asset_paths(os.path.join(args.root, "Scaled", cls))
        if not assets:
            raise FileNotFoundError(f"no scaled parquet in {args.root}/Scaled/{cls}")
        for run, ckpt in encoder_checkpoints(args.models_root, cls).items():
            if args.only and run not in args.only:
                continue
            out = Path(args.models_root) / "Encoder" / cls / "Agent" / run / AGENT_CKPT_NAME
            if out.exists() and not args.force:
                logger.info("[SOH] %s %s already done: %s", cls, run, out)
                continue
            logger.info("[SOH] %s %s <- %s", cls, run, ckpt)
            done[f"{cls}/{run}"] = build_agent_checkpoint(ckpt, assets, str(out), env_cfg, params)
    return done


if __name__ == "__main__":
    main()
