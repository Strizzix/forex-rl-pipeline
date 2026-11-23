"""
env.py — forex market environment for the Rainbow DQN agent.

MarketEnv walks one asset bar by bar (15min) on the scaled multi-TF parquet:
 - the frozen market encoder turns the last seq_len bars into an embedding, and its
   heads give trend momentum, continuous regime (vol regime, ATR, ADX, DI direction),
   breakout probabilities and the StrategyOutcomeHead P(TP) of TREND / MR, long / short
 - state vector (204 dims): embedding + head signals + market indicators + time
   buckets + position-aware features (unrealized PnL, MFE/MAE, distance to SL/TP)
   + daily risk + joint portfolio context + session validity
 - 5 actions: HOLD, LONG_TREND, SHORT_TREND, LONG_MR, SHORT_MR; SL/TP in ATR units,
   checked on the raw 15min high/low; spread by session hour, swap, force close
 - reward: realized R plus dense shaping (mark-to-market, opportunity cost, periodic
   Sharpe bonus, hold reward), training mode aware reset (scratch | retrain | eval)
 - helpers to load the encoder checkpoint and its heads without importing the encoder
   package (the heads are rebuilt from the state_dict shapes)
"""

import logging
import math
import os
from collections import deque
from datetime import timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def ensure_datetime_index_utc(df: pd.DataFrame, tz_local: str = "Europe/Rome") -> pd.DataFrame:
    """Ensure DatetimeIndex is tz-aware and in UTC."""
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"]))
        else:
            raise ValueError("DataFrame index must be a DatetimeIndex")
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz_local)
    df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    return df


# ----------------------------
# Minimal encoder classes (TCN-GRU) — UNCHANGED
# ----------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x if self.chomp == 0 else x[:, :, : -self.chomp].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, k, dilation, padding, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(n_in, n_out, k, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_out, n_out, k, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.drop1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.drop2,
        )
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoderBlock(nn.Module):
    def __init__(self, num_inputs, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i, ch in enumerate(channels):
            in_ch = num_inputs if i == 0 else channels[i - 1]
            dilation = 2**i
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_ch, ch, kernel_size, dilation, padding, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNGRUEncoder(nn.Module):
    """v23: optional asset embedding support (n_assets=0 = backward compatible)."""

    def __init__(
        self,
        input_features: int,
        seq_len: int,
        tcn_channels: list[int],
        kernel_size: int = 3,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        dropout: float = 0.1,
        layer_norm: bool = True,
        n_assets: int = 0,
        asset_embed_dim: int = 8,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_assets = n_assets
        self.asset_embed_dim = asset_embed_dim if n_assets > 0 else 0
        if n_assets > 0:
            self.asset_embedding: Optional[nn.Module] = nn.Embedding(n_assets, asset_embed_dim)
            tcn_input = input_features + asset_embed_dim
        else:
            self.asset_embedding = None
            tcn_input = input_features
        self.tcn = TCNEncoderBlock(tcn_input, tcn_channels, kernel_size, dropout)
        self.gru = nn.GRU(
            input_size=tcn_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(gru_hidden) if layer_norm else None
        self.dropout = nn.Dropout(dropout)
        self.input_features = input_features
        self.gru_hidden = gru_hidden

    def forward(self, x, asset_ids=None):
        if self.asset_embedding is not None and asset_ids is not None:
            tok = self.asset_embedding(asset_ids)  # [B, embed_dim]
            tok = tok.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, tok], dim=-1)
        x = x.permute(0, 2, 1)
        tcn_out = self.tcn(x)
        tcn_out = tcn_out.permute(0, 2, 1)
        out, h = self.gru(tcn_out)
        emb = h[-1]
        if self.layer_norm is not None:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


def strip_prefix(k: str) -> str:
    for p in ("module.encoder.", "module.", "encoder.", "model.encoder.", "model."):
        if k.startswith(p):
            return k[len(p) :]
    return k


def _load_head_from_state(
    state_dict: dict, device: str, head_name: str = "head"
) -> Optional[nn.Module]:
    """Load auxiliary head from state_dict with automatic architecture detection.

    Auto-detects architecture from fc layer keys:
      - fc1+fc2          → 2-layer MLP (TrendMomentumHeadTF, VolatilityHeadTF, etc.)
      - fc1+fc2+fc3      → 3-layer MLP with softplus (RegimeContinuousHead)
      - other            → Sequential auto-build
    """
    if not state_dict or not isinstance(state_dict, dict):
        logger.warning("%s state is empty or invalid", head_name)
        return None
    try:
        # ── v26: StrategyOutcomeHead — detected from strategy_emb.weight + shared.0.weight ──
        # typical keys: strategy_emb.weight, direction_emb.weight,
        #   shared.0.weight, shared.3.weight, hit_trunk.0.weight, side_trunk.0.weight,
        #   hit_head.weight, side_head.weight
        if "strategy_emb.weight" in state_dict and "shared.0.weight" in state_dict:
            _s_emb_w = state_dict["strategy_emb.weight"]  # [2, embed_dim]
            _sh0_w = state_dict["shared.0.weight"]  # [hidden, trunk_in]
            _embed_d = int(_s_emb_w.shape[1])
            _hidden = int(_sh0_w.shape[0])
            _trunk_in = int(_sh0_w.shape[1])
            # emb_dim inferred: trunk_in − strategy_embed − direction_embed − ctx_dim
            # ctx_dim = 0 unless there's an extra key; computed afterwards
            # Note: ctx_dim isn't known directly -> try 0 and recompute
            # in practice: trunk_in = emb_dim + 2*embed_dim + ctx_dim
            # emb_dim = gru_hidden (128), so:
            # ctx_dim = trunk_in - emb_dim - 2*embed_dim (computed by the caller)
            # Left at 0 here because _load_head_from_state doesn't know gru_hidden.
            # The caller (load_encoder_from_checkpoint) knows gru_hidden and
            # fixes ctx_dim afterwards.
            # This branch is only a fallback for generic calls.
            _dropout = 0.1  # default, not stored in the state_dict

            class _StrategyOutcomeHeadInline(nn.Module):
                """Inline copy of StrategyOutcomeHead — no external import."""

                COMBOS = [
                    (0, 0, "trend_long"),
                    (0, 1, "trend_short"),
                    (1, 0, "mr_long"),
                    (1, 1, "mr_short"),
                ]

                def __init__(self):
                    super().__init__()
                    self.embed_dim = _embed_d
                    self.hidden = _hidden
                    self.ctx_dim = 0  # updated after load_state_dict if needed
                    self.strategy_emb = nn.Embedding(2, _embed_d)
                    self.direction_emb = nn.Embedding(2, _embed_d)
                    self.shared = nn.Sequential(
                        nn.Linear(_trunk_in, _hidden),
                        nn.ReLU(),
                        nn.Dropout(_dropout),
                        nn.Linear(_hidden, _hidden // 2),
                        nn.ReLU(),
                    )
                    self.hit_trunk = nn.Sequential(
                        nn.Linear(_hidden // 2, _hidden // 2), nn.ReLU(), nn.Dropout(_dropout)
                    )
                    self.side_trunk = nn.Sequential(
                        nn.Linear(_hidden // 2, _hidden // 2), nn.ReLU(), nn.Dropout(_dropout)
                    )
                    self.hit_head = nn.Linear(_hidden // 2, 1)
                    self.side_head = nn.Linear(_hidden // 2, 1)

                def forward(self, emb, strategy, direction, extra_context=None):
                    s_emb = self.strategy_emb(strategy.long())
                    d_emb = self.direction_emb(direction.long())
                    parts = [emb, s_emb, d_emb]
                    if extra_context is not None:
                        parts.append(extra_context.to(emb.dtype))
                    x = torch.cat(parts, dim=-1)
                    h = self.shared(x)
                    return self.hit_head(self.hit_trunk(h)), self.side_head(self.side_trunk(h))

                def predict_all_combos(self, emb, extra_context=None):
                    B, device = emb.size(0), emb.device
                    out = {}
                    for si, di, pfx in self.COMBOS:
                        s_t = torch.full((B,), si, dtype=torch.long, device=device)
                        d_t = torch.full((B,), di, dtype=torch.long, device=device)
                        hl, sl = self.forward(emb, s_t, d_t, extra_context)
                        p_hit = torch.sigmoid(hl).squeeze(-1)
                        p_tp_h = torch.sigmoid(sl).squeeze(-1)
                        out[f"p_hit_{pfx}"] = p_hit
                        out[f"p_tp_given_hit_{pfx}"] = p_tp_h
                        out[f"p_tp_{pfx}"] = p_hit * p_tp_h
                        out[f"p_sl_{pfx}"] = p_hit * (1.0 - p_tp_h)
                        out[f"p_none_{pfx}"] = 1.0 - p_hit
                    return out

            head = _StrategyOutcomeHeadInline()
            head.load_state_dict(state_dict, strict=True)
            head.to(device).float().eval()
            logger.info(
                "%s loaded (StrategyOutcomeHead inline: hidden=%d embed_dim=%d)",
                head_name,
                _hidden,
                _embed_d,
            )
            return head
        fc_emb_w = state_dict.get("fc_emb.weight")
        fc_dir_w = state_dict.get("fc_dir.weight")
        fc_out_w = state_dict.get("fc_out.weight")
        if fc_emb_w is not None and fc_dir_w is not None and fc_out_w is not None:
            emb_in = int(fc_emb_w.shape[1])
            emb_h = int(fc_emb_w.shape[0])
            dir_in = int(fc_dir_w.shape[1])
            dir_h = int(fc_dir_w.shape[0])  # can differ from emb_h
            out_in = int(fc_out_w.shape[1])
            out_dim = int(fc_out_w.shape[0])

            class HeadDirCond(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc_emb = nn.Linear(emb_in, emb_h)
                    self.fc_dir = nn.Linear(dir_in, dir_h)
                    self.fc_out = nn.Linear(out_in, out_dim)

                def forward(self, x, direction=None):
                    h = F.relu(self.fc_emb(x))
                    if direction is not None:
                        if direction.dim() == 1:
                            direction = direction.unsqueeze(-1)
                        h = h + F.relu(self.fc_dir(direction.float()))
                    return self.fc_out(h)

            head = HeadDirCond()
            head.load_state_dict(state_dict, strict=True)
            head.to(device)
            head.eval()
            logger.info("%s loaded (direction-conditioned arch)", head_name)
            return head

        fc_keys = sorted([k for k in state_dict.keys() if k.startswith("fc") and "weight" in k])

        if fc_keys:
            # Detect number of fc layers
            n_layers = len(fc_keys)
            in_dim = state_dict[fc_keys[0]].shape[1]
            hidden_dim = state_dict[fc_keys[0]].shape[0]

            if n_layers == 3:
                # 3-layer MLP — RegimeContinuousHead style: fc1→relu→fc2→relu→fc3→softplus
                hidden2_dim = state_dict[fc_keys[1]].shape[0]
                out_dim = state_dict[fc_keys[2]].shape[0]

                class Head3(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.fc1 = nn.Linear(in_dim, hidden_dim)
                        self.fc2 = nn.Linear(hidden_dim, hidden2_dim)
                        self.fc3 = nn.Linear(hidden2_dim, out_dim)

                    def forward(self, x):
                        return F.softplus(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))

                head = Head3()
            else:
                # 2-layer MLP — default
                out_dim = state_dict[fc_keys[-1]].shape[0]

                class Head2(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.fc1 = nn.Linear(in_dim, hidden_dim)
                        self.relu = nn.ReLU()
                        self.fc2 = nn.Linear(hidden_dim, out_dim)

                    def forward(self, x):
                        return self.fc2(self.relu(self.fc1(x)))

                head = Head2()
            head.load_state_dict(state_dict, strict=True)
        else:
            # Fallback: Sequential auto-build
            layer_keys = sorted([k for k in state_dict.keys() if "weight" in k])
            layers = []
            for i, key in enumerate(layer_keys):
                weight = state_dict[key]
                out_dim, in_dim = weight.shape
                layers.append(nn.Linear(in_dim, out_dim))
                if i < len(layer_keys) - 1:
                    layers.append(nn.ReLU())
            head = nn.Sequential(*layers)
            head.load_state_dict(state_dict, strict=True)

        head.to(device)
        head.eval()
        logger.info("%s loaded successfully", head_name)
        return head
    except Exception as e:
        logger.exception("Failed to load %s: %s", head_name, e)
        return None


def load_encoder_from_checkpoint(
    path: str,
    device: str,
    seq_len: int,
    load_regime_head: bool = True,
    load_trend_head: bool = True,
    load_vol_head: bool = True,
    load_outcome_head: bool = True,  # v23: outcome head for the arbiter filter
    fallback_channels=None,
    fallback_gru_hidden=128,
):
    """Load encoder and optional heads. v23: outcome_head and asset_embedding support."""
    if fallback_channels is None:
        fallback_channels = [64, 64, 128]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, nn.Module):
        enc = ckpt.to(device)
        enc.eval()
        return enc, getattr(enc, "gru_hidden", fallback_gru_hidden), None, None, None
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint format not recognized (expected dict)")

    encoder_state = None
    cfg_ckpt = ckpt.get("cfg", {}) or {}
    for key in ("encoder_state_dict", "encoder_state", "encoder"):
        if key in ckpt and isinstance(ckpt[key], dict):
            encoder_state = ckpt[key]
            break
    if encoder_state is None:
        for key in ("model_state", "model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                cand = ckpt[key]
                enc_keys = {
                    strip_prefix(k): v
                    for k, v in cand.items()
                    if any(x in k.lower() for x in ("tcn", "gru", "encoder", "temporal"))
                }
                encoder_state = enc_keys if enc_keys else cand
                break
    if encoder_state is None:
        raise RuntimeError("Unable to find encoder state_dict in checkpoint")

    input_features = cfg_ckpt.get("input_dim") or cfg_ckpt.get("input_features") or 166
    tcn_channels = cfg_ckpt.get("tcn_channels") or fallback_channels
    kernel_size = cfg_ckpt.get("kernel_size", 3)
    gru_hidden = cfg_ckpt.get("gru_hidden", fallback_gru_hidden)
    gru_layers = cfg_ckpt.get("gru_layers", 1)
    dropout = cfg_ckpt.get("dropout", 0.2)
    # ── Detect n_assets / asset_embed_dim from encoder_state ─────────────
    # encoder_meta and cfg keys are missing from v25 checkpoints — derived from the weights.
    enc_meta = ckpt.get("encoder_meta", {}) or {}
    n_assets = int(enc_meta.get("n_assets", cfg_ckpt.get("n_assets", 0)))
    asset_embed_dim = int(enc_meta.get("asset_embed_dim", cfg_ckpt.get("asset_embed_dim", 8)))
    # v25: read the shape straight from the embedding weight if present
    ae_key = "asset_embedding.embedding.weight"
    if ae_key in encoder_state and isinstance(encoder_state[ae_key], torch.Tensor):
        ae_w = encoder_state[ae_key]
        if ae_w.dim() == 2:
            n_assets = int(ae_w.shape[0])
            asset_embed_dim = int(ae_w.shape[1])
            logger.info(
                "asset_embedding detected from checkpoint: n_assets=%d embed_dim=%d",
                n_assets,
                asset_embed_dim,
            )
    for k, v in encoder_state.items():
        if "conv1.weight" in k and isinstance(v, torch.Tensor):
            try:
                raw_ch = int(v.shape[1])
                if n_assets > 0:
                    raw_ch = raw_ch - asset_embed_dim
                input_features = raw_ch
                break
            except Exception:
                pass
    if n_assets > 0:
        logger.info(
            "Encoder with asset embedding: n_assets=%d embed_dim=%d", n_assets, asset_embed_dim
        )
    encoder = TCNGRUEncoder(
        input_features,
        seq_len,
        tcn_channels,
        kernel_size,
        gru_hidden,
        gru_layers,
        dropout,
        n_assets=n_assets,
        asset_embed_dim=asset_embed_dim,
    )
    model_sd = encoder.state_dict()
    mapped = {}
    for sk, v in encoder_state.items():
        sk2 = strip_prefix(sk)
        if sk2 in model_sd and tuple(model_sd[sk2].shape) == tuple(v.shape):
            mapped[sk2] = v
        else:
            for mk in model_sd.keys():
                if mk.endswith(sk2) and tuple(model_sd[mk].shape) == tuple(v.shape):
                    mapped[mk] = v
                    break
    try:
        encoder.load_state_dict(mapped, strict=False)
        logger.info("Encoder loaded: input_features=%d gru_hidden=%d", input_features, gru_hidden)
    except Exception as e:
        logger.warning("Encoder load incomplete: %s", e)
    encoder.to(device)
    encoder.eval()

    # ── Load auxiliary heads ─────────────────────────────────────────────
    # v25 checkpoint: keys "trend_head_state", "regime_cont_head_state", etc.
    # v23 legacy:     keys "trend_head", "regime_cont_head", etc.
    def _ckpt_head(primary_key: str, *fallback_keys: str) -> Optional[dict]:
        for k in (primary_key,) + fallback_keys:
            v = ckpt.get(k)
            if v is not None and isinstance(v, dict) and len(v) > 0:
                return v
        return None

    regime_head = (
        _load_head_from_state(_ckpt_head("regime_head_state", "regime_head"), device, "regime_head")
        if load_regime_head
        else None
    )
    trend_head = (
        _load_head_from_state(_ckpt_head("trend_head_state", "trend_head"), device, "trend_head")
        if load_trend_head
        else None
    )
    vol_head = (
        _load_head_from_state(_ckpt_head("vol_head_state", "vol_head"), device, "vol_head")
        if load_vol_head
        else None
    )

    # v25: head continue
    trend_mom_head = _load_head_from_state(
        _ckpt_head("trend_head_state", "trend_head"), device, "trend_mom_head"
    )
    regime_cont_head = _load_head_from_state(
        _ckpt_head("regime_cont_head_state", "regime_cont_head"), device, "regime_cont_head"
    )
    mktstr_head = _load_head_from_state(
        _ckpt_head("mktstructure_head_state", "mktstructure_head", "mktstr_head"),
        device,
        "mktstr_head",
    )

    outcome_head_sd = _ckpt_head("outcome_head_state", "outcome_head")
    outcome_head = (
        _load_head_from_state(outcome_head_sd, device, "outcome_head")
        if load_outcome_head and outcome_head_sd
        else None
    )

    # v26: StrategyOutcomeHead — loaded through _load_head_from_state (inline class)
    # No import from the encoder package — same logic as the other heads.
    # After loading, ctx_dim is fixed on the instance because _load_head_from_state
    # doesn't know gru_hidden and sets it to 0 by default.
    strategy_outcome_head = None
    _soh_sd = _ckpt_head("strategy_outcome_head_state", "strategy_outcome_head")
    if _soh_sd is not None:
        strategy_outcome_head = _load_head_from_state(
            _soh_sd, device, head_name="strategy_outcome_head"
        )
        if strategy_outcome_head is not None:
            # Fix ctx_dim: trunk_in − gru_hidden − 2×embed_dim
            try:
                _embed_d_soh = int(_soh_sd["strategy_emb.weight"].shape[1])
                _trunk_in = int(_soh_sd["shared.0.weight"].shape[1])
                strategy_outcome_head.ctx_dim = max(
                    0, _trunk_in - int(gru_hidden) - 2 * _embed_d_soh
                )
                logger.info(
                    "strategy_outcome_head: ctx_dim set to %d", strategy_outcome_head.ctx_dim
                )
            except Exception as _e:
                logger.warning("strategy_outcome_head: ctx_dim falls back to 0 (%s)", _e)

    for name, head in [
        ("regime_cont_head", regime_cont_head),
        ("mktstr_head", mktstr_head),
        ("trend_mom_head", trend_mom_head),
        ("outcome_head", outcome_head),
    ]:
        logger.info("%s: %s", name, "loaded" if head is not None else "not found in checkpoint")

    # Return tuple extended
    return (
        encoder,
        int(gru_hidden),
        regime_head,
        trend_head,
        vol_head,
        outcome_head,
        trend_mom_head,
        regime_cont_head,
        mktstr_head,
        strategy_outcome_head,
    )


# v22: session features computed from DatetimeIndex (same as encoder_dataset._compute_session_features)
_SESSION_FEATURE_NAMES = [
    "session_hour_sin",
    "session_hour_cos",
    "session_london",
    "session_ny",
    "session_overlap",
    "session_dow_sin",
    "session_dow_cos",
]


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the 7 session / time-of-day features to the DataFrame if not there yet.
    Must be called BEFORE build_encoder_feature_cols.
    Same as encoder_dataset._compute_session_features (v22).
    """
    if _SESSION_FEATURE_NAMES[0] in df.columns:
        return df  # already there
    n = len(df)
    try:
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError("Index is not DatetimeIndex")
        idx_utc = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        h = idx_utc.hour.values.astype(np.float32)
        m = idx_utc.minute.values.astype(np.float32)
        h_frac = h + m / 60.0
        h_int = idx_utc.hour.values
        dow = idx_utc.dayofweek.values.astype(np.float32)
        feats = np.stack(
            [
                np.sin(2.0 * np.pi * h_frac / 24.0).astype(np.float32),  # hour_sin
                np.cos(2.0 * np.pi * h_frac / 24.0).astype(np.float32),  # hour_cos
                ((h_int >= 7) & (h_int < 16)).astype(np.float32),  # london
                ((h_int >= 13) & (h_int < 21)).astype(np.float32),  # ny
                ((h_int >= 13) & (h_int < 16)).astype(np.float32),  # overlap
                np.sin(2.0 * np.pi * dow / 5.0).astype(np.float32),  # dow_sin
                np.cos(2.0 * np.pi * dow / 5.0).astype(np.float32),  # dow_cos
            ],
            axis=1,
        )
    except Exception as e:
        logger.warning("Session feature computation failed (%s). Using zeros.", e)
        feats = np.zeros((n, len(_SESSION_FEATURE_NAMES)), dtype=np.float32)
    sess_df = pd.DataFrame(feats, index=df.index, columns=_SESSION_FEATURE_NAMES)
    return pd.concat([df, sess_df], axis=1)


def build_encoder_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not str(c).endswith("_raw") and c != "timestamp"]


# ===========================================================================
# MARKET ENV
# ===========================================================================


class MarketEnv:
    ACTIONS = {
        0: "HOLD",
        1: "LONG_TREND",
        2: "SHORT_TREND",
        3: "LONG_MR",
        4: "SHORT_MR",
    }

    def __init__(
        self,
        df: pd.DataFrame,
        encoder: Optional[nn.Module],
        regime_head: Optional[nn.Module],
        config: dict[str, Any],
        trend_head: Optional[nn.Module] = None,
        vol_head: Optional[nn.Module] = None,
        outcome_head: Optional[nn.Module] = None,
        asset_id: int = 0,
        trend_mom_head: Optional[nn.Module] = None,
        regime_cont_head: Optional[nn.Module] = None,
        mktstr_head: Optional[nn.Module] = None,
        strategy_outcome_head: Optional[nn.Module] = None,
    ):
        # UTC checks
        if not isinstance(df.index, pd.DatetimeIndex):
            raise AssertionError("DataFrame index must be DatetimeIndex")
        if df.index.tz is None:
            raise AssertionError("DataFrame index must be tz-aware")
        if "UTC" not in str(df.index.tz):
            raise AssertionError(f"DataFrame index tz must be UTC, got {df.index.tz}")

        # v22: add the session features if the scaled dataset doesn't have them
        df = add_session_features(df)
        self.df = df.sort_index().copy()
        self.encoder = encoder
        self.regime_head = regime_head
        self.outcome_head = outcome_head
        self.asset_id = asset_id
        self.trend_head = trend_head
        self.vol_head = vol_head
        # v24: new continuous heads
        self.trend_mom_head = trend_mom_head  # TrendMomentumHeadTF — scalar per TF
        self.regime_cont_head = regime_cont_head  # RegimeContinuousHead — [vr,an,adx] per TF
        self.mktstr_head = mktstr_head  # MarketStructureHead — [R,T,B] per TF
        self.strategy_outcome_head = strategy_outcome_head  # v26: StrategyOutcomeHead 2-head
        self.cfg = config

        self.model_tf_order = config.get("model_tf_order", ["15min", "30min", "1h", "4h", "1d"])

        # Regime/trend state
        self._current_regime_15m = -1.0
        self._current_regime_30m = -1.0
        self._current_regime_1h = -1.0
        self._current_regime_4h = -1.0  # ← v2
        self._current_trend_15m = 0.0
        self._current_trend_30m = 0.0
        self._current_trend_1h = 0.0
        self._current_trend_4h = 0.0
        self._trend_conf_15m = 0.0  # ← v2: softmax max 15m
        self._trend_conf_1h = 0.0  # ← v2: softmax max 1h
        self._last_vol_pred_15m = 0.0  # ← v2: vol head pred 15m
        self._last_vol_pred_1h = 0.0  # ← v2: vol head pred 1h
        self._regime_history_1h = deque(maxlen=96)
        # Last closed trade tracking
        self._last_trade_rr = 0.0
        self._last_trade_win = 0.0

        # Training mode
        tm_raw = str(config.get("training_mode", "scratch")).lower()
        self.training_mode = {
            "full": "train",
            "retrain": "train",
            "train": "train",
            "scratch": "scratch",
            "eval": "eval",
            "live": "live",
        }.get(tm_raw, tm_raw)

        # Encoder feature cols
        self.encoder_feature_cols: list[str] = config.get("encoder_feature_cols", [])
        if not self.encoder_feature_cols:
            logger.warning("No encoder_feature_cols; using default 5")
            self.encoder_feature_cols = [
                "open_15min",
                "high_15min",
                "low_15min",
                "close_15min",
                "volume_15min",
            ]

        if self.encoder is not None:
            expected_ch = getattr(self.encoder, "input_features", None)
            if expected_ch:
                n_actual = len(self.encoder_feature_cols)
                self._encoder_expected_features = int(expected_ch)
                self._encoder_feature_pad = max(0, int(expected_ch) - n_actual)
                if self._encoder_feature_pad > 0:
                    # The missing features are the asset embedding (asset_embed_dim=8),
                    # added by the encoder itself in the forward pass.
                    # No manual padding needed — asset_ids is passed in _get_state.
                    logger.info(
                        "Encoder input: %d raw features + %d asset_embed (total %d) "
                        "— asset_ids passed in the forward",
                        n_actual,
                        self._encoder_feature_pad,
                        int(expected_ch),
                    )
                elif n_actual > int(expected_ch):
                    # too many features: truncate
                    self.encoder_feature_cols = self.encoder_feature_cols[: int(expected_ch)]
                    logger.warning(
                        "Feature mismatch: encoder expects %d, parquet has %d (truncated to %d)",
                        expected_ch,
                        n_actual,
                        int(expected_ch),
                    )
            else:
                self._encoder_expected_features = None
                self._encoder_feature_pad = 0
        else:
            self._encoder_expected_features = None
            self._encoder_feature_pad = 0

        # Core params
        self.seq_len = int(config.get("seq_len", 128))
        self.rr = float(config.get("rr", 1.8))
        self.rr_mr = float(config.get("rr_mr", 0.8))
        self.sl_atr_mult_mr = float(config.get("sl_atr_mult_mr", 1.0))
        self.max_bars_mr = int(config.get("max_bars_mr", 20))
        self.adx_trend_min = float(config.get("adx_trend_min", 0.25))
        self.adx_ranging_max = float(config.get("adx_ranging_max", 0.35))
        self.sl_atr_mult = float(config.get("sl_atr_mult", 1.5))
        self.commission_per_lot = float(config.get("commission_per_lot", 1.0))
        self.pip_value = float(config.get("pip_value", 10.0))
        self.min_lot = float(config.get("min_lot", 0.01))
        self.initial_capital = float(config.get("initial_capital", 100000.0))
        self.max_trades_per_day = int(config.get("max_trades_per_day", 5))
        self.max_dd = float(config.get("max_dd", 0.20))
        self.slippage_noise = float(config.get("slippage_noise", 0.1))
        self.spread_col = config.get("spread_col", "spread_mean_15m_raw")
        self.device = config.get("device", "cpu")
        self.daily_dd_limit = float(config.get("daily_dd_limit", 0.02))
        self.daily_soft_penalty = float(config.get("daily_soft_penalty", 0.2))
        self.max_bars_per_trade = int(config.get("max_bars_per_trade", 48))
        self.exposure_penalty_per_bar = float(config.get("exposure_penalty_per_bar", 0.01))
        self.historical_24h_bars = int(config.get("historical_24h_bars", 96))

        # ✅ dense reward params
        self.mtm_scale = float(config.get("mtm_scale", 0.3))
        self.opp_cost_scale = float(config.get("opp_cost_scale", 0.002))
        self.opp_cost_trend_threshold = float(config.get("opp_cost_trend_threshold", 0.5))
        self.sharpe_bonus_scale = float(config.get("sharpe_bonus_scale", 0.05))
        self.sharpe_window_bars = int(config.get("sharpe_window_bars", 96 * 22))

        # ATR / OHLC columns
        # atr_col_raw    -> real price (SL/TP, size, dist_sl/dist_tp, spread_norm)
        # atr_col_scaled -> StandardScaler z-score (state input of the NN)
        _atr_raw_cands = ["ATR14_15min_raw", "atr14_15min_raw"]
        _atr_scaled_cands = [
            "ATR14_15min",
            "atr14_15min",
            "atr_15min",
            "atr_norm_15m",
            "atr_norm_15min",
        ]
        self.atr_col_raw = next((c for c in _atr_raw_cands if c in self.df.columns), None)
        self.atr_col_scaled = next((c for c in _atr_scaled_cands if c in self.df.columns), None)
        # Legacy alias used in step() to open positions
        self.atr_col = self.atr_col_raw or self.atr_col_scaled

        # high/low MUST be raw columns (real prices) to compare with sl/tp
        # Scale.py writes high_15min_raw and low_15min_raw via RAW_COLUMNS_TO_REPLACE
        self.high = self.df.get("high_15min_raw", self.df.get("high_15min", self.df.iloc[:, 0]))
        self.low = self.df.get("low_15min_raw", self.df.get("low_15min", self.df.iloc[:, 0]))

        if "high_15min_raw" not in self.df.columns:
            logger.warning(
                "high_15min_raw not found — using high_15min (SCALED). "
                "SL/TP checks will be WRONG. Ensure scaled parquet has _raw columns."
            )
        if "low_15min_raw" not in self.df.columns:
            logger.warning(
                "low_15min_raw not found — using low_15min (SCALED). " "SL/TP checks will be WRONG."
            )

        self.close = self.df.get("close_15min_raw", self.df.get("close_15min", self.df.iloc[:, 0]))
        self._close_raw_col = "close_15min_raw" if "close_15min_raw" in self.df.columns else None

        self.timestamps = list(self.df.index.to_pydatetime())
        self.trades_approved = 0
        self.trades_rejected = 0

        self.reset()

        logger.info(
            "MarketEnv (PPO) init: features=%d seq_len=%d mode=%s " "mtm=%.2f opp=%.4f",
            len(self.encoder_feature_cols),
            self.seq_len,
            self.training_mode,
            self.mtm_scale,
            self.opp_cost_scale,
        )

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self, start_idx: Optional[int] = None):
        if start_idx is None:
            start_idx = self.seq_len - 1

        self.position = None
        self.bankrupt = False
        self.current_step = int(start_idx)
        self.step_counter = 0
        self.trades_today = 0

        if self.training_mode == "scratch":
            self.balance = float(self.initial_capital)
            self.equity = float(self.initial_capital)
            self.equity_curve = [self.balance]
            self.drawdown = 0.0
            self.drawdown_history = []
            self.max_drawdown_seen = 0.0
            self.daily_dd_history = []
            self.episode_trades = []
            self.episode_rewards = []
            self._bar_returns = []
        else:
            for attr, default in [
                ("balance", float(self.initial_capital)),
                ("equity_curve", [float(self.initial_capital)]),
                ("drawdown", 0.0),
                ("drawdown_history", []),
                ("max_drawdown_seen", 0.0),
                ("daily_dd_history", []),
                ("episode_trades", []),
                ("episode_rewards", []),
                ("_bar_returns", []),
            ]:
                if not hasattr(self, attr):
                    setattr(self, attr, default)

        self.daily_date = self.timestamps[min(self.current_step, len(self.timestamps) - 1)].date()
        self.last_dd_date = self.daily_date
        self.daily_start_balance = float(self.balance)
        self.daily_peak_balance = float(self.balance)
        self.daily_dd = 0.0

        self._last_z = None
        self._last_dd = 0.0
        self._last_closed_result = None
        self._last_close_price = None

        if hasattr(self, "_regime_history_1h"):
            self._regime_history_1h.clear()

        # Reset new v2 tracking vars
        self._current_trend_1h = 0.0
        self._current_trend_4h = 0.0
        self._current_regime_4h = 0.0
        self._trend_conf_15m = 0.0
        self._trend_conf_1h = 0.0
        self._last_vol_pred_15m = 0.0
        self._last_vol_pred_1h = 0.0
        self._last_trade_rr = 0.0
        self._last_trade_win = 0.0

        return self._get_state()

    def set_portfolio_context(
        self,
        open_positions_norm: float = 0.0,
        daily_pnl_pct: float = 0.0,
        other_class_active: float = 0.0,
        trades_today_norm: float = 0.0,
    ) -> None:
        """
        Set joint training portfolio context. Called by the runner before
        each _get_state() call. In single-class training all values stay 0.

        Args:
            open_positions_norm: positions open / max_concurrent in [0,1]
            daily_pnl_pct:       portfolio daily P&L%, clipped to [-0.05, 0.05]
            other_class_active:  1.0 if other class has active position(s), else 0
            trades_today_norm:   trades_opened_today / max_trades_per_day in [0,1]
        """
        self._port_open_positions_norm = float(np.clip(open_positions_norm, 0.0, 1.0))
        self._port_daily_pnl_pct = float(np.clip(daily_pnl_pct, -0.05, 0.05))
        self._port_other_class_active = float(np.clip(other_class_active, 0.0, 1.0))
        self._port_trades_today_norm = float(np.clip(trades_today_norm, 0.0, 1.0))

    def soft_reset(self):
        self.step_counter = 0
        self.trades_today = 0
        self.daily_start_balance = float(self.balance)
        self.daily_peak_balance = float(self.balance)
        self.daily_dd = 0.0
        self.episode_rewards = []
        self._last_z = None
        self._last_dd = 0.0

    # ------------------------------------------------------------------
    # ✅ PPO: PRIVILEGED STATE (Asymmetric Critic — TRAINING ONLY)
    # ------------------------------------------------------------------

    def get_privileged_state(self, n_future: int = 16) -> np.ndarray:
        """
        Future close deltas normalized by ATR.

        Used ONLY to train the critic. NEVER in inference/eval.

        Returns:
            np.ndarray (n_future,) float32
            each element = (close[t+i] - close[t]) / atr[t], clamped to [-3, 3]
            zero-padded near the end of the dataset.
        """
        idx = min(self.current_step, len(self.df) - 1)

        atr = max(float(self.df.iloc[idx].get(self.atr_col, 1e-6)), 1e-6) if self.atr_col else 1e-4

        close_now = (
            float(self.df[self._close_raw_col].iloc[idx])
            if self._close_raw_col
            else float(self.close.iloc[idx])
        )

        result = np.zeros(n_future, dtype=np.float32)
        for i in range(1, n_future + 1):
            fi = idx + i
            if fi >= len(self.df):
                break
            close_fut = (
                float(self.df[self._close_raw_col].iloc[fi])
                if self._close_raw_col
                else float(self.close.iloc[fi])
            )
            result[i - 1] = float(np.clip((close_fut - close_now) / atr, -3.0, 3.0))

        return result

    # ------------------------------------------------------------------
    # ✅ PPO: DENSE REWARD — Mark-to-Market
    # ------------------------------------------------------------------

    def _mtm_reward(self, idx: int) -> float:
        """
        Intermediate reward on an open position, computed every bar.

        unrealized_pnl / risk_amount * mtm_scale * decay → range [-0.5, 0.5]

        MTM decay: the MTM reward shrinks linearly with the number of
        bars in the position, so the agent can't hack the reward by
        keeping positions open forever to collect small rewards
        without ever closing.

        decay = max(min_decay, 1 - bars_open / max_bars_decay)
        Example with max_bars_decay=48:
          bar 0  → decay=1.00 (full reward)
          bar 24 → decay=0.50
          bar 48 → decay=min_decay (0.1)
        """
        if self.position is None:
            return 0.0

        close_now = (
            float(self.df[self._close_raw_col].iloc[idx])
            if self._close_raw_col
            else float(self.close.iloc[idx])
        )

        entry = float(self.position["entry"])
        size = float(self.position["size"])
        risk_amount = max(float(self.position.get("risk_amount", 1.0)), 1e-6)

        unrealized = (
            (close_now - entry) * size
            if self.position["dir"] == "LONG"
            else (entry - close_now) * size
        )

        # MTM decay: pushes towards closing, avoids holding forever
        bars_open = int(self.position.get("open_bars", 0))
        max_bars_decay = float(self.cfg.get("mtm_max_bars_decay", 48))  # ~12h
        min_decay = float(self.cfg.get("mtm_min_decay", 0.1))
        decay = max(min_decay, 1.0 - bars_open / max(1.0, max_bars_decay))

        return float(np.clip((unrealized / risk_amount) * self.mtm_scale * decay, -0.5, 0.5))

    # ------------------------------------------------------------------
    # ✅ PPO: DENSE REWARD — Opportunity Cost
    # ------------------------------------------------------------------

    def _opportunity_cost_penalty(self, idx: int) -> float:
        """
        Penalty for HOLD in a strong trend agreed by several TFs.
        v2: fires only if 15m AND 1h point the same way (avoids false positives
        in a sideways 15m market with 1h trending). Scaled by confidence.
        """
        if self.position is not None:
            return 0.0
        if self.trades_today >= self.max_trades_per_day:
            return 0.0

        t15 = self._current_trend_15m
        t1h = self._current_trend_1h
        # needs 15m + 1h to agree on the direction
        if abs(t15) < self.opp_cost_trend_threshold:
            return 0.0
        if t15 * t1h <= 0:  # opposite signs or one is zero -> no penalty
            return 0.0

        if self._close_raw_col and idx > 0:
            close_now = float(self.df[self._close_raw_col].iloc[idx])
            close_prev = float(self.df[self._close_raw_col].iloc[idx - 1])
            atr = (
                max(float(self.df.iloc[idx].get(self.atr_col, 1e-6)), 1e-6)
                if self.atr_col
                else 1e-4
            )
            move = abs(close_now - close_prev) / atr
        else:
            move = 0.0

        # scaled by the mean 15m+1h confidence: high conf -> full penalty
        conf_scale = (self._trend_conf_15m + self._trend_conf_1h) / 2.0
        return -self.opp_cost_scale * move * conf_scale

    # ------------------------------------------------------------------
    # ✅ PPO: SHARPE BONUS (periodic)
    # ------------------------------------------------------------------

    def _maybe_sharpe_bonus(self) -> float:
        """
        Sharpe bonus every sharpe_window_bars steps.
        Rewards the quality of the return stream, not only the raw PnL.
        """
        if len(self._bar_returns) < max(30, self.sharpe_window_bars // 4):
            return 0.0
        if self.step_counter % self.sharpe_window_bars != 0:
            return 0.0

        r = np.array(self._bar_returns[-self.sharpe_window_bars :], dtype=np.float32)
        sharpe = float(np.mean(r)) / (float(np.std(r)) + 1e-8) * np.sqrt(252 * 96)
        return float(np.clip(self.sharpe_bonus_scale * max(0.0, sharpe - 1.0), 0.0, 0.2))

    # ------------------------------------------------------------------
    # UTILITY (unchanged)
    # ------------------------------------------------------------------

    # ────────────────────────────────────────────────────────────────────
    # Time-based force close + entry lockout — UTC-based
    # All times in UTC (GMT) since parquet timestamps are UTC. No DST issues.
    #
    # Force close (positions closed by the env):
    #   • Mon-Thu 21:30 UTC: "overnight" — 30 min before the swap rollover (22:00 GMT)
    #   • Fri      21:30 UTC: "friday_weekend" — avoids the weekend gap
    #   • Saturday: defensive (the FX feed should already be closed)
    #
    # Entry lockout (no new positions):
    #   • Mon-Thu 20:30 UTC → 21:30 UTC: 1h before the close
    #   • Friday  20:30 UTC → 21:30 UTC: 1h before the weekly close
    #   • Sunday  before 22:00 UTC: market reopen — illiquid
    #   • Saturday: closed
    # ────────────────────────────────────────────────────────────────────
    # ── Action space constants (v26: 5 actions) ────────────────────────
    # ── Swap tables (pip cost per standard lot — plausible worst case) ──
    # Wednesday = triple swap (×3). Negative values = cost.
    _SWAP_PIPS: dict = {
        "AUDUSD": {"LONG": 4.0, "SHORT": 2.0},
        "EURUSD": {"LONG": 11.0, "SHORT": 1.0},
        "GBPUSD": {"LONG": 6.0, "SHORT": 4.0},
        "USDCAD": {"LONG": 2.0, "SHORT": 12.0},
        "USDCHF": {"LONG": 1.0, "SHORT": 15.0},
        "EURAUD": {"LONG": 11.0, "SHORT": 4.0},
        "EURGBP": {"LONG": 8.0, "SHORT": 3.0},
        "EURNZD": {"LONG": 18.0, "SHORT": 7.0},
        "GBPAUD": {"LONG": 14.0, "SHORT": 6.0},
        "NZDUSD": {"LONG": 5.0, "SHORT": 1.0},
    }
    _PIP_VALUE_PER_LOT: float = 10.0  # USD per pip per standard lot (approximation)
    _SWAP_APPLY_HM_UTC = (22, 0)  # NY close rollover

    # NEW v26+: daily force close REMOVED for Mon/Tue/Thu.
    # Kept ONLY for Wednesday (triple swap) and Friday (weekend).
    ACT_HOLD = 0
    ACT_LONG_TREND = 1
    ACT_SHORT_TREND = 2
    ACT_LONG_MR = 3
    ACT_SHORT_MR = 4
    N_ACTIONS = 5
    _ACT_STRATEGY = {0: "hold", 1: "trend", 2: "trend", 3: "mr", 4: "mr"}
    _ACT_DIR = {0: None, 1: "LONG", 2: "SHORT", 3: "LONG", 4: "SHORT"}

    _UTC_TZ = timezone.utc
    _DAILY_CLOSE_HM_UTC = (21, 30)  # Wed/Fri force-close only
    _DAILY_LOCKOUT_HM_UTC = (20, 30)  # no entries 1h pre Wed/Fri close
    _SUNDAY_REOPEN_LOCKOUT_UTC = (22, 0)  # Sunday before 22:00: no entries

    def _ts_utc(self, idx: int) -> Optional[tuple[int, int, int]]:
        """Return (hour, minute, dayofweek) in UTC. Parquet is already UTC."""
        if idx >= len(self.timestamps):
            return None
        ts = self.timestamps[idx]
        try:
            # Parquet timestamps are UTC; if naive, treat as UTC
            return int(ts.hour), int(ts.minute), int(ts.dayofweek)
        except Exception:
            return None

    def _is_time_force_close(self, idx: int) -> Optional[str]:
        """
        Force close ONLY on Wednesday (triple swap) and Friday (weekend gap).
        Mon/Tue/Thu: no force close — a trade can stay open overnight.

        Returns: "wednesday_swap" | "friday_weekend" | None
        """
        info = self._ts_utc(idx)
        if info is None:
            return None
        h, m, dow = info  # 0=Mon ... 4=Fri ... 5=Sat ... 6=Sun
        if dow == 5:
            return "friday_weekend"
        if dow == 4 and (h, m) >= self._DAILY_CLOSE_HM_UTC:
            return "friday_weekend"
        if dow == 2 and (h, m) >= self._DAILY_CLOSE_HM_UTC:
            return "wednesday_swap"
        return None

    def _is_entry_lockout(self, idx: int) -> bool:
        """
        Entry lockout 1h before the close ONLY on Wednesday and Friday.
        Mon/Tue/Thu: no lockout — entries allowed until the end of the NY session.

        Lockout on:
          • Wednesday 20:30 → 21:30 UTC: triple swap coming
          • Friday    20:30 → 21:30 UTC: weekly close
          • Sunday    before 22:00 UTC: illiquid market reopen
          • Saturday: closed
        """
        info = self._ts_utc(idx)
        if info is None:
            return False
        h, m, dow = info
        if dow == 5:
            return True  # Saturday: closed
        if dow == 6 and (h, m) < self._SUNDAY_REOPEN_LOCKOUT_UTC:
            return True  # Sunday before 22:00
        # Wednesday (2) and Friday (4): 1h lockout before the close
        if dow in (2, 4) and (h, m) >= self._DAILY_LOCKOUT_HM_UTC:
            return True
        return False

    def _spread_time_multiplier(ts_utc_hour: int, dow: int) -> float:
        """
        Returns spread multiplier ≥ 1.0 based on UTC hour and DOW.
        Captures: Asia thin liquidity, NY close, Sunday reopen, etc.
        Most active 06:00-21:00 UTC: multiplier 1.0 (London + NY overlap).
        """
        # Sunday 22:00-23:59 UTC: market reopen — wide spreads
        if dow == 6 and ts_utc_hour >= 22:
            return 3.5
        # Monday 00:00-04:59 UTC: Asia early week, still thin
        if dow == 0 and ts_utc_hour < 5:
            return 1.8
        # Daily 21:00-21:59 UTC: NY close → spread widens
        if ts_utc_hour == 21:
            return 1.4
        # Daily 22:00-23:59 UTC: rollover/Asia open — wide
        if ts_utc_hour >= 22:
            return 2.5
        # Daily 00:00-05:59 UTC: Asia thin
        if ts_utc_hour < 6:
            return 1.3
        # London/NY active: normal
        return 1.0

    def _get_spread(self, idx: int) -> float:
        """
        Returns realistic spread for the bar:
          base (rolling avg) × time_multiplier × volatility_multiplier
        - time_multiplier: low-liquidity windows (Asia open, NY close, etc.)
        - volatility_multiplier: 1.0 + 0.5·max(0, (vol_15m - 1)) when vol > 1σ
          captures spread widening on news/macro spikes
        """
        if self.spread_col in self.df.columns:
            val = self.df.iloc[idx].get(self.spread_col, 0.0)
            base = 0.0 if pd.isna(val) else float(val)
        else:
            base = float(self.cfg.get("default_spread", 0.0001))

        # Time-of-day multiplier (DST-aware would be marginal — UTC is sufficient
        # since liquidity windows are tied to global session opens, not local time).
        try:
            ts = self.timestamps[idx]
            if ts.tzinfo is None:
                h_utc = int(ts.hour)
                dow = int(ts.dayofweek)
            else:
                ts_utc = ts.astimezone(timezone.utc) if ts.tzinfo != timezone.utc else ts
                h_utc = int(ts_utc.hour)
                dow = int(ts_utc.dayofweek)
            time_mult = self._spread_time_multiplier(h_utc, dow)
        except Exception:
            time_mult = 1.0

        # Volatility-driven widening: when vol_15m > 1σ above mean (z-score)
        # spread widens proportionally. Captures news/macro spikes.
        vol_mult = 1.0
        try:
            vol_z = float(self.df.iloc[idx].get("volatility_15min", 0.0))
            if vol_z > 1.0:
                vol_mult = 1.0 + 0.5 * min(vol_z - 1.0, 4.0)  # cap at 1+0.5*4 = 3.0
        except Exception:
            pass

        return base * time_mult * vol_mult

    def _compute_position_size(self, atr_value: float, risk_amount: float) -> float:
        if risk_amount <= 0:
            return self.min_lot
        pip_size = float(self.cfg.get("pip_size", 0.0001))
        atr_pips = atr_value / max(1e-12, pip_size)
        sl_pips = atr_pips * self.sl_atr_mult
        lots = risk_amount / max(1e-12, sl_pips * self.pip_value)
        return min(float(self.cfg.get("max_lot", 10.0)), max(self.min_lot, round(lots, 2)))

    def get_valid_action_mask(self, max_trades_override=None):
        """5-action mask: [HOLD, LONG_TREND, SHORT_TREND, LONG_MR, SHORT_MR]"""
        if self.training_mode == "train" and self.cfg.get("mask_mode", "explore") == "explore":
            return [True, True, True, True, True]

        max_trades = max_trades_override or self.max_trades_per_day
        in_position = self.position is not None
        can_trade = (not in_position) and (self.trades_today < max_trades)

        remaining_dd = max(0.0, self.daily_dd_limit - self.daily_dd)
        risk_pct = self.cfg.get("trade_risk_pct", 0.005)
        max_trades_left = int(
            math.floor((self.balance * remaining_dd) / max(1e-12, self.balance * risk_pct))
        )
        if max_trades_left <= 0:
            can_trade = False

        adx_4h = float(getattr(self, "_adx_4h", -1.0))
        if adx_4h < 0:
            trend_ok = can_trade
            mr_ok = can_trade
        else:
            trend_ok = can_trade and (adx_4h >= self.adx_trend_min)
            mr_ok = can_trade and (adx_4h < self.adx_ranging_max)

        return [True, trend_ok, trend_ok, mr_ok, mr_ok]

    def _open_position(
        self,
        direction: str,
        price: float,
        atr: float,
        idx: int,
        external_risk_amount: Optional[float] = None,
        strategy: str = "trend",
    ) -> bool:
        if atr is None or atr <= 1e-8:
            return False

        spread = self._get_spread(idx)
        slippage = spread + np.random.normal(0, abs(spread * self.slippage_noise))
        entry_price = price + slippage if direction == "LONG" else price - slippage

        risk_target = (
            float(external_risk_amount)
            if external_risk_amount is not None
            else self.balance * self.cfg.get("trade_risk_pct", 0.005)
        )

        # Commission accounting: SINGLE charge per trade (entry only, not round-trip).
        # operational risk = risk_target - commission, so an SL hit = -risk_target
        # net P&L (commission included), not -risk_target - extra commission.
        # Example: risk_target=$50, commission=$3 → risk_operational=$47, size computed on $47
        # SL hit: raw loss $47 + commission $3 = -$50 total = exactly -risk_target.
        size_est = self._compute_position_size(atr, risk_target)
        commission_charge = size_est * self.commission_per_lot  # one-time entry charge
        risk_operational = max(0.0, risk_target - commission_charge)
        size = self._compute_position_size(atr, risk_operational)

        _rr_use = self.rr_mr if strategy == "mr" else self.rr
        _sl_use = self.sl_atr_mult_mr if strategy == "mr" else self.sl_atr_mult
        _bars_use = self.max_bars_mr if strategy == "mr" else self.max_bars_per_trade
        sl = entry_price - atr * _sl_use if direction == "LONG" else entry_price + atr * _sl_use
        tp = (
            entry_price + atr * _sl_use * _rr_use
            if direction == "LONG"
            else entry_price - atr * _sl_use * _rr_use
        )

        self.balance -= size * self.commission_per_lot
        if self.balance <= 0:
            self.balance = 0.0
            self.bankrupt = True
            return False

        self.position = {
            "dir": direction,
            "entry": entry_price,
            "sl": sl,
            "tp": tp,
            "size": size,
            "risk_amount": risk_operational,
            "entry_idx": idx,
            "open_bars": 0,
            "strategy": strategy,
            "rr": _rr_use,
            "max_bars": _bars_use,
            "max_favorable_excursion": 0.0,  # MFE: max unrealized PnL / risk_amount
            "max_adverse_excursion": 0.0,  # MAE: max unrealized loss / risk_amount ← v2
        }
        self._last_close_price = entry_price
        return True

    def _close_position(self, rr_result: float) -> tuple[float, float, int]:
        if self.position is None:
            return 0.0, 0.0, 0

        risk_amount = float(self.position.get("risk_amount", 0.0))
        if risk_amount <= 0:
            risk_amount = self.initial_capital * self.cfg.get("trade_risk_pct", 0.005)
            logger.warning("risk_amount missing, fallback: %.2f", risk_amount)

        pnl = rr_result * risk_amount
        open_bars = int(self.position.get("open_bars", 0))

        # commission charged ONLY at the entry. Not again at the close.
        # one charge per trade — already in risk_operational.
        self.balance += pnl
        if self.balance <= 0:
            self.balance = 0.0
            self.bankrupt = True

        self.position = None
        self._last_close_price = None
        return float(pnl), float(risk_amount), int(open_bars)

    def update_arbiter_counters(self, info: dict[str, Any]):
        if not info:
            return
        if info.get("approved_by_arbiter", False):
            self.trades_approved += 1
        if info.get("proposed_but_rejected", False):
            self.trades_rejected += 1

    def _check_exit(self, idx: int) -> float:
        if self.position is None:
            return 0.0
        high = float(self.high.iloc[idx])
        low = float(self.low.iloc[idx])
        d = self.position["dir"]

        # spread at the close: every exit pays half the spread (bid-ask on the exit side)
        # the opening spread is already in entry_price.
        # at TP (limit order) the fill is at the exact price -> no extra spread.
        # The SL (stop order) fills at market → pays spread + gap slippage.
        spread_raw = self._get_spread(idx)

        def _sl_fill_penalty(self_obj) -> float:
            """
            Extra slippage on the SL: models gap-through and spread widening
            when liquidity is thin (news, session open, fast market).
            Gap model: if the low/high goes past the SL by more than 30% of the ATR,
            the bar counts as "gapped" and gets extra slippage.
            Result: 0.5 × spread  for a normal bar,
                       1.5 × spread  per gap ≥ 0.3 × ATR
            """
            atr = (
                float(self_obj.df.iloc[idx].get(self_obj.atr_col_raw, 1e-4))
                if self_obj.atr_col_raw
                else 1e-4
            )
            sl_price = float(self_obj.position["sl"])
            dir_sign = 1.0 if d == "LONG" else -1.0
            gap = dir_sign * (sl_price - (low if d == "LONG" else high))
            gap_frac = gap / max(atr, 1e-8)
            base_slip = 0.5 * spread_raw
            extra = base_slip * min(gap_frac / 0.3, 2.0) if gap_frac > 0.3 else 0.0
            return base_slip + extra

        # ── SL/TP same-bar resolution ─────────────────────────────────
        # When both SL and TP fall within [low, high] of the same bar, we cannot
        # know which was hit first from OHLC data alone (no tick data available).
        # CONVENTION: SL has priority (worst-case / conservative).
        # In live, this corresponds to a stop order being filled before a limit
        # at execution venue when both prices are crossed in fast-market.
        # Rationale: in 15-min bars during news/spikes, ~70-80% of the time SL
        # fires first because price spikes against the position before reverting.
        # Conservative bias: ~5% PF underestimate during volatile periods.
        # ──────────────────────────────────────────────────────────────
        if d == "LONG":
            sl_hit = low <= self.position["sl"]
            tp_hit = high >= self.position["tp"]
            if sl_hit and tp_hit:
                # Track for diagnostics — useful to verify in OOS volatility
                self._sl_tp_collision_count = getattr(self, "_sl_tp_collision_count", 0) + 1
            if sl_hit:
                slip = _sl_fill_penalty(self)
                # worse effective rr: SL hit + slippage
                sl_price = float(self.position["sl"])
                entry = float(self.position["entry"])
                risk_dist = abs(entry - sl_price)
                # pnl = -(risk_dist + slip) / risk_dist × risk_amount → rr < -1
                rr_actual = -(risk_dist + slip) / max(risk_dist, 1e-8)
                rr_actual = float(np.clip(rr_actual, -3.0, -1.0))
                pnl, ra, ob = self._close_position(rr_actual)
                self.episode_trades.append(
                    {
                        "result": "sl",
                        "pnl": pnl,
                        "risk_amount": ra,
                        "rr": rr_actual,
                        "entry_idx": None,
                        "exit_idx": idx,
                        "duration_bars": ob,
                    }
                )
                self._last_trade_rr = rr_actual
                self._last_trade_win = 0.0
                self._last_closed_result = "sl"
                return rr_actual
            if high >= self.position["tp"]:
                pos_rr = float(self.position.get("rr", self.rr))
                pnl, ra, ob = self._close_position(pos_rr)
                self.episode_trades.append(
                    {
                        "result": "tp",
                        "pnl": pnl,
                        "risk_amount": ra,
                        "rr": pos_rr,
                        "entry_idx": None,
                        "exit_idx": idx,
                        "duration_bars": ob,
                    }
                )
                self._last_trade_rr = pos_rr
                self._last_trade_win = 1.0
                self._last_closed_result = "tp"
                return pos_rr
        else:  # SHORT
            sl_hit = high >= self.position["sl"]
            tp_hit = low <= self.position["tp"]
            if sl_hit and tp_hit:
                self._sl_tp_collision_count = getattr(self, "_sl_tp_collision_count", 0) + 1
            if sl_hit:
                slip = _sl_fill_penalty(self)
                sl_price = float(self.position["sl"])
                entry = float(self.position["entry"])
                risk_dist = abs(sl_price - entry)
                rr_actual = -(risk_dist + slip) / max(risk_dist, 1e-8)
                rr_actual = float(np.clip(rr_actual, -3.0, -1.0))
                pnl, ra, ob = self._close_position(rr_actual)
                self.episode_trades.append(
                    {
                        "result": "sl",
                        "pnl": pnl,
                        "risk_amount": ra,
                        "rr": rr_actual,
                        "entry_idx": None,
                        "exit_idx": idx,
                        "duration_bars": ob,
                    }
                )
                self._last_trade_rr = rr_actual
                self._last_trade_win = 0.0
                self._last_closed_result = "sl"
                return rr_actual
            if low <= self.position["tp"]:
                pos_rr = float(self.position.get("rr", self.rr))
                pnl, ra, ob = self._close_position(pos_rr)
                self.episode_trades.append(
                    {
                        "result": "tp",
                        "pnl": pnl,
                        "risk_amount": ra,
                        "rr": pos_rr,
                        "entry_idx": None,
                        "exit_idx": idx,
                        "duration_bars": ob,
                    }
                )
                self._last_trade_rr = pos_rr
                self._last_trade_win = 1.0
                self._last_closed_result = "tp"
                return pos_rr

        self._last_closed_result = None
        return 0.0

    def _update_drawdown(self):
        prev_dd = getattr(self, "drawdown", 0.0)
        self.drawdown = max(
            0.0, (self.initial_capital - self.balance) / max(1e-12, self.initial_capital)
        )
        if not hasattr(self, "drawdown_history"):
            self.drawdown_history = []
        if not hasattr(self, "drawdown_trailing_history"):
            self.drawdown_trailing_history = []
        self.drawdown_history.append(self.drawdown)
        if self.drawdown > getattr(self, "max_drawdown_seen", 0.0):
            self.max_drawdown_seen = self.drawdown
        self._last_dd = prev_dd
        if not hasattr(self, "peak_equity"):
            self.peak_equity = self.balance
        self.peak_equity = max(self.peak_equity, self.balance)
        self.drawdown_trailing = max(
            0.0, (self.peak_equity - self.balance) / max(1e-12, self.peak_equity)
        )
        self.drawdown_trailing_history.append(self.drawdown_trailing)

    # ------------------------------------------------------------------
    # _get_state — UNCHANGED from the DDQN version
    # ------------------------------------------------------------------

    def _get_state(self) -> np.ndarray:
        idx = min(self.current_step, len(self.df) - 1)
        start = idx - self.seq_len + 1

        if start >= 0:
            block = self.df.iloc[start : idx + 1][self.encoder_feature_cols].values.astype(
                np.float32
            )
        else:
            block = np.zeros((self.seq_len, len(self.encoder_feature_cols)), dtype=np.float32)
            n_ex = idx + 1
            if n_ex > 0:
                block[-n_ex:] = self.df.iloc[0 : idx + 1][self.encoder_feature_cols].values.astype(
                    np.float32
                )

        # No padding needed: the missing features (asset_embed_dim=8)
        # are added by the encoder itself through the asset_embedding lookup

        x_input = torch.from_numpy(block).unsqueeze(0).to(self.device).float()

        if self.encoder is not None:
            with torch.no_grad():
                # asset_id for the AssetEmbedding: encoder trained with n_assets=5, embed_dim=8
                # the TCN expects n raw features + 8 embedding channels
                _asset_ids = None
                if getattr(self.encoder, "asset_embedding", None) is not None:
                    _asset_ids = torch.tensor([self.asset_id], dtype=torch.long, device=self.device)
                emb = self.encoder(x_input, asset_ids=_asset_ids)
                n_tfs = len(self.model_tf_order)
                tf_index = {tf: i for i, tf in enumerate(self.model_tf_order)}

                # ── TrendMomentumHeadTF: ATR-normalized scalar per TF ─────
                # range [-3, 3]: positive=bullish, negative=bearish, ~0=flat
                # WF3 Majors: 30m corr=0.82, 1h=0.73, 4h=0.69, 1d=0.73
                # 15min: NOT used (left out of the loss, collapses to a bull bias)
                if self.trend_mom_head is not None:
                    tmom = self.trend_mom_head(emb)  # [1, n_tfs]
                    tmom = tmom.squeeze(0)  # [n_tfs]
                    tmom_30m = float(tmom[tf_index["30min"]].item())
                    tmom_1h = float(tmom[tf_index["1h"]].item())
                    tmom_4h = float(tmom[tf_index["4h"]].item())
                    tmom_1d = float(tmom[tf_index["1d"]].item())
                elif self.trend_head is not None:
                    # fallback: 3-class argmax → center-zero scalar
                    tl = self.trend_head(emb)
                    tl = tl.view(1, n_tfs, tl.shape[-1] // n_tfs)
                    tp_ = torch.argmax(tl, dim=-1).squeeze(0)
                    tmom_30m = float(tp_[tf_index["30min"]].item() - 1)
                    tmom_1h = float(tp_[tf_index["1h"]].item() - 1)
                    tmom_4h = float(tp_[tf_index["4h"]].item() - 1)
                    tmom_1d = float(tp_[tf_index["1d"]].item() - 1)
                else:
                    tmom_30m = tmom_1h = tmom_4h = tmom_1d = 0.0

                # ── RegimeContinuousHead: [vol_regime, atr_norm, adx, di_direction] per TF ─
                # v25: 4 signals (was 3). idx: 0=vol_regime 1=atr_norm 2=adx 3=di_direction
                # corr v25.1: vol~0.93 atr~0.90 adx~0.72-0.81 di_dir~0.93-0.95
                if self.regime_cont_head is not None:
                    rc = self.regime_cont_head(emb)  # [1, n_tfs*n_sig] or [1, n_tfs, n_sig]
                    rc = rc.squeeze(0)  # [n_tfs*n_sig] or [n_tfs, n_sig]
                    if rc.dim() == 1:
                        # auto-detect n_signals from the flat size (4 for v25, 3 for legacy)
                        _n_sig = rc.numel() // n_tfs
                        rc = rc.view(n_tfs, _n_sig)  # → [n_tfs, n_sig]
                    _n_sig = rc.shape[1]
                    # idx 0=vol_regime, 1=atr_norm, 2=adx, 3=di_direction (v25)
                    vr_30m = float(rc[tf_index["30min"], 0].item())
                    vr_1h = float(rc[tf_index["1h"], 0].item())
                    vr_4h = float(rc[tf_index["4h"], 0].item())
                    an_30m = float(rc[tf_index["30min"], 1].item())
                    an_4h = float(rc[tf_index["4h"], 1].item())
                    adx_30m = float(rc[tf_index["30min"], 2].item())
                    adx_4h = float(rc[tf_index["4h"], 2].item())
                    # di_direction in [-1, 1]: +1=strong bull, -1=strong bear, ~0=flat
                    di_dir_30m = float(rc[tf_index["30min"], 3].item()) if _n_sig >= 4 else 0.0
                    di_dir_4h = float(rc[tf_index["4h"], 3].item()) if _n_sig >= 4 else 0.0
                    # v26+: di_dir_1h (corr=0.942±0.006) and di_dir_1d (corr=0.961±0.011)
                    # di_direction is the most stable signal of the whole cycle (std≤0.011)
                    # 1h fills the 30m->4h gap; 1d adds multi-day structural context
                    di_dir_1h = float(rc[tf_index["1h"], 3].item()) if _n_sig >= 4 else 0.0
                    di_dir_1d = float(rc[tf_index["1d"], 3].item()) if _n_sig >= 4 else 0.0
                else:
                    vr_30m = vr_1h = vr_4h = 0.5
                    an_30m = an_4h = adx_30m = adx_4h = 0.5
                    di_dir_30m = di_dir_4h = 0.0

                # ── MarketStructureHead: [RANGING, TRENDING, BREAKOUT] per TF ─
                # brk_auc ~0.87 (reliable) — softmax BREAKOUT prob on 1h and 4h
                if self.mktstr_head is not None:
                    ms = self.mktstr_head(emb)  # [1, n_tfs*3] or [1, n_tfs, 3]
                    ms = ms.squeeze(0)  # [n_tfs*3] or [n_tfs, 3]
                    if ms.dim() == 1:  # Head auto-built → flat
                        ms = ms.view(n_tfs, 3)  # → [n_tfs, 3]
                    ms = torch.softmax(ms, dim=-1)  # [n_tfs, 3]
                    # v25: classes [0=RANGING, 1=LONG_BRK, 2=SHORT_BRK]
                    # P(any breakout) = 1-P(RANGING) = P(LONG_BRK)+P(SHORT_BRK)
                    # bug fix: it used to take only ms[:,2] (SHORT_BRK only)
                    brk_prob_1h = float((ms[tf_index["1h"], 1] + ms[tf_index["1h"], 2]).item())
                    brk_prob_4h = float((ms[tf_index["4h"], 1] + ms[tf_index["4h"], 2]).item())
                    # v26+: brk_prob_30m — more stable than 1h and 4h (std=0.003)
                    brk_prob_30m = float(
                        (ms[tf_index["30min"], 1] + ms[tf_index["30min"], 2]).item()
                    )
                else:
                    brk_prob_1h = brk_prob_4h = brk_prob_30m = 0.33

                # ── vol_head: future volatility (for sizing) ─────────────────
                if self.vol_head is not None:
                    vp = self.vol_head(emb)  # [1, n_tfs]
                    vol_pred_15m = float(vp[0, tf_index["15min"]].item())
                    vol_pred_1h = float(vp[0, tf_index["1h"]].item())
                else:
                    vol_pred_15m = vol_pred_1h = 0.0

                z = emb.squeeze(0).cpu().numpy().astype(np.float32)
        else:
            z = np.zeros(int(self.cfg.get("emb_dim_fallback", 64)), dtype=np.float32)
            tmom_30m = tmom_1h = tmom_4h = tmom_1d = 0.0
            vr_30m = vr_1h = vr_4h = 0.5
            an_30m = an_4h = adx_30m = adx_4h = 0.5
            di_dir_30m = di_dir_4h = di_dir_1h = di_dir_1d = 0.0
            brk_prob_1h = brk_prob_4h = brk_prob_30m = 0.33
            vol_pred_15m = vol_pred_1h = 0.0

        self._last_z = z
        # v24: scalar trend momentum (replaces 3-class OHE)
        self._tmom_30m = tmom_30m
        self._tmom_1h = tmom_1h
        self._tmom_4h = tmom_4h
        self._tmom_1d = tmom_1d
        # v24: continuous regime (replaces 3-class argmax)
        self._vr_30m = vr_30m  # vol_regime 30m — corr ~0.93
        self._vr_1h = vr_1h
        self._vr_4h = vr_4h
        self._an_30m = an_30m  # atr_norm 30m — corr ~0.88
        self._an_4h = an_4h  # atr_norm 4h  — corr ~0.95
        self._adx_30m = adx_30m  # adx 30m      — corr ~0.72
        self._adx_4h = adx_4h  # adx 4h       — corr ~0.76-0.81 (v25.1)
        self._di_dir_30m = di_dir_30m  # di_direction 30m — corr ~0.93 (v25)
        self._di_dir_4h = di_dir_4h  # di_direction 4h  — corr ~0.95 (v25)
        self._di_dir_1h = di_dir_1h  # di_direction 1h  — corr ~0.942±0.006 (v26)
        self._di_dir_1d = di_dir_1d  # di_direction 1d  — corr ~0.961±0.011 (v26)
        # v24: breakout probability from MarketStructureHead
        self._brk_prob_1h = brk_prob_1h  # brk_auc ~0.871±0.009
        self._brk_prob_4h = brk_prob_4h  # brk_auc ~0.873±0.006
        self._brk_prob_30m = brk_prob_30m  # brk_auc ~0.859±0.003 — more stable (v26)
        self._last_vol_pred_15m = vol_pred_15m
        self._last_vol_pred_1h = vol_pred_1h
        # Legacy compat — arbiter still reads _current_trend_15m/_current_regime_1h
        self._current_trend_15m = tmom_30m  # remap: use 30m as primary signal
        self._current_trend_30m = tmom_30m
        self._current_trend_1h = tmom_1h
        self._current_trend_4h = tmom_4h
        self._current_regime_15m = vr_30m  # continuous [0,1] instead of 0/1/2
        self._current_regime_30m = vr_30m
        self._current_regime_1h = vr_1h
        self._current_regime_4h = vr_4h
        self._regime_history_1h.append(vr_1h)

        # ── Market indicators ─────────────────────────────────────────
        # BUG FIX v2: the real parquet columns are volatility_15min / volatility_1H
        # (StandardScaler -> z-score, already normalized), not vol_15m / vol_1h which don't exist
        vol_15m = float(self.df.iloc[idx].get("volatility_15min", 0.0))
        vol_1h = float(self.df.iloc[idx].get("volatility_1H", 0.0))

        # NEW v2: Volatility_regime = vol_roll / vol_mean (RobustScaler)
        # > 0  -> vol above its median (rising)
        # < 0  -> vol below its median (falling)
        vol_regime_15m = float(self.df.iloc[idx].get("Volatility_regime_15min", 0.0))
        vol_regime_1h = float(self.df.iloc[idx].get("Volatility_regime_1H", 0.0))

        # atr_norm for the NN: scaled column (StandardScaler z-score) -> range ~[-3,3]
        # atr_ref for price maths: raw column (e.g. 0.00120 for EURUSD)
        atr_norm = (
            float(self.df.iloc[idx].get(self.atr_col_scaled, 0.0)) if self.atr_col_scaled else 0.0
        )
        atr_ref_raw = (
            float(self.df.iloc[idx].get(self.atr_col_raw, 1e-4)) if self.atr_col_raw else 1e-4
        )
        atr_ref_raw = max(atr_ref_raw, 1e-6)

        ts = pd.to_datetime(self.timestamps[idx])
        bkt = [0] * 6
        bkt[min(5, ts.hour // 4)] = 1
        dow = [1 if i == ts.weekday() else 0 for i in range(7)]

        deltas = [
            float(self.df.iloc[idx].get(n, 0.0)) if n in self.df.columns else 0.0
            for n in [
                "delta_EMA50_1D_15min",
                "delta_EMA20_4H_15min",
                "delta_RSI14_4H_15min",
                "delta_CCI20_4H_15min",
            ]
        ]

        start_step = max(0, idx - self.historical_24h_bars)
        recent = sum(
            1 for t in getattr(self, "episode_trades", []) if (t.get("exit_idx") or 0) >= start_step
        )
        trades_norm = recent / max(1.0, self.max_trades_per_day)

        rp = 0
        for pr in reversed(self._regime_history_1h):
            if abs(pr - vr_1h) < 0.05:
                rp += 1  # continuous: same regime within tolerance
            else:
                break

        def oht(t):
            v = np.zeros(3, dtype=np.float32)
            n = int(t + 1)
            if 0 <= n < 3:
                v[n] = 1.0
            return v

        # ── Position-aware features ──────────────────────────────────
        # The agent must know WHERE it is in its current position.
        # Without these it can't learn exit timing,
        # intra-trade risk management, caution after a DD.

        atr_ref = atr_ref_raw  # raw ATR to normalize distances (same space as the prices)

        # NEW v2: spread normalized by the raw ATR -> relative cost of opening
        # the spread column is RAW-ONLY (not scaled) -> divided directly by the raw ATR
        spread_col = self.cfg.get("spread_col", "spread_mean_15m_raw")
        spread_raw = float(self.df.iloc[idx].get(spread_col, 0.0))
        spread_norm = float(np.clip(spread_raw / (atr_ref + 1e-9), 0.0, 1.0))

        if self.position is not None:
            pos_entry = float(self.position["entry"])
            pos_size = float(self.position["size"])
            pos_risk = max(float(self.position.get("risk_amount", 1.0)), 1e-6)
            close_now_raw = (
                float(self.df[self._close_raw_col].iloc[idx])
                if self._close_raw_col
                else float(self.close.iloc[idx])
            )
            unreal = (
                (close_now_raw - pos_entry) * pos_size
                if self.position["dir"] == "LONG"
                else (pos_entry - close_now_raw) * pos_size
            )
            unrealized_pnl_norm = float(np.clip(unreal / pos_risk, -3.0, 3.0))

            # bars since the entry -> [0, 1+]
            bars_open = int(self.position.get("open_bars", 0))
            bars_norm = float(
                np.tanh(bars_open / 96.0)
            )  # tanh: 24h→0.6, 48h→0.9 — no hard max_bars

            # MFE — max favourable excursion, normalized by ATR×sl_mult → [0, 3]
            mfe_raw = float(self.position.get("max_favorable_excursion", 0.0))
            mfe_norm = float(np.clip(mfe_raw / (atr_ref * self.sl_atr_mult + 1e-9), 0.0, 3.0))

            # MAE — max loss seen, normalized by ATR×sl_mult -> [0, 3]  ← v2
            # MAE = lowest unrealized PnL / risk_amount seen during the trade
            # updated in step() together with MFE
            mae_raw = float(self.position.get("max_adverse_excursion", 0.0))
            mae_norm = float(np.clip(mae_raw / (atr_ref * self.sl_atr_mult + 1e-9), 0.0, 3.0))

            # distance from SL and TP normalized by ATR  ← v2
            # sl/tp are raw prices (computed with ATR14_15min_raw in _open_position)
            # atr_ref is ATR14_15min_raw -> same unit -> dimensionless ratio
            # LONG:  dist_sl = (close - sl) / atr   [positive = safe distance]
            #        dist_tp = (tp - close) / atr   [positive = how far the TP still is]
            # SHORT: signs flipped, same logic
            sl_price = float(self.position["sl"])
            tp_price = float(self.position["tp"])
            if self.position["dir"] == "LONG":
                dist_sl = float(np.clip((close_now_raw - sl_price) / (atr_ref + 1e-9), -3.0, 3.0))
                dist_tp = float(np.clip((tp_price - close_now_raw) / (atr_ref + 1e-9), -3.0, 3.0))
            else:
                dist_sl = float(np.clip((sl_price - close_now_raw) / (atr_ref + 1e-9), -3.0, 3.0))
                dist_tp = float(np.clip((close_now_raw - tp_price) / (atr_ref + 1e-9), -3.0, 3.0))

            # Direction one-hot: [is_long, is_short]
            pos_dir = np.array(
                [1.0, 0.0] if self.position["dir"] == "LONG" else [0.0, 1.0], dtype=np.float32
            )
        else:
            unrealized_pnl_norm = 0.0
            bars_norm = 0.0
            mfe_norm = 0.0
            mae_norm = 0.0  # ← v2
            dist_sl = 0.0  # ← v2
            dist_tp = 0.0  # ← v2
            pos_dir = np.array([0.0, 0.0], dtype=np.float32)

        # daily DD as a fraction of the limit -> [0, 1.5]
        daily_dd_limit_ref = max(float(self.cfg.get("daily_dd_limit", 0.02)), 1e-6)
        daily_dd_pct = float(np.clip(self.daily_dd / daily_dd_limit_ref, 0.0, 1.5))

        # Trades left today → [0, 1]
        max_trades = float(self.max_trades_per_day)
        trades_remaining_norm = float(
            np.clip((max_trades - self.trades_today) / max(1.0, max_trades), 0.0, 1.0)
        )

        # result of the last closed trade: rr clipped [-1, rr_max] -> norm, and win flag  ← v2
        # lets the agent recalibrate its confidence after SL/TP streaks
        last_rr_norm = float(np.clip(self._last_trade_rr / max(self.rr, 1e-6), -1.0, 1.0))
        last_win = float(self._last_trade_win)  # 0.0 o 1.0

        # ── Joint training context (v25) ─────────────────────────────
        # set by the runner via env.set_portfolio_context() before get_state.
        # Lets the agent see the global state: cross-class positions,
        # portfolio P&L, asset usage. 0 when not in joint mode.
        port_open_pos = float(getattr(self, "_port_open_positions_norm", 0.0))
        port_daily_pnl = float(getattr(self, "_port_daily_pnl_pct", 0.0))
        port_other_active = float(getattr(self, "_port_other_class_active", 0.0))
        port_trades_today = float(getattr(self, "_port_trades_today_norm", 0.0))

        # ── STATE VECTOR (186 dim) — v25 with joint context ─────────
        # dim   0-127: encoder embedding                             128
        # ── Encoder head outputs (continuous, not OHE) ──────────────────
        # dim 128-131: tmom [30m, 1h, 4h, 1d] scalars in [-3,3]      4
        # dim 132-134: vol_regime [30m, 1h, 4h] in [0,1]             3
        # dim 135-136: atr_norm [30m, 4h] in [0,1]                   2
        # dim 137-138: adx [30m, 4h] in [0,1]                        2
        # dim 139-140: di_direction [30m, 4h] in [-1,1]              2
        # dim 141-142: mktstr brk_prob [1h, 4h] softmax in [0,1]      2
        # ── Market features ─────────────────────────────────────────
        # dim 143:     daily_dd                                        1
        # dim 144-145: volatility_15min + volatility_1H               2
        # dim 146-147: Volatility_regime_15min + _1H                   2
        # dim 148:     ATR_norm (scaled)                               1
        # dim 149:     spread_norm                                     1
        # dim 150-155: hour_bucket OHE                                 6
        # dim 156-162: day_of_week OHE                                 7
        # dim 163-164: trades_norm + regime_persistence                2
        # dim 165-168: delta cross-TF                                  4
        # ── Position features ───────────────────────────────────────
        # dim 169:     unrealized_pnl_norm                             1
        # dim 170:     bars_norm                                        1
        # dim 171:     mfe_norm                                         1
        # dim 172:     mae_norm                                         1
        # dim 173-174: pos_dir [long, short]                           2
        # dim 175-176: dist_sl + dist_tp                               2
        # dim 177:     daily_dd_pct                                     1
        # dim 178:     trades_remaining_norm                            1
        # ── v26: StrategyOutcomeHead — 12 explicit features ──────────────────
        # gives the DQN a direct signal on "which strategy fits now"
        # without having to decode it from the embedding alone.
        #
        # p_tp_tl: P(TP) TREND LONG   = P(hit) × P(TP|hit)
        # p_tp_ts: P(TP) TREND SHORT
        # p_tp_ml: P(TP) MR LONG      (SL = TP = mr_r×ATR, symmetric)
        # p_tp_ms: P(TP) MR SHORT
        # best_trend_tp = max(p_tp_tl, p_tp_ts)
        # best_mr_tp    = max(p_tp_ml, p_tp_ms)
        # trend_vs_mr   = best_trend_tp - best_mr_tp  (>0 → TREND favoured)
        # dir_edge      = max(p_tp_tl, p_tp_ml) - max(p_tp_ts, p_tp_ms)
        p_tp_tl = p_tp_ts = p_tp_ml = p_tp_ms = 0.25  # neutral default
        p_hit_tl = p_hit_ts = p_hit_ml = p_hit_ms = 0.5
        if self.strategy_outcome_head is not None and self._last_z is not None:
            try:
                with torch.no_grad():
                    _z_t = torch.tensor(
                        self._last_z, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    # ctx_dim detected from the architecture: zero context is passed.
                    # The full context (tmom, regime, mktstr, timing) was used in
                    # training but isn't free at inference without
                    # a second forward pass. With ctx=0 the probabilities are slightly
                    # degraded but the hit/side structure stays informative.
                    _ctx_dim_soh = getattr(self.strategy_outcome_head, "ctx_dim", 0)
                    _ctx = (
                        torch.zeros(1, _ctx_dim_soh, device=self.device, dtype=torch.float32)
                        if _ctx_dim_soh > 0
                        else None
                    )
                    _soh_out = self.strategy_outcome_head.predict_all_combos(
                        _z_t, extra_context=_ctx
                    )
                    p_hit_tl = float(_soh_out["p_hit_trend_long"][0].item())
                    p_hit_ts = float(_soh_out["p_hit_trend_short"][0].item())
                    p_hit_ml = float(_soh_out["p_hit_mr_long"][0].item())
                    p_hit_ms = float(_soh_out["p_hit_mr_short"][0].item())
                    p_tp_tl = float(_soh_out["p_tp_trend_long"][0].item())
                    p_tp_ts = float(_soh_out["p_tp_trend_short"][0].item())
                    p_tp_ml = float(_soh_out["p_tp_mr_long"][0].item())
                    p_tp_ms = float(_soh_out["p_tp_mr_short"][0].item())
            except Exception:
                pass
        best_trend_tp = max(p_tp_tl, p_tp_ts)
        best_mr_tp = max(p_tp_ml, p_tp_ms)
        trend_vs_mr = float(np.clip(best_trend_tp - best_mr_tp, -1.0, 1.0))
        dir_edge = float(np.clip(max(p_tp_tl, p_tp_ml) - max(p_tp_ts, p_tp_ms), -1.0, 1.0))
        # cache for the arbiter (read in step_once without recomputing)
        self._strat_p_tp_tl = p_tp_tl
        self._strat_p_tp_ts = p_tp_ts
        self._strat_p_tp_ml = p_tp_ml
        self._strat_p_tp_ms = p_tp_ms
        self._strat_p_hit_tl = p_hit_tl
        self._strat_p_hit_ts = p_hit_ts
        self._strat_p_hit_ml = p_hit_ml
        self._strat_p_hit_ms = p_hit_ms
        self._strat_best_trend = best_trend_tp
        self._strat_best_mr = best_mr_tp
        self._strat_trend_vs_mr = trend_vs_mr
        self._strat_dir_edge = dir_edge

        # ── v26: bars_to_force_close — trading session left ────────────
        # how many 15min bars are left before the next force close (21:30 UTC)?
        # used as a state feature, as an arbiter penalty and as a hard gate.
        _fc_h, _fc_m = self._DAILY_CLOSE_HM_UTC  # (21, 30)
        _ts_now = pd.to_datetime(self.timestamps[idx])
        if _ts_now.tzinfo is not None:

            _ts_utc = _ts_now.tz_convert("UTC")
        else:
            _ts_utc = _ts_now
        _cur_min = _ts_utc.hour * 60 + _ts_utc.minute
        _close_min = _fc_h * 60 + _fc_m
        if _cur_min < _close_min:
            _delta_min = _close_min - _cur_min
        else:
            _delta_min = (24 * 60 - _cur_min) + _close_min
        _bars_to_fc = max(1, _delta_min // 15)  # 15min bars to the force close
        bars_to_fc_norm = float(np.clip(_bars_to_fc / 96.0, 0.0, 1.0))  # norm su 24h
        is_near_fc = 1.0 if _bars_to_fc < 16 else 0.0  # <4h = danger zone
        # Cache for the arbiter
        self._bars_to_fc = _bars_to_fc
        self._bars_to_fc_norm = bars_to_fc_norm
        self._is_near_fc = is_near_fc

        # ── State vector assembly ─────────────────────────────────────────────
        # dim 0-127:    encoder embedding                             128
        # dim 128-131:  tmom (4 TF: 30m,1h,4h,1d)                    4
        # dim 132-134:  vol_regime (3 TF: 30m,1h,4h)                  3
        # dim 135-136:  atr_norm (2 TF: 30m,4h)                       2
        # dim 137-138:  adx (2 TF: 30m,4h)                            2
        # dim 139-142:  di_direction (4 TF: 30m,4h,1h,1d)             4  ← +2 vs v25
        # dim 143-145:  brk_prob (3 TF: 1h,4h,30m)                    3  ← +1 vs v25
        # dim 146-165:  market features (vol, atr, spread, bkt, dow…) 20
        # dim 166-173:  position features                               8
        # dim 174-181:  daily_dd, trades, rp, deltas                   8
        # dim 182-185:  last_rr, last_win, vol_pred                    4
        # dim 186-189:  portfolio context (joint)                       4
        # dim 190-201:  v26 StrategyOutcomeHead features               12
        #   p_hit_tl, p_hit_ts, p_hit_ml, p_hit_ms  (P(hit) per combo)
        #   p_tp_tl, p_tp_ts, p_tp_ml, p_tp_ms      (P(TP) per combo)
        #   best_trend_tp, best_mr_tp, trend_vs_mr, dir_edge
        # dim 202-203:  session validity (bars_to_fc_norm, is_near_fc) 2
        # TOTAL: 204 dims
        state = np.concatenate(
            [
                z.astype(np.float32),  # 128
                np.array([tmom_30m, tmom_1h, tmom_4h, tmom_1d], dtype=np.float32),  #   4
                np.array([vr_30m, vr_1h, vr_4h], dtype=np.float32),  #   3
                np.array([an_30m, an_4h], dtype=np.float32),  #   2
                np.array([adx_30m, adx_4h], dtype=np.float32),  #   2
                np.array(
                    [di_dir_30m, di_dir_4h, di_dir_1h, di_dir_1d], dtype=np.float32
                ),  #   4  ← +2
                np.array([brk_prob_1h, brk_prob_4h, brk_prob_30m], dtype=np.float32),  #   3  ← +1
                np.array([self.daily_dd], dtype=np.float32),  #   1
                np.array([vol_15m, vol_1h], dtype=np.float32),  #   2
                np.array([vol_regime_15m, vol_regime_1h], dtype=np.float32),  #   2
                np.array([atr_norm], dtype=np.float32),  #   1
                np.array([spread_norm], dtype=np.float32),  #   1
                np.array(bkt, dtype=np.float32),  #   6
                np.array(dow, dtype=np.float32),  #   7
                np.array([trades_norm, float(rp)], dtype=np.float32),  #   2
                np.array(deltas, dtype=np.float32),  #   4
                np.array([unrealized_pnl_norm], dtype=np.float32),  #   1
                np.array([bars_norm], dtype=np.float32),  #   1
                np.array([mfe_norm], dtype=np.float32),  #   1
                np.array([mae_norm], dtype=np.float32),  #   1
                pos_dir,  #   2
                np.array([dist_sl, dist_tp], dtype=np.float32),  #   2
                np.array([daily_dd_pct], dtype=np.float32),  #   1
                np.array([trades_remaining_norm], dtype=np.float32),  #   1
                np.array([last_rr_norm, last_win], dtype=np.float32),  #   2
                np.array([vol_pred_15m, vol_pred_1h], dtype=np.float32),  #   2
                # Joint portfolio context (zero in single-class training, populated in joint)
                np.array(
                    [port_open_pos, port_daily_pnl, port_other_active, port_trades_today],
                    dtype=np.float32,
                ),  # 4
                # v26: StrategyOutcomeHead — 12 explicit TREND/MR features
                np.array(
                    [
                        p_hit_tl,
                        p_hit_ts,
                        p_hit_ml,
                        p_hit_ms,  # P(hit) x4
                        p_tp_tl,
                        p_tp_ts,
                        p_tp_ml,
                        p_tp_ms,  # P(TP)  x4
                        best_trend_tp,
                        best_mr_tp,  # best per strategy
                        trend_vs_mr,
                        dir_edge,
                    ],  # edge signals
                    dtype=np.float32,
                ),  # 12
                # v26: session validity — the DQN sees the time left directly
                np.array([bars_to_fc_norm, is_near_fc], dtype=np.float32),  # 2
            ]
        ).astype(np.float32)

        return state

    # ────────────────────────────────────────────────────────────────────
    # SWAP MECHANICS (v26+)
    # ────────────────────────────────────────────────────────────────────

    def _get_swap_pips(self, direction: str) -> float:
        """Swap pips for the current direction (absolute value)."""
        pair = getattr(self, "asset", "EURUSD").upper()
        table = self._SWAP_PIPS.get(pair, {"LONG": 5.0, "SHORT": 5.0})
        return float(table.get(direction, 5.0))

    def _apply_swap_cost(self, idx: int) -> float:
        """
        Applies the swap cost when the position goes through the NY close (22:00 UTC).
        Wednesday: triple swap (×3).

        Returns a negative reward proportional to the cost in R.

        Formula:
          swap_R = (swap_pips × pip_value_per_lot × lot_size) / risk_amount
                 = swap_pips / sl_pips           ← simplified for USD-quoted
          sl_pips = sl_atr_dist / pip_size

        Non-USD-quoted pairs are approximated with a fixed pip_value_per_lot ($10/pip/lot).
        """
        if self.position is None:
            return 0.0
        info = self._ts_utc(idx)
        if info is None:
            return 0.0
        h, m, dow = info
        # the swap is applied once a night at 22:00 UTC (NY close rollover)
        if (h, m) != self._SWAP_APPLY_HM_UTC:
            return 0.0

        direction = self.position["dir"]
        swap_pips = self._get_swap_pips(direction)
        size = float(self.position.get("size", 0.0))
        risk_amount = max(float(self.position.get("risk_amount", 1.0)), 1.0)

        # Cost in $ = swap_pips × pip_value × lots
        swap_cost_usd = swap_pips * self._PIP_VALUE_PER_LOT * size
        # Wednesday: triple swap
        if dow == 2:
            swap_cost_usd *= 3.0
        # Convert to R
        swap_cost_r = swap_cost_usd / risk_amount
        # apply to the balance
        self.balance = max(0.0, self.balance - swap_cost_usd)

        # update the days-held counter and the swap accumulator
        days_held = int(self.position.get("open_bars", 0)) // 96 + 1  # ~96 bars/day @15min
        self.position["swap_cost_r"] = float(self.position.get("swap_cost_r", 0.0)) + swap_cost_r
        self.position["days_held"] = max(int(self.position.get("days_held", 0)), days_held)
        return -swap_cost_r  # negative reward

    def _swap_open_penalty(self, idx: int) -> float:
        """
        Extra penalty for trades that 'hit' the swap: rewards not opening
        close to the rollover. Proportional to the days already held.
        More days = more penalty (the agent learns not to open late).

        Only if the trade has already gone through at least one rollover (days_held>=1).
        """
        if self.position is None:
            return 0.0
        days_held = int(self.position.get("days_held", 0))
        if days_held < 1:
            return 0.0
        # base penalty × days held — pushes to close before the 2nd/3rd swap
        base_swap_pen = float(self.cfg.get("swap_hold_penalty_per_day_r", 0.03))
        return -base_swap_pen * days_held  # e.g. -0.03R after 1 day, -0.06R after 2 days

    def _hold_reward(self, idx: int) -> float:
        """
        Reward for HOLD without an open position.

        Two parts:
        1. legacy confidence-based (embedding norm) — unchanged
        2. v26+: regime-uncertainty bonus — rewards HOLD when the encoder
           signals are ambiguous / conflicting (transition market).
           Teaches the agent that waiting is right when the regime is unclear,
           instead of opening random trades "to do something".

        Uncertainty score ∈ [0, 1]:
          - High (→ 1.0): flat / contradictory signals → HOLD is right → bonus
          - Low (→ 0.0): clear signals → HOLD is a missed chance → 0
        """
        # warmup: zero HOLD reward in the first N steps so the buffer isn't polluted
        early_cutoff = int(self.cfg.get("hold_reward_warmup_steps", 5_000))
        if self.training_mode == "train" and self.step_counter < early_cutoff:
            return 0.0

        # ── Component 1: embedding confidence (legacy) ───────────────────
        r = 0.0
        z_conf = float(np.linalg.norm(self._last_z)) if self._last_z is not None else 0.0
        lo = self.cfg.get("low_conf_threshold", 5.0)
        hi = self.cfg.get("high_conf_threshold", 20.0)
        if z_conf < lo:
            r += float(self.cfg.get("hold_reward_good", 0.005))
        elif z_conf > hi:
            r -= float(self.cfg.get("hold_reward_bad", 0.010))

        # Bonus low-volatility
        vol15 = float(self.df.iloc[idx].get("volatility_15min", 0.0))
        if vol15 < float(self.cfg.get("low_vol_threshold", -0.5)):
            r += float(self.cfg.get("hold_low_vol_bonus", 0.010))

        # ── Component 2: regime uncertainty bonus (v26+) ─────────────────
        # how ambiguous / flat the encoder signals are:
        #   |tmom_4h|   low → weak trend
        #   |di_dir_4h| low -> unclear direction
        #   |brk_4h - 0.5| low -> market between ranging and breakout (transition)
        # bonus scaled: max when everything is ambiguous, 0 when everything is clear.
        _uncertainty_bonus = float(self.cfg.get("hold_uncertainty_bonus", 0.0))
        _uncertainty_thr = float(self.cfg.get("hold_uncertainty_thr", 0.60))
        if _uncertainty_bonus > 0:
            _tmom_abs = float(np.clip(abs(self._tmom_4h or 0.0) / 1.5, 0.0, 1.0))
            _di_abs = float(np.clip(abs(self._di_dir_4h or 0.0), 0.0, 1.0))
            _brk_mid = float(np.clip(abs((self._brk_prob_4h or 0.33) - 0.5) * 2, 0.0, 1.0))
            # uncertainty = 1 - signal clarity
            _clarity = _tmom_abs * 0.40 + _di_abs * 0.40 + _brk_mid * 0.20
            _uncertainty = float(np.clip(1.0 - _clarity, 0.0, 1.0))
            if _uncertainty > _uncertainty_thr:
                # bonus proportional to the uncertainty: max +0.020 when everything is flat
                r += (
                    _uncertainty_bonus
                    * (_uncertainty - _uncertainty_thr)
                    / (1.0 - _uncertainty_thr)
                )

        return r

    # ------------------------------------------------------------------
    # STEP — dense reward integrated
    # ------------------------------------------------------------------

    def step(self, action: int, external_risk_amount: Optional[float] = None):
        if self.current_step >= len(self.df) - 1:
            return self._get_state(), 0.0, True, {"terminated": "end_of_data"}

        reward = 0.0
        done = False
        info: dict[str, Any] = {}
        idx = self.current_step
        ts = self.timestamps[idx]

        # Day change
        if ts.date() != self.daily_date:
            if not hasattr(self, "daily_dd_history"):
                self.daily_dd_history = []
            if self.daily_date != getattr(self, "last_dd_date", None):
                self.daily_dd_history.append(self.daily_dd)
                self.last_dd_date = self.daily_date
            self.trades_today = 0
            self.daily_date = ts.date()
            self.daily_start_balance = float(self.balance)
            self.daily_peak_balance = float(self.balance)
            self.daily_dd = 0.0

        mask = self.get_valid_action_mask()
        if self.training_mode != "train":
            if not mask[action]:
                reward -= 0.05
                action = 0

        if self.position is not None:
            self.position["open_bars"] = int(self.position.get("open_bars", 0)) + 1
            # update the Max Favorable Excursion (MFE) for the state features
            close_cur = (
                float(self.df[self._close_raw_col].iloc[idx])
                if self._close_raw_col
                else float(self.close.iloc[idx])
            )
            pos_entry = float(self.position["entry"])
            pos_size = float(self.position["size"])
            pos_risk = max(float(self.position.get("risk_amount", 1.0)), 1e-6)
            cur_unreal = (
                (close_cur - pos_entry) * pos_size
                if self.position["dir"] == "LONG"
                else (pos_entry - close_cur) * pos_size
            )
            cur_mfe = cur_unreal / pos_risk
            self.position["max_favorable_excursion"] = max(
                float(self.position.get("max_favorable_excursion", 0.0)), cur_mfe
            )
            # MAE: max loss seen (absolute value, always ≥ 0)  ← v2
            # cur_unreal < 0 -> losing -> MAE grows
            self.position["max_adverse_excursion"] = max(
                float(self.position.get("max_adverse_excursion", 0.0)),
                -cur_unreal / pos_risk,  # positive when losing
            )

        # Check exit
        if self.position:
            reward += self._check_exit(idx)
        else:
            self._last_closed_result = None

        if getattr(self, "_last_closed_result", None) is not None:
            info["closed_trade_result"] = self._last_closed_result
            self._last_closed_result = None

        # Open position
        if not self.position and self.trades_today < self.max_trades_per_day:
            if action in (1, 2, 3, 4):
                if self._is_entry_lockout(idx):
                    info["entry_blocked"] = "time_lockout"
                    reward += self._hold_reward(idx)
                else:
                    price = float(self.close.iloc[idx])
                    atrv = (
                        float(self.df.iloc[idx].get(self.atr_col_raw, 0.0))
                        if self.atr_col_raw
                        else 0.0
                    )
                    direc = self._ACT_DIR[action]
                    strategy = self._ACT_STRATEGY[action]
                    if self._open_position(
                        direc, price, atrv, idx, external_risk_amount, strategy=strategy
                    ):
                        self.trades_today += 1
            else:
                reward += self._hold_reward(idx)
        else:
            if not self.position:
                reward += self._hold_reward(idx)

        # ✅ PPO DENSE REWARDS
        reward += self._mtm_reward(idx)
        if action == 0:
            reward += self._opportunity_cost_penalty(idx)

        # Drawdown update
        self._update_drawdown()

        if getattr(self, "bankrupt", False):
            done = True
            reward -= 5.0
            info["kill_reason"] = "bankruptcy"
            return self._get_state(), reward, done, info

        if self.position is not None and self._close_raw_col:
            close_cur = float(self.df[self._close_raw_col].iloc[idx])
            pos_entry = float(self.position["entry"])
            pos_size = float(self.position["size"])
            unrealized = (
                (close_cur - pos_entry) * pos_size
                if self.position["dir"] == "LONG"
                else (pos_entry - close_cur) * pos_size
            )
        else:
            unrealized = 0.0

        equity_now = self.balance + unrealized

        self.daily_peak_balance = max(getattr(self, "daily_peak_balance", equity_now), equity_now)
        self.daily_dd = max(
            0.0, (self.daily_peak_balance - equity_now) / max(1e-12, self.daily_peak_balance)
        )

        if self.position is not None:
            ob = int(self.position.get("open_bars", 0))

            # ── Time-based force close (v26+: Wednesday and Friday only) ────
            # Mon/Tue/Thu: no force close — a trade can stay open overnight.
            # Wednesday 21:30: "wednesday_swap" (avoids the triple swap)
            # Friday    21:30: "friday_weekend" (avoids the weekend gap)
            # entry lockout: only Wed/Fri 20:30->21:30 + Sunday before 22:00
            time_force_close = self._is_time_force_close(idx)
            if time_force_close is not None:
                close_force = (
                    float(self.df[self._close_raw_col].iloc[idx])
                    if self._close_raw_col
                    else float(self.close.iloc[idx])
                )
                spread_raw = self._get_spread(idx)
                entry = float(self.position["entry"])
                size = float(self.position["size"])
                pos_risk = max(float(self.position.get("risk_amount", 1.0)), 1e-6)
                if self.position["dir"] == "LONG":
                    raw_pnl = (close_force - spread_raw - entry) * size
                else:
                    raw_pnl = (entry - close_force - spread_raw) * size
                rr_force = float(np.clip(raw_pnl / pos_risk, -3.0, 3.0))

                # v26+: force-close penalty — teaches the agent not to hold
                # positions until the force close. Proportional to the lost RR.
                _fc_pen_scale = float(self.cfg.get("force_close_penalty_scale", 0.10))
                _rr_vs_tp = float(self.position.get("rr", 1.8))
                _rr_gap = max(0.0, _rr_vs_tp - abs(rr_force)) if rr_force > 0 else 0.0
                force_close_pen = -_fc_pen_scale * max(0.0, _rr_gap)
                reward += force_close_pen

                # swap cost accumulated in the position
                swap_cost_r = float(self.position.get("swap_cost_r", 0.0))
                days_held = int(self.position.get("days_held", 0))

                pnl, ra, _ = self._close_position(rr_force)
                self.episode_trades.append(
                    {
                        "result": time_force_close,
                        "pnl": pnl,
                        "risk_amount": ra,
                        "rr": rr_force,
                        "entry_idx": None,
                        "exit_idx": idx,
                        "duration_bars": ob,
                        # v26+: tracking force-close per JSON stats
                        "is_force_close": True,
                        "force_close_reason": time_force_close,
                        "swap_cost_r": swap_cost_r,
                        "days_held": days_held,
                    }
                )
                self._last_trade_rr = rr_force
                self._last_trade_win = 1.0 if rr_force > 0 else 0.0
                self._last_closed_result = time_force_close
                info["closed_trade_result"] = time_force_close
                info["is_force_close"] = True
                # update the force-close counters for the stats
                self._force_close_count = getattr(self, "_force_close_count", 0) + 1
                self._force_close_pnl = getattr(self, "_force_close_pnl", 0.0) + pnl
                self._force_close_rr_sum = getattr(self, "_force_close_rr_sum", 0.0) + rr_force
            # ── v26: no max_bars timeout ─────────────────────────────────
            # positions live until TP, SL or the time force close (above).
            # very soft zombie penalty for trades open > 5 days (300 bars @15min):
            # tells the Q-network that holding forever isn't optimal,
            # without forcing arbitrary exits like max_bars did.
            else:
                # v26+: swap cost applied once a night at 22:00 UTC
                _swap_r = self._apply_swap_cost(idx)
                if _swap_r != 0.0:
                    reward += _swap_r
                    # extra penalty for multi-day trades (discourages excessive holding)
                    reward += self._swap_open_penalty(idx)

            if ob > 300:  # 300 × 15min = 75h ≈ 3 working days
                reward -= self.exposure_penalty_per_bar * 0.3  # very soft vs the old -2.0

        # ✅ Periodic Sharpe bonus
        reward += self._maybe_sharpe_bonus()

        # Track the bar return
        if len(self.equity_curve) > 0:
            br = (self.balance - self.equity_curve[-1]) / max(1e-12, self.equity_curve[-1])
            self._bar_returns.append(float(br))
            if len(self._bar_returns) > self.sharpe_window_bars * 2:
                self._bar_returns = self._bar_returns[-self.sharpe_window_bars :]

        self.equity_curve.append(self.balance)
        if not hasattr(self, "episode_rewards"):
            self.episode_rewards = []
        self.episode_rewards.append(reward)

        self.current_step += 1
        self.step_counter += 1

        if self.drawdown > self.max_dd:
            done = True
            # scaled penalty: proportional to how far past the limit
            # -1.0 base + up to -4.0 extra for a very high DD
            # Example: max_dd=0.20, drawdown=0.25 → overshoot=0.05/0.20=0.25 → penalty=-2.0
            overshoot = max(0.0, self.drawdown - self.max_dd) / max(self.max_dd, 1e-6)
            kill_penalty = -(1.0 + min(overshoot * 4.0, 4.0))
            reward += kill_penalty
            info["kill_reason"] = "drawdown_exceeded"
            info["kill_penalty"] = float(kill_penalty)

        if getattr(self, "bankrupt", False):
            reward -= 10.0 * (1 - self.balance / self.initial_capital)

        return self._get_state(), float(reward), bool(done), info

    # ------------------------------------------------------------------
    # RENDER (unchanged)
    # ------------------------------------------------------------------

    def render_last_trade_summary(self):
        trades = getattr(self, "episode_trades", [])
        n = len(trades)
        wins = sum(1 for t in trades if t.get("pnl", 0.0) > 0)
        losses = n - wins
        sum_profit = sum(float(t.get("pnl", 0.0)) for t in trades if t.get("pnl", 0.0) > 0)
        sum_loss = -sum(float(t.get("pnl", 0.0)) for t in trades if t.get("pnl", 0.0) < 0)
        pnl = float(sum_profit - sum_loss)
        win_rate = (wins / n) if n > 0 else 0.0
        pf = (
            (sum_profit / (sum_loss + 1e-12))
            if sum_loss > 0
            else (float("inf") if sum_profit > 0 else 0.0)
        )
        final_bal = float(getattr(self, "balance", self.initial_capital))
        init_cap = float(getattr(self, "initial_capital", final_bal))
        max_dd_s = float(getattr(self, "max_drawdown_seen", 0.0))
        dth = getattr(self, "drawdown_trailing_history", [])
        ddh = getattr(self, "daily_dd_history", [])
        rr_vals = [float(t.get("rr", 0.0)) for t in trades if float(t.get("rr", 0.0)) != 0.0]
        avg_rr = float(np.mean(rr_vals)) if rr_vals else 0.0
        exp = 0.0
        if n > 0:
            aw = (sum_profit / max(1, wins)) if wins > 0 else 0.0
            al = (sum_loss / max(1, losses)) if losses > 0 else 0.0
            exp = (win_rate * aw) - ((1 - win_rate) * al)
        mdt = float(max(dth)) if dth else 0.0
        mdd = float(max(ddh)) if ddh else 0.0
        return {
            "initial_capital": int(init_cap),
            "trades": int(n),
            "wins": int(wins),
            "losses": int(losses),
            "win_rate": float(win_rate),
            "profit_factor": float(pf),
            "sum_profit": float(sum_profit),
            "sum_loss": float(sum_loss),
            "pnl": float(pnl),
            "final_balance": float(final_bal),
            "max_dd_static": float(max_dd_s),
            "max_daily_dd": float(mdd),
            "avg_daily_dd": float(np.mean(ddh)) if ddh else 0.0,
            "return_over_max_dd": pnl / (init_cap * max_dd_s) if max_dd_s > 0 else 0.0,
            "max_dd_trailing": float(mdt),
            "avg_dd_trailing": float(np.mean(dth)) if dth else 0.0,
            "return_over_trailing_dd": pnl / (init_cap * mdt) if mdt > 0 else 0.0,
            "avg_rr_realized": float(avg_rr),
            "expectancy": float(exp),
            "trades_approved": int(getattr(self, "trades_approved", 0)),
            "trades_rejected": int(getattr(self, "trades_rejected", 0)),
            "sl_tp_collision_count": int(getattr(self, "_sl_tp_collision_count", 0)),
            "force_close_count": int(getattr(self, "_force_close_count", 0)),
            "force_close_pnl": float(getattr(self, "_force_close_pnl", 0.0)),
        }


if __name__ == "__main__":
    logger.info("env_fx (PPO-upgraded) module loaded.")
