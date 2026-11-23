"""
components.py — model, augmentations and losses of the market encoder.

TCN + GRU backbone with a learnable asset embedding, the supervised heads,
augmentations, contrastive losses, projection head, memory bank and the
contrastive pretrainer.

Checkpoints store state_dicts: attribute names here must not change.
"""

import logging
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.utils.checkpoint import checkpoint as _torch_ckpt

    def torch_checkpoint(fn, *args, **kwargs):
        kwargs.pop("use_reentrant", None)
        return _torch_ckpt(fn, *args, use_reentrant=False)

except Exception:

    def torch_checkpoint(fn, *args, **kwargs):
        kwargs.pop("use_reentrant", None)
        return fn(*args)


logger = logging.getLogger("encoder_trainer")


# ----------------------------
# Helpers: validation & diagnostics
# ----------------------------
def _tensor_stats(t):
    try:
        if t is None:
            return "missing"
        if isinstance(t, torch.Tensor):
            a = t.detach().cpu().numpy()
        else:
            a = np.asarray(t)
    except Exception as e:
        return f"uninspectable({e})"
    if a.size == 0:
        return {"shape": tuple(a.shape), "empty": True}
    d = {}
    d["shape"] = tuple(a.shape)
    d["dtype"] = str(a.dtype)
    try:
        d["min"] = float(np.nanmin(a))
        d["max"] = float(np.nanmax(a))
        d["mean"] = float(np.nanmean(a))
    except Exception:
        d["min"] = None
        d["max"] = None
        d["mean"] = None
    try:
        d["nan_count"] = int(np.isnan(a).sum())
    except Exception:
        d["nan_count"] = -1
    try:
        d["inf_count"] = int(np.isinf(a).sum())
    except Exception:
        d["inf_count"] = -1
    return d


def _validate_batch(batch: dict, seq_len: int):
    reasons = []
    bad = False
    # v28: vol removed. Added trend_momentum, regime_continuous, mktstructure
    keys_to_check = [
        "seq",
        "masked_seq",
        "mask",
        "trend",
        "regime",
        "is_close",
        "trend_momentum",
        "regime_continuous",
    ]
    for key in keys_to_check:
        val = batch.get(key, None)
        st = _tensor_stats(val)
        logger.debug("Batch stat %s: %s", key, st)
        if isinstance(st, dict):
            if st.get("nan_count", 0) > 0 or st.get("inf_count", 0) > 0:
                bad = True
                reasons.append(
                    f"{key} has NaN/Inf (nan={st.get('nan_count')}, inf={st.get('inf_count')})"
                )
            if key == "seq" and isinstance(st.get("shape"), tuple):
                if len(st["shape"]) != 3:
                    bad = True
                    reasons.append(f"seq ndim != 3 ({st['shape']})")
                else:
                    if st["shape"][1] != seq_len:
                        reasons.append(f"seq length {st['shape'][1]} != expected seq_len {seq_len}")
        else:
            if st == "missing":
                bad = True
                reasons.append(f"{key} missing from batch")
            elif isinstance(st, str) and st.startswith("uninspectable"):
                bad = True
                reasons.append(f"{key} uninspectable: {st}")
    # v28: trend can be int8 (labels -1/0/1), it's cast to float in the trainer
    # don't flag it as corrupted
    if "regime" in batch and isinstance(batch["regime"], torch.Tensor):
        if batch["regime"].dtype not in (torch.int64, torch.int32):
            reasons.append(f"regime dtype {batch['regime'].dtype} not integer")
    return bad, reasons


# ----------------------------
# Model components
# ----------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        padding: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoderBlock(nn.Module):
    def __init__(
        self, num_inputs: int, channels: list[int], kernel_size: int = 3, dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        num_levels = len(channels)
        for i in range(num_levels):
            in_ch = num_inputs if i == 0 else channels[i - 1]
            out_ch = channels[i]
            dilation_size = 2**i
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AssetEmbedding(nn.Module):
    """Learnable per-asset token concatenated to the TCN input at every timestep."""

    def __init__(self, n_assets: int, embed_dim: int = 8):
        super().__init__()
        self.n_assets = n_assets
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(n_assets, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, asset_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(asset_ids)


class TCNGRUEncoder(nn.Module):
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
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.n_assets = n_assets
        self.asset_embed_dim = asset_embed_dim if n_assets > 0 else 0

        if n_assets > 0:
            self.asset_embedding: Optional[AssetEmbedding] = AssetEmbedding(
                n_assets, asset_embed_dim
            )
            tcn_input = input_features + asset_embed_dim
        else:
            self.asset_embedding = None
            tcn_input = input_features

        self.tcn = TCNEncoderBlock(
            num_inputs=tcn_input, channels=tcn_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.gru = nn.GRU(
            input_size=tcn_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(gru_hidden) if layer_norm else None
        self.dropout = nn.Dropout(dropout)
        self.use_checkpoint = False

    def forward(self, x: torch.Tensor, asset_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.asset_embedding is not None and asset_ids is not None:
            tok = self.asset_embedding(asset_ids)
            tok = tok.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, tok], dim=-1)
        x = x.permute(0, 2, 1)
        if self.use_checkpoint:
            out = x
            for layer in self.tcn.network:
                out = torch_checkpoint(layer, out, use_reentrant=False)
            tcn_out = out
        else:
            tcn_out = self.tcn(x)
        tcn_out = tcn_out.permute(0, 2, 1)
        out, h = self.gru(tcn_out)
        emb = h[-1]
        if self.layer_norm is not None:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


# ----------------------------
# Heads
# ----------------------------


class TrendMomentumHeadTF(nn.Module):
    """
    Per-TF scalar regression: predicts the forward return normalized by ATR.

    target = log_return_future / ATR_current  (clipped to [-3, 3])
    Typical values: ±0.3 = normal move, ±1.0 = strong move, ±2.0 = spike

    Why not the old 3-class TrendHeadTF:
    - no class effect -> no UP/DOWN flip-flopping
    - gradient proportional to the error -> stable learning
    - encodes both direction (sign) and strength (magnitude) in one scalar
    - for the DDQN: 4 scalars (one per TF) instead of 12 softmax dims

    Output: [B, n_tfs]  (values in [-3, 3], positive=bullish, negative=bearish)
    Label -1e9 = skip (excluded samples or no data).
    Loss: Huber (delta=0.5) — robust to FX outliers
    """

    def __init__(self, emb_dim: int, n_tfs: int, hidden: int = 128):
        super().__init__()
        self.n_tfs = n_tfs
        self.fc1 = nn.Linear(emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_tfs)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # tanh keeps the output in a finite range, scaled to [-3, 3]
        return torch.tanh(self.fc2(F.relu(self.fc1(emb)))) * 3.0  # [B, n_tfs]


class RegimeContinuousHead(nn.Module):
    """
    Predicts the regime of each TF as continuous scalars, no artificial thresholds.

    Output [B, n_tfs, 4]:
      [:, :, 0] = vol_regime    — volatility / rolling vol mean (~[0,3])
      [:, :, 1] = atr_norm×100  — ATR14 / price × 100 (~[0.05, 2.0] on FX)
      [:, :, 2] = adx/100       — ADX14 / 100 (~[0, 1], trend strength)
      [:, :, 3] = di_direction  — tanh((DI+−DI−)/20) in [−1,+1] (v25)
                                  +1=strong uptrend, −1=strong downtrend, 0=neutral

    Skip target = -999.0. Loss: one Huber per signal.
    di_direction = -999 on old parquets (backward compatible).

    Why not the old 3-class RegimeHeadTF:
    - no arbitrary LOW/MID/HIGH thresholds
    - stable gradient even with few 4H samples per batch
    - orthogonal + directional signals: v25 adds di_direction,
      which forces the backbone to tell up trends from down trends
    - DDQN state vector size: n_tfs × 4 (was n_tfs × 3)
    """

    def __init__(
        self, emb_dim: int, n_tfs: int, n_signals: int = 4, hidden: int = 128
    ):  # v25: 4 signals (di_direction added)
        super().__init__()
        self.n_tfs = n_tfs
        self.n_signals = n_signals
        self.fc1 = nn.Linear(emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, n_tfs * n_signals)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(emb))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # [B, n_tfs * n_signals]
        x = x.view(-1, self.n_tfs, self.n_signals)  # [B, n_tfs, n_signals]
        if self.n_signals >= 4:
            # non-negative signals: vol_regime, atr_norm, adx -> softplus
            x_pos = F.softplus(x[:, :, :3])
            # di_direction must be in [-1,+1] -> tanh
            x_dir = torch.tanh(x[:, :, 3:4])
            return torch.cat([x_pos, x_dir], dim=-1)  # [B, n_tfs, 4]
        else:
            # v24 compatibility: non-negative signals only
            return F.softplus(x)


class MarketStructureHead(nn.Module):
    """
    3-class classifier per TF: DIRECTIONAL market structure.
      0 = RANGING        — price close to the MA/EMA, no breakout (sideways)
      1 = LONG_BREAKOUT  — close >= rolling_max(high, 20 candles) * 0.9995
      2 = SHORT_BREAKOUT — close <= rolling_min(low,  20 candles) * 1.0005

    v25: TRENDING removed, BREAKOUT split into LONG/SHORT to give the backbone
    explicit directional information. It now has to predict not only
    "is there a breakout" but also "which way" -> separates bull/bear embeddings.

    Input:  emb [B, emb_dim]
    Output: logits [B, n_tfs, 3]  — softmax[...,1]=P(LONG_BRK), [...,2]=P(SHORT_BRK)
    Label -1 = skip.
    """

    def __init__(self, emb_dim: int, n_tfs: int, hidden: int = 64, n_classes: int = 3):
        super().__init__()
        self.n_tfs = n_tfs
        self.n_classes = n_classes
        self.fc1 = nn.Linear(emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_tfs * n_classes)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        out = self.fc2(F.relu(self.fc1(emb)))  # [B, n_tfs * 3]
        return out.view(-1, self.n_tfs, self.n_classes)  # [B, n_tfs, 3]


class SessionBreakoutHead(nn.Module):
    """
    Head for the London Session Breakout (sess_sl_third).

    Predicts LONG and SHORT independently:
      P(hit_long)  — will price reach 3 ATR UP within 32 bars?
      P(tp_long)   — if something is hit, is it the TP (not the SL)?
      P(hit_short) — will price reach 3 ATR DOWN within 32 bars?
      P(tp_short)  — if something is hit, is it the TP (not the SL)?

    No strategy_emb/direction_emb: the backbone already encodes
    the direction through di_direction (RegimeContinuousHead).
    The weights specialize entirely on the London breakout pattern:
      - quality of the Asian compression (ATR_norm, BB_width, volatility)
      - momentum at the London open (session_london, MACD_1H, di_dir_4H)
      - range width vs ATR (range_ratio_15min)

    Live: called ONLY when sess_sl_third gives an entry signal.
    P(TP) = P(hit) × P(tp|hit) — used as a quality filter (threshold 0.4-0.5).

    Labels: tp_r=3.0, sl_r=1.0, horizon=32 bars (8h London session)
    """

    LONG = 0
    SHORT = 1

    def __init__(self, emb_dim: int, hidden: int = 96, ctx_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.ctx_dim = ctx_dim
        trunk_in = emb_dim + ctx_dim

        self.shared = nn.Sequential(
            nn.Linear(trunk_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        # separate trunks for LONG and SHORT
        self.long_trunk = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
        )
        self.short_trunk = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
        )
        # side-only: hit_rate≈90% on random bars -> hit logit useless, removed
        self.long_side = nn.Linear(hidden // 4, 1)
        self.short_side = nn.Linear(hidden // 4, 1)
        for head in [self.long_side, self.short_side]:
            nn.init.zeros_(head.bias)

    def forward(
        self, emb: torch.Tensor, extra_context: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (long_side_logit, short_side_logit) — [B,1] ciascuno.
        Predicts P(TP first) for LONG and SHORT.
        """
        x = (
            torch.cat([emb, extra_context.to(emb.dtype)], dim=-1)
            if extra_context is not None
            else emb
        )
        h = self.shared(x)
        hl = self.long_trunk(h)
        hs = self.short_trunk(h)
        return self.long_side(hl), self.short_side(hs)

    def predict_proba(
        self, emb: torch.Tensor, extra_context: Optional[torch.Tensor] = None, direction: int = LONG
    ) -> torch.Tensor:
        """P(TP first) for the requested direction. Suggested threshold: 0.35."""
        ls, ss = self.forward(emb, extra_context)
        if direction == self.LONG:
            return torch.sigmoid(ls).squeeze(-1)
        else:
            return torch.sigmoid(ss).squeeze(-1)


class PDHLBreakoutHead(nn.Module):
    """
    Head for PDH/PDL + London confluence (pdhl_london_di).

    Same structure as SessionBreakoutHead, specialized on the
    PDH/PDL pattern:
      - respect of the previous daily levels (Pivot_1D, R1_1D, S1_1D)
      - London directional confirmation (session_london, di_dir_4H, di_dir_1H)
      - breakout strength (ADX14_4H, MACD_1H, momentum_proxy_4H)

    Live: called ONLY when pdhl_london_di gives a signal.

    Labels: tp_r=3.0, sl_r=1.0, horizon=64 bars (16h, full day)
    """

    LONG = 0
    SHORT = 1

    def __init__(self, emb_dim: int, hidden: int = 96, ctx_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.ctx_dim = ctx_dim
        trunk_in = emb_dim + ctx_dim

        self.shared = nn.Sequential(
            nn.Linear(trunk_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.long_trunk = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
        )
        self.short_trunk = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
        )
        # side-only: hit_rate≈98% on random bars -> hit logit useless, removed
        self.long_side = nn.Linear(hidden // 4, 1)
        self.short_side = nn.Linear(hidden // 4, 1)
        for head in [self.long_side, self.short_side]:
            nn.init.zeros_(head.bias)

    def forward(
        self, emb: torch.Tensor, extra_context: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns: (long_side_logit, short_side_logit) — [B,1] ciascuno."""
        x = (
            torch.cat([emb, extra_context.to(emb.dtype)], dim=-1)
            if extra_context is not None
            else emb
        )
        h = self.shared(x)
        hl = self.long_trunk(h)
        hs = self.short_trunk(h)
        return self.long_side(hl), self.short_side(hs)

    def predict_proba(
        self, emb: torch.Tensor, extra_context: Optional[torch.Tensor] = None, direction: int = LONG
    ) -> torch.Tensor:
        """P(TP first) for the requested direction. Suggested threshold: 0.35."""
        ls, ss = self.forward(emb, extra_context)
        if direction == self.LONG:
            return torch.sigmoid(ls).squeeze(-1)
        else:
            return torch.sigmoid(ss).squeeze(-1)


class RegimeQualityHead(nn.Module):
    """
    Proxy head for how tradable the market is for the strategies.

    One score in [0,1] answering:
    "On this bar, would at least one of the downstream strategies have
    a high probability of TP-first?"

    Supervised label (from _precompute_strategy_outcome_labels):
      1  = at least one of the valid side labels
           (session_long, session_short, pdhl_long, pdhl_short) is TP-first
      0  = all the valid side labels are SL-first
      -1 = no valid side label (skip)

    Note:
    this head is NOT supervised on a generic expansion measure
    (e.g. max excursion >= 2 ATR) but on an aggregate of the downstream
    strategy targets. So it's a proxy of "tradability for deployed strategies",
    not a pure measure of the physical market regime.

    Live: used as a global quality filter.
    If score < threshold, no entry on that bar.

    Loss: masked BCE (-1 sentinels excluded)
    """

    def __init__(self, emb_dim: int, hidden: int = 64, ctx_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.ctx_dim = ctx_dim
        trunk_in = emb_dim + ctx_dim

        self.net = nn.Sequential(
            nn.Linear(trunk_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, emb: torch.Tensor, extra_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Returns: logit [B,1] — sigmoid → P(good regime)"""
        x = (
            torch.cat([emb, extra_context.to(emb.dtype)], dim=-1)
            if extra_context is not None
            else emb
        )
        return self.net(x)

    def predict_score(
        self, emb: torch.Tensor, extra_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Returns: score [B] in [0,1]"""
        return torch.sigmoid(self.forward(emb, extra_context)).squeeze(-1)


class BEReachableHead(nn.Module):
    """
    v29 — optional head for breakeven reachability.

    For each direction (long/short) one logit answering:
    "With the BE rule at trigger×R, would this trade reach breakeven
    (SL moved to BE+costs) BEFORE dying at the original SL?"

    Label (from dataset.py, be_reachable):
      1  = the BE trigger is touched before the original SL (protected trade)
      0  = no
      -1 = skip (no valid trade on that bar)

    PURPOSE: a dimension SEPARATE from P(TP first). The live filter can use it
    to prefer trades "protected by BE" when the goal is a lower drawdown,
    WITHOUT changing the P(TP) based selection. It doesn't replace the side
    head, it sits next to it.

    Same architecture as RegimeQualityHead (reuses the embedding, does NOT touch
    the backbone). Two outputs: long and short. Loss: masked BCE (-1 excluded).
    """

    def __init__(self, emb_dim: int, hidden: int = 64, ctx_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.ctx_dim = ctx_dim
        trunk_in = emb_dim + ctx_dim

        def _branch():
            net = nn.Sequential(
                nn.Linear(trunk_in, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )
            nn.init.zeros_(net[-1].bias)
            return net

        self.net_long = _branch()
        self.net_short = _branch()

    def forward(self, emb: torch.Tensor, extra_context: Optional[torch.Tensor] = None):
        """Returns: (logit_long [B,1], logit_short [B,1])."""
        x = (
            torch.cat([emb, extra_context.to(emb.dtype)], dim=-1)
            if extra_context is not None
            else emb
        )
        return self.net_long(x), self.net_short(x)

    def predict_score(self, emb: torch.Tensor, extra_context: Optional[torch.Tensor] = None):
        """Returns: (score_long [B], score_short [B]) in [0,1]."""
        ll, ls = self.forward(emb, extra_context)
        return torch.sigmoid(ll).squeeze(-1), torch.sigmoid(ls).squeeze(-1)


class StrategyOutcomeHead(nn.Module):
    """
    Outcome of the trading agent's four trades: TREND / MR x LONG / SHORT.

    A trade opened at the close of the last bar of the window, SL and TP in ATR units
    (the agent's own rules, see agent/strategy_head.py). Two outputs per combo:
      hit  = P(SL or TP touched within the horizon)   (0 = still open, timeout)
      side = P(TP first | hit)
    P(TP) = P(hit) x P(TP first | hit).

    Strategy and direction are learned embeddings concatenated to the market
    embedding, so the four combos share one trunk. Trained on top of a FROZEN
    encoder (the embedding the agent sees), not inside the encoder pretraining.
    Parameter names match the v26 checkpoints the agent was built on.
    """

    STRATEGIES = ("trend", "mr")
    DIRECTIONS = ("long", "short")
    COMBOS = [(0, 0, "trend_long"), (0, 1, "trend_short"), (1, 0, "mr_long"), (1, 1, "mr_short")]

    def __init__(
        self,
        emb_dim: int,
        hidden: int = 64,
        embed_dim: int = 8,
        ctx_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.strategy_emb = nn.Embedding(2, embed_dim)
        self.direction_emb = nn.Embedding(2, embed_dim)
        self.shared = nn.Sequential(
            nn.Linear(emb_dim + 2 * embed_dim + ctx_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.hit_trunk = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 2), nn.ReLU(), nn.Dropout(dropout)
        )
        self.side_trunk = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 2), nn.ReLU(), nn.Dropout(dropout)
        )
        self.hit_head = nn.Linear(hidden // 2, 1)
        self.side_head = nn.Linear(hidden // 2, 1)

    @classmethod
    def from_state_dict(cls, state: dict, emb_dim: int) -> "StrategyOutcomeHead":
        """Builds a head with the sizes of a saved state_dict (hidden, embed_dim, ctx_dim)."""
        embed_dim = int(state["strategy_emb.weight"].shape[1])
        hidden, trunk_in = (int(s) for s in state["shared.0.weight"].shape)
        head = cls(
            emb_dim, hidden=hidden, embed_dim=embed_dim, ctx_dim=trunk_in - emb_dim - 2 * embed_dim
        )
        head.load_state_dict(state)
        return head

    def forward(
        self,
        emb: torch.Tensor,
        strategy: torch.Tensor,
        direction: torch.Tensor,
        extra_context: Optional[torch.Tensor] = None,
    ):
        """Returns: (hit_logit [B,1], side_logit [B,1])."""
        parts = [emb, self.strategy_emb(strategy.long()), self.direction_emb(direction.long())]
        if self.ctx_dim > 0:
            if extra_context is None:
                extra_context = emb.new_zeros(emb.size(0), self.ctx_dim)
            parts.append(extra_context.to(emb.dtype))
        h = self.shared(torch.cat(parts, dim=-1))
        return self.hit_head(self.hit_trunk(h)), self.side_head(self.side_trunk(h))

    def predict_all_combos(
        self, emb: torch.Tensor, extra_context: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """p_hit_<combo>, p_tp_given_hit_<combo>, p_tp_<combo>, p_sl_<combo>, p_none_<combo>."""
        out = {}
        for si, di, name in self.COMBOS:
            s = torch.full((emb.size(0),), si, dtype=torch.long, device=emb.device)
            d = torch.full((emb.size(0),), di, dtype=torch.long, device=emb.device)
            hit_logit, side_logit = self.forward(emb, s, d, extra_context)
            p_hit = torch.sigmoid(hit_logit).squeeze(-1)
            p_side = torch.sigmoid(side_logit).squeeze(-1)
            out[f"p_hit_{name}"] = p_hit
            out[f"p_tp_given_hit_{name}"] = p_side
            out[f"p_tp_{name}"] = p_hit * p_side
            out[f"p_sl_{name}"] = p_hit * (1.0 - p_side)
            out[f"p_none_{name}"] = 1.0 - p_hit
        return out


# ----------------------------
# Augmentations
# ----------------------------
def augment_timeseries(x: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    s = x.copy()
    L, _Ff = s.shape
    jitter = float(config.get("jitter", 0.0))
    if jitter > 0:
        s = s + np.random.normal(0, jitter, size=s.shape).astype(np.float32)
    scaling = float(config.get("scaling", 0.0))
    if scaling > 0:
        factor = np.random.normal(1.0, scaling)
        s = (s * factor).astype(np.float32)
    tmask = int(config.get("time_mask", 0))
    if tmask > 0 and np.random.rand() < 0.5:
        start = np.random.randint(0, max(1, L - 1))
        size = np.random.randint(1, min(tmask, L - start))
        s[start : start + size, :] = 0.0
    cut = int(config.get("cutout", 0))
    if cut > 0 and np.random.rand() < 0.5:
        start = np.random.randint(0, max(1, L - 1))
        size = np.random.randint(1, min(cut, L - start))
        s[start : start + size, :] = 0.0
    perm_prob = float(config.get("permute_prob", 0.0))
    if np.random.rand() < perm_prob:
        n_chunks = max(2, int(L * 0.05))
        idx = np.arange(L)
        chunks = np.array_split(idx, n_chunks)
        np.random.shuffle(chunks)
        new_idx = np.concatenate(chunks)
        s = s[new_idx]
    drop = float(config.get("dropout", 0.0))
    if drop > 0:
        mask = np.random.rand(L) < drop
        s[mask, :] = 0.0
    return s.astype(np.float32)


def augment_timeseries_torch_batch(x: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    device = x.device
    B, L, _F = x.shape
    out = x.clone()
    deterministic = bool(config.get("deterministic", False))
    base_seed = int(config.get("seed", 0)) if deterministic else None

    jitter = float(config.get("jitter", 0.0))
    if jitter > 0:
        if deterministic:
            g = torch.Generator(device=device).manual_seed(base_seed or 0)
            noise = torch.normal(mean=0.0, std=jitter, size=out.shape, generator=g, device=device)
        else:
            noise = torch.randn_like(out, device=device)
        out = out + noise

    scaling = float(config.get("scaling", 0.0))
    if scaling > 0:
        if deterministic:
            g = torch.Generator(device=device).manual_seed((base_seed or 0) + 1)
            factors = torch.normal(
                mean=1.0, std=scaling, size=(B, 1, 1), generator=g, device=device
            )
        else:
            factors = torch.randn((B, 1, 1), device=device) * scaling + 1.0
        out = out * factors

    tmask = int(config.get("time_mask", 0))
    if tmask > 0:
        if deterministic:
            g = torch.Generator(device=device).manual_seed((base_seed or 0) + 2)
            rand = torch.rand(B, generator=g, device=device)
        else:
            rand = torch.rand(B, device=device)
        mask_apply = rand < 0.5
        if mask_apply.any():
            if deterministic:
                g1 = torch.Generator(device=device).manual_seed((base_seed or 0) + 3)
                starts = torch.randint(0, max(1, L - 1), (B,), generator=g1, device=device)
                g2 = torch.Generator(device=device).manual_seed((base_seed or 0) + 4)
                sizes = torch.randint(1, max(1, tmask), (B,), generator=g2, device=device)
            else:
                starts = torch.randint(0, max(1, L - 1), (B,), device=device)
                sizes = torch.randint(1, max(1, tmask), (B,), device=device)
            for i in torch.nonzero(mask_apply, as_tuple=False).squeeze(1).tolist():
                s0 = int(starts[i].item())
                sz = int(min(sizes[i].item(), L - s0))
                out[i, s0 : s0 + sz, :] = 0.0

    cut = int(config.get("cutout", 0))
    if cut > 0:
        if deterministic:
            g = torch.Generator(device=device).manual_seed((base_seed or 0) + 5)
            rand = torch.rand(B, generator=g, device=device)
        else:
            rand = torch.rand(B, device=device)
        mask_apply = rand < 0.5
        if mask_apply.any():
            if deterministic:
                g1 = torch.Generator(device=device).manual_seed((base_seed or 0) + 6)
                starts = torch.randint(0, max(1, L - 1), (B,), generator=g1, device=device)
                g2 = torch.Generator(device=device).manual_seed((base_seed or 0) + 7)
                sizes = torch.randint(1, max(1, cut), (B,), generator=g2, device=device)
            else:
                starts = torch.randint(0, max(1, L - 1), (B,), device=device)
                sizes = torch.randint(1, max(1, cut), (B,), device=device)
            for i in torch.nonzero(mask_apply, as_tuple=False).squeeze(1).tolist():
                s0 = int(starts[i].item())
                sz = int(min(sizes[i].item(), L - s0))
                out[i, s0 : s0 + sz, :] = 0.0

    drop = float(config.get("dropout", 0.0))
    if drop > 0:
        if deterministic:
            g = torch.Generator(device=device).manual_seed((base_seed or 0) + 8)
            mask = (torch.rand((B, L), generator=g, device=device) < drop).unsqueeze(-1)
        else:
            mask = (torch.rand((B, L), device=device) < drop).unsqueeze(-1)
        out = out.masked_fill(mask, 0.0)

    perm_prob = float(config.get("permute_prob", 0.0))
    if perm_prob > 0:
        if deterministic:
            g = torch.Generator(device=device).manual_seed((base_seed or 0) + 9)
            rand = torch.rand(B, generator=g, device=device)
        else:
            rand = torch.rand(B, device=device)
        mask_apply = rand < perm_prob
        if mask_apply.any():
            idx_cpu = np.arange(L)
            n_chunks = max(2, int(L * 0.05))
            np_chunks = np.array_split(idx_cpu, n_chunks)
            chunks = [
                torch.from_numpy(c.astype(np.int64)).to(device) for c in np_chunks if c.size > 0
            ]
            for i in torch.nonzero(mask_apply, as_tuple=False).squeeze(1).tolist():
                if deterministic:
                    gperm = torch.Generator(device=device).manual_seed((base_seed or 0) + 10 + i)
                    perm = torch.randperm(len(chunks), generator=gperm, device=device)
                else:
                    perm = torch.randperm(len(chunks), device=device)
                new_idx = torch.cat([chunks[int(j)] for j in perm], dim=0)
                if new_idx.numel() > L:
                    new_idx = new_idx[:L]
                elif new_idx.numel() < L:
                    needed = L - new_idx.numel()
                    pad = chunks[0][:needed]
                    new_idx = torch.cat([new_idx, pad], dim=0)
                out[i] = out[i, new_idx, :]
    return out


def _augment_np_worker(args):
    arr, cfg = args
    return augment_timeseries(arr, cfg)


# ----------------------------
# Loss functions
# ----------------------------
def nt_xent_simclr(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    device = z1.device
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    mask = (~torch.eye(2 * B, device=device).bool()).float()
    labels = torch.arange(0, 2 * B, device=device)
    positives = (labels + B) % (2 * B)
    sim = sim * mask + (1 - mask) * (-1e9)
    log_prob = F.log_softmax(sim, dim=1)
    loss = -log_prob[torch.arange(2 * B, device=device), positives].mean()
    return loss + 1e-12


def info_nce_from_pos_neg(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: Optional[torch.Tensor],
    temperature: float = 0.1,
) -> torch.Tensor:
    device = anchors.device
    N, _d = anchors.shape
    eps_local = 1e-12
    if N < 1:
        return (anchors.sum() * 0.0) + eps_local

    if positives.dim() == 2:
        pos_col = (anchors * positives).sum(dim=-1, keepdim=True) / temperature
    elif positives.dim() == 3:
        pos_sim = (
            torch.matmul(anchors.unsqueeze(1), positives.transpose(1, 2)).squeeze(1) / temperature
        )
        pos_col = pos_sim.mean(dim=1, keepdim=True)
    else:
        raise ValueError("positives dims not supported")

    if negatives is None or negatives.numel() == 0:
        return (anchors.sum() * 0.0) + eps_local

    if negatives.dim() == 2:
        # [N, K] — every negative as its own column (standard InfoNCE)
        neg_sim = torch.matmul(anchors, negatives.T) / temperature
    elif negatives.dim() == 3:
        # [N, K] — every negative as its own column
        neg_sim = torch.sum(anchors.unsqueeze(1) * negatives, dim=-1) / temperature
    else:
        raise ValueError("negatives dims not supported")

    logits = torch.cat([pos_col, neg_sim], dim=1)
    labels = torch.zeros(N, dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, labels)
    return loss + eps_local


def variance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """VICReg-style variance loss to prevent collapse. z: (B, d)"""
    if z is None or z.numel() == 0:
        return torch.tensor(0.0, device=z.device if isinstance(z, torch.Tensor) else "cpu")
    std = torch.sqrt(z.var(dim=0) + eps)
    loss = torch.mean(F.relu(1.0 - std))
    return loss


class ProjectionHead(nn.Module):
    def __init__(self, emb_dim: int, proj_dim: int = 128, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, proj_dim))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        z = self.net(emb)
        z = F.normalize(z, dim=-1)
        return z


class MemoryBank:
    def __init__(self, dim: int, capacity: int = 16384, device: Optional[torch.device] = None):
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.device = device if device is not None else torch.device("cpu")
        self.ptr = 0
        self.size = 0
        self.zs = torch.zeros((self.capacity, self.dim), dtype=torch.float32, device=self.device)
        self.meta = [None] * self.capacity

    def add(self, z: torch.Tensor, metas: list[dict]):
        z = z.detach().to(self.device)
        B = z.size(0)
        for i in range(B):
            self.zs[self.ptr] = z[i].detach().clone()
            self.meta[self.ptr] = metas[i]
            self.ptr = (self.ptr + 1) % self.capacity
            if self.size < self.capacity:
                self.size += 1

    def sample_negatives(
        self, k: int, avoid_asset: Optional[str] = None, avoid_regime: Optional[int] = None
    ):
        if self.size == 0:
            return torch.empty((0, self.dim), device=self.device), []
        idxs = list(range(self.size))
        if avoid_asset is not None:
            idxs = [
                i
                for i in idxs
                if self.meta[i] is not None and self.meta[i].get("asset") != avoid_asset
            ]
        if avoid_regime is not None:
            idxs = [
                i
                for i in idxs
                if self.meta[i] is not None and self.meta[i].get("regime") != avoid_regime
            ]
        if len(idxs) == 0:
            return torch.empty((0, self.dim), device=self.device), []
        if len(idxs) >= k:
            cho = np.random.choice(idxs, size=k, replace=False)
        else:
            cho = np.random.choice(idxs, size=k, replace=True)
        return self.zs[cho], [self.meta[i] for i in cho]


# ----------------------------
# Contrastive Pretrainer
# ----------------------------
def _to_asset_id_tensor(asset_ids, device) -> Optional[torch.Tensor]:
    if asset_ids is None:
        return None
    if isinstance(asset_ids, torch.Tensor):
        return asset_ids.long().to(device)
    try:
        arr = [int(a) if a is not None else 0 for a in asset_ids]
        return torch.tensor(arr, dtype=torch.long, device=device)
    except Exception:
        return None


class DirectionalContrastiveLoss(nn.Module):
    """Directional contrastive loss: pushes bull embeddings away from bear ones.

    Uses the sign of tmom_30m as supervision:
      - tmom > +threshold  -> BULL sample
      - tmom < -threshold  -> BEAR sample
      - |tmom| < threshold -> NEUTRAL sample (ignored)

    Loss per pair (bull_i, bear_j):
      L = max(0, cos(bull_i, bear_j) - margin)

    Goal: the backbone encodes bull and bear in separate regions of the
    embedding space, bringing their cosine similarity from 0.91 towards < 0.5.

    It adds to SimCLR, doesn't replace it: SimCLR pulls views of the same
    sample together (augmentation invariance), the directional loss pushes
    samples of opposite direction apart.
    """

    def __init__(
        self,
        margin: float = 0.3,
        tmom_threshold: float = 0.3,
        max_pairs: int = 2048,
        tmom_tf_idx: int = 1,
    ):
        super().__init__()
        self.margin = margin  # cosine similarity margin to respect
        self.tmom_threshold = tmom_threshold  # min |tmom| to call a sample bull/bear
        self.max_pairs = max_pairs  # cap on pairs, for memory
        self.tmom_tf_idx = tmom_tf_idx  # 0=15min, 1=30min, 2=1h (standard TF_LIST)

    def forward(
        self,
        emb: torch.Tensor,  # [B, D] normalized or raw embedding
        tmom: torch.Tensor,  # [B, n_tfs] trend momentum scalars
    ) -> torch.Tensor:
        """
        emb:  [B, D] — backbone embedding (doesn't need to be normalized)
        tmom: [B, n_tfs] — tmom scalars, the configured TF is the primary signal

        Returns: scalar loss (0 if there are no valid pairs)
        """
        device = emb.device
        thr = self.tmom_threshold

        # tmom of the configured TF (default idx=1 -> 30min with the standard TF_LIST)
        # TF_LIST = ["15min","30min","1h","4h","1d"] → idx 0=15min, 1=30min
        tmom_30m = tmom[:, self.tmom_tf_idx]  # [B]

        bull_idx = (tmom_30m > thr).nonzero(as_tuple=True)[0]
        bear_idx = (tmom_30m < -thr).nonzero(as_tuple=True)[0]

        n_bull = len(bull_idx)
        n_bear = len(bear_idx)

        if n_bull == 0 or n_bear == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # normalize the embeddings for the cosine similarity
        emb_n = F.normalize(emb.float(), p=2, dim=1)  # [B, D]

        emb_bull = emb_n[bull_idx]  # [n_bull, D]
        emb_bear = emb_n[bear_idx]  # [n_bear, D]

        # bull × bear cosine similarity matrix
        cos_mat = emb_bull @ emb_bear.T  # [n_bull, n_bear]

        # loss: penalize pairs with cosine > margin
        # max(0, cos - margin) -> 0 if already apart, > 0 if too similar
        loss_mat = F.relu(cos_mat - self.margin)

        # memory cap: sample max_pairs pairs if the matrix is too big
        n_pairs = n_bull * n_bear
        if n_pairs > self.max_pairs:
            # random rows and columns
            n_b_sample = min(n_bull, int((self.max_pairs / n_bear) ** 0.5) + 1)
            n_r_sample = min(n_bear, self.max_pairs // n_b_sample)
            b_sel = torch.randperm(n_bull, device=device)[:n_b_sample]
            r_sel = torch.randperm(n_bear, device=device)[:n_r_sample]
            loss_mat = loss_mat[b_sel][:, r_sel]

        loss = loss_mat.mean()
        return loss


class ContrastivePretrainer:
    def __init__(
        self, encoder: nn.Module, proj_head: nn.Module, device: torch.device, config: dict[str, Any]
    ):
        self.encoder = encoder
        self.proj = proj_head
        self.device = device
        self.config = config
        self.temperature = float(config.get("temperature", 0.1))
        mem_cfg = config.get("memory_bank", {"enabled": False})
        if mem_cfg.get("enabled", False):
            cap = int(mem_cfg.get("capacity", 16384))
            proj_dim = int(config.get("proj_dim", 128))
            self.memory_bank = MemoryBank(dim=proj_dim, capacity=cap, device=device)
        else:
            self.memory_bank = None
        self.aug_workers = int(config.get("aug_workers", 4))
        _pool_workers = max(1, self.aug_workers)
        try:
            self._executor = ThreadPoolExecutor(max_workers=_pool_workers)
            logger.info("[CP] ThreadPoolExecutor created with %d workers", _pool_workers)
        except Exception as e:
            logger.warning("[CP] ThreadPoolExecutor failed (%s), fallback sequential", e)
            self._executor = None

    def compute_simclr_batch_pairs(
        self,
        seq_batch: torch.Tensor,
        simclr_aug_config: dict[str, Any],
        asset_ids_t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = seq_batch.device
        use_gpu_aug = simclr_aug_config.get("use_gpu_augment", False) and device.type == "cuda"
        deterministic = bool(simclr_aug_config.get("deterministic", False))
        base_seed = int(simclr_aug_config.get("seed", 0)) if deterministic else None

        if use_gpu_aug:
            if deterministic:
                a_t = augment_timeseries_torch_batch(
                    seq_batch, dict(simclr_aug_config, deterministic=True, seed=base_seed or 0)
                )
                b_t = augment_timeseries_torch_batch(
                    seq_batch,
                    dict(simclr_aug_config, deterministic=True, seed=(base_seed or 0) + 1),
                )
            else:
                a_t = augment_timeseries_torch_batch(seq_batch, simclr_aug_config)
                b_t = augment_timeseries_torch_batch(seq_batch, simclr_aug_config)
        else:
            seq_np = seq_batch.detach().cpu().numpy()
            args_a = [(s, simclr_aug_config) for s in seq_np]
            args_b = [(s, simclr_aug_config) for s in seq_np]
            if self._executor is not None:
                a_list = list(self._executor.map(_augment_np_worker, args_a))
                b_list = list(self._executor.map(_augment_np_worker, args_b))
            else:
                a_list = [_augment_np_worker(a) for a in args_a]
                b_list = [_augment_np_worker(b) for b in args_b]
            a_t = torch.from_numpy(np.stack(a_list)).to(device, non_blocking=True)
            b_t = torch.from_numpy(np.stack(b_list)).to(device, non_blocking=True)

        emb_a = self.encoder(a_t, asset_ids=asset_ids_t)
        emb_b = self.encoder(b_t, asset_ids=asset_ids_t)
        z_a = self.proj(emb_a)
        z_b = self.proj(emb_b)
        return z_a, z_b

    def compute_cross_asset_positive_pairs(
        self,
        z: torch.Tensor,
        asset_ids: list[Any],
        timestamps: Optional[list[Any]] = None,
        regimes: Optional[list[Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = z.device
        B = z.size(0)

        def regime_scalar(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                try:
                    if x.numel() == 0:
                        return None
                    arr = x.detach().cpu().numpy()
                except Exception:
                    return None
            else:
                arr = np.array(x)
            if arr.size == 0:
                return None
            if arr.size == 1:
                try:
                    return int(np.round(float(arr.item())))
                except Exception:
                    return None
            return int(np.round(np.mean(arr)))

        anchors = []
        positives = []

        if regimes is not None:
            regime_map = {}
            for i in range(B):
                r = regime_scalar(regimes[i])
                regime_map.setdefault(r, []).append(i)
            for i in range(B):
                r = regime_scalar(regimes[i])
                candidates = [
                    j for j in regime_map.get(r, []) if j != i and asset_ids[j] != asset_ids[i]
                ]
                if len(candidates) == 0:
                    continue
                j = random.choice(candidates)
                anchors.append(z[i])
                positives.append(z[j])
        elif timestamps is not None:
            ts_map = {}
            for i, ts in enumerate(timestamps):
                key = str(ts) if (ts is not None) else f"__missing_ts__{i}"
                ts_map.setdefault(key, []).append(i)
            for i in range(B):
                key = str(timestamps[i]) if (timestamps[i] is not None) else f"__missing_ts__{i}"
                cands = ts_map.get(key, [])
                cands = [j for j in cands if j != i and asset_ids[j] != asset_ids[i]]
                if len(cands) == 0:
                    continue
                j = random.choice(cands)
                anchors.append(z[i])
                positives.append(z[j])

        if len(anchors) == 0:
            idxs = list(range(B))
            for i in idxs:
                candidates = [j for j in idxs if j != i and asset_ids[j] != asset_ids[i]]
                if len(candidates) > 0:
                    j = random.choice(candidates)
                    anchors.append(z[i])
                    positives.append(z[j])

        if len(anchors) == 0:
            return (
                torch.empty((0, z.size(1)), device=device),
                torch.empty((0, z.size(1)), device=device),
            )
        return torch.stack(anchors, dim=0), torch.stack(positives, dim=0)

    def compute_cross_asset_negative_pairs(
        self,
        z: torch.Tensor,
        asset_ids: list[Any],
        timestamps: Optional[list[Any]],
        k_per_anchor: int = 32,
        regimes: Optional[list[Any]] = None,
    ) -> torch.Tensor:
        device = z.device
        B, d = z.size()
        negs_list = []
        for i in range(B):
            candidate_idxs = [
                j
                for j in range(B)
                if asset_ids[j] != asset_ids[i]
                and (timestamps is None or str(timestamps[j]) != str(timestamps[i]))
            ]
            negs = torch.empty((0, d), device=device)
            if len(candidate_idxs) > 0:
                if len(candidate_idxs) >= k_per_anchor:
                    chosen = random.sample(candidate_idxs, k_per_anchor)
                else:
                    chosen = list(np.random.choice(candidate_idxs, size=k_per_anchor, replace=True))
                negs = z[chosen]
            if (
                negs.size(0) < k_per_anchor
                and self.memory_bank is not None
                and self.memory_bank.size > 0
            ):
                need = k_per_anchor - negs.size(0)
                avoid_reg = None
                if regimes is not None:
                    try:
                        avoid_reg = (
                            int(np.round(np.array(regimes[i]).mean()))
                            if regimes[i] is not None
                            else None
                        )
                    except Exception:
                        avoid_reg = None
                mb_negs, _ = self.memory_bank.sample_negatives(
                    need, avoid_asset=asset_ids[i], avoid_regime=avoid_reg
                )
                if mb_negs is not None and mb_negs.size(0) > 0:
                    negs = (
                        mb_negs.to(device)
                        if negs.size(0) == 0
                        else torch.cat([negs, mb_negs.to(device)], dim=0)
                    )
            if negs.size(0) < k_per_anchor:
                pool = [j for j in range(B) if asset_ids[j] != asset_ids[i]]
                if len(pool) == 0:
                    pool = [j for j in range(B) if j != i]
                if len(pool) == 0:
                    pool = list(range(B))  # last fallback: the whole batch
                chosen = list(
                    np.random.choice(pool, size=k_per_anchor - negs.size(0), replace=True)
                )
                more = z[chosen]
                negs = more if negs.size(0) == 0 else torch.cat([negs, more], dim=0)
            if negs.size(0) > k_per_anchor:
                negs = negs[:k_per_anchor]
            negs_list.append(negs)
        if len(negs_list) == 0:
            return torch.empty((0, k_per_anchor, d), device=device)
        return torch.stack(negs_list, dim=0)

    def compute_total_contrastive_loss(
        self,
        seq_batch: torch.Tensor,
        asset_ids: list[Any],
        timestamps: Optional[list[Any]],
        loss_weights: dict[str, float],
        simclr_aug_config: dict[str, Any],
        k_neg: int = 64,
        regimes: Optional[list[Any]] = None,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        eps_local = 1e-12

        _aid = _to_asset_id_tensor(asset_ids, seq_batch.device)
        emb = self.encoder(seq_batch, asset_ids=_aid)
        z = self.proj(emb)

        if seq_batch.size(0) < 2:
            zero_connected = (z.sum() * 0.0) + eps_local
            return zero_connected, {"cross_pos": 0.0, "cross_neg": 0.0, "simclr_batch": 0.0}, z

        anchors_pos, positives_pos = self.compute_cross_asset_positive_pairs(
            z, asset_ids, timestamps=timestamps, regimes=regimes
        )
        if anchors_pos.size(0) > 0:
            negs_all = self.compute_cross_asset_negative_pairs(
                z, asset_ids, timestamps, k_per_anchor=k_neg, regimes=regimes
            )
            if negs_all.size(0) == anchors_pos.size(0):
                loss_cross_pos = info_nce_from_pos_neg(
                    anchors_pos, positives_pos, negs_all, temperature=self.temperature
                )
            else:
                loss_cross_pos = info_nce_from_pos_neg(
                    anchors_pos, positives_pos, z, temperature=self.temperature
                )
        else:
            loss_cross_pos = (z.sum() * 0.0) + eps_local

        negs_for_batch = self.compute_cross_asset_negative_pairs(
            z, asset_ids, timestamps, k_per_anchor=k_neg, regimes=regimes
        )
        if negs_for_batch.size(0) > 0:
            sim_per_anchor = torch.sum(z.unsqueeze(1) * negs_for_batch, dim=-1)
            sim_neg = sim_per_anchor.mean(dim=1).mean() / max(1.0, float(z.size(1)))
            loss_cross_neg = F.softplus(sim_neg) + eps_local
        else:
            loss_cross_neg = (z.sum() * 0.0) + eps_local

        try:
            z_a, z_b = self.compute_simclr_batch_pairs(
                seq_batch, simclr_aug_config, asset_ids_t=_aid
            )
            loss_simclr = nt_xent_simclr(z_a, z_b, temperature=self.temperature) + eps_local
        except Exception:
            loss_simclr = (z.sum() * 0.0) + eps_local

        w_pos = float(loss_weights.get("cross_pos", 1.0))
        w_neg = float(loss_weights.get("cross_neg", 1.0))
        w_sim = float(loss_weights.get("simclr_batch", 1.0))
        total = w_pos * loss_cross_pos + w_neg * loss_cross_neg + w_sim * loss_simclr

        if not torch.isfinite(total):
            total = (z.sum() * 0.0) + eps_local

        stats = {
            "cross_pos": (
                float(loss_cross_pos.detach().cpu().item())
                if isinstance(loss_cross_pos, torch.Tensor)
                else float(loss_cross_pos)
            ),
            "cross_neg": (
                float(loss_cross_neg.detach().cpu().item())
                if isinstance(loss_cross_neg, torch.Tensor)
                else float(loss_cross_neg)
            ),
            "simclr_batch": (
                float(loss_simclr.detach().cpu().item())
                if isinstance(loss_simclr, torch.Tensor)
                else float(loss_simclr)
            ),
        }
        return total, stats, z

    def compute_total_contrastive_loss_partial(
        self,
        seq_batch: torch.Tensor,
        asset_ids: list[Any],
        timestamps: Optional[list[Any]],
        loss_weights: dict[str, float],
        simclr_aug_config: dict[str, Any],
        do_simclr: bool = True,
        k_neg: int = 64,
        regimes: Optional[list[Any]] = None,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        eps_local = 1e-12
        _aid = _to_asset_id_tensor(asset_ids, seq_batch.device)
        emb = self.encoder(seq_batch, asset_ids=_aid)
        z = self.proj(emb)

        if seq_batch.size(0) < 2:
            zero_connected = (z.sum() * 0.0) + eps_local
            return zero_connected, {"cross_pos": 0.0, "cross_neg": 0.0, "simclr_batch": 0.0}, z

        if do_simclr:
            return self.compute_total_contrastive_loss(
                seq_batch,
                asset_ids,
                timestamps,
                loss_weights,
                simclr_aug_config,
                k_neg=k_neg,
                regimes=regimes,
            )

        anchors_pos, positives_pos = self.compute_cross_asset_positive_pairs(
            z, asset_ids, timestamps=timestamps, regimes=regimes
        )
        if anchors_pos.size(0) > 0:
            negs_all = self.compute_cross_asset_negative_pairs(
                z, asset_ids, timestamps, k_per_anchor=k_neg, regimes=regimes
            )
            if negs_all.size(0) == anchors_pos.size(0):
                loss_cross_pos = info_nce_from_pos_neg(
                    anchors_pos, positives_pos, negs_all, temperature=self.temperature
                )
            else:
                loss_cross_pos = info_nce_from_pos_neg(
                    anchors_pos, positives_pos, z, temperature=self.temperature
                )
            loss_cross_pos = loss_cross_pos + eps_local
        else:
            loss_cross_pos = (z.sum() * 0.0) + eps_local

        negs_for_batch = self.compute_cross_asset_negative_pairs(
            z, asset_ids, timestamps, k_per_anchor=k_neg, regimes=regimes
        )
        if negs_for_batch.size(0) > 0:
            sim_per_anchor = torch.sum(z.unsqueeze(1) * negs_for_batch, dim=-1)
            sim_neg = sim_per_anchor.mean(dim=1).mean() / max(1.0, float(z.size(1)))
            loss_cross_neg = F.softplus(sim_neg) + eps_local
        else:
            loss_cross_neg = (z.sum() * 0.0) + eps_local

        total = (
            float(loss_weights.get("cross_pos", 1.0)) * loss_cross_pos
            + float(loss_weights.get("cross_neg", 1.0)) * loss_cross_neg
        )
        if not torch.isfinite(total):
            total = (z.sum() * 0.0) + eps_local

        stats = {
            "cross_pos": (
                float(loss_cross_pos.detach().cpu().item())
                if isinstance(loss_cross_pos, torch.Tensor)
                else float(loss_cross_pos)
            ),
            "cross_neg": (
                float(loss_cross_neg.detach().cpu().item())
                if isinstance(loss_cross_neg, torch.Tensor)
                else float(loss_cross_neg)
            ),
            "simclr_batch": 0.0,
        }
        # update the memory bank AFTER the loss, so the current batch can't be
        # sampled as a negative of itself in the same forward
        if self.memory_bank is not None:
            _metas = []
            for i in range(z.size(0)):
                _reg = regimes[i] if (regimes is not None and i < len(regimes)) else None
                _metas.append(
                    {
                        "asset": asset_ids[i] if asset_ids else None,
                        "regime": _reg,
                    }
                )
            self.memory_bank.add(z.detach(), _metas)
        return total, stats, z


# ----------------------------
# CudaPrefetcher
# ----------------------------
class CudaPrefetcher:
    """Moves DataLoader batches to GPU using a background CUDA stream.

    On non-CUDA backends (DirectML, CPU, MPS) the stream is disabled and
    batches are moved synchronously — the class still works, just without
    the async speedup.
    """

    def __init__(self, loader, device=None, non_blocking=True):
        self.loader = loader
        self.device = device if device is not None else torch.device("cuda")
        self.non_blocking = non_blocking
        self.iter = iter(self.loader)
        # CUDA stream only on actual CUDA (not DirectML, not CPU)
        _is_cuda = getattr(device, "type", str(device)) == "cuda" and torch.cuda.is_available()
        self.stream = torch.cuda.Stream(device=device) if _is_cuda else None
        self.next_batch = None
        self._preload()

    def _preload(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream is None:
            self.next_batch = batch
            return
        with torch.cuda.stream(self.stream):
            moved = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    try:
                        moved[k] = v.to(self.device, non_blocking=self.non_blocking)
                    except Exception:
                        moved[k] = v.to(self.device)
                else:
                    moved[k] = v
            self.next_batch = moved

    def next(self):
        if self.stream is None:
            try:
                return next(self.iter)
            except StopIteration:
                return None
        torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch

    def reset(self):
        self.iter = iter(self.loader)
        self._preload()
