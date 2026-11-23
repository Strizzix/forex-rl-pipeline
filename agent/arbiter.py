"""
arbiter.py — trade selection, risk and portfolio for the Rainbow DQN (v2)

TradeProposal, TradeArbiter, Portfolio and MultiAssetRunner.

LAYERED SAFETY:
  1. Q-gap filter      — enters only if Rainbow is convinced (q_gap > MIN_Q_GAP)
  2. Regime risk       — smaller size in a sideways regime (regime=0 -> 60%)
  3. DD ladder         — size cut at 30/55/75% of the baseline DD:
                          30% -> size×0.50, 55% -> size×0.20, 75% -> size×0.10
  4. Daily DD ladder   — size cut at 50/75% of the daily limit
  5. Trailing DD block — stops trading if the drawdown from the peak > baseline
  6. Consecutive SL    — after N SLs in a row the size goes down (default N=3)
  7. Daily loss limit  — in eval/live the day is blocked after N daily SLs
  8. Correlation cap   — max 2 trades in the same direction at the same time
  9. Hard block        — blocks everything if DD > baseline; unblocks only below 70%

PORTFOLIO:
  - tracks the absolute drawdown (from initial_capital) — prop firm rule
  - tracks the trailing drawdown (from peak equity) — extra safety
  - blocks on both
"""

import dataclasses
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ══════════════════════════════════════════════════════════════════════════════
# TradeProposal
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class TradeProposal:
    asset: str
    action: int
    q_value: float
    confidence: float
    regime: float  # legacy compat (vol_regime_30m continuous)
    trend_alignment: float  # legacy compat (tmom_30m scalar)
    atr: float
    timestamp: any
    state: any
    approved: bool = False
    filter_reason: Optional[str] = None
    rank: Optional[int] = None
    q_vals_raw: Optional[np.ndarray] = None
    q_vals_masked: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None
    regime_model: Optional[float] = None  # legacy
    trend_model: Optional[float] = None  # legacy
    # ── v24: continuous encoder signals ──────────────────────────────────
    # Trend momentum scalars (TrendMomentumHeadTF): [-3,3], WF3 corr ~0.73-0.82
    tmom_30m: Optional[float] = None  # most reliable
    tmom_1h: Optional[float] = None
    tmom_4h: Optional[float] = None
    tmom_1d: Optional[float] = None
    # Regime continuous (RegimeContinuousHead): [0,1], WF3 corr ~0.93
    vol_regime_30m: Optional[float] = None  # primary regime signal
    vol_regime_1h: Optional[float] = None
    vol_regime_4h: Optional[float] = None
    atr_norm_30m: Optional[float] = None  # WF3 corr ~0.88
    atr_norm_4h: Optional[float] = None  # WF3 corr ~0.95
    adx_30m: Optional[float] = None  # WF3 corr ~0.72 (trend strength)
    adx_4h: Optional[float] = None  # WF4 corr ~0.76-0.81 (trend strength longer)
    # v25: di_direction (RegimeContinuousHead signal idx=3) — corr ~0.93-0.95
    di_dir_30m: Optional[float] = None  # in [-1,1]: +1=bull, -1=bear
    di_dir_4h: Optional[float] = None  # most reliable directional signal
    # v26+: di_dir_1h corr=0.942±0.006 — fills the 30m->4h gap, triple-TF confirmation
    di_dir_1h: Optional[float] = None
    # MarketStructure (MarketStructureHead): brk_auc ~0.87
    brk_prob_1h: Optional[float] = None  # P(BREAKOUT) on 1H
    brk_prob_4h: Optional[float] = None  # P(BREAKOUT) on 4H
    # v26+: brk_prob_30m — more stable (std=0.003), primary for the MR gate
    brk_prob_30m: Optional[float] = None  # P(BREAKOUT) on 30min
    # v26+: regime uncertainty — [0,1]: high = ambiguous / transition signals
    # computed from |tmom_4h|, |di_dir_4h|, |brk_prob_4h - 0.5|
    # used in passes_entry_filter to block entries in transition regimes
    regime_uncertainty: Optional[float] = None
    hidden: Optional[any] = None
    log_prob: Optional[float] = None
    # ── v26: StrategyOutcomeHead signals ─────────────────────────────────
    # P(TP) of the proposed action — from the StrategyOutcomeHead 2-head
    # p_tp = p_hit × p_tp_given_hit
    proposal_tp_prob: Optional[float] = None  # P(TP) of this specific action
    strategy_edge: Optional[float] = None  # proposal_tp - best_alternative_tp
    # raw values of all 4 combos (filled if strategy_outcome_head is available)
    p_tp_trend_long: Optional[float] = None
    p_tp_trend_short: Optional[float] = None
    p_tp_mr_long: Optional[float] = None
    p_tp_mr_short: Optional[float] = None
    bars_to_force_close: Optional[int] = None  # v26: 15min bars to the next 21:30 UTC
    value: Optional[float] = None
    rollout_step: Optional[any] = None
    is_ppo: bool = False
    strategy_type: str = "trend"  # v26: "trend" | "mr"
    # advantage score and diagnostics
    q_hold: Optional[float] = None  # Q(HOLD) of the same state
    filter_check: Optional[str] = None  # entry-filter verdict, also with the filters off
    reject_stage: Optional[str] = None  # where a trade proposal was stopped, if it was
    shadow_action: int = 0  # eval: the agent's trade choice its conviction filters stopped


# ══════════════════════════════════════════════════════════════════════════════
# TradeArbiter  (scoring + selection)
# ══════════════════════════════════════════════════════════════════════════════
class TradeArbiter:
    """
    Ranks the agent's trade proposals and keeps the best ones within the daily quota.

    By default it doesn't second-guess the agent: the score is the agent's own advantage
    over HOLD, Q(action) - Q(HOLD), and no entry filter is applied — the trend / regime /
    breakout signals the filters use are already in the agent's state.

    Options (arbiter_cfg):
      score_mode     "advantage" (default) | "legacy": the v26 score, Q plus trend / regime /
                     breakout / strategy-outcome bonuses and an ATR and time penalty
      entry_filters  False (default) | True: the v26 entry filters (trend: tmom / ADX 4h /
                     DI 4h agreeing with the direction, regime uncertainty; MR: low ADX and
                     breakout probability; time to the force close), with the SAME thresholds
                     in training and eval
    The filter verdict is always computed (proposal.filter_check), also with the filters
    off, so the eval report can show what the filters would have removed.
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self._arbiter_cfg = cfg  # alias for property access
        self.score_mode = str(cfg.get("score_mode", "advantage"))
        if self.score_mode not in ("advantage", "legacy"):
            raise ValueError(f"score_mode must be 'advantage' or 'legacy', got {self.score_mode}")
        self.entry_filters = bool(cfg.get("entry_filters", False))
        self.max_trades_per_day = int(cfg.get("max_trades_per_day", 5))
        self.min_score_threshold = cfg.get("min_score_threshold", -1e9)
        # legacy score weights
        self.w_q = cfg.get("w_q", 1.0)
        self.w_conf = cfg.get("w_conf", 0.5)
        self.w_trend = cfg.get("w_trend", 0.4)
        self.w_atr = cfg.get("w_atr", 0.5)
        self.w_outcome = float(cfg.get("w_outcome", 0.0))
        self.w_strategy_edge = float(cfg.get("w_strategy_edge", 0.0))
        self.tmom_multi_tf = bool(cfg.get("tmom_multi_tf", True))
        # entry filter thresholds (also used by the filter verdict when the filters are off)
        self.tmom_min_eval = float(cfg.get("tmom_min_eval", 0.20))
        self.adx_min_eval = float(cfg.get("adx_min_eval", 0.0))  # legacy 30m ADX gate, off
        self.adx_4h_min = float(cfg.get("adx_4h_min", 0.25))
        self.di_dir_min = float(cfg.get("di_dir_min", 0.15))
        self.brk_min_eval = float(cfg.get("brk_min_eval", 0.35))
        self.adx_ranging_max = float(cfg.get("adx_ranging_max", 0.28))
        self.brk_min_mr = float(cfg.get("brk_min_mr", 0.35))
        self._max_rank_eval = 99

        if cfg.get("allowed_regimes"):
            logger.warning("allowed_regimes configured but regime hard-filter is disabled.")

    # ── score ─────────────────────────────────────────────────────────────────
    def score(self, p: TradeProposal) -> float:
        """Advantage over HOLD (default), or the v26 score with score_mode="legacy"."""
        if self.score_mode == "legacy":
            return self._legacy_score(p)
        q_hold = p.q_hold if p.q_hold is not None else 0.0
        return float(p.q_value - q_hold)

    def _legacy_score(self, p: TradeProposal) -> float:
        """v26 score: Q-value + tmom alignment and encoder bonuses - ATR / time penalties."""
        base = self.w_q * p.q_value + self.w_conf * p.confidence - self.w_atr * p.atr

        tmom = p.tmom_30m if p.tmom_30m is not None else p.trend_alignment
        trend_score = 0.0
        if p.action == 1:  # LONG: reward positive tmom
            trend_score = float(np.clip(tmom / 3.0, -1.0, 1.0))
        elif p.action == 2:  # SHORT: reward negative tmom
            trend_score = float(np.clip(-tmom / 3.0, -1.0, 1.0))

        # Multi-TF confirmation: 30m + 1h in the same direction -> +0.15
        if self.tmom_multi_tf and p.tmom_1h is not None and p.action != 0:
            if p.action == 1 and tmom > 0 and p.tmom_1h > 0:
                trend_score += 0.15
            elif p.action == 2 and tmom < 0 and p.tmom_1h < 0:
                trend_score += 0.15

        # di_direction_4h agreeing with tmom 30m -> +0.10
        if p.di_dir_4h is not None and p.action != 0:
            di_4h = p.di_dir_4h
            if p.action == 1 and tmom > 0 and di_4h > 0.2:
                trend_score += 0.10
            elif p.action == 2 and tmom < 0 and di_4h < -0.2:
                trend_score += 0.10

        # di_direction_1h triple-TF alignment -> +0.08 (TREND only)
        if p.di_dir_1h is not None and p.action not in (0, 3, 4):
            di_1h = p.di_dir_1h
            if p.action == 1 and tmom > 0 and di_1h > 0.15:
                trend_score += 0.08
            elif p.action == 2 and tmom < 0 and di_1h < -0.15:
                trend_score += 0.08

        # breakout probability soft penalty (probably RANGING), max -0.30
        brk_4h = p.brk_prob_4h
        brk_penalty = 0.0
        if brk_4h is not None and self.brk_min_eval > 0 and brk_4h < self.brk_min_eval:
            brk_penalty = float(np.clip((self.brk_min_eval - brk_4h), 0.0, 0.30))

        # MR bonus in a deep-ranging market (low brk_30m), max +0.08
        mr_ranging_bonus = 0.0
        if p.action in (3, 4) and p.brk_prob_30m is not None and self.brk_min_mr > 0:
            gap = float(np.clip(self.brk_min_mr - p.brk_prob_30m, 0.0, self.brk_min_mr))
            mr_ranging_bonus = 0.08 * (gap / self.brk_min_mr)

        # StrategyOutcomeHead bonus (only if w_outcome / w_strategy_edge > 0)
        outcome_bonus = 0.0
        if self.w_outcome > 0 and p.proposal_tp_prob is not None:
            outcome_bonus += self.w_outcome * (p.proposal_tp_prob - 0.25)
        if self.w_strategy_edge > 0 and p.strategy_edge is not None:
            outcome_bonus += self.w_strategy_edge * p.strategy_edge

        # StrategyOutcomeHead direction agreement: +/-0.05 when the P(TP) gap > 0.03
        pairs = {
            1: ("p_tp_trend_long", "p_tp_trend_short", 1),
            2: ("p_tp_trend_long", "p_tp_trend_short", -1),
        }
        pairs.update(
            {3: ("p_tp_mr_long", "p_tp_mr_short", 1), 4: ("p_tp_mr_long", "p_tp_mr_short", -1)}
        )
        if p.action in pairs:
            long_k, short_k, sign = pairs[p.action]
            p_long, p_short = getattr(p, long_k), getattr(p, short_k)
            if p_long is not None and p_short is not None:
                gap = sign * (p_long - p_short)
                if gap > 0.03:
                    outcome_bonus += 0.05
                elif gap < -0.03:
                    outcome_bonus -= 0.05

        # TREND trades close to the force close: malus, doubled below the hard threshold
        time_penalty = 0.0
        bars_fc = getattr(p, "bars_to_force_close", None)
        if bars_fc is not None and p.action in (1, 2):
            soft = float(self._arbiter_cfg.get("trend_soft_close_bars", 24))
            hard = float(self._arbiter_cfg.get("trend_hard_close_bars", 12))
            w_time = float(self._arbiter_cfg.get("w_time_penalty", 0.20))
            if bars_fc < soft:
                time_penalty = w_time * (1.0 - bars_fc / soft)
                if bars_fc < hard:
                    time_penalty *= 2.0

        # TREND bonus in training (balances the MR drift), off in eval
        trend_train_bonus = 0.0
        if getattr(self, "_current_mode", "train") == "train" and p.action in (1, 2):
            trend_train_bonus = float(self._arbiter_cfg.get("trend_train_bonus", 0.0))

        return (
            base
            + self.w_trend * trend_score
            - brk_penalty
            + trend_train_bonus
            + mr_ranging_bonus
            + outcome_bonus
            - time_penalty
        )

    # ── entry filters ─────────────────────────────────────────────────────────
    def passes_entry_filter(self, p: TradeProposal, mode: str = "train") -> bool:
        """True if the proposal passes the entry filters (always, with the filters off)."""
        return self.entry_filter_reason(p, mode) == "pass"

    def entry_filter_reason(self, p: TradeProposal, mode: str = "train") -> str:
        """ "pass", "hold", or the first filter that failed (only with entry_filters on)."""
        if p.action == 0:
            return "hold"
        if not self.entry_filters:
            return "pass"
        return self.filter_check(p)

    def filter_check(self, p: TradeProposal) -> str:
        """
        Verdict of the entry filters, whether they're on or not: "pass", "hold", or the
        reason of the first filter that fails. Same thresholds in training and eval.
        """
        if p.action == 0:
            return "hold"
        cfg = self._arbiter_cfg
        is_trend = p.action in (1, 2)

        if p.bars_to_force_close is not None:
            key = "trend_min_bars_to_close" if is_trend else "mr_min_bars_to_close"
            min_bars = int(cfg.get(key, 0))
            if min_bars > 0 and p.bars_to_force_close < min_bars:
                return "trend_time_lockout" if is_trend else "mr_time_lockout"

        if not is_trend:  # MR: only in a ranging market
            if (p.adx_4h if p.adx_4h is not None else 1.0) >= self.adx_ranging_max:
                return "mr_adx_too_high"
            if (p.brk_prob_4h if p.brk_prob_4h is not None else 1.0) >= self.brk_min_mr:
                return "mr_brk_4h_too_high"
            if (p.brk_prob_30m if p.brk_prob_30m is not None else 1.0) >= self.brk_min_mr:
                return "mr_brk_30m_too_high"
            return "pass"

        max_unc = float(cfg.get("max_regime_uncertainty", 0.72))
        if p.regime_uncertainty is not None and p.regime_uncertainty > max_unc:
            return "regime_uncertainty"
        tmom = p.tmom_30m if p.tmom_30m is not None else 0.0
        adx_30m_v = p.adx_30m if p.adx_30m is not None else 1.0
        adx_4h_v = p.adx_4h if p.adx_4h is not None else 1.0
        di_4h = p.di_dir_4h if p.di_dir_4h is not None else 0.0
        if self.adx_4h_min > 0 and adx_4h_v < self.adx_4h_min:
            return "trend_adx_4h_too_low"
        if self.adx_min_eval > 0 and adx_30m_v < self.adx_min_eval:
            return "trend_adx_30m_too_low"
        if abs(tmom) < self.tmom_min_eval:
            return "trend_tmom_too_low"
        if p.action == 1 and tmom < 0:
            return "trend_long_wrong_sign"
        if p.action == 2 and tmom > 0:
            return "trend_short_wrong_sign"
        if self.di_dir_min > 0:
            if abs(di_4h) < self.di_dir_min:
                return "trend_di_too_low"
            if p.action == 1 and di_4h < 0:
                return "trend_long_di_wrong_sign"
            if p.action == 2 and di_4h > 0:
                return "trend_short_di_wrong_sign"
        return "pass"

    # ── selection ─────────────────────────────────────────────────────────────
    def select(
        self,
        proposals: list[TradeProposal],
        remaining_trades: int,
    ) -> dict[str, TradeProposal]:
        """Approves the best-scored trade proposals, at most remaining_trades of them."""
        mode = getattr(self, "_current_mode", "train")
        candidates = []
        for p in proposals:
            if p.action == 0:
                continue
            p.filter_check = self.filter_check(p)
            reason = self.entry_filter_reason(p, mode)
            if reason == "pass":
                candidates.append(p)
            else:
                p.filter_reason = reason
                p.reject_stage = "entry_filter"

        scored = sorted(((self.score(p), p) for p in candidates), key=lambda x: x[0], reverse=True)
        for rank, (_, p) in enumerate(scored):
            p.rank = rank
        hold_base = len(scored)
        for p in proposals:
            if p.action == 0:
                p.rank = hold_base
                hold_base += 1

        # optional cap on the rank approved in eval (max_rank_eval, off by default)
        if mode != "train" and self._max_rank_eval < 99:
            for _, p in scored[self._max_rank_eval :]:
                p.reject_stage = "max_rank"
            scored = scored[: self._max_rank_eval]

        approved = {}
        for i, (sval, p) in enumerate(scored):
            if i < remaining_trades and sval > self.min_score_threshold:
                p.approved = True
                approved[p.asset] = p
            else:
                p.approved = False
                p.reject_stage = "daily_quota" if i >= remaining_trades else "min_score"
        return approved


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio
# ══════════════════════════════════════════════════════════════════════════════
class Portfolio:
    """
    Tracks the balance, the absolute drawdown (from initial) and the trailing one (from peak equity).

    Absolute drawdown   -> prop firm rule  (initial - balance) / initial
    Trailing drawdown   -> extra safety     (peak - balance)   / peak
    block_new_trades in MultiAssetRunner turns on if BOTH go past the baseline.
    """

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial = float(initial_capital)
        self.balance = float(initial_capital)
        self.peak_balance = float(initial_capital)  # ← for the trailing DD
        self.reserved_map: dict[str, float] = {}
        self.equity_curve: list[float] = [float(initial_capital)]
        self.drawdown = 0.0
        self.drawdown_trailing = 0.0  # ← trailing from the peak
        self.max_dd_seen = 0.0
        self.drawdown_history: list[float] = [0.0]
        self.daily_dd = 0.0
        self.current_day = None
        self.day_start_balance = self.initial
        self.intraday_peak = self.initial
        self.daily_pnl_pct = 0.0  # P&L % from the balance at the start of the day

        # Streak tracking
        self.consecutive_losses = 0  # SLs in a row (reset on TP)
        self.daily_losses = 0  # SLs today
        self._loss_day = None

    def available(self) -> float:
        return max(0.0, self.balance - sum(self.reserved_map.values()))

    def reserve(self, asset: str, amount: float) -> bool:
        if amount <= 0:
            return False
        if amount <= self.available() + 1e-12:
            self.reserved_map[asset] = self.reserved_map.get(asset, 0.0) + float(amount)
            return True
        return False

    def release(self, asset: str):
        if asset in self.reserved_map:
            del self.reserved_map[asset]

    def apply_pnl(self, asset: str, pnl: float, timestamp=None):
        if asset in self.reserved_map:
            del self.reserved_map[asset]

        self.balance += float(pnl)
        self.equity_curve.append(self.balance)

        # update peak equity and trailing DD
        self.peak_balance = max(self.peak_balance, self.balance)
        self.drawdown_trailing = max(
            0.0, (self.peak_balance - self.balance) / max(self.peak_balance, 1e-8)
        )

        # absolute drawdown
        self.drawdown = max(0.0, (self.initial - self.balance) / max(self.initial, 1e-8))
        self.drawdown_history.append(self.drawdown)
        if self.drawdown > self.max_dd_seen:
            self.max_dd_seen = self.drawdown

        # daily P&L %
        dsb = max(self.day_start_balance, 1e-8)
        self.daily_pnl_pct = (self.balance - dsb) / dsb

        # ✅ FIX: daily_losses reset on the date, whatever the sign of the PnL
        today = timestamp.date() if timestamp is not None else None
        if today is not None and today != self._loss_day:
            self._loss_day = today
            self.daily_losses = 0  # always reset when the day changes, not only on SL

        if pnl < 0:
            self.consecutive_losses += 1
            if today is not None:
                self.daily_losses += 1
        else:
            self.consecutive_losses = 0

        # intraday daily DD (unchanged)
        if timestamp is None:
            return
        today = timestamp.date()
        if self.current_day != today:
            self.current_day = today
            self.day_start_balance = self.balance
            self.intraday_peak = self.balance
            self.daily_dd = 0.0
            return
        if self.balance > self.intraday_peak:
            self.intraday_peak = self.balance
        if self.intraday_peak > 0:
            self.daily_dd = max(0.0, (self.intraday_peak - self.balance) / self.intraday_peak)


# ══════════════════════════════════════════════════════════════════════════════
# MultiAssetRunner
# ══════════════════════════════════════════════════════════════════════════════
class MultiAssetRunner:
    """
    Multi-asset runner for the Rainbow DQN.

    Every step:
      1. collects the state of every active asset
      2. Rainbow: q_values -> Q-gap filter -> act -> proposal
      3. correlation cap: max 2 trades in the same direction
      4. arbiter: picks the best proposals within the daily quota
      5. DD risk ladder: scales risk_amount with the DD level
      6. runs env.step() on every asset
      7. applies the PnL to the portfolio, updates streaks and safety blocks
    """

    def __init__(
        self,
        envs: dict[str, Any],
        asset_class_map: dict[str, str],
        agents: dict[str, Any],
        arbiter: TradeArbiter,
        runner_cfg: dict[str, Any],
    ):
        self.envs = envs
        self.asset_class_map = asset_class_map
        self.agents = agents
        self.arbiter = arbiter
        self.runner_cfg = runner_cfg
        self.mode = runner_cfg.get("mode", "train")
        self.device = runner_cfg.get("device", "cpu")

        # Reward shaping
        self.arbiter_bonus = float(runner_cfg.get("arbiter_bonus", 0.10))
        self.arbiter_sl_penalty = float(runner_cfg.get("arbiter_sl_penalty", 0.12))
        self.rejection_penalty_weak = float(runner_cfg.get("rejection_penalty_weak", 0.05))
        self.hold_bias_penalty = float(runner_cfg.get("hold_bias_penalty", 0.01))

        # portfolio and baseline DD
        self.portfolio = Portfolio(runner_cfg.get("initial_capital", 100_000.0))
        self.BASELINE_MAX_DD = float(runner_cfg.get("baseline_max_dd", 0.12))
        self.BASELINE_DAILY_DD = float(runner_cfg.get("baseline_daily_dd", 0.04))
        self.BASELINE_AVG_DD = float(runner_cfg.get("baseline_avg_dd", 0.06))
        self.block_new_trades = False

        # Rainbow conviction filter
        # MIN_Q_GAP applied ONLY in eval/live.
        # In training = 0.0: NoisyLinear explores freely, the Qs diverge with experience.
        # Applying it in training blocks exploration -> the agent never learns.
        self.MIN_Q_GAP_EVAL = (
            float(runner_cfg.get("min_q_gap", 0.05)) if self.mode != "train" else 0.0
        )
        # ── Eval: filter on the highest approvable rank ───────────────────────
        # max_rank_eval=2 -> only rank1+rank2 (0-indexed: 0,1)
        # default 99 = no filter (Run1 compatibility)
        self._max_rank_eval = int(runner_cfg.get("max_rank_eval", 99))
        # ── Eval only: vote filter + positive Q requirement ──────────────
        # vote_n: number of NoisyLinear forward passes for the vote
        # vote_threshold: minimum agreeing votes to accept an action
        # require_positive_q: reject if best_trade_q <= 0 (the agent expects a loss)
        self._eval_vote_n = int(runner_cfg.get("eval_vote_n", 3))
        self._eval_vote_threshold = int(runner_cfg.get("eval_vote_threshold", 2))
        self._eval_require_positive_q = bool(runner_cfg.get("eval_require_positive_q", False))
        # Strong signal bypass: if Q-gap >= mult × min_q_gap, skip the vote
        # Avoids 0 trades when the Q-values are all negative and NoisyLinear never agrees
        # Default 2.0: bypass at gap ≥ 0.14 with min_q_gap=0.07
        self._eval_vote_strong_bypass = float(
            runner_cfg.get("eval_vote_strong_bypass", 1.5)
        )  # Run1: 1.5 (→1.75→2.0)
        self.MIN_Q_GAP_TRAIN = 0.0  # always 0 — the training filter is off by design
        self.MIN_Q_GAP = self.MIN_Q_GAP_EVAL  # alias for logging

        # after N consecutive SLs -> lower risk
        self.STREAK_TRIGGER = int(runner_cfg.get("streak_protection_n", 3))
        self.STREAK_RISK_SCALE = float(runner_cfg.get("streak_risk_factor", 0.50))

        # in eval/live: blocks the day's trading after N daily SLs
        self.MAX_DAILY_LOSSES = int(runner_cfg.get("max_daily_losses", 3))
        # daily P&L thresholds for the concurrent positions gate
        # -1% -> down to 1 position; -2% -> blocks everything
        self.DAILY_REDUCE_PCT = float(runner_cfg.get("daily_reduce_pct", -0.01))
        self.DAILY_BLOCK_PCT = float(runner_cfg.get("daily_block_pct", -0.02))
        # concurrent positions: normal=2, after -1%=1, after -2%=0
        self.MAX_CONCURRENT = int(runner_cfg.get("max_concurrent_positions", 1))
        # In eval a lower limit keeps the trade volume under control.
        # eval_max_concurrent=2 -> max 2 open positions in eval (default: same as training).
        # With max_rank_eval=2 already filtering to rank1+rank2, concurrent cuts further.
        self.EVAL_MAX_CONCURRENT = int(runner_cfg.get("eval_max_concurrent", self.MAX_CONCURRENT))

        # v23: OutcomeHead filter — probabilistic TP/SL filter from the encoder
        # eval/live only (off in training, so exploration isn't blocked)
        # outcome_threshold=0.0 -> off; threshold=0.55 -> enter only if P(TP)>=55%
        self.outcome_threshold = 0.0  # disabled: pretraining auc ~0.50 → random filter
        # asset -> outcome_head map (loaded by pipeline.py)
        self.outcome_heads: dict[str, Any] = runner_cfg.get("outcome_heads", {})
        # statistics counter
        self.arbiter_stats_outcome_filtered = 0

        # ── Curriculum, shaping, soft reset ──────────────────────────
        self.curriculum = runner_cfg.get(
            "curriculum", {"steps": [0, 20000, 50000], "max_trades": [2, 4, 5]}
        )
        self.warmup_steps = int(runner_cfg.get("warmup_steps", 0))
        self.decay_steps = int(runner_cfg.get("decay_steps", 150_000))
        self.min_shaping_scale = float(runner_cfg.get("min_shaping_scale", 0.2))
        self.soft_reset_every_steps = int(runner_cfg.get("soft_reset_every_steps", 10_000))

        # ── Asset health (optional) ───────────────────────────────────
        # ── Asset health (soft penalty, applied in _score) ────────────
        # On by default in v25. The penalty starts only after health_min_trades.
        # Threshold PF=1.05 (not 0.85): a clearly broken asset.
        # In Run 1 with min_trades=30 it rarely fires (~once in 6 months).
        # From Run 2 lower it to min_trades=20 to react sooner.
        self.health_enabled = bool(runner_cfg.get("health_filter_enabled", False))
        self.health_min_trades = int(runner_cfg.get("health_min_trades", 30))
        self.health_min_expectancy = float(runner_cfg.get("health_min_expectancy", 0.0))
        self.health_min_pf = float(runner_cfg.get("health_min_pf", 1.05))
        self.health_penalty_dist_all = float(runner_cfg.get("health_penalty_dist_all", 0.25))
        # cap on the penalty, so an asset never freezes completely
        self.health_max_penalty = float(runner_cfg.get("health_max_penalty", 0.60))
        self.asset_health = {
            a: {"recent_pnl": deque(maxlen=int(runner_cfg.get("health_window_size", 50)))}
            for a in self.envs.keys()
        }

        # ── v25 joint training mode ──────────────────────────────────
        # If True: state vector dim 183-186 populated with portfolio-wide
        # context. Otherwise zero (single-class). Run 1=False, Run 2/3=True.
        # The agent networks can stay class-specific with joint=True too;
        # it's the portfolio context that is shared, not the weights.
        self.joint_mode = bool(runner_cfg.get("joint_mode", False))
        # ON only in eval/live. Filters single trades when the situation
        # is anomalous (regime mismatch, ATR spike, uncertain agent, low confidence).
        # 4 conditions, each one can reject the trade.
        dg_cfg = runner_cfg.get("decision_gate", {})
        self.dg_enabled = bool(dg_cfg.get("enabled", False))
        self.dg_regime_mismatch = bool(dg_cfg.get("regime_mismatch_block", True))
        self.dg_atr_spike_mult = float(dg_cfg.get("atr_spike_mult", 2.5))
        self.dg_atr_window = int(dg_cfg.get("atr_window", 50))
        self.dg_action_pingpong_n = int(dg_cfg.get("action_pingpong_n", 3))
        self.dg_min_conf_product = float(dg_cfg.get("min_conf_product", 0.05))
        # State per asset: deque of recent actions, ATR history for spike detection
        self._dg_recent_actions: dict[str, deque] = {a: deque(maxlen=5) for a in self.envs.keys()}
        self._dg_atr_history: dict[str, deque] = {
            a: deque(maxlen=self.dg_atr_window) for a in self.envs.keys()
        }
        self._dg_stats = {
            "regime_mismatch": 0,
            "atr_spike": 0,
            "action_pingpong": 0,
            "low_confidence": 0,
        }

        # ── Kill switch (v25) — portfolio-level emergency stop ────────
        # ON only in eval/live. Blocks the WHOLE portfolio for persistence_min
        # minutes after a trigger. Persistent state (once on, it stays on).
        # 4 conditions, any one of them turns it on.
        ks_cfg = runner_cfg.get("kill_switch", {})
        self.ks_enabled = bool(ks_cfg.get("enabled", False))
        self.ks_dd_velocity_pct = float(ks_cfg.get("dd_velocity_pct", -0.01))  # -1% in window
        self.ks_dd_velocity_window = int(ks_cfg.get("dd_velocity_window_min", 30))  # minutes
        self.ks_loss_streak = int(ks_cfg.get("loss_streak_n", 5))
        self.ks_pnl_anomaly_mult = float(ks_cfg.get("pnl_anomaly_mult", 3.0))  # loss > 3× risk
        self.ks_stuck_n = int(ks_cfg.get("stuck_positions_n", 3))
        self.ks_stuck_min_bars_pct = float(ks_cfg.get("stuck_min_bars_pct", 0.80))
        self.ks_persistence_min = int(ks_cfg.get("persistence_min", 240))  # 4h block
        # kill switch state
        self._ks_active = False
        self._ks_active_since: Optional[pd.Timestamp] = None
        self._ks_active_reason = ""
        # P&L history for the drawdown velocity (timestamp + balance)
        self._ks_balance_history: deque = deque(maxlen=200)
        # Recent trade outcomes portfolio-wide (for the loss streak)
        self._ks_recent_outcomes: deque = deque(maxlen=10)
        self._ks_stats = {
            "dd_velocity": 0,
            "loss_streak": 0,
            "pnl_anomaly": 0,
            "stuck_positions": 0,
        }

        # ─── v25: Decision gate ──────────────────────────────────────────
        # Per-trade filter applied AFTER score and BEFORE arbiter.select().
        # 4 conditions that block a single trade in eval/live:
        #   1. Regime mismatch: brk_prob_4h high but vol_regime_4h low (likely fakeout)
        #   2. ATR explosion: current ATR > N× rolling ATR (volatility spike)
        #   3. Q-disagreement: action ping-pong in the last K steps
        #   4. Encoder confidence product: tmom·di_dir·adx < threshold
        self.decision_gate_enabled = bool(runner_cfg.get("decision_gate_enabled", False))
        self.dg_brk_high = float(runner_cfg.get("dg_brk_high", 0.70))
        self.dg_vol_low = float(runner_cfg.get("dg_vol_low", 0.30))
        self.dg_atr_explosion_mult = float(runner_cfg.get("dg_atr_explosion_mult", 2.5))
        self.dg_atr_window = int(runner_cfg.get("dg_atr_window", 50))
        self.dg_action_ping_window = int(runner_cfg.get("dg_action_ping_window", 5))
        self.dg_action_ping_max_changes = int(runner_cfg.get("dg_action_ping_max", 3))
        self.dg_confidence_product_min = float(runner_cfg.get("dg_conf_product_min", 0.05))
        # per-asset action history for the Q-disagreement detection
        self._action_history: dict[str, deque] = {
            a: deque(maxlen=max(2, self.dg_action_ping_window)) for a in self.envs.keys()
        }
        # Per-asset ATR rolling buffer
        self._atr_history: dict[str, deque] = {
            a: deque(maxlen=self.dg_atr_window) for a in self.envs.keys()
        }
        # Stats
        self.arbiter_stats_dg_filtered = {
            "regime_mismatch": 0,
            "atr_explosion": 0,
            "action_ping": 0,
            "low_confidence": 0,
        }

        # ─── v25: Kill switch ────────────────────────────────────────────
        # Portfolio-wide circuit breaker. Once on it stays on for
        # `ks_cooldown_minutes`. Four conditions:
        #   1. DD velocity: -X% in Y minutes across assets -> block
        #   2. Portfolio-wide loss streak: N consecutive SLs on different assets
        #   3. P&L anomaly: a single trade losing > N× the planned risk
        #   4. Stuck positions: M+ stale positions (no MFE/MAE) for > T bars
        self.kill_switch_enabled = bool(runner_cfg.get("kill_switch_enabled", False))
        self.ks_dd_velocity_pct = float(runner_cfg.get("ks_dd_velocity_pct", -0.01))  # -1%
        self.ks_dd_velocity_window_min = int(
            runner_cfg.get("ks_dd_velocity_window_min", 30)
        )  # 30 min
        self.ks_dd_velocity_min_assets = int(runner_cfg.get("ks_dd_velocity_min_assets", 3))
        self.ks_loss_streak = int(runner_cfg.get("ks_loss_streak_portfolio", 5))
        self.ks_pnl_anomaly_mult = float(runner_cfg.get("ks_pnl_anomaly_mult", 3.0))
        self.ks_stuck_positions_n = int(runner_cfg.get("ks_stuck_positions_n", 3))
        self.ks_stuck_bars_threshold = int(runner_cfg.get("ks_stuck_bars_threshold", 30))
        self.ks_cooldown_steps = int(
            runner_cfg.get("ks_cooldown_steps", 100)
        )  # ~25 min @ 15min bars
        # State
        self._kill_active_until_step = -1
        self._kill_reason = None
        # Detection buffers
        self._pnl_recent_window = deque(maxlen=200)  # (step, pnl_pct, asset)
        self._loss_streak_portfolio = 0  # consecutive SLs across assets
        self.arbiter_stats_ks = {
            "dd_velocity": 0,
            "loss_streak": 0,
            "pnl_anomaly": 0,
            "stuck_positions": 0,
            "total_blocks": 0,
        }

        # ── Statistics ───────────────────────────────────────────────
        self.arbiter_stats = {
            "approved": 0,
            "rejected": 0,
            "total_proposals": 0,
            "qgap_filtered": 0,
            "corr_filtered": 0,
            "streak_filtered": 0,
            "daily_loss_blocked": 0,
            "outcome_filtered": 0,
            "sum_rank_of_approved": 0.0,
            "count_ranked": 0,
            "sum_q_of_approved": 0.0,
            "asset_usage": {},
            # ── Rank performance tracking ───────────────────────────
            # rank0 = best proposal of the step (already in eval: rank1_tp/sl)
            # rank1 = second proposal, rank2 = third
            "rank1_tp": 0,
            "rank1_sl": 0,  # rank0 (best)
            "rank2_tp": 0,
            "rank2_sl": 0,  # rank1 (second)
            "rank3_tp": 0,
            "rank3_sl": 0,  # rank2 (third)
            # ── Confidence bracket tracking (on the Q-gap) ──────────
            # low:  min_q_gap ≤ q_gap < 2× min_q_gap
            # mid:  2× ≤ q_gap < 4× min_q_gap
            # high: q_gap ≥ 4× min_q_gap
            "conf_low_tp": 0,
            "conf_low_sl": 0,
            "conf_mid_tp": 0,
            "conf_mid_sl": 0,
            "conf_high_tp": 0,
            "conf_high_sl": 0,
            # ── Strategy tracking (trend vs mean-reversion) ─────────
            "trend_tp": 0,
            "trend_sl": 0,
            "mr_tp": 0,
            "mr_sl": 0,
            "entry_filter_blocked": 0,
            "decision_gate_blocked": 0,
            "vote_blocked": 0,
            "positive_q_blocked": 0,
            "rank_blocked": 0,
            "concurrent_blocked": 0,
        }
        self.usage_history: list[str] = []
        self.rolling_window = int(runner_cfg.get("arbiter_rolling_window", 20))
        self.global_step = 0
        self.max_steps = 0  # set by run_loop through set_max_steps()
        self.global_daily_date = None
        # the portfolio daily quota binds in eval / live; in training only if asked:
        # an exploring agent fills it in the first bars of the day and would learn
        # the value of its trades almost only at 00:00 UTC
        self.train_daily_quota = bool(runner_cfg.get("train_daily_quota", False))
        self._allowed_max_trades = None
        self._entry_block = ""
        self.decision_log: list[dict] = []  # eval: the agent's trade decisions, for the report
        self._daily_sl_warned_day = None  # ✅ explicit initialization
        self.trades_opened_today = 0

        logger.info(
            "MultiAssetRunner | mode=%s | DD base=%.3f daily=%.3f avg=%.3f | "
            "Q-gap train=%.3f eval=%.3f | concurrent=%d | streak_N=%d | "
            "daily_SL=%d reduce@%.0f%% block@%.0f%%",
            self.mode,
            self.BASELINE_MAX_DD,
            self.BASELINE_DAILY_DD,
            self.BASELINE_AVG_DD,
            self.MIN_Q_GAP_TRAIN,
            self.MIN_Q_GAP_EVAL,
            self.MAX_CONCURRENT,
            self.STREAK_TRIGGER,
            self.MAX_DAILY_LOSSES,
            self.DAILY_REDUCE_PCT * 100,
            self.DAILY_BLOCK_PCT * 100,
        )
        logger.info(
            "OutcomeHead filter: OFF (auc ~0.50 in pretraining — enable post DDQN fine-tuning)"
        )

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────
    def _resolve_schedule(self, schedule: list) -> list:
        """
        Turns a schedule into absolute steps if the thresholds are fractions [0.0, 1.0].
        Example: [(0.0, 0.50), (0.25, 0.38)] with max_steps=40000
                 → [(0, 0.50), (10000, 0.38)]
        If the thresholds are already absolute steps (>= 1.0 for any non-zero threshold),
        they're returned as they are, for legacy configs.
        """
        if not schedule:
            return schedule
        # fraction based if any non-zero threshold is < 2.0
        non_zero = [t for t, _ in schedule if t > 0]
        if non_zero and all(t < 2.0 for t in non_zero):
            max_s = max(1, getattr(self, "max_steps", 40_000))
            return [(int(round(t * max_s)), v) for t, v in schedule]
        return schedule

    def _apply_max_dd_schedule(self):
        """Updates max_dd and daily_dd_limit of the envs + the arbiter BASELINE from the schedule."""
        if self.mode in ("eval", "live"):
            ev_max = float(self.runner_cfg.get("eval_max_dd", 0.045))
            ev_daily = float(self.runner_cfg.get("eval_daily_dd_limit", 0.025))
            for env in self.envs.values():
                env.max_dd = ev_max
                env.daily_dd_limit = ev_daily
            return

    def _apply_concurrent_schedule(self):
        """Scales MAX_CONCURRENT and the DD limits with the schedule (training only)."""
        if self.mode != "train":
            return

        # ── Concurrent schedule ───────────────────────────────────────
        schedule = self._resolve_schedule(self.runner_cfg.get("concurrent_schedule", []))
        if schedule:
            curr = schedule[0][1]
            for thr, val in sorted(schedule):
                if self.global_step >= thr:
                    curr = val
                else:
                    break
            self.MAX_CONCURRENT = int(curr)

        # ── Env max_dd schedule ───────────────────────────────────────
        dd_sched = self._resolve_schedule(self.runner_cfg.get("max_dd_schedule", [(0, 0.30)]))
        curr = dd_sched[-1][1]
        for thr, val in sorted(dd_sched):
            if self.global_step >= thr:
                curr = val
            else:
                break

        daily_sched = self._resolve_schedule(self.runner_cfg.get("daily_dd_schedule", [(0, 0.08)]))
        curr_d = daily_sched[-1][1]
        for thr, val in sorted(daily_sched):
            if self.global_step >= thr:
                curr_d = val
            else:
                break

        for env in self.envs.values():
            env.max_dd = float(curr)
            env.daily_dd_limit = float(curr_d)
        baseline_scale = float(self.runner_cfg.get("baseline_dd_env_ratio", 0.60))
        self.BASELINE_MAX_DD = float(curr) * baseline_scale
        self.BASELINE_DAILY_DD = float(curr_d) * baseline_scale

    def _maybe_soft_reset(self):
        if self.mode != "train" or self.soft_reset_every_steps <= 0:
            return
        if self.global_step % self.soft_reset_every_steps != 0:
            return
        if any(env.position is not None for env in self.envs.values()):
            return
        for env in self.envs.values():
            try:
                env.soft_reset()
            except Exception:
                logger.exception("soft_reset failed")

    def _shaping_scale(self) -> float:
        scale = 1.0 - float(self.global_step) / max(1.0, float(self.decay_steps))
        scale = max(self.min_shaping_scale, scale)
        return scale if self.mode == "train" else max(self.min_shaping_scale * 0.3, 0.0)

    def _asset_health_penalty(self, asset: str) -> float:
        """
        Progressive penalty for broken assets (eval/live).
        Returns penalty in [0, health_max_penalty] to subtract from _score().

        Components:
          - Expectancy penalty: |exp| / (|exp| + 100), normalized to [0, 1]
            non-linear saturation -> the penalty grows but doesn't explode on outliers
          - PF penalty: (min_pf - pf) clipped at [0, 0.5]
            more weight when pf < 1.0 (the asset really loses)

        Minimum window: health_min_trades. Below it -> penalty=0.
        Total cap: health_max_penalty (default 0.60).
        """
        if not self.health_enabled:
            return 0.0
        pnl_list = self.asset_health.get(asset, {}).get("recent_pnl", [])
        if len(pnl_list) < self.health_min_trades:
            return 0.0
        recent = np.array(pnl_list)
        exp = float(np.mean(recent))
        profits = recent[recent > 0].sum()
        losses = abs(recent[recent < 0].sum()) + 1e-8
        pf = profits / losses

        penalty = 0.0
        if exp < self.health_min_expectancy:
            # non-linear saturation: -50 USD -> 0.33 | -200 -> 0.67 | -500 -> 0.83
            penalty += abs(exp - self.health_min_expectancy) / (
                abs(exp - self.health_min_expectancy) + 100.0
            )
        if pf < self.health_min_pf:
            # linear, clipped: PF=0.5 -> 0.35 | PF=0.0 -> 0.5 (cap)
            pf_pen = min(self.health_min_pf - pf, 0.5)
            penalty += pf_pen

        return min(penalty, self.health_max_penalty)

    # ─────────────────────────────────────────────────────────────────
    # Decision gate (v25): final pre-trade check on individual proposals
    # 4 conditions, each can reject. Active only in eval/live.
    # ─────────────────────────────────────────────────────────────────
    def _passes_decision_gate(self, p: TradeProposal) -> tuple[bool, str]:
        """
        Check 4 conditions on the proposal and asset state. Returns (ok, reason).
        Reasons logged in self._dg_stats.
        """
        if not self.dg_enabled or self.mode == "train" or p.action == 0:
            return True, ""

        # 1) Regime mismatch: brk_prob_4h high but vol_regime_4h low (compressed)
        # = likely fakeout. Threshold: brk > 0.6 AND vol < 0.3
        if self.dg_regime_mismatch:
            brk = p.brk_prob_4h if p.brk_prob_4h is not None else 0.0
            vol = p.vol_regime_4h if p.vol_regime_4h is not None else 0.5
            if brk > 0.6 and vol < 0.3:
                self._dg_stats["regime_mismatch"] += 1
                return False, f"regime_mismatch (brk={brk:.2f}, vol={vol:.2f})"

        # 2) ATR explosion: current ATR > spike_mult × rolling mean
        # catches news/macro/spikes. Skipped if the window history is too short.
        env = self.envs.get(p.asset)
        if env is not None and env.atr_col_raw and self.dg_atr_spike_mult > 0:
            try:
                idx = min(env.current_step, len(env.df) - 1)
                cur_atr = float(env.df.iloc[idx].get(env.atr_col_raw, 0.0))
                self._dg_atr_history[p.asset].append(cur_atr)
                hist = self._dg_atr_history[p.asset]
                if len(hist) >= int(self.dg_atr_window * 0.5) and cur_atr > 0:
                    median_atr = float(np.median(list(hist)))
                    if median_atr > 0 and cur_atr > self.dg_atr_spike_mult * median_atr:
                        self._dg_stats["atr_spike"] += 1
                        return False, f"atr_spike ({cur_atr/median_atr:.1f}× median)"
            except Exception:
                pass

        # 3) Action ping-pong: in the last N steps the agent changed direction N times
        # e.g. with n=3, sequence [1,2,1] on the asset -> reject
        recent = self._dg_recent_actions[p.asset]
        recent.append(int(p.action))
        if len(recent) >= self.dg_action_pingpong_n:
            last_n = list(recent)[-self.dg_action_pingpong_n :]
            non_zero = [a for a in last_n if a != 0]
            if len(non_zero) == self.dg_action_pingpong_n:
                # all different in a row = ping-pong (1->2->1 or 2->1->2)
                changes = sum(1 for i in range(1, len(non_zero)) if non_zero[i] != non_zero[i - 1])
                if changes >= self.dg_action_pingpong_n - 1:
                    self._dg_stats["action_pingpong"] += 1
                    return False, f"action_pingpong (last={last_n})"

        # 4) Encoder confidence product: |tmom_30m| × |di_dir_4h| × adx_4h < threshold
        # Every signal must be convinced. Non-linear saturation.
        tmom = abs(p.tmom_30m if p.tmom_30m is not None else 0.0)
        didi = abs(p.di_dir_4h if p.di_dir_4h is not None else 0.0)
        adxv = abs(p.adx_4h if p.adx_4h is not None else 0.0)
        # tmom is on a [0, 3] scale; normalized to [0, 1]
        tmom_n = min(tmom / 1.5, 1.0)  # tmom=1.5 -> conf=1.0
        conf_product = tmom_n * didi * adxv
        if conf_product < self.dg_min_conf_product:
            self._dg_stats["low_confidence"] += 1
            return False, f"low_conf_product ({conf_product:.3f})"

        return True, ""

    # ─────────────────────────────────────────────────────────────────
    # Kill switch (v25): portfolio-level emergency stop
    # 4 conditions. Once triggered, blocks portfolio for persistence_min.
    # ─────────────────────────────────────────────────────────────────
    def _check_kill_switch(self) -> tuple[bool, str]:
        """
        Returns (active, reason). Active=True blocks ALL new trades.
        Auto-deactivates after persistence_min minutes.
        Sets self._ks_active state. Once activated, persists.
        """
        if not self.ks_enabled or self.mode == "train":
            return False, ""

        # auto reset after persistence
        if self._ks_active and self._ks_active_since is not None:
            try:
                any_env = next(iter(self.envs.values()))
                cur_ts = any_env.timestamps[min(any_env.current_step, len(any_env.timestamps) - 1)]
                elapsed_min = (cur_ts - self._ks_active_since).total_seconds() / 60.0
                if elapsed_min >= self.ks_persistence_min:
                    logger.warning(
                        "[KILL SWITCH] Auto-reset after %d min (was: %s)",
                        self.ks_persistence_min,
                        self._ks_active_reason,
                    )
                    self._ks_active = False
                    self._ks_active_since = None
                    self._ks_active_reason = ""
                else:
                    return True, self._ks_active_reason
            except Exception:
                return True, self._ks_active_reason

        if self._ks_active:
            return True, self._ks_active_reason

        # Snapshot current timestamp + balance for the drawdown velocity
        try:
            any_env = next(iter(self.envs.values()))
            cur_ts = any_env.timestamps[min(any_env.current_step, len(any_env.timestamps) - 1)]
            cur_balance = self.portfolio.balance
            self._ks_balance_history.append((cur_ts, cur_balance))
        except Exception:
            cur_ts = None

        # 1) Drawdown velocity: P&L < -X% within window minutes
        if cur_ts is not None and self.ks_dd_velocity_window > 0:
            cutoff = cur_ts - pd.Timedelta(minutes=self.ks_dd_velocity_window)
            recent_balances = [(t, b) for (t, b) in self._ks_balance_history if t >= cutoff]
            if len(recent_balances) >= 2:
                start_bal = recent_balances[0][1]
                if start_bal > 0:
                    pct_change = (cur_balance - start_bal) / start_bal
                    if pct_change <= self.ks_dd_velocity_pct:
                        self._ks_stats["dd_velocity"] += 1
                        return self._activate_kill_switch(
                            cur_ts,
                            f"dd_velocity ({pct_change:.2%} in {self.ks_dd_velocity_window}min)",
                        )

        # 2) Portfolio-wide loss streak: the last N trades all losses across assets
        if len(self._ks_recent_outcomes) >= self.ks_loss_streak:
            last = list(self._ks_recent_outcomes)[-self.ks_loss_streak :]
            if all(o < 0 for o in last):
                self._ks_stats["loss_streak"] += 1
                return self._activate_kill_switch(
                    cur_ts, f"loss_streak ({self.ks_loss_streak} consecutive losses)"
                )

        # 3) P&L anomaly: last trade with a loss > anomaly_mult × the planned risk
        # Based on the last closed trade of ANY asset
        for asset, env in self.envs.items():
            if not env.episode_trades:
                continue
            last_trade = env.episode_trades[-1]
            # only recently closed trades (rr < -anomaly_mult)
            rr = float(last_trade.get("rr", 0.0))
            # marker so the same trade doesn't trigger twice
            tid = id(last_trade)
            if rr < -self.ks_pnl_anomaly_mult and tid != getattr(
                self, "_ks_last_anomaly_tid", None
            ):
                self._ks_last_anomaly_tid = tid
                self._ks_stats["pnl_anomaly"] += 1
                return self._activate_kill_switch(
                    cur_ts, f"pnl_anomaly (rr={rr:.2f}, asset={asset})"
                )

        # 4) Stuck positions: N+ positions open for > stuck_min_bars_pct × max_bars
        # without moving (mae+mfe both small)
        if self.ks_stuck_n > 0:
            stuck_count = 0
            for env in self.envs.values():
                if env.position is None:
                    continue
                ob = int(env.position.get("open_bars", 0))
                threshold_bars = int(env.max_bars_per_trade * self.ks_stuck_min_bars_pct)
                if ob < threshold_bars:
                    continue
                mfe = abs(float(env.position.get("max_favorable_excursion", 0.0)))
                mae = abs(float(env.position.get("max_adverse_excursion", 0.0)))
                # "Stuck" = MFE and MAE both < 0.3 (in rr units)
                if mfe < 0.3 and mae < 0.3:
                    stuck_count += 1
            if stuck_count >= self.ks_stuck_n:
                self._ks_stats["stuck_positions"] += 1
                return self._activate_kill_switch(cur_ts, f"stuck_positions ({stuck_count} stale)")

        return False, ""

    def _activate_kill_switch(self, ts, reason: str) -> tuple[bool, str]:
        """Activate kill switch with persistence."""
        self._ks_active = True
        self._ks_active_since = ts
        self._ks_active_reason = reason
        logger.warning(
            "[KILL SWITCH] ACTIVATED: %s | blocking %d min | balance=%.2f",
            reason,
            self.ks_persistence_min,
            self.portfolio.balance,
        )
        return True, reason

    def _record_trade_outcome(self, pnl: float):
        """Called when a trade closes — feeds kill switch outcome history."""
        self._ks_recent_outcomes.append(float(pnl))

    def _risk_reduction_factor(self, current_dd: float, daily_dd: float) -> float:
        """
        Risk reduction by DD level.

        4-level ladder on the fraction of the baseline:
          < 30%  -> factor 1.00  (full risk)
          30-55% -> factor 0.50  (halved)
          55-75% -> factor 0.20  (drastic cut)
          > 75%  -> factor 0.10  (almost stopped, tiny positions only)

        Daily DD ladder:
          < 50%  -> no extra reduction
          50-75% → factor 0.50
          > 75%  → factor 0.20

        Streak protection:
          consecutive_losses >= STREAK_TRIGGER → factor *= STREAK_RISK_SCALE
        """
        mod = self.BASELINE_MAX_DD
        dmod = self.BASELINE_DAILY_DD

        # absolute DD
        if current_dd > mod * 0.75:
            factor = 0.10
        elif current_dd > mod * 0.55:
            factor = 0.20
        elif current_dd > mod * 0.30:
            factor = 0.50
        else:
            factor = 1.00

        # Daily DD
        if daily_dd > dmod * 0.75:
            factor = min(factor, 0.20)
        elif daily_dd > dmod * 0.50:
            factor = min(factor, 0.50)

        # Rolling avg DD
        if len(self.portfolio.drawdown_history) > 500:
            avg_dd = float(np.mean(self.portfolio.drawdown_history[-500:]))
            if avg_dd > self.BASELINE_AVG_DD:
                factor = min(factor, 0.70)

        # Consecutive SL streak
        streak = self.portfolio.consecutive_losses
        if streak >= self.STREAK_TRIGGER:
            streak_factor = max(0.20, self.STREAK_RISK_SCALE ** (streak - self.STREAK_TRIGGER + 1))
            factor = min(factor, streak_factor)

        return factor

    # ──────────────────────────────────────────────────────────────────
    # MAIN STEP
    # ──────────────────────────────────────────────────────────────────
    def _remaining_trades(self) -> int:
        """Trades left today under the portfolio quota (none in training by default)."""
        if self.mode == "train" and not self.train_daily_quota:
            return 10**9
        return max(0, self.arbiter.max_trades_per_day - self.trades_opened_today)

    def _max_concurrent(self, soft_reduce: bool = False) -> int:
        """Open positions allowed now: training schedule, eval_max_concurrent in eval."""
        if soft_reduce:
            return 1
        return self.MAX_CONCURRENT if self.mode == "train" else self.EVAL_MAX_CONCURRENT

    def _portfolio_entry_state(self) -> tuple[bool, str]:
        """(True, "") if the portfolio can take one more trade now, else (False, reason)."""
        if self.block_new_trades:
            return False, "dd_block"
        soft_reduce = False
        if self.mode in ("eval", "live"):
            if self.ks_enabled and self._check_kill_switch()[0]:
                return False, "kill_switch"
            daily_pnl = self.portfolio.daily_pnl_pct
            if daily_pnl <= self.DAILY_BLOCK_PCT or (
                self.portfolio.daily_losses >= self.MAX_DAILY_LOSSES
            ):
                return False, "daily_gate"
            soft_reduce = daily_pnl <= self.DAILY_REDUCE_PCT
        if self._remaining_trades() <= 0:
            return False, "daily_quota"
        n_open = sum(1 for env in self.envs.values() if env.position is not None)
        if n_open >= self._max_concurrent(soft_reduce):
            return False, "position_slots"
        return True, ""

    def _next_mask(self, env) -> list[bool]:
        """Action mask of the env's next state, with the portfolio limits after this step."""
        mask = env.get_valid_action_mask(max_trades_override=self._allowed_max_trades)
        if not self._portfolio_entry_state()[0]:
            mask = [True] + [False] * (len(mask) - 1)
        return mask

    def step_once(self) -> tuple[dict, dict]:
        """
        Main Rainbow DQN step.

        Flow:
        1. daily date reset; can the portfolio open anything now? (DD block, daily gate,
           kill switch, daily quota, free position slots)
        2. for every active asset: state + action mask (env limits + portfolio limits),
           q_values → act → TradeProposal. Trades the portfolio can't take are masked,
           so the agent's choice is executable instead of silently turned into HOLD
        3. health penalty (if on)
        4. Arbiter.select(): rank by advantage over HOLD, daily quota
        5. position slots, risk scaling, Portfolio.reserve()
        6. env.step() → reward shaping, next-state mask for the replay
        7. Portfolio.apply_pnl() -> updates streak/block
        """

        self.global_step += 1
        self._apply_max_dd_schedule()
        self._apply_concurrent_schedule()
        self._maybe_soft_reset()

        active = [a for a, env in self.envs.items() if env.current_step < len(env.df) - 1]
        if not active:
            return {}, {}

        if self.global_step < self.warmup_steps:
            self.usage_history.clear()

        # ── Curriculum max_trades ─────────────────────────────────────
        allowed_max_trades = self.curriculum["max_trades"][-1]
        for s_idx, s_val in enumerate(self.curriculum["steps"]):
            if self.global_step < s_val:
                allowed_max_trades = self.curriculum["max_trades"][max(0, s_idx - 1)]
                break
        self._allowed_max_trades = allowed_max_trades

        # ── Daily date reset ──────────────────────────────────────────
        try:
            current_date = max(
                env.timestamps[env.current_step].date()
                for env in self.envs.values()
                if env.current_step < len(env.timestamps)
            )
        except (ValueError, IndexError):
            any_env = next(iter(self.envs.values()))
            current_date = any_env.timestamps[
                min(any_env.current_step, len(any_env.timestamps) - 1)
            ].date()

        if self.global_daily_date is None or current_date != self.global_daily_date:
            self.global_daily_date = current_date
            self.trades_opened_today = 0
            self.portfolio.daily_losses = 0
            self.portfolio._loss_day = current_date
            self.portfolio.daily_pnl_pct = 0.0  # ✅ explicit P&L% reset when the day changes

        # can the portfolio take one more trade at all? if not, every trade action is
        # masked: the agent sees HOLD as its only choice instead of being overruled
        entry_ok, self._entry_block = self._portfolio_entry_state()

        proposals: list[TradeProposal] = []

        # ── Joint training: compute portfolio-wide context once per step ───
        # Used by env._get_state() to populate dim 183-186 of state vector.
        # In single-class run all envs are same class; values still meaningful.
        if self.joint_mode:
            n_open_total = sum(1 for env in self.envs.values() if env.position is not None)
            max_conc = max(
                1, self.EVAL_MAX_CONCURRENT if self.mode != "train" else self.MAX_CONCURRENT
            )
            port_pos_norm = float(np.clip(n_open_total / max_conc, 0.0, 1.0))
            port_pnl_pct = float(self.portfolio.daily_pnl_pct)
            port_trd_norm = float(
                np.clip(
                    self.trades_opened_today / max(1, self.arbiter.max_trades_per_day), 0.0, 1.0
                )
            )
            # per asset: 1.0 if the other class has open positions
            other_class_open = {}
            for asset in self.envs.keys():
                my_class = self.asset_class_map.get(asset, "majors")
                other_active = any(
                    env.position is not None
                    for a, env in self.envs.items()
                    if self.asset_class_map.get(a, "majors") != my_class
                )
                other_class_open[asset] = 1.0 if other_active else 0.0
        else:
            port_pos_norm = port_pnl_pct = port_trd_norm = 0.0
            other_class_open = {a: 0.0 for a in self.envs.keys()}

        # ── PHASE 1: state + q_values for every asset ─────────────────
        for asset in active:
            env = self.envs[asset]
            # Set portfolio context BEFORE _get_state() (joint training)
            env.set_portfolio_context(
                open_positions_norm=port_pos_norm,
                daily_pnl_pct=port_pnl_pct,
                other_class_active=other_class_open.get(asset, 0.0),
                trades_today_norm=port_trd_norm,
            )
            state = env._get_state()
            mask = env.get_valid_action_mask(max_trades_override=allowed_max_trades)
            if not entry_ok:
                mask = [True] + [False] * (len(mask) - 1)
            agent = self.agents[self.asset_class_map[asset]]

            state_t = torch.from_numpy(state).float().to(agent.device)
            mask_arr = np.array(mask, dtype=bool)
            q_tensor = agent.q_values(state_t)
            q_np = q_tensor.detach().cpu().numpy()
            q_masked = q_np.copy()
            q_masked[~mask_arr] = -1e9

            q_hold = float(q_masked[0])
            # v26: best_trade_q includes EVERY non-HOLD action (1-4)
            # Bug: the previous version ignored the MR actions 3,4
            # -> in eval with a mask setting TREND to -1e9, q_gap was always -inf -> 0 trades
            best_trade_q = max(
                float(q_masked[1]) if len(q_masked) > 1 else -1e9,
                float(q_masked[2]) if len(q_masked) > 2 else -1e9,
                float(q_masked[3]) if len(q_masked) > 3 else -1e9,
                float(q_masked[4]) if len(q_masked) > 4 else -1e9,
            )
            # v26: absolute q_gap (raw difference, not relative).
            # Relative -> unstable when q_hold ≈ 0 (early training),
            # dividing by ~1e-8 gives huge gaps on Qs that haven't converged.
            # Absolute -> direct meaning: "how much more than HOLD do I expect"
            q_gap = float(best_trade_q - q_hold)
            env._last_q_gap = q_gap  # per confidence bracket tracking

            # Q-gap filter: different threshold for train vs eval/live
            # - train: 0.0 -> always explores (NoisyLinear handles the exploration)
            # - eval/live: layered filters against NoisyLinear over-trading:
            #   1. MIN_Q_GAP_EVAL (default 0.15): minimum advantage over HOLD
            #   2. require_positive_q: only if best_trade_q > 0 (the agent expects a gain)
            #   3. vote_n (default 5): N forward passes with different noise,
            #      the action is taken only if ≥ vote_threshold (default 4/5) agree
            #      This filters the cases where NoisyLinear gives a random action
            if self.mode == "train":
                # no q_gap filter in training — always explore
                best_action = agent.act(state_t, mask=mask, deterministic=False)
                # Rank curriculum off here:
                # the rank isn't defined yet at this point of the pipeline.
                # Rank-based filters go AFTER arbiter.select().
            else:
                # Eval/live step 1: Q-gap filter
                agent_choice = int(np.argmax(q_masked))
                conviction_stage = None
                action_candidate = 0
                if q_gap >= self.MIN_Q_GAP_EVAL:
                    # Step 2: positive Q filter (the agent expects a positive return)
                    if self._eval_require_positive_q and best_trade_q <= 0:
                        action_candidate = 0
                        conviction_stage = "positive_q"
                        self.arbiter_stats["positive_q_blocked"] += 1
                    else:
                        action_candidate = int(np.argmax(q_masked))

                # Step 3: vote filter — N noise samples, majority needed
                # BYPASS: if Q-gap >= strong_bypass_mult × min_q_gap the model
                # is already very convinced -> skip the vote (avoids 0 trades with negative Q-values)
                strong_bypass = q_gap >= self._eval_vote_strong_bypass * self.MIN_Q_GAP_EVAL
                if action_candidate != 0 and self._eval_vote_n > 1 and not strong_bypass:
                    votes = [action_candidate]
                    for _ in range(self._eval_vote_n - 1):
                        agent.online.sample_noise()
                        q_v = agent.q_values(state_t)
                        q_mv = q_v.cpu().numpy().copy()
                        q_mv[~mask_arr] = -1e9  # every masked action (MR included)
                        v_a = int(np.argmax(q_mv))
                        votes.append(v_a)
                    # count the votes for the candidate action
                    vote_count = votes.count(action_candidate)
                    if vote_count < self._eval_vote_threshold:
                        action_candidate = 0
                        conviction_stage = "vote"
                        self.arbiter_stats["vote_blocked"] += 1
                    else:
                        # refresh the noise for the final normal forward pass
                        agent.online.sample_noise()

                best_action = action_candidate
                if best_action == 0 and q_gap < self.MIN_Q_GAP_EVAL:
                    self.arbiter_stats["qgap_filtered"] += 1
                    conviction_stage = "q_gap"

            idx_read = min(env.current_step, len(env.df) - 1)
            atr = float(env.df.iloc[idx_read].get(env.atr_col, 0.0)) if env.atr_col else 0.0
            ts = (
                env.timestamps[env.current_step] if env.current_step < len(env.timestamps) else None
            )
            z = getattr(env, "_last_z", None)
            conf = float(np.linalg.norm(z)) if z is not None else 0.0

            # v24: continuous encoder signals (fallback to legacy if head not loaded)
            tmom_30m = float(getattr(env, "_tmom_30m", 0.0) or 0.0)
            tmom_1h = float(getattr(env, "_tmom_1h", 0.0) or 0.0)
            tmom_4h = float(getattr(env, "_tmom_4h", 0.0) or 0.0)
            tmom_1d = float(getattr(env, "_tmom_1d", 0.0) or 0.0)
            vr_30m = float(getattr(env, "_vr_30m", 0.5) or 0.5)
            vr_1h = float(getattr(env, "_vr_1h", 0.5) or 0.5)
            vr_4h = float(getattr(env, "_vr_4h", 0.5) or 0.5)
            an_30m = float(getattr(env, "_an_30m", 0.5) or 0.5)
            an_4h = float(getattr(env, "_an_4h", 0.5) or 0.5)
            adx_30m_v = float(getattr(env, "_adx_30m", 0.5) or 0.5)
            adx_4h_v = float(getattr(env, "_adx_4h", 0.5) or 0.5)
            di_dir_30m_v = float(getattr(env, "_di_dir_30m", 0.0) or 0.0)
            di_dir_4h_v = float(getattr(env, "_di_dir_4h", 0.0) or 0.0)
            di_dir_1h_v = float(getattr(env, "_di_dir_1h", 0.0) or 0.0)  # v26+
            brk_1h = float(getattr(env, "_brk_prob_1h", 0.33) or 0.33)
            brk_4h = float(getattr(env, "_brk_prob_4h", 0.33) or 0.33)
            brk_30m = float(getattr(env, "_brk_prob_30m", 0.33) or 0.33)  # v26+

            # v26+: regime uncertainty — 1 = ambiguous, 0 = clear signal
            _tmom4 = float(getattr(env, "_tmom_4h", 0.0) or 0.0)
            _di4 = float(getattr(env, "_di_dir_4h", 0.0) or 0.0)
            _clarity = (
                min(abs(_tmom4) / 1.5, 1.0) * 0.40
                + min(abs(_di4), 1.0) * 0.40
                + min(abs(brk_4h - 0.5) * 2, 1.0) * 0.20
            )
            _regime_uncertainty = float(np.clip(1.0 - _clarity, 0.0, 1.0))
            # Legacy compat
            regime_model = float(getattr(env, "_current_regime_1h", vr_1h) or vr_1h)
            trend_model = tmom_30m

            # v26: strategy from the action (4=MR, 5=MR, else TREND)
            _strategy_type = "mr" if int(best_action) in (3, 4) else "trend"

            prop = TradeProposal(
                asset=asset,
                action=int(best_action),
                q_value=float(q_masked[best_action]),
                confidence=conf,
                regime=regime_model,
                trend_alignment=trend_model,
                atr=atr,
                timestamp=ts,
                state=state,
                regime_model=regime_model,
                trend_model=trend_model,
                tmom_30m=tmom_30m,
                tmom_1h=tmom_1h,
                tmom_4h=tmom_4h,
                tmom_1d=tmom_1d,
                vol_regime_30m=vr_30m,
                vol_regime_1h=vr_1h,
                vol_regime_4h=vr_4h,
                atr_norm_30m=an_30m,
                atr_norm_4h=an_4h,
                adx_30m=adx_30m_v,
                adx_4h=adx_4h_v,
                di_dir_30m=di_dir_30m_v,
                di_dir_4h=di_dir_4h_v,
                di_dir_1h=di_dir_1h_v,
                brk_prob_1h=brk_1h,
                brk_prob_4h=brk_4h,
                brk_prob_30m=brk_30m,
                regime_uncertainty=_regime_uncertainty,
                is_ppo=False,
                strategy_type=_strategy_type,
            )
            prop.q_vals_raw = q_np
            prop.q_vals_masked = q_masked
            prop.valid_mask = mask_arr
            prop.q_hold = q_hold

            # v26: strategy outcome signals from the env cache (precomputed in _get_state)
            _p_tl = getattr(env, "_strat_p_tp_tl", None)
            if _p_tl is not None:
                prop.p_tp_trend_long = float(getattr(env, "_strat_p_tp_tl", 0.25))
                prop.p_tp_trend_short = float(getattr(env, "_strat_p_tp_ts", 0.25))
                prop.p_tp_mr_long = float(getattr(env, "_strat_p_tp_ml", 0.25))
                prop.p_tp_mr_short = float(getattr(env, "_strat_p_tp_ms", 0.25))
                # proposal_tp_prob: P(TP) of the chosen action
                _action_to_tp = {
                    1: prop.p_tp_trend_long,  # LONG_TREND
                    2: prop.p_tp_trend_short,  # SHORT_TREND
                    3: prop.p_tp_mr_long,  # LONG_MR
                    4: prop.p_tp_mr_short,  # SHORT_MR
                }
                prop.proposal_tp_prob = _action_to_tp.get(prop.action, 0.25)
                # strategy_edge: how much better the proposed strategy is than the alternative
                _best_tr = float(getattr(env, "_strat_best_trend", 0.25))
                _best_mr = float(getattr(env, "_strat_best_mr", 0.25))
                if prop.action in (1, 2):  # TREND
                    prop.strategy_edge = (prop.proposal_tp_prob or 0.25) - _best_mr
                elif prop.action in (3, 4):  # MR
                    prop.strategy_edge = (prop.proposal_tp_prob or 0.25) - _best_tr
                else:
                    prop.strategy_edge = 0.0

            # v26: session validity — bars to the next force close
            prop.bars_to_force_close = getattr(env, "_bars_to_fc", None)

            # eval: the agent wanted a trade, its own conviction filters said no
            if self.mode != "train" and best_action == 0 and agent_choice != 0:
                prop.reject_stage = conviction_stage or "q_gap"
                prop.filter_check = self.arbiter.filter_check(
                    dataclasses.replace(prop, action=agent_choice)
                )
                prop.shadow_action = agent_choice
            proposals.append(prop)

        # ── Health penalty ────────────────────────────────────────────
        if self.health_enabled:
            for p in proposals:
                penalty = self._asset_health_penalty(p.asset)
                if penalty > 0:
                    p.q_value -= penalty

        self.arbiter_stats["total_proposals"] += len(proposals)

        remaining_trades = self._remaining_trades()

        # ── PHASE 1.5: portfolio-level kill switch (eval/live only) ──
        # 4 conditions: drawdown velocity, loss streak, P&L anomaly, stuck.
        # If on -> blocks every new trade (open positions can still exit).
        if self.mode in ("eval", "live") and self.ks_enabled:
            ks_active, ks_reason = self._check_kill_switch()
            if ks_active:
                for p in proposals:
                    p.approved = False
                return self._execute_holds(proposals, {})

        # ── PHASE 2: hybrid daily gate P&L% + SL count ────────────────
        # The block is based on the real daily P&L%,
        # not on the raw SL count. E.g.:
        #   3TP + 4SL with net P&L > -1% -> no block
        #   0TP + 2SL with P&L < -2%     -> full block
        # The SL count stays as an absolute safety net (MAX_DAILY_LOSSES).
        if self.mode in ("eval", "live"):
            daily_pnl = self.portfolio.daily_pnl_pct  # e.g. -0.015 = -1.5%
            daily_sl = self.portfolio.daily_losses

            # hard block: P&L < threshold or too many SLs
            hard_block = daily_pnl <= self.DAILY_BLOCK_PCT or daily_sl >= self.MAX_DAILY_LOSSES
            if hard_block:
                _today = self.portfolio.current_day
                if not getattr(self, "_daily_sl_warned_day", None) == _today:
                    reason = (
                        f"P&L {daily_pnl:.2%}"
                        if daily_pnl <= self.DAILY_BLOCK_PCT
                        else f"{daily_sl} SL"
                    )
                    logger.warning(
                        "Daily gate HARD BLOCK (%s). Blocking rest of day %s.", reason, _today
                    )
                    self._daily_sl_warned_day = _today
                self.arbiter_stats["daily_loss_blocked"] += 1
                for p in proposals:
                    p.approved = False
                return self._execute_holds(proposals, {})

            # soft reduce: P&L < reduce threshold -> max 1 concurrent
            soft_reduce = daily_pnl <= self.DAILY_REDUCE_PCT
        else:
            soft_reduce = False

        if self.global_step % 1000 == 0:
            logger.info(
                "[DEBUG step=%d] block=%s daily_pnl=%.4f daily_sl=%d "
                "n_open=%d slots=%d soft_reduce=%s",
                self.global_step,
                self.block_new_trades,
                self.portfolio.daily_pnl_pct,
                self.portfolio.daily_losses,
                sum(1 for env in self.envs.values() if env.position is not None),
                max(
                    0,
                    (1 if soft_reduce else self.MAX_CONCURRENT)
                    - sum(1 for env in self.envs.values() if env.position is not None),
                ),
                soft_reduce,
            )

        # ── PHASE 3: hard block (DD breach) ──────────────────────────
        if self.block_new_trades:
            logger.warning(
                "Trading BLOCKED — DD breach. Abs=%.4f Trail=%.4f",
                self.portfolio.drawdown,
                self.portfolio.drawdown_trailing,
            )
            for p in proposals:
                p.approved = False
            return self._execute_holds(proposals, {})

        # ── PHASE 4: arbiter select ───────────────────────────────────
        self.arbiter._current_mode = self.mode  # for entry filter
        approved_map = self.arbiter.select(proposals, remaining_trades)

        self.arbiter_stats["entry_filter_blocked"] += sum(
            1
            for p in proposals
            if p.action != 0
            and not getattr(p, "approved", False)
            and getattr(p, "filter_reason", None) not in (None, "pass")
        )

        # ── PHASE 4.5: decision gate (eval/live only) ─────────────────
        # 4 conditions: regime mismatch, ATR spike, action ping-pong, low conf product.
        # Every approved proposal is checked again; if rejected, it's removed.
        if self.dg_enabled and self.mode in ("eval", "live"):
            for asset in list(approved_map.keys()):
                prop = approved_map[asset]
                ok, reason = self._passes_decision_gate(prop)
                if not ok:
                    prop.approved = False
                    prop.reject_stage = "decision_gate"
                    prop.filter_reason = f"decision_gate:{reason}"
                    del approved_map[asset]
                    self.arbiter_stats["decision_gate_blocked"] += 1
                    logger.debug("[DECISION GATE] reject %s: %s", asset, reason)

        # ── PHASE 5: dynamic concurrent position cap ─────────────────
        # MAX_CONCURRENT positions open at the same time.
        # After daily_reduce (P&L < -1%): down to 1.
        # Always picks the best-ranked trades among the approved ones.
        n_open = sum(1 for env in self.envs.values() if env.position is not None)
        effective_max = self._max_concurrent(soft_reduce)
        slots_available = max(0, effective_max - n_open)

        if slots_available == 0:
            # no free slot: block everything
            for asset in list(approved_map.keys()):
                approved_map[asset].approved = False
                approved_map[asset].reject_stage = "position_slots"
                self.arbiter_stats["corr_filtered"] += 1
                self.arbiter_stats["concurrent_blocked"] += 1
            approved_map.clear()
            logger.debug("Concurrent cap: %d/%d open, no slots.", n_open, effective_max)
        elif len(approved_map) > slots_available:
            # more approved trades than free slots: keep the best by rank
            ranked = sorted(
                approved_map.keys(),
                key=lambda a: approved_map[a].rank if approved_map[a].rank is not None else 999,
            )
            for asset in ranked[slots_available:]:
                approved_map[asset].approved = False
                approved_map[asset].reject_stage = "position_slots"
                self.arbiter_stats["corr_filtered"] += 1
                self.arbiter_stats["concurrent_blocked"] += 1
                del approved_map[asset]
            logger.debug(
                "Concurrent cap: keeping top-%d by rank (soft_reduce=%s).",
                slots_available,
                soft_reduce,
            )

        # ── PHASE 6: risk scaling + reservation ──────────────────────
        current_dd = self.portfolio.drawdown
        daily_dd = self.portfolio.daily_dd
        reduction = self._risk_reduction_factor(current_dd, daily_dd)

        reserved_assets: dict[str, float] = {}
        for prop in proposals:
            if prop.asset not in approved_map or prop.action == 0:
                continue

            asset = prop.asset
            env = self.envs[asset]

            # risk always fixed — dynamic sizing is up to the DQN
            BASE_RISK = min(float(env.cfg.get("trade_risk_pct", 0.004)), 0.004)
            risk_scaled = BASE_RISK

            risk_scaled *= reduction
            risk_floor = max(BASE_RISK * 0.05, 0.0002)
            risk_scaled = float(np.clip(risk_scaled, risk_floor, 0.004))

            risk_amount = self.portfolio.balance * risk_scaled

            if self.portfolio.reserve(asset, risk_amount):
                reserved_assets[asset] = risk_amount
            else:
                prop.approved = False
                prop.reject_stage = "no_capital"
                del approved_map[asset]

        # ── Approval stats ──────────────────────────────────────
        for p in proposals:
            if p.approved:
                self.arbiter_stats["approved"] += 1
                if p.rank is not None:
                    self.arbiter_stats["sum_rank_of_approved"] += float(p.rank)
                    self.arbiter_stats["count_ranked"] += 1
                self.arbiter_stats["sum_q_of_approved"] += float(p.q_value)
                if p.tmom_30m is not None:
                    logger.debug(
                        "[ARBITER] Approved %s %s tmom30m=%.2f vr30m=%.2f adx30m=%.2f brk4h=%.2f",
                        p.asset,
                        "LONG" if p.action == 1 else "SHORT",
                        p.tmom_30m,
                        p.vol_regime_30m or 0,
                        p.adx_30m or 0,
                        p.brk_prob_4h or 0,
                    )
            else:
                self.arbiter_stats["rejected"] += 1

        # ── PHASE 7: env.step() + reward shaping ─────────────────────
        shaping_scale = self._shaping_scale()
        transitions = {}
        pre_trade_counts = {a: len(env.episode_trades) for a, env in self.envs.items()}
        pre_steps = {a: env.current_step for a, env in self.envs.items()}

        for prop in proposals:
            asset = prop.asset
            env = self.envs[asset]

            external_risk = reserved_assets.get(asset)
            if external_risk is None and prop.asset in approved_map:
                external_risk = max(
                    1e-6, self.portfolio.balance * env.cfg.get("trade_risk_pct", 0.004)
                )

            action_to_exec = approved_map[asset].action if asset in approved_map else 0
            prev_state = prop.state

            next_state, base_reward, done, info = env.step(
                action_to_exec, external_risk_amount=external_risk
            )
            info = info or {}
            if prop.approved:
                info["approved_by_arbiter"] = True
            elif prop.action != 0:
                info["proposed_but_rejected"] = True
            env.update_arbiter_counters(info)

            shaped_reward = float(base_reward)
            pre_step = pre_steps[asset]
            opened = False
            pos = env.position
            if pos is not None and int(pos.get("entry_idx", -999)) == pre_step:
                opened = True

            closed = False
            last_result = None
            last_pnl = 0.0
            if len(env.episode_trades) > pre_trade_counts[asset]:
                closed = True
                last_trade = env.episode_trades[-1]
                last_result = last_trade.get("result")
                last_pnl = last_trade.get("pnl", 0.0)

            # Reward shaping
            scale = (
                shaping_scale
                if self.mode == "train"
                else (0.0 if self.mode == "live" else max(self.min_shaping_scale * 0.3, 0.0))
            )

            if prop.approved and prop.action != 0:
                if opened:
                    shaped_reward += self.arbiter_bonus * scale
                    # v26: small low-quality cost — opening rank2+ isn't free
                    if prop.rank is not None and prop.rank >= 2:
                        shaped_reward -= 0.01 * scale
                if closed and last_result == "sl":
                    shaped_reward -= self.arbiter_sl_penalty * scale

            if closed:
                result_tp = last_result == "tp"
                result_sl = last_result == "sl"
                # v26: rank-aware shaping — rewards rank0 TP, punishes rank2+ SL
                # It doesn't reward the rank itself, it teaches that the top-ranked
                # ones should be preferred and the low-ranked ones cost when they fail.
                if self.mode == "train":
                    if prop.rank == 0 and result_tp:
                        shaped_reward += 0.03 * scale
                    elif prop.rank is not None and prop.rank >= 2 and result_sl:
                        shaped_reward -= 0.03 * scale

            if not prop.approved and prop.action != 0:
                if prop.q_value < 0:
                    shaped_reward -= self.rejection_penalty_weak * scale

            # Hold bias penalty
            if prop.action == 0 and prop.q_vals_raw is not None and len(prop.q_vals_raw) >= 3:
                q_hold_r = float(prop.q_vals_raw[0])
                q_long_r = float(prop.q_vals_raw[1])
                q_short_r = float(prop.q_vals_raw[2])
                max_gap = max(q_hold_r - q_long_r, q_hold_r - q_short_r)
                # threshold lowered from 0.2 to 0.05: acts before the collapse
                # on HOLD. With hold_bias_penalty=0.02 and threshold 0.05 the penalty
                # follows the actual gap, it doesn't wait for it to diverge.
                if max_gap > 0.05:
                    shaped_reward -= self.hold_bias_penalty * (max_gap - 0.05) * scale
                if prop.rank == 0:
                    shaped_reward -= self.hold_bias_penalty * 0.5 * scale

            if closed:
                result_tp = last_result == "tp"
                result_sl = last_result == "sl"
                # ── Rank tracking (rank0=best, rank1=second, rank2=third) ──
                if prop.rank == 0:
                    if result_tp:
                        self.arbiter_stats["rank1_tp"] += 1
                    elif result_sl:
                        self.arbiter_stats["rank1_sl"] += 1
                elif prop.rank == 1:
                    if result_tp:
                        self.arbiter_stats["rank2_tp"] += 1
                    elif result_sl:
                        self.arbiter_stats["rank2_sl"] += 1
                elif prop.rank == 2:
                    if result_tp:
                        self.arbiter_stats["rank3_tp"] += 1
                    elif result_sl:
                        self.arbiter_stats["rank3_sl"] += 1
                # ── Confidence bracket tracking (via Q-gap) ──────────────
                env = self.envs.get(prop.asset)
                if env is not None:
                    q_gap_val = getattr(env, "_last_q_gap", 0.0)
                    lo = self.MIN_Q_GAP_EVAL if self.mode != "train" else 0.03
                    if q_gap_val < 2 * lo:
                        key = "conf_low"
                    elif q_gap_val < 4 * lo:
                        key = "conf_mid"
                    else:
                        key = "conf_high"
                    if result_tp:
                        self.arbiter_stats[f"{key}_tp"] += 1
                    elif result_sl:
                        self.arbiter_stats[f"{key}_sl"] += 1
                # ── Strategy tracking (trend vs MR) ───────────────────────
                strategy = getattr(prop, "strategy_type", "trend")
                if result_tp:
                    self.arbiter_stats[f"{strategy}_tp"] += 1
                elif result_sl:
                    self.arbiter_stats[f"{strategy}_sl"] += 1

            transitions[asset] = {
                "state": prev_state,
                "action": action_to_exec,
                "reward": shaped_reward,
                "next_state": next_state,
                "done": done,
                "info": info,
                "proposal": prop,
                "is_ppo": False,
                "rollout_step": None,
            }
            self._log_decision(prop, pre_step, action_to_exec, opened)

            if prop.approved and opened:
                self.trades_opened_today += 1
                self.usage_history.append(asset)
                if len(self.usage_history) > self.rolling_window:
                    self.usage_history.pop(0)
                unique, counts = np.unique(self.usage_history, return_counts=True)
                for u, c in zip(unique, counts):
                    self.arbiter_stats["asset_usage"][u] = int(c)

            if closed:
                ts = (
                    env.timestamps[env.current_step]
                    if env.current_step < len(env.timestamps)
                    else None
                )
                self.portfolio.apply_pnl(asset, last_pnl, timestamp=ts)
                self.asset_health[asset]["recent_pnl"].append(float(last_pnl))
                # Kill switch: track portfolio-wide outcomes for loss streak detection
                self._record_trade_outcome(float(last_pnl))

                # ── Block / unblock logic ─────────────────────────────
                abs_breach = self.portfolio.drawdown > self.BASELINE_MAX_DD

                if self.mode in ("eval", "live"):
                    # EVAL/LIVE: hard block ONLY on the absolute DD from the initial capital
                    # — the prop firm metric (e.g. "max 10% loss from starting balance")
                    # The trailing DD (from peak equity) is informative only: you could be
                    # at +8% and drop to +2% -> trailing=6% but absolute DD=0% -> no block
                    if abs_breach and not self.block_new_trades:
                        logger.error(
                            "⚠️ CRITICAL: absolute DD breach → BLOCK. Abs=%.4f (limit=%.4f) Trail=%.4f",
                            self.portfolio.drawdown,
                            self.BASELINE_MAX_DD,
                            self.portfolio.drawdown_trailing,
                        )
                        self.block_new_trades = True
                    elif self.block_new_trades:
                        if self.portfolio.drawdown < self.BASELINE_MAX_DD * 0.70:
                            logger.info(
                                "✅ RECOVERY: absolute DD < 70%% of baseline → RE-ENABLING."
                            )
                            self.block_new_trades = False

                else:
                    # TRAINING: never block. Portfolio reset on the absolute DD only.
                    # Trailing DD = informative in the log, never triggers anything.
                    # (same principle as eval/live: the prop firm measures DD from the initial capital)
                    if abs_breach:
                        logger.warning(
                            "absolute DD breach in training (Abs=%.4f limit=%.4f Trail=%.4f) "
                            "→ portfolio reset, training goes on.",
                            self.portfolio.drawdown,
                            self.BASELINE_MAX_DD,
                            self.portfolio.drawdown_trailing,
                        )
                        init_cap = self.portfolio.initial
                        self.portfolio.balance = init_cap
                        self.portfolio.peak_balance = init_cap
                        self.portfolio.drawdown = 0.0
                        self.portfolio.drawdown_trailing = 0.0
                        self.portfolio.daily_dd = 0.0
                        self.portfolio.intraday_peak = init_cap
                        self.portfolio.consecutive_losses = 0
                        self.portfolio.reserved_map.clear()
                        self.block_new_trades = False

                        # after reserved_map.clear():
                        # reset the open positions in the envs too
                        for env in self.envs.values():
                            if env.position is not None:
                                env.position = None
                                env.trades_today = 0

        # release the reservation of assets without an open position
        for asset in list(self.portfolio.reserved_map.keys()):
            env = self.envs.get(asset)
            if env is None or env.position is None:
                self.portfolio.release(asset)

        # actions available in each next state, with the portfolio as it is now
        for asset, tr in transitions.items():
            tr["next_mask"] = self._next_mask(self.envs[asset])

        return transitions, {k: v for k, v in approved_map.items()}

    def _log_decision(self, prop: TradeProposal, step_idx: int, executed: int, opened: bool):
        """Eval: every trade the agent wanted, with what happened to it (counterfactual report)."""
        if self.mode == "train":
            return
        action = prop.action or prop.shadow_action
        if action == 0:
            return
        if prop.filter_check is None:
            prop.filter_check = self.arbiter.filter_check(dataclasses.replace(prop, action=action))
        self.decision_log.append(
            {
                "asset": prop.asset,
                "step": int(step_idx),
                "action": int(action),
                "opened": bool(opened and executed == action),
                "stage": "opened" if opened and executed == action else prop.reject_stage,
                "filter_check": prop.filter_check,
            }
        )

    def _execute_holds(self, proposals, approved_map):
        transitions = {}
        pre_counts = {a: len(env.episode_trades) for a, env in self.envs.items()}
        for prop in proposals:
            asset = prop.asset
            env = self.envs[asset]
            ns, r, done, info = env.step(0, external_risk_amount=None)
            info = info or {}
            # stored in the replay by the training loop like every other transition
            transitions[asset] = {
                "state": prop.state,
                "action": 0,
                "reward": float(r),
                "next_state": ns,
                "done": done,
                "info": info,
                "proposal": prop,
                "is_ppo": False,
                "rollout_step": None,
                "next_mask": self._next_mask(env),
            }
            if prop.action != 0 and prop.reject_stage is None:
                prop.reject_stage = self._entry_block or "blocked"
            self._log_decision(prop, env.current_step - 1, 0, False)

            if len(env.episode_trades) > pre_counts[asset]:
                last = env.episode_trades[-1]
                ts = (
                    env.timestamps[env.current_step]
                    if env.current_step < len(env.timestamps)
                    else None
                )
                self.portfolio.apply_pnl(asset, last.get("pnl", 0.0), timestamp=ts)
        return transitions, {}


if __name__ == "__main__":
    logger.info("arbiter v2 (Rainbow) loaded.")
