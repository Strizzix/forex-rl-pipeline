"""
rainbow.py — Rainbow DQN for forex trading (v2)
================================================
Components:
  1. NoisyLinear        — parametric exploration (replaces epsilon-greedy)
  2. SumTree            — O(log N) sampling for PER
  3. PrioritizedBuffer  — PER (α=0.6, β 0.4→1.0) + high initial priority for TP/SL
  4. NStepBuffer        — accumulates n steps per asset (n=5 default)
  5. DuelingQRNet       — MLP + Dueling + QR-DQN (32 quantiles)
  6. RainbowAgent       — the full agent, drop-in interface for pipeline.py

Pipeline interface (same as DoubleDQNAgent):
  - agent.act(state, mask, deterministic)  → int
  - agent.q_values(state_tensor)           → Tensor (n_actions,)  [mean Q]
  - agent.q_diagnostics(state, mask)       → dict
  - agent.store(s, a, r, ns, done, tag, env_id)  → None
  - agent.train_step(freeze_target_steps)  → float
  - agent.prepare_for_retrain(...)         → None
  - agent.set_eval_mode() / set_train_mode() → None
  - agent.save(path) / agent.load(path)    → None
  - agent.device     → str
  - agent.n_actions  → 5  (HOLD, LONG_TREND, SHORT_TREND, LONG_MR, SHORT_MR)
  - agent.gru_hidden → 0  (dummy — only PPO uses a RolloutBuffer with gru_hidden)

Why QR-DQN instead of C51:
  - no fixed atom bounds: forex returns are heavy-tailed, the range isn't known
  - the asymmetric quantile-Huber loss is robust to outliers
  - more stable convergence under the distribution shift between WFs

Why Dueling:
  - HOLD is optimal ~95% of the time -> V(s) separates "good state" from "best action"
  - less bias towards overtrading

Why N-step = 5:
  - TP/SL happen 5-20 steps after the entry
  - propagates the trade reward 5 steps back towards the entry decision
  - CRITICAL: NStepBuffer is per asset (env_id), so different sequences never mix

Why PER + TP/SL boost:
  - TP/SL naturally have a high TD error -> PER samples them already
  - the initial boost keeps HOLD from dominating the first batches
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
_N_ACTIONS = 5  # HOLD=0, LONG_TREND=1, SHORT_TREND=2, LONG_MR=3, SHORT_MR=4  (v26)
_N_QUANT = 32  # quantili QR-DQN


# ══════════════════════════════════════════════════════════════════════════════
# 1.  NoisyLinear
# ══════════════════════════════════════════════════════════════════════════════
class NoisyLinear(nn.Module):
    """
    Factorized Gaussian Noisy Linear (Fortunato et al. 2017).
    Replaces epsilon-greedy: the noise shrinks by itself where the
    gradient says the action is already decided.
    """

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma0 = sigma0

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma0 / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma0 / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def sample_noise(self):
        """Resamples the noise — call before every forward in training."""
        p = self._scale_noise(self.in_features)
        q = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(q.outer(p))
        self.bias_epsilon.copy_(q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w = self.weight_mu  # eval: deterministic (mean)
            b = self.bias_mu
        return F.linear(x, w, b)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SumTree  (per PER)
# ══════════════════════════════════════════════════════════════════════════════
class SumTree:
    """
    Segment tree per Prioritized Experience Replay.
    update / query / sample in O(log N).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data_ptr = 0
        self.size = 0

    def _propagate(self, idx: int, delta: float):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent > 0:
            self._propagate(parent, delta)

    def update(self, data_idx: int, priority: float):
        tree_idx = data_idx + self.capacity - 1
        delta = float(priority) - self.tree[tree_idx]
        self.tree[tree_idx] = float(priority)
        self._propagate(tree_idx, delta)

    def add(self, priority: float) -> int:
        idx = self.data_ptr
        self.update(idx, priority)
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

    def _retrieve(self, idx: int, value: float) -> int:
        # Iterative: no recursion, no edge cases on the leaves
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):  # we're on a leaf
                return idx
            if right >= len(self.tree):  # last level with only a left child
                return left
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right

    def get(self, value: float) -> tuple[int, float]:
        # clamp: floating-point drift can make value > total
        value = float(np.clip(value, 0.0, max(self.total - 1e-8, 0.0)))
        tree_idx = self._retrieve(0, value)
        # final defensive clamp: always a valid array index
        tree_idx = int(np.clip(tree_idx, self.capacity - 1, 2 * self.capacity - 2))
        data_idx = tree_idx - self.capacity + 1
        return data_idx, float(self.tree[tree_idx])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def max_priority(self, default: float = 1.0) -> float:
        if self.size == 0:
            return default
        leaves = self.tree[self.capacity - 1 : self.capacity - 1 + self.size]
        return float(np.max(leaves)) if len(leaves) > 0 else default


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PrioritizedReplayBuffer
# ══════════════════════════════════════════════════════════════════════════════
class PrioritizedReplayBuffer:
    """
    Replay buffer with PER on a SumTree + initial priority boost for TP/SL.

    P(i) ∝ p_i^α
    w_i  = (N · P(i))^{-β} / max_j(w_j)   (importance sampling, normalized)
    """

    _TAG_PRIORITY_MULT: dict[str, float] = {
        "tp": 5.0,  # winning trade — very informative
        "sl": 4.0,  # losing trade — just as informative
        "other": 2.0,  # forced close
        "hold": 0.5,  # HOLD — common, base priority
        # ── rank-boosted priority ─────────────────────────────────────
        # rank1/rank2 trades are better quality (WR 89%/57% in Run1).
        # Boost: rank1 × 2.0, rank2 × 1.4 over the base tag.
        # E.g. rank1_tp = 5.0 × 2.0 = 10.0 — highest priority in the PER.
        # The network sees these samples proportionally more often.
        "rank1_tp": 10.0,  # rank1 TP: 5.0 × 2.0
        "rank1_sl": 8.0,  # rank1 SL: 4.0 × 2.0
        "rank2_tp": 7.0,  # rank2 TP: 5.0 × 1.4
        "rank2_sl": 5.6,  # rank2 SL: 4.0 × 1.4
    }

    def __init__(
        self,
        capacity: int = 200_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 300_000,
        min_priority: float = 1e-6,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.min_priority = min_priority
        self.device = device

        self.tree = SumTree(capacity)
        self._step = 0

        # Storage on CPU (saves VRAM)
        self._states = np.empty(capacity, dtype=object)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.empty(capacity, dtype=object)
        self._dones = np.zeros(capacity, dtype=np.bool_)

    # ── write ───────────────────────────────────────────────────────
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        tag: str = "hold",
    ):
        max_p = self.tree.max_priority(default=1.0)
        mult = self._TAG_PRIORITY_MULT.get(tag, 1.0)
        init_p = max(max_p * mult, self.min_priority)

        idx = self.tree.add(init_p**self.alpha)

        # Normalize to a numpy float32 array
        self._states[idx] = (
            state if isinstance(state, np.ndarray) else np.array(state, dtype=np.float32)
        )
        self._actions[idx] = int(action)
        self._rewards[idx] = float(reward)
        self._next_states[idx] = (
            next_state
            if isinstance(next_state, np.ndarray)
            else np.array(next_state, dtype=np.float32)
        )
        self._dones[idx] = bool(done)

    # ── read ────────────────────────────────────────────────────────
    @property
    def beta(self) -> float:
        frac = min(1.0, self._step / max(1, self.beta_steps))
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def sample(self, batch_size: int) -> tuple | None:
        """
        Samples proportionally to the priority.
        Returns (states, actions, rewards, next_states, dones, indices, weights) or None.
        """
        n = self.tree.size
        if n < batch_size:
            return None

        self._step += 1
        beta = self.beta
        segment = self.tree.total / batch_size

        indices = np.zeros(batch_size, dtype=np.int64)
        weights = np.zeros(batch_size, dtype=np.float32)

        min_prob = self.min_priority**self.alpha / max(self.tree.total, 1e-8)
        max_w = (n * min_prob) ** (-beta)

        for i in range(batch_size):
            v = random.uniform(segment * i, segment * (i + 1))
            try:
                data_idx, priority = self.tree.get(v)
            except (IndexError, ValueError):
                # defensive fallback: sample a random valid index
                data_idx = random.randint(0, n - 1)
                priority = 1.0
            data_idx = int(np.clip(data_idx, 0, n - 1))
            indices[i] = data_idx
            prob = max(float(priority) / max(self.tree.total, 1e-8), 1e-10)
            weights[i] = ((n * prob) ** (-beta)) / max_w

        states = torch.tensor(np.stack(self._states[indices].tolist()), dtype=torch.float32).to(
            self.device
        )
        next_states = torch.tensor(
            np.stack(self._next_states[indices].tolist()), dtype=torch.float32
        ).to(self.device)
        actions = torch.tensor(self._actions[indices], dtype=torch.long).to(self.device)
        rewards = torch.tensor(self._rewards[indices], dtype=torch.float32).to(self.device)
        dones = torch.tensor(self._dones[indices], dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, p in zip(indices, priorities):
            clipped = max(float(abs(p)), self.min_priority) ** self.alpha
            self.tree.update(int(idx), clipped)

    def __len__(self) -> int:
        return self.tree.size

    def reset(self):
        self.tree = SumTree(self.capacity)
        self._step = 0
        self._states = np.empty(self.capacity, dtype=object)
        self._actions = np.zeros(self.capacity, dtype=np.int64)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._next_states = np.empty(self.capacity, dtype=object)
        self._dones = np.zeros(self.capacity, dtype=np.bool_)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  NStepBuffer  (per-asset)
# ══════════════════════════════════════════════════════════════════════════════
class NStepBuffer:
    """
    Rolling buffer for N-step returns, per asset.
    Accumulates the last n transitions and computes the discounted return:
        R = r_t + γ·r_{t+1} + ... + γ^{n-1}·r_{t+n-1}
    then releases (s_t, a_t, R, s_{t+n}, done_{t+n}).

    CRITICAL: one instance per asset, so independent sequences
    never get mixed.
    """

    def __init__(self, n: int = 5, gamma: float = 0.99):
        self.n = n
        self.gamma = gamma
        self._buf: deque = deque(maxlen=n)

    def add(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
        tag: str,
    ) -> list[tuple]:
        """
        Adds a transition.
        Returns a list of (s, a, R, ns, done, tag) ready for the replay buffer.
        """
        self._buf.append((state, action, reward, next_state, done, tag))
        ready = []

        if done:
            # flush: n-step return for every transition left
            buf_list = list(self._buf)
            for i in range(len(buf_list)):
                s_i, a_i, _, _, _, tag_i = buf_list[i]
                R = 0.0
                actual_ns = buf_list[-1][3]
                actual_done = True
                for j, (_, _, r_j, ns_j, d_j, _) in enumerate(buf_list[i:]):
                    R += (self.gamma**j) * r_j
                    actual_ns = ns_j
                    actual_done = d_j
                    if d_j:
                        break
                ready.append((s_i, a_i, R, actual_ns, actual_done, tag_i))
            self._buf.clear()

        elif len(self._buf) == self.n:
            # buffer full: release the oldest transition
            buf_list = list(self._buf)
            s0, a0, _, _, _, tag0 = buf_list[0]
            R = 0.0
            actual_ns = buf_list[-1][3]
            actual_done = buf_list[-1][4]
            for j, (_, _, r_j, ns_j, d_j, _) in enumerate(buf_list):
                R += (self.gamma**j) * r_j
                if d_j:
                    actual_ns = ns_j
                    actual_done = True
                    break
            ready.append((s0, a0, R, actual_ns, actual_done, tag0))

        return ready

    def reset(self):
        self._buf.clear()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  DuelingQRNet
# ══════════════════════════════════════════════════════════════════════════════
class DuelingQRNet(nn.Module):
    """
    Dueling + QR-DQN (Quantile Regression DQN).

    Architecture:
        state → trunk (MLP + LayerNorm) → split:
            V-stream → NoisyLinear → N_QUANT scalars
            A-stream → NoisyLinear → N_QUANT × n_actions
        Q_τ(s,a) = V_τ(s) + A_τ(s,a) − mean_a[A_τ(s,a)]
        Q(s,a)   = mean_τ[Q_τ(s,a)]

    Notes:
    - LayerNorm in the trunk (not BatchNorm) — works with batch_size=1 at inference
    - orthogonal init in the trunk for stability
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 5,
        hidden_dims: list[int] = None,
        n_quant: int = _N_QUANT,
        noisy_sigma: float = 0.5,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        self.n_actions = n_actions
        self.n_quant = n_quant

        # ── Shared trunk ─────────────────────────────────────────────
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        trunk_out = hidden_dims[-1]  # usually 128
        dueling_h = 64

        # ── Value stream ─────────────────────────────────────────────
        self.v_fc = NoisyLinear(trunk_out, dueling_h, sigma0=noisy_sigma)
        self.v_out = NoisyLinear(dueling_h, n_quant, sigma0=noisy_sigma)

        # ── Advantage stream ─────────────────────────────────────────
        self.a_fc = NoisyLinear(trunk_out, dueling_h, sigma0=noisy_sigma)
        self.a_out = NoisyLinear(dueling_h, n_quant * n_actions, sigma0=noisy_sigma)

        self._init_trunk()

    def _init_trunk(self):
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def sample_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x  → (B, state_dim)
        out → (B, n_actions, n_quant)
        """
        feat = self.trunk(x)  # (B, trunk_out)

        v = F.relu(self.v_fc(feat))  # (B, dueling_h)
        v = self.v_out(v).unsqueeze(1)  # (B, 1, n_quant)

        a = F.relu(self.a_fc(feat))  # (B, dueling_h)
        a = self.a_out(a).view(x.size(0), self.n_actions, self.n_quant)  # (B, na, nq)

        q = v + a - a.mean(dim=1, keepdim=True)  # (B, n_actions, n_quant)
        return q

    def q_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Q(s,a) = mean_τ[Q_τ(s,a)]  → (B, n_actions)"""
        return self.forward(x).mean(dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  RainbowAgent
# ══════════════════════════════════════════════════════════════════════════════
class RainbowAgent:
    """
    Rainbow DQN agent for multi-asset forex trading.

    Components:
      ✅ Double DQN         — online picks the action, target evaluates it
      ✅ Dueling Networks   — V(s) + A(s,a) to handle the dominant HOLD
      ✅ QR-DQN             — 32 quantiles, asymmetric Huber loss
      ✅ PER                — α=0.6, β annealing 0.4→1.0
      ✅ N-step returns     — n=5, PER ASSET so sequences never mix
      ✅ Noisy Networks     — adaptive exploration, no epsilon

    Config keys (rainbow_cfg in DEFAULT_CFG of pipeline.py):
        n_actions         — 3  (HOLD / LONG / SHORT)
        hidden_dims       — [256, 256, 128]
        n_quant           — 32
        noisy_sigma       — 0.5
        lr                — 3e-4
        batch_size        — 256
        buffer_capacity   — 200_000
        per_alpha         — 0.6
        per_beta_start    — 0.4
        per_beta_end      — 1.0
        per_beta_steps    — 300_000
        gamma             — 0.99
        n_step            — 5
        target_update_tau — 0.005
        grad_clip         — 1.0
        reward_clip       — 1.0  (0 = off)
        warmup_steps      — 2_000
    """

    def __init__(
        self,
        state_dim: int,
        cfg: dict[str, Any],
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.cfg = cfg
        self.device = device
        self.n_actions = int(cfg.get("n_actions", _N_ACTIONS))
        self.gru_hidden = 0  # dummy, for the PPO pipeline (RolloutBuffer)

        # ── Hyperparameters ────────────────────────────────────────
        self.lr = float(cfg.get("lr", 3e-4))
        self.batch_size = int(cfg.get("batch_size", 256))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.n_step = int(cfg.get("n_step", 5))
        self.target_tau = float(cfg.get("target_update_tau", 0.005))
        self.grad_clip = float(cfg.get("grad_clip", 1.0))
        self.reward_clip = float(cfg.get("reward_clip", 1.0))
        self.warmup_steps = int(cfg.get("warmup_steps", 2_000))

        hidden_dims = list(cfg.get("hidden_dims", [256, 256, 128]))
        n_quant = int(cfg.get("n_quant", _N_QUANT))
        noisy_sigma = float(cfg.get("noisy_sigma", 0.5))

        buf_cap = int(cfg.get("buffer_capacity", 200_000))
        per_alpha = float(cfg.get("per_alpha", 0.6))
        per_beta_start = float(cfg.get("per_beta_start", 0.4))
        per_beta_end = float(cfg.get("per_beta_end", 1.0))
        per_beta_steps = int(cfg.get("per_beta_steps", 300_000))

        # ── Networks ────────────────────────────────────────────────
        self.online = DuelingQRNet(state_dim, self.n_actions, hidden_dims, n_quant, noisy_sigma).to(
            device
        )
        self.target = DuelingQRNet(state_dim, self.n_actions, hidden_dims, n_quant, noisy_sigma).to(
            device
        )
        self._hard_update_target()
        self.target.eval()

        # ── Optimizer ────────────────────────────────────────────────
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.lr, eps=1.5e-4)

        # ── Quantile taus (constant, on device) ─────────────────────
        self._tau = torch.linspace(
            0.5 / n_quant, 1.0 - 0.5 / n_quant, n_quant, device=device
        )  # (n_quant,)

        # ── Replay buffer ────────────────────────────────────────────
        self.buffer = PrioritizedReplayBuffer(
            capacity=buf_cap,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_end=per_beta_end,
            beta_steps=per_beta_steps,
            device=device,
        )

        # ── Elite buffer ─────────────────────────────────────────────
        # Keeps the rank1/rank2 transitions for guaranteed sampling.
        # In every batch, elite_sample_frac of the transitions come from here.
        # The priority boost is handled by _TAG_PRIORITY_MULT in PER.add().
        _elite_cap = int(cfg.get("elite_buffer_capacity", 20_000))
        self.elite_buffer = deque(maxlen=_elite_cap)
        self.elite_sample_frac = float(cfg.get("elite_sample_frac", 0.25))

        # ── N-step buffers PER ASSET ─────────────────────────────────
        # CRITICAL: one NStepBuffer per asset, because every asset is an
        # independent episode — mixing the transitions corrupts the returns.
        # Keyed by env_id (str, e.g. "EURUSD"); created lazily on the first store().
        self._nstep_bufs: dict[str, NStepBuffer] = {}
        self._nstep_default_key = "__single__"  # fallback when env_id is not given

        # ── Internal state ───────────────────────────────────────────
        self._train_steps = 0
        self._eval_mode = False
        self._freeze_until_step = 0  # ✅ avoids the getattr fallback in train_step

        logger.info(
            "RainbowAgent | state_dim=%d | hidden=%s | n_quant=%d | "
            "buffer=%d | n_step=%d | device=%s",
            state_dim,
            hidden_dims,
            n_quant,
            buf_cap,
            self.n_step,
            device,
        )

    # ──────────────────────────────────────────────────────────────
    # Internal utilities
    # ──────────────────────────────────────────────────────────────
    def _hard_update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def _soft_update_target(self):
        tau = self.target_tau
        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)

    def _to_numpy(self, x) -> np.ndarray:
        """Converts a tensor/array to a float32 numpy array."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32)
        return np.asarray(x, dtype=np.float32)

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float().to(self.device)
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _mask_q(self, q: torch.Tensor, mask: list[bool] | None) -> torch.Tensor:
        """Sets -inf on the invalid actions."""
        if mask is None:
            return q
        mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device)
        q = q.clone()
        q[~mask_t] = -1e9
        return q

    def _get_nstep_buf(self, env_id: str | None) -> NStepBuffer:
        """Gets or creates the NStepBuffer of the given asset."""
        key = env_id if env_id is not None else self._nstep_default_key
        if key not in self._nstep_bufs:
            self._nstep_bufs[key] = NStepBuffer(n=self.n_step, gamma=self.gamma)
        return self._nstep_bufs[key]

    # ──────────────────────────────────────────────────────────────
    # public API — compatible with DoubleDQNAgent
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def q_values(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Q(s, a) = mean_τ[Q_τ(s,a)]  → Tensor (n_actions,).
        Used by the arbiter for the Q-gap filter and the scoring.
        Accepts both (state_dim,) and (1, state_dim).

        Behaviour:
        - train mode : uses the noise already sampled by the NoisyLinear layers
        - eval mode  : deterministic (NoisyLinear uses weight_mu)
        """
        x = state_tensor.float().to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        q = self.online.q_mean(x).squeeze(0)  # (n_actions,)
        return q

    @torch.no_grad()
    def q_diagnostics(
        self,
        state: np.ndarray,
        mask: list[bool] | None = None,
    ) -> dict[str, Any]:
        """Q diagnostics for the arbiter / logging (same as DoubleDQNAgent)."""
        st = self._to_tensor(state).unsqueeze(0)
        q = self.online.q_mean(st).squeeze(0)  # (n_actions,)
        qm = self._mask_q(q, mask)

        best_a = int(qm.argmax().item())
        q_arr = q.cpu().numpy()
        q_hold = float(q_arr[0])
        q_trade = float(max(q_arr[1:])) if len(q_arr) > 1 else -1e9  # v26: all 5 actions
        q_gap = q_trade - q_hold

        return {
            "q_max": float(qm.max().item()),
            "q_mean": float(q.mean().item()),
            "q_std": float(q.std().item()),
            "q_gap": q_gap,
            "action_best": best_a,
            "q_vals": q_arr.tolist(),
        }

    @torch.no_grad()
    def act(
        self,
        state: Any,
        mask: list[bool] | None = None,
        deterministic: bool = False,
    ) -> int:
        """
        Picks an action.

        Rules:
        - training, deterministic=False → resample the noise, noisy policy
        - eval mode                     → deterministic (online.eval())
        - deterministic=True            → deterministic
        """
        if self.online.training and not deterministic and not self._eval_mode:
            self.online.sample_noise()

        st = self._to_tensor(state).unsqueeze(0)
        q = self.online.q_mean(st).squeeze(0)
        q = self._mask_q(q, mask)
        return int(q.argmax().item())

    def store(self, state, action, reward, next_state, done, tag="hold", env_id=None):
        """
        Adds a transition to the N-step buffer of the asset (env_id),
        then moves the ready transitions into the PER replay buffer.

        env_id:  asset id, e.g. "EURUSD".
                 None = a single buffer (backward compatible).
        """
        if self.reward_clip > 0:
            # Asymmetric: positive rewards clipped at reward_clip (1.0),
            # negative rewards clipped at -reward_clip * 0.5 (-0.5).
            # Softens the "trading is always wrong" signal in bootstrap
            # without removing it: SLs still give a negative signal,
            # but don't flood the PER buffer with a fixed reward = -1.0.
            reward = float(np.clip(reward, -self.reward_clip * 0.5, self.reward_clip))

        s_np = self._to_numpy(state)
        ns_np = self._to_numpy(next_state)

        nstep_buf = self._get_nstep_buf(env_id)
        ready = nstep_buf.add(s_np, action, reward, ns_np, done, tag)

        for s, a, r, ns, d, t in ready:
            # ✅ clip the n-step return — the sum of n clipped rewards
            # can reach reward_clip * n_step (e.g. 1.0 * 5 = 5.0)
            if self.reward_clip > 0:
                r = float(
                    np.clip(r, -self.reward_clip * self.n_step, self.reward_clip * self.n_step)
                )

            # PER priority boost per rank — handled by
            # _TAG_PRIORITY_MULT in PrioritizedReplayBuffer.add().
            # The "rank1_tp"/"rank1_sl"/"rank2_tp"/"rank2_sl" tags turn on
            # the right priority multiplier automatically.
            self.buffer.add(s, a, r, ns, d, t)

            # ── Change 1: elite buffer per rank1/rank2 ───────────────
            # rank1/2 transitions kept apart for the guaranteed
            # sampling (elite_sample_frac of the batch) in train_step().
            if isinstance(t, str) and (t.startswith("rank1") or t.startswith("rank2")):
                self.elite_buffer.append((s, a, r, ns, d))

    def train_step(self, freeze_target_steps: int = 0) -> float:
        """
        One QR-DQN update with Double DQN and PER IS weights.
        Returns the loss (0.0 while the buffer is still warming up).
        """
        if len(self.buffer) < self.warmup_steps:
            return 0.0

        # ── guaranteed elite sampling ────────────────────────────────
        # If the elite buffer has enough samples, it replaces a fraction
        # of the batch with rank1/rank2 transitions. IS weights = 1.0 for these
        # (no IS correction needed — they're over-represented by design).
        _n_elite = 0
        _elite_states = _elite_actions = _elite_rewards = _elite_ns = _elite_dones = None
        if (
            len(self.elite_buffer) >= max(32, int(self.batch_size * self.elite_sample_frac))
            and self.elite_sample_frac > 0
        ):
            _n_elite = int(self.batch_size * self.elite_sample_frac)
            _elite_idxs = random.sample(range(len(self.elite_buffer)), _n_elite)
            _elite_batch = [self.elite_buffer[i] for i in _elite_idxs]
            _es, _ea, _er, _ens, _ed = zip(*_elite_batch)
            _elite_states = torch.tensor(np.stack(_es), dtype=torch.float32, device=self.device)
            _elite_actions = torch.tensor(np.array(_ea), dtype=torch.long, device=self.device)
            _elite_rewards = torch.tensor(np.array(_er), dtype=torch.float32, device=self.device)
            _elite_ns = torch.tensor(np.stack(_ens), dtype=torch.float32, device=self.device)
            _elite_dones = torch.tensor(np.array(_ed), dtype=torch.float32, device=self.device)

        # sample the remaining (batch_size - n_elite) from the PER
        _n_per = self.batch_size - _n_elite
        batch = self.buffer.sample(_n_per)
        if batch is None:
            return 0.0

        states, actions, rewards, next_states, dones, indices, weights = batch

        # concatenate elite + PER if there are elite samples
        if _n_elite > 0:
            states = torch.cat([_elite_states, states], dim=0)
            actions = torch.cat([_elite_actions, actions], dim=0)
            rewards = torch.cat([_elite_rewards, rewards], dim=0)
            next_states = torch.cat([_elite_ns, next_states], dim=0)
            dones = torch.cat([_elite_dones, dones], dim=0)
            # IS weights = 1.0 for the elite transitions (no correction needed)
            _elite_w = torch.ones(_n_elite, dtype=torch.float32, device=self.device)
            weights = torch.cat([_elite_w, weights], dim=0)

        self._train_steps += 1
        self.online.train()
        self.online.sample_noise()
        self.target.sample_noise()

        # ── Target: Double DQN ────────────────────────────────────
        with torch.no_grad():
            # the online network picks the best action...
            next_q_online = self.online.q_mean(next_states)  # (B, na)
            best_actions = next_q_online.argmax(dim=1)  # (B,)

            # ...the target network evaluates the distribution of that action
            target_dist = self.target(next_states)  # (B, na, nq)
            idx_e = best_actions.view(-1, 1, 1).expand(-1, 1, self._tau.size(0))
            target_next = target_dist.gather(1, idx_e).squeeze(1)  # (B, nq)

            gamma_n = self.gamma**self.n_step
            target_q = rewards.unsqueeze(1) + gamma_n * (1.0 - dones.unsqueeze(1)) * target_next
            # (B, nq)

        # ── Online: quantiles of the action taken ────────────────
        online_dist = self.online(states)  # (B, na, nq)
        idx_e2 = actions.view(-1, 1, 1).expand(-1, 1, self._tau.size(0))
        online_q = online_dist.gather(1, idx_e2).squeeze(1)  # (B, nq)

        # ── Quantile Huber Loss ───────────────────────────────────
        # td_err: (B, nq_target, nq_online) = target[:,j] - online[:,i]
        target_exp = target_q.unsqueeze(2)  # (B, nq, 1)
        online_exp = online_q.unsqueeze(1)  # (B, 1, nq)
        td_err = target_exp - online_exp  # (B, nq, nq)

        k = 1.0  # Huber threshold
        huber = torch.where(
            td_err.abs() <= k, 0.5 * td_err.pow(2), k * (td_err.abs() - 0.5 * k)
        )  # (B, nq, nq)

        # τ midpoints: (1, 1, nq)
        tau_exp = self._tau.unsqueeze(0).unsqueeze(0)
        indicator = (td_err < 0).float()
        q_weights = (tau_exp - indicator).abs()  # (B, nq, nq)

        loss_per_sample = (q_weights * huber).mean(dim=2).sum(dim=1)  # (B,)
        weighted_loss = (weights * loss_per_sample).mean()

        # ── Backprop ──────────────────────────────────────────────
        self.optimizer.zero_grad()
        weighted_loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()

        # ── update the PER priorities ─────────────────────────────
        with torch.no_grad():
            new_prio = loss_per_sample.detach().cpu().numpy() + 1e-6
        self.buffer.update_priorities(indices, new_prio)

        # ── soft target update ────────────────────────────────────
        # priority: _freeze_until_step (from prepare_for_retrain) > local parameter
        _freeze_limit = getattr(self, "_freeze_until_step", 0)
        if self._train_steps > max(freeze_target_steps, _freeze_limit):
            self._soft_update_target()

        self.online.eval()
        return float(weighted_loss.item())

    def set_eval_mode(self):
        """
        Deterministic eval.

        In eval:
        - online.eval()  → NoisyLinear uses weight_mu / bias_mu
        - target.eval()
        - no noise resampling

        Makes backtests comparable and matches the behaviour
        described in the pipeline.
        """
        self._eval_mode = True
        self.online.eval()
        self.target.eval()

    def set_train_mode(self):
        self._eval_mode = False
        self.online.train()
        self.target.eval()

    # ──────────────────────────────────────────────────────────────
    # Retrain / Walk-forward
    # ──────────────────────────────────────────────────────────────
    def prepare_for_retrain(
        self,
        reset_buffer: bool = True,
        reset_epsilon: bool = True,
        epsilon_value: float = 0.2,
        lr_scale: float = 0.3,
        reset_target: bool = False,
        freeze_target_steps: int = 3_000,
        reset_trunk: bool = False,
        reset_elite_buffer: bool = True,
    ):
        """
        Prepares the agent for a new WF period (retrain).
        - reset_buffer:        empties the PER and every per-asset NStepBuffer
        - lr_scale:            lower LR for fine-tuning
        - reset_target:        if True, hard-copy online -> target
        - freeze_target_steps: blocks the target soft update for N steps
        """
        if reset_buffer:
            self.buffer.reset()
            self._nstep_bufs.clear()
            logger.info("RainbowAgent: PER + NStepBuffer reset")

        if reset_elite_buffer:
            self.elite_buffer.clear()
            logger.info("RainbowAgent: elite buffer reset")

        # Re-initialize the Noisy layers for fresh exploration
        for m in self.online.modules():
            if isinstance(m, NoisyLinear):
                m.reset_parameters()
                m.sample_noise()
        logger.info("RainbowAgent: NoisyLinear re-initialized")

        if reset_trunk:
            self.online._init_trunk()
            self.target._init_trunk()
            logger.info("RainbowAgent: trunk re-initialized (encoder space changed)")

        new_lr = self.lr * lr_scale
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr
        logger.info("RainbowAgent: LR → %.2e", new_lr)

        if reset_target:
            self._hard_update_target()
            logger.info("RainbowAgent: target network hard-reset")

        # block the soft update for the next N steps
        self._freeze_until_step = self._train_steps + freeze_target_steps

        self._eval_mode = False
        self.online.train()

    # ──────────────────────────────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────────────────────────────
    def save(self, path: str):
        # elite buffer serialized as a list of numpy tuples (portable)
        _elite_list = []
        for s, a, r, ns, d in self.elite_buffer:
            _elite_list.append(
                (
                    s.numpy() if hasattr(s, "numpy") else np.array(s, dtype=np.float32),
                    int(a),
                    float(r),
                    ns.numpy() if hasattr(ns, "numpy") else np.array(ns, dtype=np.float32),
                    bool(d),
                )
            )
        torch.save(
            {
                "online_state_dict": self.online.state_dict(),
                "target_state_dict": self.target.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "train_steps": self._train_steps,
                "buffer_step": self.buffer._step,
                "cfg": self.cfg,
                "state_dim": self.state_dim,
                "agent_type": "rainbow",
                # v26+: the elite buffer persists across WFs — accumulates rank1/2 over every WF
                "elite_buffer": _elite_list,
            },
            path,
        )
        logger.info("RainbowAgent saved → %s (elite_buf=%d)", path, len(_elite_list))

    def load(self, path: str, reset_elite: bool = False):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(ckpt["online_state_dict"])
        self.target.load_state_dict(ckpt["target_state_dict"])
        if "optimizer_state" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception as e:
                logger.warning("Optimizer state not loaded: %s", e)
        self._train_steps = ckpt.get("train_steps", 0)
        self.buffer._step = ckpt.get("buffer_step", 0)
        self._freeze_until_step = 0

        # v26+: restore the elite buffer — not reset between consecutive WFs
        # (reset_elite=True only at the start of a new RUN, e.g. Bootstrap)
        _elite_list = [] if reset_elite else ckpt.get("elite_buffer", [])
        self.elite_buffer.clear()
        if _elite_list:
            for s_np, a, r, ns_np, d in _elite_list:
                self.elite_buffer.append(
                    (
                        torch.tensor(s_np, dtype=torch.float32),
                        a,
                        r,
                        torch.tensor(ns_np, dtype=torch.float32),
                        d,
                    )
                )
        logger.info(
            "RainbowAgent loaded ← %s (steps=%d, elite_buf=%d)",
            path,
            self._train_steps,
            len(self.elite_buffer),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Quick sanity check
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os
    import tempfile

    STATE_DIM = 204  # v26+ state vector: 128 emb + heads + market + position + SOH + session
    cfg = {
        "n_actions": 5,  # HOLD, LONG_TREND, SHORT_TREND, LONG_MR, SHORT_MR (v26)
        "hidden_dims": [256, 256, 128],
        "n_quant": 32,
        "lr": 3e-4,
        "batch_size": 64,
        "buffer_capacity": 5_000,
        "n_step": 5,
        "warmup_steps": 200,
        "gamma": 0.99,
        "reward_clip": 1.0,
        "noisy_sigma": 0.5,
        "target_update_tau": 0.005,
    }

    agent = RainbowAgent(STATE_DIM, cfg, device="cpu")
    total_params = sum(p.numel() for p in agent.online.parameters())
    print(f"  Params online net: {total_params:,}")
    assert agent.gru_hidden == 0, "gru_hidden must be 0"
    assert agent.n_actions == 5  # v26: 5 actions

    # check the q_values shape and the tensor compatibility
    state_np = np.random.randn(STATE_DIM).astype(np.float32)
    q = agent.q_values(torch.tensor(state_np))
    assert q.shape == (5,), f"wrong q_values shape: {q.shape}"
    print(f"  q_values: {q.numpy().round(3)}")

    # warm up the PER on two separate assets (per-asset NStep test)
    ASSETS = ["EURUSD", "GBPUSD"]
    for step in range(300):
        for asset in ASSETS:
            s = np.random.randn(STATE_DIM).astype(np.float32)
            ns = np.random.randn(STATE_DIM).astype(np.float32)
            a = random.randint(0, 4)
            r = random.uniform(-1, 1)
            d = step % 20 == 19
            t = "tp" if (d and a != 0) else "hold"
            agent.store(torch.tensor(s), a, r, torch.tensor(ns), d, tag=t, env_id=asset)

    # check the separate NStep buffers
    assert "EURUSD" in agent._nstep_bufs
    assert "GBPUSD" in agent._nstep_bufs
    print("  NStepBuffer per-asset: OK")

    # Check train_step
    loss = agent.train_step(freeze_target_steps=0)
    print(f"  train_step loss: {loss:.4f}")

    # check the q_values shape and the tensor compatibility
    state_np = np.random.randn(STATE_DIM).astype(np.float32)
    q = agent.q_values(torch.tensor(state_np))
    assert q.shape == (5,), f"wrong q_values shape: {q.shape}"
    print(f"  q_values: {q.numpy().round(3)}")

    # Check act()
    a_tr = agent.act(state_np, mask=[True, True, True, True, True], deterministic=False)
    a_ev = agent.act(state_np, mask=[True, False, False, False, False], deterministic=True)
    assert a_tr in (0, 1, 2, 3, 4)
    assert a_ev == 0, f"only HOLD is valid, got {a_ev}"
    print(f"  act() train={a_tr}, eval(mask=[T,F,F,F,F])={a_ev}")

    # Check q_diagnostics
    diag = agent.q_diagnostics(state_np)
    assert "q_gap" in diag and "action_best" in diag
    print(f"  q_diagnostics: q_gap={diag['q_gap']:.4f} best={diag['action_best']}")

    # Check set_eval_mode
    agent.set_eval_mode()
    assert agent._eval_mode
    agent.set_train_mode()
    assert not agent._eval_mode
    print("  eval/train mode: OK")

    # save/load round trip
    # NoisyLinear resamples the noise on every forward -> compare only in eval mode
    # (eval = mean weights, no noise, deterministic output)
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp = f.name
    agent.set_eval_mode()
    agent.save(tmp)
    q_eval = agent.q_values(torch.tensor(state_np))
    agent2 = RainbowAgent(STATE_DIM, cfg, device="cpu")
    agent2.load(tmp)
    agent2.set_eval_mode()
    q2 = agent2.q_values(torch.tensor(state_np))
    assert torch.allclose(q_eval, q2, atol=1e-5), f"Q differs after load: {q_eval} vs {q2}"
    agent.set_train_mode()
    os.unlink(tmp)
    print("  save/load round-trip: OK")

    # Check prepare_for_retrain
    agent.prepare_for_retrain(reset_buffer=True, lr_scale=0.3, freeze_target_steps=1000)
    assert len(agent.buffer) == 0
    assert len(agent._nstep_bufs) == 0
    print("  prepare_for_retrain: OK")

    print("\n✅ RainbowAgent v2 — all checks passed")
