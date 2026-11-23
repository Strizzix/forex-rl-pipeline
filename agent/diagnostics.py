"""
diagnostics.py — what the arbiter and the agent's own filters did to its trades (eval).

During an eval the runner logs every trade the agent wanted (runner.decision_log): the
action, whether it was opened, the stage that stopped it otherwise (q_gap / vote /
positive_q conviction filters, entry_filter, daily_quota, position_slots, daily_gate,
kill_switch...) and the verdict of the entry filters (filter_check), computed also when
the filters are off.

counterfactual_report() replays every logged decision with the env's trade rules
(agent.strategy_head.trade_labels: entry at the close + spread, SL / TP on the real
high / low, force close) and gives, per group, how those trades would have ended. If the
trades a filter removes do as well as the ones it keeps, the filter only costs trades.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from .strategy_head import strategy_labels

logger = logging.getLogger(__name__)


def _summary(r: np.ndarray) -> dict:
    known = r[np.isfinite(r)]
    if len(known) == 0:
        return {"n": int(len(r)), "known": 0}
    return {
        "n": int(len(r)),
        "known": int(len(known)),
        "mean_r": float(known.mean()),
        "sl_rate": float((known <= -1.0 + 1e-9).mean()),
        "win_rate": float((known > 0).mean()),
    }


def counterfactual_report(runner, env_cfg: dict | None = None) -> dict:
    """
    Outcome of the agent's logged trade decisions, grouped by what happened to them
    ("by_stage") and by the entry-filter verdict ("by_filter"). R = result in risk units
    (TP = rr, SL = -1, force close = the move to that close), spread at the entry only.
    """
    log = getattr(runner, "decision_log", None) or []
    if not log:
        return {"decisions": 0}
    r_cache = {}
    by_stage, by_filter = defaultdict(list), defaultdict(list)
    for d in log:
        env = runner.envs[d["asset"]]
        if d["asset"] not in r_cache:
            r_cache[d["asset"]] = strategy_labels(env.df, env_cfg or env.cfg, return_r=True)[2]
        r = r_cache[d["asset"]][d["step"], d["action"] - 1]
        by_stage[d["stage"] or "unknown"].append(r)
        verdict = "pass" if d["filter_check"] == "pass" else "fail"
        by_filter[f"{'opened' if d['opened'] else 'not_opened'}/{verdict}"].append(r)
        if verdict == "fail":
            by_filter[f"fail:{d['filter_check']}"].append(r)
    report = {
        "decisions": len(log),
        "by_stage": {k: _summary(np.array(v)) for k, v in sorted(by_stage.items())},
        "by_filter": {k: _summary(np.array(v)) for k, v in sorted(by_filter.items())},
    }
    for group in ("by_stage", "by_filter"):
        for k, s in report[group].items():
            if s.get("known"):
                logger.info(
                    "[COUNTERFACTUAL] %-9s %-32s n=%5d  mean R=%+.3f  win=%.1f%%",
                    group[3:],
                    k,
                    s["known"],
                    s["mean_r"],
                    100 * s["win_rate"],
                )
    return report
