"""
market_calendar.py — trading calendar of the environment (force close, entry lockout, swap).

The FX day rolls over at 17:00 New York time: 22:00 UTC in winter, 21:00 UTC in summer.
The rules are written in "market time" = New York time + 5h, which is UTC in winter and
one hour earlier in summer, so the force close, the lockouts and the swap stay at the same
distance from the real rollover all year. A fixed UTC clock put them one hour late from
March to November: the Friday 21:30 force close came after the weekly close (positions
held over the weekend) and the Wednesday 21:30 close after the triple swap.

Times are the row labels of the data, i.e. bar close times: a row is the moment its bar
is known, which is when the env acts on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_management.config import MARKET_TZ

MARKET_SHIFT = pd.Timedelta(hours=5)  # New York + 5h = UTC in winter

FORCE_CLOSE_HM = (21, 30)  # Wednesday (triple swap) and Friday (weekend) force close
LOCKOUT_HM = (20, 30)  # no entries from here on Wednesday and Friday
SUNDAY_REOPEN_HM = (22, 0)  # no entries on Sunday before the reopening
ROLLOVER_HM = (22, 0)  # swap rollover (17:00 New York)
SWAP_DAYS = {0: 1.0, 1: 1.0, 2: 3.0, 3: 1.0, 4: 1.0}  # Wednesday = triple swap


def market_clock(index) -> tuple[np.ndarray, np.ndarray]:
    """(minutes of the day, weekday 0=Mon) in market time, for a tz-aware index."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    t = idx.tz_convert(MARKET_TZ).tz_localize(None) + MARKET_SHIFT
    return np.asarray(t.hour * 60 + t.minute, dtype=np.int64), np.asarray(t.day_of_week)


def _hm(hm: tuple[int, int]) -> int:
    return hm[0] * 60 + hm[1]


def force_close_mask(index) -> np.ndarray:
    """Rows where MarketEnv force-closes: Wednesday / Friday from 21:30 market time, Saturday."""
    hm, dow = market_clock(index)
    late = hm >= _hm(FORCE_CLOSE_HM)
    return (dow == 5) | ((dow == 4) & late) | ((dow == 2) & late)


def entry_lockout_mask(index) -> np.ndarray:
    """Rows where MarketEnv opens nothing: Wed / Fri from 20:30, Sunday before 22:00, Saturday."""
    hm, dow = market_clock(index)
    return (
        (dow == 5)
        | ((dow == 6) & (hm < _hm(SUNDAY_REOPEN_HM)))
        | (np.isin(dow, (2, 4)) & (hm >= _hm(LOCKOUT_HM)))
    )


def swap_days(index) -> np.ndarray:
    """
    Swap charged on a position held from the previous row to this one: the rollovers
    (22:00 market time, Mon-Fri) in (t[i-1], t[i]], Wednesday counting three. Row 0 is 0.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    t = idx.tz_convert(MARKET_TZ).tz_localize(None) + MARKET_SHIFT
    roll = pd.Timedelta(minutes=_hm(ROLLOVER_HM))
    day = np.asarray(((t - roll).normalize() - pd.Timestamp("1970-01-01")).days, dtype=np.int64)
    out = np.zeros(len(idx))
    for i in np.flatnonzero(np.diff(day) > 0) + 1:
        for d in range(day[i - 1] + 1, day[i] + 1):
            out[i] += SWAP_DAYS.get(int((d + 3) % 7), 0.0)  # 1970-01-01 was a Thursday
    return out


def rows_to_force_close(index, cap: int = 10_000) -> np.ndarray:
    """Rows left until the next force-close row (0 on it), capped at ``cap``."""
    fc = force_close_mask(index)
    n = len(fc)
    out = np.full(n, cap, dtype=np.int64)
    nxt = None
    for i in range(n - 1, -1, -1):
        if fc[i]:
            nxt = i
        if nxt is not None:
            out[i] = min(cap, nxt - i)
    return out
