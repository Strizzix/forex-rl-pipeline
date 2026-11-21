"""Trade simulation behind the session / PDHL strategy labels."""

import datetime as dt

import numpy as np
import pytest

from encoder.dataset import (
    W_AMBIGUOUS,
    W_BE_CUT,
    previous_trading_days,
    simulate_trade,
    stop_order_filled,
)

NO_BREAK = np.zeros(8, dtype=bool)


def _bars(*hl):
    hi = np.array([h for h, _ in hl], dtype=float)
    lo = np.array([lo for _, lo in hl], dtype=float)
    return hi, lo


# ── fills ─────────────────────────────────────────────────────────────────────


def test_stop_fills_only_at_the_entry_price():
    # level 1.0000, entry 1.0005: a bar reaching 1.0003 broke the level, not the entry
    hi = np.array([0.9990, 1.0003, 1.0006])
    lo = np.array([0.9980, 0.9985, 0.9995])
    cl = np.array([0.9985, 1.0001, 1.0004])
    assert not stop_order_filled(hi, lo, cl, NO_BREAK, 1, 1.0005, +1)
    assert stop_order_filled(hi, lo, cl, NO_BREAK, 2, 1.0005, +1)


def test_stop_does_not_fill_on_a_bar_opening_beyond_the_entry():
    hi = np.array([1.0010, 1.0020])
    lo = np.array([0.9990, 1.0008])
    cl = np.array([1.0009, 1.0015])  # previous close already above the entry
    assert not stop_order_filled(hi, lo, cl, NO_BREAK, 1, 1.0005, +1)
    # short side, mirrored
    cl_s = np.array([0.9990, 0.9985])
    assert not stop_order_filled(hi, lo, cl_s, NO_BREAK, 1, 0.9995, -1)
    short_lo = np.array([0.9990, 0.9990])
    assert stop_order_filled(hi, short_lo, np.array([1.0000, 0.0]), NO_BREAK, 1, 0.9995, -1)


def test_stop_does_not_fill_across_a_market_break():
    hi = np.array([0.9990, 1.0010])
    lo = np.array([0.9980, 0.9990])
    cl = np.array([0.9985, 1.0005])
    brk = np.array([True, False])
    assert not stop_order_filled(hi, lo, cl, brk, 1, 1.0005, +1)


# ── outcomes ──────────────────────────────────────────────────────────────────
# long: entry 1.0, SL 0.99, TP 1.02 (RR 2)


def test_sl_in_the_fill_bar_is_an_ambiguous_loss():
    hi, lo = _bars((1.001, 0.989), (1.021, 1.0))
    hit, side, _, w = simulate_trade(hi, lo, 0, 1, 1.02, 0.99, +1)
    assert (hit, side, w) == (1, 0, W_AMBIGUOUS)
    # historical rules skipped the fill bar and saw a clean win
    hit, side, _, w = simulate_trade(hi, lo, 0, 1, 1.02, 0.99, +1, strict=False)
    assert (hit, side, w) == (1, 1, 1.0)


def test_tp_in_the_fill_bar_is_a_win():
    hi, lo = _bars((1.021, 0.999), (1.0, 0.98))
    assert simulate_trade(hi, lo, 0, 1, 1.02, 0.99, +1)[:2] == (1, 1)


def test_tp_and_sl_in_the_same_bar_is_a_loss():
    hi, lo = _bars((1.001, 0.999), (1.021, 0.989))
    hit, side, _, w = simulate_trade(hi, lo, 0, 1, 1.02, 0.99, +1)
    assert (hit, side, w) == (1, 0, W_AMBIGUOUS)
    assert simulate_trade(hi, lo, 0, 1, 1.02, 0.99, +1, strict=False)[1] == 1


def test_short_mirror():
    # short: entry 1.0, SL 1.01, TP 0.98
    hi, lo = _bars((1.0005, 0.999), (1.0, 0.979))
    assert simulate_trade(hi, lo, 0, 1, 0.98, 1.01, -1)[:2] == (1, 1)
    hi, lo = _bars((1.0005, 0.999), (1.011, 0.995))
    assert simulate_trade(hi, lo, 0, 1, 0.98, 1.01, -1)[:2] == (1, 0)


def test_unresolved_trade_cut_by_a_break_is_skipped():
    hi, lo = _bars((1.001, 0.999), (1.005, 0.995), (1.021, 1.0))
    # break after row 1: the TP on row 2 is beyond the weekend
    hit, side, _, w = simulate_trade(hi, lo, 0, 2, 1.02, 0.99, +1, last_row=1)
    assert (hit, side, w) == (-1, -1, 0.0)
    # the full horizon fitting with no outcome is a real "no outcome"
    hit, side, _, _ = simulate_trade(hi[:2], lo[:2], 0, 1, 1.02, 0.99, +1)
    assert (hit, side) == (0, -1)


# ── breakeven weights ─────────────────────────────────────────────────────────
# RR 2, trigger 0.25R = 1.0025, BE stop 0.07R = 1.0007


def test_be_win_without_pullback_keeps_full_weight():
    hi, lo = _bars((1.001, 0.999), (1.003, 1.001), (1.021, 1.002), (1.0, 0.98))
    hit, side, be, w = simulate_trade(hi, lo, 0, 3, 1.02, 0.99, +1, be_rr=2.0)
    assert (side, be, w) == (1, 1, 1.0)
    # historical rules cut it because an SL came after the TP
    assert simulate_trade(hi, lo, 0, 3, 1.02, 0.99, +1, be_rr=2.0, strict=False)[3] == W_BE_CUT


def test_be_win_with_pullback_to_the_be_stop_is_down_weighted():
    hi, lo = _bars((1.001, 0.999), (1.003, 1.001), (1.002, 1.0005), (1.021, 1.002))
    assert simulate_trade(hi, lo, 0, 3, 1.02, 0.99, +1, be_rr=2.0)[3] == W_BE_CUT


def test_be_saved_loss_is_down_weighted():
    hi, lo = _bars((1.001, 0.999), (1.003, 1.001), (1.002, 0.989), (1.021, 1.0))
    hit, side, be, w = simulate_trade(hi, lo, 0, 3, 1.02, 0.99, +1, be_rr=2.0)
    assert (side, be, w) == (0, 1, W_BE_CUT)


# ── PDH/PDL day ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "skip, expected", [(True, dt.date(2024, 1, 5)), (False, dt.date(2024, 1, 7))]
)
def test_monday_previous_day(skip, expected):
    days = [dt.date(2024, 1, d) for d in (4, 5, 7, 8)]  # Thu, Fri, Sun stub, Mon
    assert previous_trading_days(days, skip_sunday=skip)[dt.date(2024, 1, 8)] == expected
