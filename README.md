# forex-rl-pipeline

End-to-end deep RL pipeline for forex: data download, preprocessing, multi-head
encoder training and DDQN agent.

I'm building it one module at a time. This first part is the **data layer**
(`data_management/`): it downloads 1-minute bid/ask data from Dukascopy, builds
one multi-timeframe parquet per asset with the indicators of every TF aligned
on a 15min grid, and scales it with per-column scalers fit without touching the
holdout.

| Module | Status |
|---|---|
| `data_management/` — download, features, scaling | ✅ here |
| multi-head encoder training | coming next |
| DDQN agent | coming next |
| live / paper trading | coming next |

---

## How it works

**`downloader.py`**

1. Downloads 1min **bid and ask** in contiguous weekly blocks (retry on errors).
   Bid and ask are merged with an outer join: a minute published on one side only
   is kept, using the other side for that timestamp (no forward fill in time).
   Mid OHLC + volume + spread.
2. Resamples to **15min, 30min, 1H, 4H, 1D** and computes the indicators on each TF.
3. **Anti-lookahead:** every TF above 15min is shifted by one bar before being
   forward-filled on the 15min grid, so a row only sees higher-TF bars that are
   already closed. The last 15min bar is dropped if it's still incomplete.
4. Warmup: it downloads extra history before `start_date` (auto: 314 days, enough
   for EMA200 on 1D, counted in trading days) and trims it.
5. **Update mode:** reloads the existing parquet, re-downloads from
   `last_ts − warmup`, rebuilds the features and appends only the new rows.

**`scale.py`**

- One sklearn transformer per column, picked from the feature name:

  | Scaler | Columns |
  |---|---|
  | StandardScaler | OHLC, EMA/WMA, VWP, pivots, ATR, returns, MACD, volatility |
  | RobustScaler | EMA slopes, CCI, BB width, candle shapes, volatility regime |
  | MinMaxScaler | RSI, ADX, Stoch |
  | log1p + RobustScaler | volume |
  | not scaled | spread mean/max (kept only as `*_raw`) |

- **Leak-safe:** scalers are fit only on rows `<= 2022-12-31`; 2023+ is a holdout
  the scalers never see. There is no silent fallback to a fit on the whole dataset.
- Saved scalers are **reused** when they cover all the columns, so new data goes
  through exactly the same transform (use `--refit` to force a new fit).
- 15min OHLCV and ATR14 are also kept as `*_raw` (real prices for returns / targets);
  the other prices can be recovered with `inverse_transform` from `all_scalers.pkl`.

## Features (per TF)

OHLCV · EMA / EMA slope / WMA (10-20-50 intraday, 20-50 on 1H/4H, 50-200 on 1D) ·
ADX14, DI+, DI− · RSI14 · CCI20 · ATR14, ATR50, ATR / price · Bollinger width ·
volatility and volatility regime · range, body, shadow, candle pressure, range ratio ·
log / abs return · rolling VWP (50) · 3-bar momentum.
Plus: spread mean/max (15min), Stoch (15min, 30min), MACD (1H), pivots R1/S1/R2/S2 (1D).

178 columns per asset in the raw parquet, 184 after scaling (6 `*_raw` columns added,
spread columns moved to `*_raw`).

---

## Layout

```
data_management/
    config.py        asset lists, default dates, cutoff
    downloader.py    Dukascopy download, multi-TF features, full / update mode
    scale.py         per-column scalers, leak-safe fit, apply saved scalers
tests/               downloader and scaler tests on synthetic data (no download)
```

Data (root set with `--root` or the `FX_DATA_ROOT` env var):

```
<root>/Raw/<Majors|Minors>/<ASSET>/<ASSET>_multiTF_<start>_to_<end>.parquet
<root>/Scaled/<Majors|Minors>/<ASSET>/<ASSET>_scaled_<start>_to_<end>.parquet
<root>/Scaled/<Majors|Minors>/<ASSET>/all_scalers.pkl
```

Assets: **Majors** AUDUSD, EURUSD, GBPUSD, USDCAD, USDCHF — **Minors** EURAUD, GBPAUD,
EURGBP, EURNZD, NZDUSD (edit `config.ASSETS`).

## Usage

```bash
pip install -r requirements.txt

# download (first time) / update
python -m data_management.downloader --root D:/Data --mode full --start 2013-01-01
python -m data_management.downloader --root D:/Data --mode update

# scale (reuses saved scalers if complete, otherwise fits on <= 2022-12-31)
python -m data_management.scale --root D:/Data
python -m data_management.scale --root D:/Data --refit       # force a new fit
python -m data_management.scale --root D:/Data --diagnose    # print the scaler of each column

# apply saved scalers to any parquet
python -m data_management.scale --apply all_scalers.pkl --input new.parquet --output new_scaled.parquet
```

`--group Majors|Minors` and `--assets EURUSD ...` work on a subset.

## Tests

```bash
pytest
```

Offline, on synthetic data:
- cutting the 1min data at T doesn't change any row whose bar closed by T (no lookahead)
- an update gives the same parquet as a full rebuild
- 4H bars don't depend on the day the download starts; no partial last bar
- the scaler fit ignores everything after the cutoff, raw columns and saved scalers round trip

**Known limit:** EMA200 on 1D needs more history than the update warmup to fully
converge, so after an update `EMA200_1D` can differ by ~0.1% from a full rebuild
(and its slope more). A full rebuild from time to time keeps it exact.

---

Data: [Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/) via
[`dukascopy-python`](https://pypi.org/project/dukascopy-python/).
Personal research project, not financial advice.
