# forex-rl-pipeline

End-to-end deep RL pipeline for forex: data download, preprocessing, multi-head
encoder training and DDQN agent.

I'm building it one module at a time. This first part is the **data layer**
(`data_management/`): it downloads 1-minute bid/ask data from Dukascopy, builds
one multi-timeframe parquet per asset with the indicators you choose, for every TF,
aligned on a 15min grid, and scales it with per-column scalers fit without touching the
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

1. Downloads 1min **bid and ask** in contiguous weekly blocks (retry with back-off).
   Bid and ask are merged with an outer join: a minute published on one side only
   is kept, using the other side for that timestamp (no forward fill in time); its
   spread is unknown and left out of the 15min spread mean/max. Mid OHLC + volume + spread.
2. Builds **15min bars** and saves them as a cache (`*_bars15min_*.parquet`, warm-up
   included). The 30min, 1H, 4H and 1D bars are aggregated from them.
3. Computes the selected **indicators** on every TF (see below) and aligns all the TFs on
   the 15min grid (`*_multiTF_*.parquet`).
4. **Timestamps are bar close times**: the row `10:15` holds the bar `[10:00, 10:15)` and
   everything in it is known at `10:15`. Higher TFs are close-labelled too and
   forward-filled, so a row only sees bars that closed at or before its timestamp. An
   incomplete last bar is never written.
5. Warm-up: extra history is downloaded before `start` (auto: the longest lookback of the
   selected indicators; 314 days with EMA200 on 1D, counted in trading days).
6. **Update mode** downloads only the new data (plus a 1-day overlap), appends it to the
   bars cache and rebuilds the features from the whole cache: the result is **identical
   to a full rebuild**, long EMAs included.

**`indicators.py`** — every indicator is registered once with the function that computes
it, the scaler of each of its columns, the history it needs and the TFs it applies to.

| Name | Columns (per TF) | TFs |
|---|---|---|
| `ema` | `EMA{p}`, `EMA{p}_slope` (p = 10-20-50 intraday, 20-50 on 1H/4H, 50-200 on 1D) | all |
| `wma` | `WMA{p}` (same periods) | all |
| `adx` | `ADX14`, `DI_plus14`, `DI_minus14` | all |
| `rsi` | `RSI14` | all |
| `cci` | `CCI20` | all |
| `stoch` | `Stoch` | 15min, 30min |
| `macd` | `MACD`, `MACD_signal`, `MACD_diff` | 1H |
| `atr` | `ATR14`, `ATR50`, `ATR_norm` | all |
| `bollinger` | `BB_width` | all |
| `volatility` | `volatility`, `Volatility_regime` | all |
| `returns` | `log_return`, `abs_return`, `momentum_proxy` | all |
| `candles` | `high_low_range`, `close_open_diff`, `candle_body`, `candle_shadow`, `candle_pressure`, `range_ratio` | all |
| `vwp` | `vwp` (volume-weighted price, 50 bars) | all |
| `pivots` | `Pivot`, `R1`, `S1`, `R2`, `S2` | 1D |

Every TF also has `open`, `high`, `low`, `close`, `volume`; the 15min one adds
`spread_mean` and `spread_max`. With all the indicators: 178 columns per asset in the raw
parquet, 184 after scaling.

**Choosing the indicators:** edit `config.INDICATORS`, or pass `--indicators ema rsi atr ...`.
Changing the selection doesn't need a new download: `--mode features` rebuilds the
features from the bars cache (an `update` also notices the change). Adding a new indicator
is one function plus one `@register(...)` in `indicators.py`; `scale.py` picks up its
scalers automatically. `python -m data_management.downloader --list-indicators` prints the list.

**`scale.py`**

- One sklearn transformer per column, the one declared by the indicator that produces it:

  | Scaler | Columns |
  |---|---|
  | StandardScaler | OHLC, EMA/WMA, VWP, pivots, ATR, returns, MACD, volatility |
  | RobustScaler | EMA slopes, CCI, BB width, candle shapes, volatility regime |
  | MinMaxScaler | RSI, ADX, DI+/DI−, Stoch |
  | log1p + RobustScaler | volume |
  | not scaled | spread mean/max (kept only as `*_raw`) |

  A column no indicator declares gets a skewness-based fallback and a warning.
- **Leak-safe:** scalers are fit only on rows up to `2022-12-31` (the whole day); 2023+ is
  a holdout the scalers never see. There is no silent fallback to a fit on the whole dataset.
- Saved scalers are **reused** when they cover all the columns, so new data goes through
  exactly the same transform (`--refit` forces a new fit). Scalers fit on data of an older
  schema are never reused, and raw files from the older downloader (open-labelled rows)
  are refused until rebuilt with `--mode full`.
- 15min OHLCV and ATR14 are also kept as `*_raw` (real prices for returns / targets);
  the other prices can be recovered with `inverse_transform` from `all_scalers.pkl`.
- Every parquet carries its metadata (schema version, timestamp convention, indicators),
  readable with `storage.read_meta(path)`.

---

## Layout

```
data_management/
    config.py        asset lists, default dates, cutoff, indicator selection
    indicators.py    indicator registry: computation, scaler, lookback, TFs
    downloader.py    Dukascopy download, 15min bars, multi-TF features; full / update / features
    scale.py         per-column scalers, leak-safe fit, apply saved scalers
    storage.py       parquet I/O with metadata, one file per kind and asset
tests/               downloader and scaler tests on synthetic data (no download)
```

Data (root set with `--root` or the `FX_DATA_ROOT` env var):

```
<root>/Raw/<Majors|Minors>/<ASSET>/<ASSET>_bars15min_<start>_to_<end>.parquet
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

# a different indicator set, rebuilt from the bars cache (no download)
python -m data_management.downloader --root D:/Data --mode features --indicators ema rsi atr adx

# scale (reuses saved scalers if complete, otherwise fits on <= 2022-12-31)
python -m data_management.scale --root D:/Data
python -m data_management.scale --root D:/Data --refit       # force a new fit
python -m data_management.scale --root D:/Data --diagnose    # print the scaler of each column

# apply saved scalers to any parquet
python -m data_management.scale --apply all_scalers.pkl --input new.parquet --output new_scaled.parquet
```

`--group Majors|Minors` and `--assets EURUSD ...` work on a subset.

> Data built before the close-label convention (no `*_bars15min_*` file next to it) is
> rebuilt from scratch on the next `update`; scale it again with `--refit` afterwards.

## Development

```bash
pip install -r requirements-dev.txt
pytest            # offline test suite
black . && ruff check --fix .
```

Formatting (Black) and linting (Ruff) are configured in `pyproject.toml`; GitHub Actions runs
both plus the tests on every push. Changes are tracked in [CHANGELOG.md](CHANGELOG.md).

## Tests

Offline, on synthetic data:
- cutting the 1min data at T doesn't change any row labelled <= T (no lookahead)
- rows are labelled at bar close; higher-TF bars equal a direct resampling of the 1min data
- an update gives exactly the same bars and features as a full rebuild
- indicator selection, `features` mode, old-format data rebuilt, every column has a declared scaler
- 4H bars don't depend on the day the download starts; no partial last bar
- the scaler fit ignores everything after the cutoff, raw columns and saved scalers round trip,
  old-schema scalers are refit, the scaler reads the features file and not the bars cache
- one-sided bid/ask minutes: no double-counted volume, unknown spread left out of the mean

---

Data: [Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/) via
[`dukascopy-python`](https://pypi.org/project/dukascopy-python/).
Personal research project, not financial advice.
