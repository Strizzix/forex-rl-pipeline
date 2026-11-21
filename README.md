# forex-rl-pipeline

End-to-end deep RL pipeline for forex: data download, preprocessing, multi-head
encoder training and DDQN agent.

I'm building it one module at a time. So far:

- the **data layer** (`data_management/`): downloads 1-minute bid/ask data from Dukascopy,
  builds one multi-timeframe parquet per asset with the indicators you choose, for every TF,
  aligned on a 15min grid, and scales it with per-column scalers fit without touching the
  holdout;
- the **market encoder** (`encoder/`): a TCN-GRU network pretrained with several supervised
  heads and contrastive losses on those parquets, retrained walk-forward every six months.
  Its embedding is the state the DDQN agent will work on.

| Module | Status |
|---|---|
| `data_management/` — download, features, scaling | ✅ here |
| `encoder/` — multi-head encoder, bootstrap + walk-forward retrain | ✅ here |
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
6. **Closed market**: Dukascopy sometimes fills weekends and holidays with flat volume-0
   bars and sometimes leaves them empty. Since indicators count bars, those flat bars would
   change ATR / RSI / EMA after some weekends and not after others, so they are left out of
   the features (they stay in the bars cache). Only volume-0 bars are dropped, and only when
   the market is closed: in the weekly close (Friday 17:00 → Sunday 17:00 New York time, DST
   included) or in a run of volume-0 bars lasting ≥ 4h outside it (holidays). A quiet bar of
   an open session is always kept. Settings: `DROP_CLOSED_BARS`, `CLOSED_RUN_HOURS` in `config.py`.
7. **Update mode** downloads only the new data (plus a 1-day overlap), appends it to the
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

## Encoder

**Input.** Every sample is a window of `seq_len` 15min rows (256 by default) of the
scaled features of every TF, plus 7 session features (hour and weekday as sin/cos,
London / New York / overlap flags) and, optionally, 5 structural features
(`use_structural_features`). With `score_offset=1` the window ends at the bar *before*
the signal: the model only sees closed bars, as it would live, while the labels stay on
the signal bar. The data rows are labelled at bar close; sessions, days and the session
features are computed on each bar's open time (the session it trades in), the weekday on
the FX trading day (Sunday evening = Monday). Structural features use real prices and are
normalized with the training window's statistics (`structural_stats.json`).

**Model** (`encoder/components.py`): dilated causal TCN + GRU with a learnable asset
embedding concatenated to every timestep. On top of the embedding:

| Head | Target |
|---|---|
| `TrendMomentumHeadTF` | forward log-return / ATR per TF, clipped to [-3, 3] (Huber + sign penalty) |
| `RegimeContinuousHead` | volatility regime, ATR / price, ADX, DI direction per TF |
| `MarketStructureHead` | RANGING / LONG_BREAKOUT / SHORT_BREAKOUT per TF |
| `SessionBreakoutHead` | P(TP before SL) of the London session breakout, long and short |
| `PDHLBreakoutHead` | P(TP before SL) of the previous-day high/low breakout, long and short (London direction + 4H DI gate) |
| `RegimeQualityHead` | is at least one of the strategies TP-first on this bar |
| `ProjectionHead` | SimCLR + cross-asset contrastive, directional contrastive (bull vs bear) |

The strategy labels (`encoder/dataset.py`) are simulated on the raw 15min prices with the
same rules as the backtest: one entry per day per direction, TP/SL hit order, breakeven
reachability, and a sample weight that follows what the live system would actually cash in.
With `outcome_strict_execution` (default) the simulation is executable: the stop order fills
only if it was resting at the bar open and the bar reaches the entry price; the fill bar is
part of the TP/SL scan; anything whose order inside a bar is unknown counts as a loss with a
low weight; the horizon stops at weekend/holiday breaks; Monday's previous day is Friday.
`outcome_strict_execution=false` gives back the historical labels.

**Training** (`encoder/trainer.py`):

- **bootstrap** — from scratch on a fixed split (default train 2014-2022, val 2023).
- **retrain** — walk-forward, every six months: starts from the previous WF checkpoint,
  trains on a rolling window (4 years by default) with a phase schedule per asset class
  (heads first, then GRU / last TCN blocks at low LR, then heads again), a KL anchor to the
  previous model and embedding / regime drift checks. The previous model is evaluated on
  the new validation period before training, as a baseline.
- Loss balancing with EMA normalization, AMP on GPU, one score per epoch (trend +
  regime + strategy-head AUCs weighted by sample coverage) to pick the best checkpoint;
  a checkpoint is saved as `retrain_safe` only if the regime and drift checks pass.

Every knob is in `encoder/config.py` (`DEFAULT_CONFIG`, commented). Checkpoints store
plain `state_dict`s:

```
<models>/Encoder/<class>/Checkpoint/Bootstrap/best_full_checkpoint.pth   encoder + heads + optimizer
<models>/Encoder/<class>/Checkpoint/Bootstrap/best_encoder_only.pth      encoder only
<models>/Encoder/<class>/Checkpoint/Retrain/<tag>/retrain_full_<tag>.pth
```
plus `training_log_<tag>.txt`, `pretrain_history_*.json`, `best_score.json` and the drift
baselines in the same folder.

---

## Layout

```
data_management/
    config.py        asset lists, default dates, cutoff, indicator selection
    indicators.py    indicator registry: computation, scaler, lookback, TFs
    downloader.py    Dukascopy download, 15min bars, multi-TF features; full / update / features
    scale.py         per-column scalers, leak-safe fit, apply saved scalers
    storage.py       parquet I/O with metadata, one file per kind and asset
    checks.py        quality checks on the real raw / scaled data
    compare_versions.py   old (open-labelled) vs new dataset comparison
encoder/
    config.py        DEFAULT_CONFIG: every training setting, paths
    dataset.py       input windows, session/structural features, all the labels
    components.py    TCN-GRU encoder, heads, augmentations, contrastive losses
    trainer.py       bootstrap / walk-forward retrain, metrics, checkpoints, CLI
tests/               data, scaler and encoder tests on synthetic data (no download, CPU)
```

Data (root set with `--root` or the `FX_DATA_ROOT` env var; encoder checkpoints under
`--models-root` or `FX_MODELS_ROOT`):

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

### Training the encoder

```bash
# for a GPU, install the CUDA build of torch first (pytorch.org), then the requirements

# bootstrap (train 2014-2022, val 2023 by default)
python -m encoder.trainer --class Majors --mode bootstrap --root D:/Data --models-root D:/Models

# walk-forward retrain from the previous WF
python -m encoder.trainer --class Minors --mode retrain --root D:/Data --models-root D:/Models \
    --wf-tag WF7_2026H1 --train-end 2025-12-31 --val-start 2026-01-01 --val-end 2026-06-30 \
    --start-from D:/Models/Encoder/Minors/Checkpoint/Retrain/WF6_2025H2/retrain_full_WF6_2025H2.pth

# any other setting of encoder/config.py
python -m encoder.trainer --class Majors --mode bootstrap --set use_structural_features=true seq_len=128
python -m encoder.trainer --config my_run.json      # several overrides in a JSON file
```

The scaled data must come from the current `data_management` (bar-close labels): an encoder
trained on the older open-labelled data should be bootstrapped again on the new data.

### Checking the data

```bash
# quality checks on the real data (raw + scaled); exit status 1 if anything is wrong
python -m data_management.checks --root D:/Data
python -m data_management.checks --root D:/Data --only raw --assets EURUSD

# old dataset (open-labelled rows) vs new one, built from the same source data
python -m data_management.compare_versions --old D:/DataOld --new D:/Data --assets EURUSD
```

`checks` verifies, per asset: one bars cache and one features file of the current schema;
bars on the 15min grid with consistent OHLC; no closed-market bar left in the features and
no open-market bar of the cache missing from them; exactly the columns of the selected indicators;
**bar-close labels** (`close_15min` equals the cached bar, and at every 1H / 4H / 1D close the
higher-TF close equals the 15min close); NaN in the first rows (warm-up too short); time gaps
classified as weekends, holidays, short illiquid gaps or anomalies (including a gap repeating
on the same weekday with the same length, the signature of a download bug). On the scaled
data: same rows and indicators as raw, `*_raw` columns identical to raw, every column scaled,
no NaN/inf added, and the scalers really fit on the cutoff window (Standard columns with mean
0 / std 1, Robust ones with median 0, MinMax ones spanning [0, 1] on the rows up to the cutoff).
A table at the end compares the assets and flags outliers in the share of zero-volume bars.

`compare_versions` checks that the new 15min columns equal the old ones one bar earlier and
the higher-TF columns equal the old ones at the same timestamp. Spread, volume/VWP and the 1D
long EMAs are reported apart, since the fixes in the CHANGELOG can change them. Two datasets of
the current version (e.g. one built with `full`, one with `full` + `update`) are compared
directly instead: on the common rows every column must be identical.

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
- encoder: dataset items and label ranges, the input window ending before the signal with
  `score_offset=1`, structural features (live, training-window stats), head output shapes, a
  one-epoch bootstrap through the CLI whose checkpoint loads back strictly
- encoder time convention: session features and weekday on the bar open time, 4H closes in
  summer time, TF closes matching the rows where a new bar appears, PDHL DI gate applied,
  inputs without bar-close labels refused

---

Data: [Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/) via
[`dukascopy-python`](https://pypi.org/project/dukascopy-python/).
Personal research project, not financial advice.
