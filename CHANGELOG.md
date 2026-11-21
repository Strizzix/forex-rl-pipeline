# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

> Entries marked **(AI-assisted)** were found or implemented with the help of an AI code review.
> Every change was reviewed and is covered by the test suite.

## [Unreleased]

### Breaking
- Rows are now labelled at the bar **close** time (the row `10:15` holds the bar `[10:00, 10:15)`),
  so a timestamp is the moment its data is known and downstream code can't take a 15-minute
  look-ahead by mistake. Higher TFs are close-labelled as well (no more `shift(1)`) and reach the
  grid one bar earlier, still without look-ahead. Existing datasets are rebuilt automatically by
  the next `update`; scale them again with `--refit`. **(AI-assisted)**

### Added
- `encoder`: `outcome_session_sl_r` / `outcome_pdhl_sl_r` now set the SL (multiples of range/3);
  the default 1.0 gives the same labels as before. **(AI-assisted)**
- `encoder`: the trainer refuses inputs without bar-close labels (older downloader) or raw /
  bars-cache files (`allow_open_labels=true` to override), and never picks the bars cache as an
  asset's input. **(AI-assisted)**
- `encoder/`: the multi-head market encoder (dataset, model components, trainer), moved in from
  the research scripts `encoder_dataset.py`, `encoder_components.py`, `encoder_trainer.py`:
  package layout, a CLI (`python -m encoder.trainer`) instead of the config block with hard-coded
  paths, every setting in `encoder/config.py`, comments and logs in English, unused helpers
  removed, Black/Ruff clean. It produces the same samples, labels, sample weights and
  checkpoints as the scripts (checked bit for bit on synthetic data, bootstrap and retrain), and
  old checkpoints load unchanged (same `state_dict` names). **(AI-assisted)**
- `tests/test_encoder.py`: dataset items and label ranges, `score_offset` window, structural
  features, head shapes, one-epoch bootstrap through the CLI with a strict checkpoint reload.
  CI installs the CPU build of torch. **(AI-assisted)**
- Closed-market bars left out of the features: Dukascopy fills some weekends/holidays with flat
  volume-0 bars and leaves others empty, which made indicators after a weekend depend on how it
  was recorded. Only volume-0 bars inside the weekly close (Fri 17:00 -> Sun 17:00 New York time)
  or in a >= 4h volume-0 run outside it are dropped; quiet bars of open sessions are kept. The
  bars cache is unchanged, so `--mode features` applies it without downloading; `checks` verifies
  it. **(AI-assisted)**
- `indicators.py`: indicator registry. Each indicator declares its computation, the scaler of each
  of its columns, the history it needs and its TFs. The dataset contains the indicators listed in
  `config.INDICATORS` (or `--indicators`); `--list-indicators` prints them. **(AI-assisted)**
- 15min bars cache (`*_bars15min_*.parquet`): updates download only the new data and rebuild the
  features from the cache, so an update is **identical** to a full rebuild (this removes the old
  `EMA200_1D` known limit). **(AI-assisted)**
- `--mode features`: rebuild the features from the cache without downloading (e.g. after changing
  the indicator selection). **(AI-assisted)**
- `checks.py`: quality checks on the real raw and scaled data, ported from the earlier
  `Raw_Parquet_Tester` / `Scaled_Parquet_Tester` and adapted to the new layout: bar-close labels,
  bars cache vs features, expected indicator columns, gaps (weekends no longer counted as
  "suspicious recurrences"; holes longer than 4 days flagged), `*_raw` columns equal to raw, and
  scalers verified against the saved cutoff window. **(AI-assisted)**
- `compare_versions.py`: compares a dataset of the old downloader with one of the new one
  (same bar one label later for 15min columns, same value for higher TFs), or two builds of the
  current version directly (e.g. full build vs full build + update). **(AI-assisted)**
- `storage.py`: every parquet carries its metadata (schema version, timestamp convention,
  indicators), also copied to the scaled parquet. **(AI-assisted)**

### Changed
- `encoder`: a checkpoint that doesn't load strictly is still loaded with `strict=False`, but
  now with a warning instead of silently. **(AI-assisted)**
- `scale`: scalers come from the indicator registry instead of a separate name table (same choice
  for every existing column); undeclared columns get a skewness fallback with a warning.
  **(AI-assisted)**
- `scale`: saved scalers fit on an older schema are refit, raw files from the older downloader are
  refused until rebuilt, and only `*_multiTF_*` files are read (never the bars cache). **(AI-assisted)**
- One-sided bid/ask minutes now have an unknown (`NaN`) spread instead of `0.0`, so they no longer
  pull `spread_mean_15min` towards 0. **(AI-assisted)**
- Indicators are built in a single concat (no fragmented DataFrames, no pandas
  `PerformanceWarning`). **(AI-assisted)**
- The feature schema no longer depends on the data: all-NaN columns are kept instead of dropped.
  **(AI-assisted)**

### Fixed
- `checks`: `DatetimeIndex.dayofweek` (deprecated, removed in pandas 4) replaced with
  `day_of_week`; deprecation warnings raised by the package now fail the test suite. **(AI-assisted)**
- `encoder`, time convention: sessions (Asian range 00-07 UTC, London, PDHL window), days and the
  session input features are computed on the bar **open** time. With the close-labelled rows the
  bar `[23:45, 00:00)` fell into the next day's Asian range and every session edge was one bar off.
  On the same bars the strategy labels are now identical to what the encoder computed on the old
  open-labelled data (checked bit for bit). **(AI-assisted)**
- `encoder`, PDHL DI gate: the 4H DI columns were looked up as `DI_plus14_4H` in a frame with
  lower-cased columns, so they were never found and the gate was silently always open (more PDHL
  labels than the simulator trades). It now uses the real DI, `tanh((DI+ - DI-) / 20) > 0.15`
  (short `< -0.15`) on the 4H DI known when the order is placed, as the simulator. Settings
  `outcome_pdhl_di_gate` / `outcome_pdhl_di_thresh`; `outcome_pdhl_di_gate=false` gives back the
  labels of the earlier runs. **(AI-assisted)**
- `encoder`, TF closes: `hour % 4 == 0` marked the 4H closes, but the 4H bars are aligned in
  absolute time and close at 01/05/09.. Rome in summer: the 4H trend/regime labels were taken on the
  wrong rows half of the year. Now aligned with the data bins. **(AI-assisted)**
- `encoder`, structural features (`use_structural_features`): `prebreak_range_atr` and
  `day_extension_atr` were always 0 (ATR / daily columns looked up with the wrong case) and the
  other three were computed on scaled prices, where open/high/low/close have different scalers.
  They now use real prices (`*_raw` and inverse-transformed daily levels), and the median/IQR
  normalization comes from the training window (saved as `structural_stats.json` next to the
  checkpoint for live use) instead of each period's own data. **(AI-assisted)**
- `encoder`, weekday feature: the calendar weekday made Sunday-evening bars (6) look like Tuesday
  with the 5-day period; it is now the FX trading day (rolls at 17:00 New York time), so Sunday
  evening counts as Monday. **(AI-assisted)**

### Suggested, not applied
- **License**: the repository has no `LICENSE` file; without one, nobody else may legally reuse the
  code. **(AI-assisted review)**
- **Encoder, days of the PDHL / Asian range**: days are UTC calendar days, so on Mondays the
  "previous day" is the 2-3 hours of Sunday evening and the PDH/PDL of Monday are that short
  range. Using the FX trading day (rolls at 17:00 New York) would give Monday Friday's range; to be
  changed together with the backtest simulator, so labels and backtest keep matching.
  **(AI-assisted review)**

## [0.1.1] - 2026-11-20

### Fixed
- `downloader`: minutes quoted on one side only (bid or ask) counted that side's volume twice,
  because the missing side was filled from the other before summing. The missing side now adds 0. **(AI-assisted)**
- `downloader`: full builds named the file after the requested `--start`/`--end` instead of the first/last row,
  and left older `<asset>_multiTF_*.parquet` files next to the new one. Files are now always named after the data
  they contain and there is exactly one per asset. **(AI-assisted)**
- `scale`: a date-only cutoff (`2022-12-31`) excluded the rows of the cutoff day after 00:00. The whole day is now
  part of the fit window (`<= 2022`); a cutoff with a time is still exact. **(AI-assisted)**
- `scale`: when saved scalers were reused, `all_scalers.pkl` recorded the `--cutoff` of the current run instead of
  the cutoff the scalers were actually fit with. The original cutoff is kept and a different `--cutoff` is reported,
  pointing to `--refit`. **(AI-assisted)**

### Changed
- `scale`: `DI_plus14` / `DI_minus14` (bounded 0-100) use `MinMaxScaler` like `ADX14`, instead of a skewness
  heuristic whose choice depended on the data. Existing `all_scalers.pkl` files are unaffected until `--refit`. **(AI-assisted)**
- `downloader`: linear back-off between Dukascopy retries instead of retrying immediately. **(AI-assisted)**
- Both CLIs exit with status 1 when an asset fails, so scheduled runs can detect errors. **(AI-assisted)**

### Added
- `scale --apply` warns about numeric columns that have no saved scaler and would be left unscaled. **(AI-assisted)**
- Black + Ruff configuration (`pyproject.toml`), `requirements-dev.txt` and a GitHub Actions workflow running lint and
  the offline test suite on every push. **(AI-assisted)**
- Tests for every fix above.

