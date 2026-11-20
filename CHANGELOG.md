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
- `indicators.py`: indicator registry. Each indicator declares its computation, the scaler of each
  of its columns, the history it needs and its TFs. The dataset contains the indicators listed in
  `config.INDICATORS` (or `--indicators`); `--list-indicators` prints them. **(AI-assisted)**
- 15min bars cache (`*_bars15min_*.parquet`): updates download only the new data and rebuild the
  features from the cache, so an update is **identical** to a full rebuild (this removes the old
  `EMA200_1D` known limit). **(AI-assisted)**
- `--mode features`: rebuild the features from the cache without downloading (e.g. after changing
  the indicator selection). **(AI-assisted)**
- `storage.py`: every parquet carries its metadata (schema version, timestamp convention,
  indicators), also copied to the scaled parquet. **(AI-assisted)**

### Changed
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

### Suggested, not applied
- **License**: the repository has no `LICENSE` file; without one, nobody else may legally reuse the
  code. **(AI-assisted review)**

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

