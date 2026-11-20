# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

> Entries marked **(AI-assisted)** were found or implemented with the help of an AI code review.
> Every change was reviewed and is covered by the test suite.

## [Unreleased]

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

### Suggested, not applied
These would change the produced data or need a design decision, so they are left for later. **(AI-assisted review)**
- **Spread of one-sided minutes**: it is set to `0.0`, which biases `spread_mean_15min` low around market closures.
  Leaving it `NaN` would make the 15min mean/max ignore those minutes.
- **Timestamp convention**: rows are labelled with the bar *open* time but contain the bar close (see the README
  note). Labelling bars at their close time would remove the risk of a 15-minute look-ahead in downstream code.
- **`EMA200_1D` after an update**: the update warm-up is too short for the 1D EMA200 to converge exactly. Carrying the
  last EMA values over from the existing parquet (or a longer warm-up for that TF only) would make updates match a
  full rebuild.
- **Performance**: `add_features` inserts columns one by one, which fragments the DataFrame (pandas
  `PerformanceWarning`). Building the columns in a dict and concatenating once would be faster.
- **License**: the repository has no `LICENSE` file; without one, nobody else may legally reuse the code.
