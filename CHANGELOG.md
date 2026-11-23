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

- `agent`: aligned with the current encoder (see Fixed). The state is now `gru_hidden + 72`
  values (200 with the default encoder, was 204) and the agent needs the encoder checkpoints
  with the strategy head (`python -m agent.strategy_head`); agents trained before must be
  trained again. **(AI-assisted)**

- `agent`: the arbiter no longer second-guesses the agent. Trades are ranked by the agent's
  advantage over HOLD, Q(a) − Q(HOLD), instead of a score where the trend / breakout bonuses
  (±0.5) outweighed the Q differences (±0.02) and the "confidence" was the embedding norm (a
  constant after the LayerNorm); the v26 entry filters and decision gates are off. Both stay
  available (`score_mode: "legacy"`, `entry_filters: true`), with the same thresholds in
  training and eval. **(AI-assisted)**
- `agent`: what can't be done is an action mask, in training too: position open, entry
  lockouts, the day's trades, daily DD budget and the portfolio limits (position slots, daily
  gate, DD block, kill switch). Before, the trades the arbiter rejected were stored as HOLD:
  in a synthetic Run1 the global 5-trades-a-day quota turned 98% of the agent's trade choices
  into HOLD and 85% of the trades it did learn from opened at 00:00 UTC, right after the quota
  reset (now: 94% executed, opens spread over the day). The portfolio quota binds in eval only
  (`train_daily_quota`). The replay stores the next state's mask and the Double DQN target
  picks among the available actions only. **(AI-assisted)**

### Added
- `agent/diagnostics.py`: `counterfactual` report in every eval / val summary — each trade the
  agent wanted, replayed with the env rules (mean R, win rate) per stage that stopped it and
  per entry-filter verdict, computed also with the filters off. **(AI-assisted)**
- `agent`: the joint runs (Run2 / Run3) evaluate every WF out of sample on the shared
  portfolio and save `eval_summary_<wf>.json`; before they only trained. **(AI-assisted)**
- `agent/strategy_head.py`: the agent's `StrategyOutcomeHead` (P(hit) and P(TP first) of TREND /
  MR x LONG / SHORT) trained on top of each frozen encoder checkpoint, with labels made by the
  env's own trade rules, on the checkpoint's training window (early stopping on its validation
  window). Writes `<models>/Encoder/<class>/Agent/<run>/encoder_agent.pth`, which the agent
  pipeline now reads; the encoder's own checkpoints and the PDHL heads are untouched.
  `StrategyOutcomeHead` is in `encoder.components` (v26 parameter names). **(AI-assisted)**
- `encoder`: checkpoints store `extra["inputs"]` (feature columns, session / structural flags,
  structural statistics per asset, asset map, TFs, `seq_len`), and
  `encoder.dataset.encoder_input_matrix` rebuilds the exact input from a parquet; session and
  structural features are module functions shared with the agent. Training unchanged (same
  weights, checked bit for bit). **(AI-assisted)**
- `agent/`: the Rainbow DQN trading agent, moved in from the research scripts `env_fx.py`,
  `agent_rainbow.py`, `arbiter.py`, `main_pipeline_ppo.py`: package layout, the configuration
  (`DEFAULT_CFG`, `RUN_CONFIGS`) in `agent/config.py`, a CLI (`python -m agent.pipeline --run ...`)
  that finds the encoder checkpoint of every WF and the Run1 agents for the joint runs instead of
  the hard-coded paths, comments and logs in English, Black/Ruff clean. Same agent weights and
  evaluation results as the scripts (checked bit for bit on synthetic data, single class and
  joint run). **(AI-assisted)**
- `tests/test_agent.py`: Rainbow train step and save/load, entry filters, environment state and
  steps, config and checkpoint discovery, one WF through the CLI. **(AI-assisted)**
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
- `agent`: settings that never reached the code: `train_filter_scale`, `adx_ranging_max`,
  `brk_min_mr` and `trend_train_bonus` were in `runner_cfg` but read from `arbiter_cfg` (the
  "filters at 40% / 100% in training" of Run2 / Run3 never applied: training was always
  unfiltered, eval always filtered); `class_overrides` was inside `runner_cfg` but read at the
  top level (the per-class `min_q_gap` 0.04 / 0.008 was never used); `eval_max_concurrent` only
  scaled a state feature, the eval cap was the training one; the per-rank Q-gap curriculum was
  never read. Moved, applied or removed. **(AI-assisted)**
- `agent`: the eval vote masked only actions 1-2, so a masked MR action could win a vote.
  **(AI-assisted)**
- `agent`: the env read the encoder differently from how it was trained — its own copy of the
  encoder, heads rebuilt from the `state_dict` shapes (trend without `tanh * 3`, DI direction
  through `softplus`, so never negative: DI gates and bonuses broken), columns in parquet order
  instead of the encoder's order, its own session features on the row time, no structural
  features, `seq_len` 128 against 256. It now uses `encoder.components`, the input saved in the
  checkpoint (`encoder_input_matrix`), the encoder's `seq_len` and asset ids, and stops on a
  feature-count mismatch instead of padding / truncating. **(AI-assisted)**
- `agent`: Wednesday / Friday force close, the entry lockouts, the overnight swap and the spread
  by session hour never ran: they read `dayofweek` on Python datetimes, the error was swallowed
  and treated as "no rule". **(AI-assisted)**
- `agent`: the spread column was `spread_mean_15m_raw` (the data has `spread_mean_15min_raw`), so
  every trade paid the fixed 1-pip fallback and the spread state feature was 0; the four
  `delta_*` state features read columns that don't exist (always 0) and are dropped.
  **(AI-assisted)**
- `agent`: the environments were created before the seeds were set (their reset draws random
  start points), so two runs with the same seed differed. **(AI-assisted)**
- `agent`: `TradeArbiter.passes_entry_filter` was called by the runner but no longer existed
  (the scripts crashed at the first proposal); it is back, as `entry_filter_reason(...) == "pass"`.
  **(AI-assisted)**
- `agent`: `run_pipeline` copied `DEFAULT_CFG` shallowly, so the overrides of one run changed the
  nested defaults of the next one in the same process; it now uses a deep copy. **(AI-assisted)**
- `agent`: `pytz` replaced by `datetime.timezone` (it wasn't in the requirements);
  `python-dateutil` added to them. **(AI-assisted)**
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

- `encoder`, strategy label execution (session breakout and PDHL), setting
  `outcome_strict_execution` (default on; `false` gives back the earlier labels bit for bit):
  **(AI-assisted)**
  - fills: a trade was labelled as soon as the bar crossed the Asian high/low or PDH/PDL, but the
    stop order sits 0.05 ATR beyond it, so bars that touched the level without reaching the entry
    became trades that were never filled. The arming check also used the low of the same bar,
    so a bar that opened beyond the entry counted as a fill at the entry. Now the order must be
    resting at the bar open (previous close on the near side) and the bar must reach the entry;
  - fill bar: the TP/SL scan started at the bar after the fill, so an SL hit in the fill bar was
    ignored. The fill bar is now scanned: a TP there is certain, an SL is ambiguous (loss, weight 0.30);
  - TP and SL in the same bar were labelled as a **win** (`tb <= sb`) although the comment and
    the simulator take the SL: now a loss, weight 0.30;
  - breakeven weights: every win with a later SL in the horizon got weight 0.35 (the BE trigger is
    always reached on the way to TP), and a win that came back to the BE stop with no SL after it
    kept 1.0. A win is now down-weighted only if price came back to the BE stop (entry + 0.07 R,
    `outcome_be_offset_r`, which was read but unused) after the trigger; a BE-saved loss gets 0.35
    also when the TP came later;
  - horizon: it counted rows, so Friday PDHL trades were decided by Sunday's reopening gap; it now
    stops at market breaks, and trades left open by a break or by the end of the data are skipped
    instead of counted as "no outcome";
  - Monday PDH/PDL came from the 2-3 hours of Sunday evening (UTC days): Monday now uses Friday;
  - daily ATR for the entry buffer: when a day had no Asian bars it fell back to the whole day,
    bars after the entry included; it now uses the Asian bars only.
  The trade simulation is now in `simulate_trade` / `stop_order_filled`, covered by
  `tests/test_strategy_labels.py`. The backtest simulator should follow the same rules, or the
  labels and the backtest will no longer match.

### Suggested, not applied
- **License**: the repository has no `LICENSE` file; without one, nobody else may legally reuse the
  code. **(AI-assisted review)**
- **Encoder, spread and costs**: the strategy labels use mid prices with no spread; a long stop
  fills on the ask and its SL triggers on the bid. Adding half the spread (`spread_mean_15min`) to
  entry and exits would make the labels closer to live fills. **(AI-assisted review)**

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

