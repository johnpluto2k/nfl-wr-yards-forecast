# nfl-wr-yards-forecast

INST414 Module 5 (supervised learning). A regression model that predicts an NFL wide receiver's receiving
yards per game **next** season from his stats **this** season, tested on the 2025 season, with a close look
at the five players it got most wrong.

The write-up is in `article/medium_draft.md`.

## Question

Using only what a wide receiver did this season, how well can we predict his receiving yards per game next
season, and which of this season's numbers actually carry over? The stakeholder is an NFL pro-personnel staff
deciding which veteran receivers to pursue and how much production to expect.

## Data

All data is from [nflverse](https://github.com/nflverse/nflverse-data/releases), downloaded with `requests`:

- `stats_player_week_<season>.csv` for 2015-2025 (one row per player per game)
- `stats_player_reg_2024.csv`, `stats_player_reg_2025.csv` (season totals, only used to check my own totals)
- `players.csv` (birth date and rookie season)

`data/raw/` is not in the repository because of its size (about 100 MB). `src/collect.py` downloads it again.
URLs, download times and sha256 hashes are in `data/collection_metadata.json`.

## Method

- One row per receiver per pair of seasons: features from season N, label from season N+1.
- Rows need 50+ targets in season N and 8+ games in season N+1: 706 rows.
- Label: receiving yards per game in season N+1 (continuous, so regression).
- 13 features: yards per game, targets per game, target share, air yards share, catch rate, yards per target,
  yards after catch per reception, touchdowns per target, EPA per target, games, average depth of target,
  age and experience.
- Train on season pairs 2015-2023 (630 rows), test once on 2024 -> 2025 (76 rows).
- Models compared with leave-one-season-out cross-validation: linear regression, ridge, k-nearest neighbors,
  decision tree, random forest, against two baselines (league average, same as last season).

## Results

| | Cross-validated MAE | Test MAE | Test R2 |
|---|---|---|---|
| Baseline: league average | 18.27 | 18.34 | -0.06 |
| Baseline: same as last season | 14.45 | 14.20 | 0.32 |
| Ridge regression (alpha = 30), chosen | 13.05 | 12.95 | 0.48 |

MAE is in receiving yards per game. Usage and last season's yards per game carry almost all of the signal;
the efficiency stats carry very little on their own. The five biggest misses on the test season were
Jaxon Smith-Njigba, Rashod Bateman, Allen Lazard, Adam Thielen and Puka Nacua.

## Files

```
src/collect.py          download the nflverse files into data/raw/
src/build_dataset.py    build data/wr_seasons_2015_2025.csv and data/wr_season_pairs.csv
notebooks/wr_yards_model.ipynb   model comparison, test, feature groups, error analysis, figures
src/validate.py         24 checks on the input data and the model output
data/                   the two built CSVs and the collection metadata
results/                cross-validation results, test predictions, biggest misses, validation report
article/                Medium draft and figures
```

## How to run

Python 3.14 with the packages in `requirements.txt`.

```
python src/collect.py
python src/build_dataset.py
jupyter nbconvert --to notebook --execute --inplace notebooks/wr_yards_model.ipynb
python src/validate.py
```

The notebook writes the tables in `results/` and the figures in `article/figures/`. All random steps use
seed 42.

## Limitations

- Receivers who played fewer than 8 games the next season are dropped (149 of 855), so the model describes
  receivers who stayed healthy and kept a job.
- The model knows nothing about the offseason: a new quarterback, teammates arriving or leaving, injuries.
- It pulls extreme seasons toward the middle and under-predicts the best receivers.
- One held-out season (76 players) is a small test set.

## Credits

Data: nflverse (https://github.com/nflverse/nflverse-data). Code and a first draft of the article were
written with an AI assistant (Claude, through Claude Code) and checked with `src/validate.py`.
