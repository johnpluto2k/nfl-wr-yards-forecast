"""Build the supervised-learning dataset: one row per wide receiver per pair of seasons.

Reads data/raw/ (see collect.py) and writes:
  data/wr_seasons_2015_2025.csv  one row per WR per regular season (every WR with 1+ target)
  data/wr_season_pairs.csv       one row per WR with MIN_TARGETS+ targets in season N,
                                 with that season's features and the NEXT season's result

Features come from season N (regular season only):
  games              games with a stat line
  targets_per_game   targets / games
  target_share       player targets / team targets, only in games the player played
  air_yards_share    player air yards / team air yards, only in games the player played
  adot               average depth of target = air yards / targets
  catch_rate         receptions / targets
  yac_per_rec        yards after catch / receptions
  yards_per_target   receiving yards / targets
  td_per_target      receiving touchdowns / targets
  epa_per_target     receiving EPA / targets
  yards_per_game     receiving yards / games
  age                age in years on Sept 1 of season N+1 (the season being predicted)
  experience         NFL seasons completed before season N+1 (a rookie in season N has 1)

Label (the thing the model predicts), from season N+1:
  next_yards_per_game = receiving yards / games in season N+1

A row is used for modeling (in_model = True) when the player played MIN_NEXT_GAMES or more
games in season N+1. Rows that fail this are kept in the file with in_model = False so the
number of dropped players can be reported.

The context columns (team, next_team, qb, next_qb, ...) are NOT features. They are only used
afterwards to look at the players the model got wrong.
"""
from pathlib import Path

import numpy as np
import pandas as pd

FIRST_SEASON = 2015
LAST_SEASON = 2025
MIN_TARGETS = 50
MIN_NEXT_GAMES = 8

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

cols = ["player_id", "player_display_name", "position", "season", "season_type", "game_id", "team",
        "attempts", "targets", "receptions", "receiving_yards", "receiving_tds",
        "receiving_air_yards", "receiving_yards_after_catch", "receiving_epa"]

frames = []
for season in range(FIRST_SEASON, LAST_SEASON + 1):
    w = pd.read_csv(RAW / f"stats_player_week_{season}.csv", low_memory=False, usecols=cols)
    frames.append(w[w["season_type"] == "REG"])
weekly = pd.concat(frames, ignore_index=True)
weekly["receiving_epa"] = weekly["receiving_epa"].fillna(0)   # EPA is blank when a player had no targets

# team targets and air yards in each game
team_game = weekly.groupby(["season", "game_id", "team"]).agg(
    team_targets=("targets", "sum"),
    team_air_yards=("receiving_air_yards", "sum"),
).reset_index()
weekly = weekly.merge(team_game, on=["season", "game_id", "team"], how="left")

# each team's main quarterback in each season = the player with the most pass attempts
qb_att = weekly.groupby(["season", "team", "player_display_name"])["attempts"].sum().reset_index()
qb_att = qb_att.sort_values("attempts", ascending=False).drop_duplicates(["season", "team"])
team_qb = qb_att.rename(columns={"player_display_name": "qb"})[["season", "team", "qb"]]

wr = weekly[weekly["position"] == "WR"]

# season totals for every WR; team totals only count the games the player played
seasons = wr.groupby(["player_id", "season"]).agg(
    player=("player_display_name", "last"),
    games=("game_id", "nunique"),
    targets=("targets", "sum"),
    receptions=("receptions", "sum"),
    receiving_yards=("receiving_yards", "sum"),
    receiving_tds=("receiving_tds", "sum"),
    air_yards=("receiving_air_yards", "sum"),
    yards_after_catch=("receiving_yards_after_catch", "sum"),
    receiving_epa=("receiving_epa", "sum"),
    team_targets_when_active=("team_targets", "sum"),
    team_air_yards_when_active=("team_air_yards", "sum"),
    n_teams=("team", "nunique"),
).reset_index()

# main team that season = the team the player played the most games for
team_games = wr.groupby(["player_id", "season", "team"])["game_id"].nunique().reset_index(name="team_games")
main_team = team_games.sort_values("team_games", ascending=False).drop_duplicates(["player_id", "season"])
seasons = seasons.merge(main_team[["player_id", "season", "team"]], on=["player_id", "season"], how="left")
seasons = seasons.merge(team_qb, on=["season", "team"], how="left")

seasons = seasons[seasons["targets"] > 0].copy()
seasons["targets_per_game"] = seasons["targets"] / seasons["games"]
seasons["target_share"] = seasons["targets"] / seasons["team_targets_when_active"]
seasons["air_yards_share"] = seasons["air_yards"] / seasons["team_air_yards_when_active"]
seasons["adot"] = seasons["air_yards"] / seasons["targets"]
seasons["catch_rate"] = seasons["receptions"] / seasons["targets"]
seasons["yac_per_rec"] = seasons["yards_after_catch"] / seasons["receptions"].replace(0, np.nan)
seasons["yards_per_target"] = seasons["receiving_yards"] / seasons["targets"]
seasons["td_per_target"] = seasons["receiving_tds"] / seasons["targets"]
seasons["epa_per_target"] = seasons["receiving_epa"] / seasons["targets"]
seasons["yards_per_game"] = seasons["receiving_yards"] / seasons["games"]

# birth date and rookie season from the nflverse players file
players = pd.read_csv(RAW / "players.csv", low_memory=False, usecols=["gsis_id", "birth_date", "rookie_season"])
players = players.rename(columns={"gsis_id": "player_id"}).drop_duplicates("player_id")
seasons = seasons.merge(players, on="player_id", how="left")

seasons = seasons.sort_values(["season", "targets"], ascending=[True, False])
seasons.to_csv(ROOT / "data" / "wr_seasons_2015_2025.csv", index=False)
print(f"{len(seasons)} WR seasons with 1+ target, {FIRST_SEASON}-{LAST_SEASON}")

# pair each season N with the same player's season N+1
this_year = seasons[(seasons["targets"] >= MIN_TARGETS) & (seasons["season"] < LAST_SEASON)].copy()
next_year = seasons[["player_id", "season", "games", "targets", "receiving_yards", "yards_per_game",
                     "targets_per_game", "target_share", "team", "qb"]].copy()
next_year["season"] = next_year["season"] - 1
next_year = next_year.rename(columns={
    "games": "next_games", "targets": "next_targets", "receiving_yards": "next_receiving_yards",
    "yards_per_game": "next_yards_per_game", "targets_per_game": "next_targets_per_game",
    "target_share": "next_target_share", "team": "next_team", "qb": "next_qb"})
pairs = this_year.merge(next_year, on=["player_id", "season"], how="left")

pairs["next_season"] = pairs["season"] + 1
sept1 = pd.to_datetime(pairs["next_season"].astype(str) + "-09-01")
pairs["age"] = (sept1 - pd.to_datetime(pairs["birth_date"])).dt.days / 365.25
pairs["experience"] = pairs["next_season"] - pairs["rookie_season"]

pairs["next_games"] = pairs["next_games"].fillna(0).astype(int)
pairs["in_model"] = pairs["next_games"] >= MIN_NEXT_GAMES
pairs["changed_team"] = pairs["team"] != pairs["next_team"]
pairs["changed_qb"] = pairs["qb"] != pairs["next_qb"]
pairs["split"] = np.where(pairs["season"] == LAST_SEASON - 1, "test", "train")

keep = ["player_id", "player", "season", "next_season", "split", "in_model",
        "games", "targets_per_game", "target_share", "air_yards_share", "adot", "catch_rate",
        "yac_per_rec", "yards_per_target", "td_per_target", "epa_per_target", "yards_per_game",
        "age", "experience",
        "next_yards_per_game",
        "targets", "receptions", "receiving_yards", "receiving_tds", "team", "qb", "n_teams",
        "next_games", "next_targets", "next_receiving_yards", "next_targets_per_game", "next_target_share",
        "next_team", "next_qb", "changed_team", "changed_qb", "birth_date", "rookie_season"]
pairs = pairs[keep].sort_values(["season", "targets"], ascending=[True, False])
pairs.to_csv(ROOT / "data" / "wr_season_pairs.csv", index=False)

print(f"{len(pairs)} WR seasons with {MIN_TARGETS}+ targets in {FIRST_SEASON}-{LAST_SEASON - 1}")
print(f"  no stats at all the next season: {(pairs['next_games'] == 0).sum()}")
print(f"  1 to {MIN_NEXT_GAMES - 1} games the next season: {((pairs['next_games'] > 0) & ~pairs['in_model']).sum()}")
print(f"  used for modeling ({MIN_NEXT_GAMES}+ games the next season): {pairs['in_model'].sum()}")
used = pairs[pairs["in_model"]]
print(used.groupby(["split", "season"]).size().to_string())
print("missing values in the modeling rows:")
print(used[keep[6:20]].isna().sum().to_string())
print("wrote data/wr_seasons_2015_2025.csv and data/wr_season_pairs.csv")
