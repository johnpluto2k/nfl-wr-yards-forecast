"""Checks on the input data and on the model output. Writes results/validation_report.md.

Run after collect.py, build_dataset.py and the notebook. Exits with an error if any check fails.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RES = ROOT / "results"

FEATURES = ["games", "targets_per_game", "target_share", "air_yards_share", "adot", "catch_rate",
            "yac_per_rec", "yards_per_target", "td_per_target", "epa_per_target", "yards_per_game",
            "age", "experience"]
LABEL = "next_yards_per_game"

results = []

def check(section, name, passed, detail):
    results.append((section, name, bool(passed), detail))
    print("PASS" if passed else "FAIL", "|", name, "|", detail)

# ---------------------------------------------------------------- input data
weekly = []
for season in range(2015, 2026):
    w = pd.read_csv(RAW / f"stats_player_week_{season}.csv", low_memory=False,
                    usecols=["player_id", "position", "season", "season_type", "game_id", "team"])
    weekly.append(w[w["season_type"] == "REG"])
weekly = pd.concat(weekly, ignore_index=True)

check("Input", "11 seasons of weekly data, 2015-2025",
      sorted(weekly["season"].unique()) == list(range(2015, 2026)), f"{len(weekly):,} regular-season player-game rows")
teams = weekly.groupby("season")["team"].nunique()
check("Input", "32 teams in every season", (teams == 32).all(), f"teams per season: {sorted(teams.unique().tolist())}")
dupes = int(weekly.duplicated(["player_id", "game_id"]).sum())
check("Input", "no player appears twice in the same game", dupes == 0, f"{dupes} duplicate player-game rows")

seasons = pd.read_csv(ROOT / "data" / "wr_seasons_2015_2025.csv")
pairs = pd.read_csv(ROOT / "data" / "wr_season_pairs.csv")
data = pairs[pairs["in_model"]]
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]

# my season totals (summed from the weekly files) against nflverse's own season file
for season in (2024, 2025):
    official = pd.read_csv(RAW / f"stats_player_reg_{season}.csv", low_memory=False)
    official = official[(official["position"] == "WR") & (official["targets"] >= 50)]
    mine = seasons[seasons["season"] == season]
    both = official.merge(mine, on="player_id", suffixes=("_nflverse", "_mine"))
    cols = ["games", "targets", "receptions", "receiving_yards", "receiving_tds"]
    mismatches = sum(int((both[c + "_nflverse"] != both[c + "_mine"]).sum()) for c in cols)
    check("Input", f"{season} season totals match nflverse's season file",
          len(both) == len(official) and mismatches == 0,
          f"{len(both)} WRs with 50+ targets x {len(cols)} stats, {mismatches} mismatches")

    # 17 = a full season; a traded player can reach 18 because of bye weeks, so don't use the maximum
    full = both[(both["games_mine"] == 17) & (both["n_teams"] == 1)]
    gap = (full["target_share_nflverse"] - full["target_share_mine"]).abs().max()
    check("Input", f"{season} target share matches nflverse for WRs who played every game for one team",
          gap < 0.001, f"{len(full)} WRs, largest difference {gap:.5f}")

jsn = seasons[(seasons["player"] == "Jaxon Smith-Njigba") & (seasons["season"] == 2025)].iloc[0]
nacua = seasons[(seasons["player"] == "Puka Nacua") & (seasons["season"] == 2025)].iloc[0]
check("Input", "two 2025 totals I checked against ESPN for my Module 3 post",
      jsn["receiving_yards"] == 1793 and nacua["receptions"] == 129,
      f"Smith-Njigba {int(jsn['receiving_yards'])} yards (ESPN 1,793), Nacua {int(nacua['receptions'])} catches (ESPN 129)")

check("Input", "every candidate row has 50+ targets", (pairs["targets"] >= 50).all(), f"{len(pairs)} rows, minimum {int(pairs['targets'].min())} targets")
check("Input", "every modeling row played 8+ games the next season", (data["next_games"] >= 8).all(),
      f"{len(data)} rows used, {len(pairs) - len(data)} dropped")
dupes = int(data.duplicated(["player_id", "season"]).sum())
check("Input", "one row per player per season", dupes == 0, f"{dupes} duplicates")
missing = int(data[FEATURES + [LABEL]].isna().sum().sum())
check("Input", "no missing values in the features or label", missing == 0, f"{missing} missing cells")
rates_ok = all(data[c].between(0, 1).all() for c in ["target_share", "air_yards_share", "catch_rate", "td_per_target"])
check("Input", "shares and rates are between 0 and 1", rates_ok, "target share, air yards share, catch rate, TD rate")
check("Input", "ages are plausible", data["age"].between(20, 40).all() and (data["experience"] >= 1).all(),
      f"age {data['age'].min():.1f} to {data['age'].max():.1f}, experience {int(data['experience'].min())} to {int(data['experience'].max())}")

# the label must be exactly the next season's yards per game from the season table
lookup = seasons.set_index(["player_id", "season"])["yards_per_game"]
recomputed = np.array([lookup[(pid, s + 1)] for pid, s in zip(data["player_id"], data["season"])])
gap = np.abs(recomputed - data[LABEL].values).max()
check("Input", "label equals the next season's yards per game", gap < 1e-9, f"largest difference {gap:.2e}")

check("Input", "no leakage between train and test",
      train["season"].max() == 2023 and train["next_season"].max() == 2024 and (test["season"] == 2024).all(),
      f"train features 2015-2023 and labels 2016-2024 ({len(train)} rows); test features 2024, labels 2025 ({len(test)} rows)")

# ---------------------------------------------------------------- model output
summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
saved = pd.read_csv(RES / "test_predictions.csv")
alpha = float(summary["chosen_model"].split("alpha=")[1].rstrip(")"))

model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(train[FEATURES], train[LABEL])
refit = pd.DataFrame({"player": test["player"].values, "predicted": model.predict(test[FEATURES])})
both = saved.merge(refit, on="player", suffixes=("_saved", "_refit"))
gap = (both["predicted_saved"] - both["predicted_refit"]).abs().max()
check("Output", "refitting from scratch reproduces the saved test predictions", len(both) == len(test) and gap < 1e-6,
      f"{len(both)} players, largest difference {gap:.2e}")

mae = mean_absolute_error(test[LABEL], model.predict(test[FEATURES]))
chosen = [v for k, v in summary["test"].items() if k.endswith("- chosen")][0]
check("Output", "test error in summary.json matches a fresh calculation", abs(mae - chosen["mae"]) < 0.001,
      f"MAE {mae:.3f} vs saved {chosen['mae']}")

same = mean_absolute_error(test[LABEL], test["yards_per_game"])
average = mean_absolute_error(test[LABEL], np.full(len(test), train[LABEL].mean()))
check("Output", "model beats both baselines on the test season", mae < same and mae < average,
      f"model {mae:.2f}, same as last season {same:.2f}, league average {average:.2f}")

by_season = pd.read_csv(RES / "cv_by_season.csv")
wins = int((by_season["model_mae"] < by_season["same_as_last_season_mae"]).sum())
check("Output", "model beats 'same as last season' in every held-out training season", wins == len(by_season),
      f"{wins} of {len(by_season)} seasons")

# if the labels are shuffled, the model should have nothing to learn
rng = np.random.default_rng(42)
shuffled_r2 = []
for _ in range(5):
    y = rng.permutation(train[LABEL].values)
    pred = cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=alpha)), train[FEATURES], y,
                             groups=train["season"], cv=LeaveOneGroupOut())
    shuffled_r2.append(r2_score(y, pred))
check("Output", "with shuffled labels the model learns nothing", max(shuffled_r2) < 0.05,
      f"cross-validated R2 over 5 shuffles: {min(shuffled_r2):.3f} to {max(shuffled_r2):.3f} (real labels: {summary['cv'][summary['chosen_model']]['r2']})")

check("Output", "predictions are in a plausible range", saved["predicted"].between(0, 130).all(),
      f"{saved['predicted'].min():.1f} to {saved['predicted'].max():.1f} yards per game")

misses = pd.read_csv(RES / "biggest_misses.csv")
actual_ok = True
for _, row in misses.iterrows():
    source = pairs[(pairs["player"] == row["player"]) & (pairs["season"] == 2024)].iloc[0]
    if abs(source["next_receiving_yards"] / source["next_games"] - row["actual"]) > 1e-9:
        actual_ok = False
    if abs((row["predicted"] - row["actual"]) - row["error"]) > 1e-9:
        actual_ok = False
top5 = saved.sort_values("abs_error", ascending=False).head(5)["player"].tolist()
check("Output", "the five biggest misses are the five largest errors and their numbers trace back to the data",
      actual_ok and top5 == misses["player"].tolist(), ", ".join(misses["player"]))

check("Output", "row counts in summary.json match the dataset",
      summary["rows_used"] == len(data) and summary["train_rows"] == len(train) and summary["test_rows"] == len(test),
      f"{len(data)} used = {len(train)} train + {len(test)} test")

# ---------------------------------------------------------------- report
passed = sum(1 for r in results if r[2])
lines = ["# Validation report", "", f"{passed} of {len(results)} checks passed.", ""]
for section in ("Input", "Output"):
    lines += [f"## {section} checks", ""]
    for sec, name, ok, detail in results:
        if sec == section:
            lines.append(f"- {'PASS' if ok else 'FAIL'}: {name}. {detail}")
    lines.append("")
(RES / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
print(f"{passed} of {len(results)} checks passed -> results/validation_report.md")
if passed != len(results):
    sys.exit(1)
