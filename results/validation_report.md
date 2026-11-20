# Validation report

24 of 24 checks passed.

## Input checks

- PASS: 11 seasons of weekly data, 2015-2025. 191,281 regular-season player-game rows
- PASS: 32 teams in every season. teams per season: [32]
- PASS: no player appears twice in the same game. 0 duplicate player-game rows
- PASS: 2024 season totals match nflverse's season file. 89 WRs with 50+ targets x 5 stats, 0 mismatches
- PASS: 2024 target share matches nflverse for WRs who played every game for one team. 31 WRs, largest difference 0.00000
- PASS: 2025 season totals match nflverse's season file. 78 WRs with 50+ targets x 5 stats, 0 mismatches
- PASS: 2025 target share matches nflverse for WRs who played every game for one team. 22 WRs, largest difference 0.00000
- PASS: two 2025 totals I checked against ESPN for my Module 3 post. Smith-Njigba 1793 yards (ESPN 1,793), Nacua 129 catches (ESPN 129)
- PASS: every candidate row has 50+ targets. 855 rows, minimum 50 targets
- PASS: every modeling row played 8+ games the next season. 706 rows used, 149 dropped
- PASS: one row per player per season. 0 duplicates
- PASS: no missing values in the features or label. 0 missing cells
- PASS: shares and rates are between 0 and 1. target share, air yards share, catch rate, TD rate
- PASS: ages are plausible. age 21.8 to 37.3, experience 1 to 16
- PASS: label equals the next season's yards per game. largest difference 0.00e+00
- PASS: no leakage between train and test. train features 2015-2023 and labels 2016-2024 (630 rows); test features 2024, labels 2025 (76 rows)

## Output checks

- PASS: refitting from scratch reproduces the saved test predictions. 76 players, largest difference 7.11e-15
- PASS: test error in summary.json matches a fresh calculation. MAE 12.951 vs saved 12.951
- PASS: model beats both baselines on the test season. model 12.95, same as last season 14.20, league average 18.34
- PASS: model beats 'same as last season' in every held-out training season. 9 of 9 seasons
- PASS: with shuffled labels the model learns nothing. cross-validated R2 over 5 shuffles: -0.027 to -0.018 (real labels: 0.486)
- PASS: predictions are in a plausible range. 18.0 to 87.2 yards per game
- PASS: the five biggest misses are the five largest errors and their numbers trace back to the data. Jaxon Smith-Njigba, Rashod Bateman, Allen Lazard, Adam Thielen, Puka Nacua
- PASS: row counts in summary.json match the dataset. 706 used = 630 train + 76 test
