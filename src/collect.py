"""Download the nflverse files used in this project into data/raw/.

Files come from the public nflverse-data GitHub releases:
https://github.com/nflverse/nflverse-data/releases

- stats_player_week_<season>.csv  one row per player per game, seasons 2015-2025
- stats_player_reg_<season>.csv   one row per player per regular season (2024 and 2025 only,
                                  used to check my own season totals against nflverse's)
- players.csv                     one row per player, used for birth date and rookie season

Writes data/collection_metadata.json with URLs, download time, sizes and sha256 hashes.
Run with --refresh to download again even if the files already exist.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://github.com/nflverse/nflverse-data/releases/download"
SEASONS = range(2015, 2026)

FILES = {}
for season in SEASONS:
    FILES[f"stats_player_week_{season}.csv"] = f"{BASE}/stats_player/stats_player_week_{season}.csv"
for season in (2024, 2025):
    FILES[f"stats_player_reg_{season}.csv"] = f"{BASE}/stats_player/stats_player_reg_{season}.csv"
FILES["players.csv"] = f"{BASE}/players/players.csv"

refresh = "--refresh" in sys.argv
RAW.mkdir(parents=True, exist_ok=True)
meta_path = ROOT / "data" / "collection_metadata.json"
meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"files": {}}

for name, url in FILES.items():
    out = RAW / name
    if out.exists() and not refresh:
        print("already have", name)
    else:
        print("downloading", url)
        r = requests.get(url, timeout=180, headers={"User-Agent": "INST414 student project (nfl-wr-yards-forecast)"})
        r.raise_for_status()
        out.write_bytes(r.content)
        meta["files"][name] = {"url": url, "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    data = out.read_bytes()
    meta["files"].setdefault(name, {"url": url})
    meta["files"][name]["bytes"] = len(data)
    meta["files"][name]["sha256"] = hashlib.sha256(data).hexdigest()

meta["source"] = "nflverse-data GitHub releases, https://github.com/nflverse/nflverse-data"
meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", meta_path)
