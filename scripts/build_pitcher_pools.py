"""
Pre-builds pitch pools (with real ~20Hz trajectory frames) for every
eligible pitcher and writes them to Data/derived/pitcher_pools.json.

This is the cloud-deployment equivalent of build_game_pitches.py: the
Streamlit app runs on Streamlit Cloud where Data/ball-positions/ is not
available (gitignored). Loading trajectories from raw partition files at
runtime therefore isn't possible in that environment.  This script runs
once locally (where the raw data IS available) and produces a single JSON
file that gets committed to the repo alongside the other derived CSVs.

Eligible: pitchers with >= PITCHER_MIN_PITCHES total pitches in
pitch_type.csv (the same threshold the app uses for pitcher selection).

Output schema:
  { "PHD-4341": [ {pitch_dict}, ... ], ... }

Each pitch_dict contains game_string, play_per_game, pitch_type_label,
zone, plate_x, plate_z, release_speed_mph, and frames (list of
[x, y, z] lists, already downsampled to MAX_FRAMES).
"""
import json
import os
import re

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "Data")
DERIVED = os.path.join(DATA_DIR, "derived")
OUT_PATH = os.path.join(DERIVED, "pitcher_pools.json")

PITCHER_MIN_PITCHES = 40
MAX_FRAMES = 45

GAME_STRING_RE = re.compile(r"^(y\d+)_(d[\d.]+)_([A-Za-z]+)_([A-Za-z]+)$")


def _partition_path(game_string: str, dataset: str) -> str | None:
    m = GAME_STRING_RE.match(game_string)
    if not m:
        return None
    year, day, away, home = m.groups()
    fname = "ball_events.csv" if dataset == "ball-events" else "ball-positions.csv"
    return os.path.join(DATA_DIR, dataset, home, away, year, day, fname)


def _pitch_window(be: pd.DataFrame, play_per_game) -> tuple | None:
    """Same rule as derive_zone.py / build_game_pitches.py / app.py:
    window ends at the first ball_eventcode 2 or 4 after the Pitch event."""
    play = be[be["play_per_game"] == play_per_game]
    pitch_rows = play[play["ball_eventcode"] == 1]
    if pitch_rows.empty:
        return None
    t_release = pitch_rows["timestamp"].min()
    end_events = play[
        (play["timestamp"] > t_release) &
        (play["ball_eventcode"].isin([2, 4]))
    ]
    if end_events.empty:
        return None
    return t_release, end_events["timestamp"].min()


def _downsample(frames: list, n: int) -> list:
    if len(frames) <= n:
        return frames
    step = len(frames) / n
    return [frames[int(i * step)] for i in range(n)]


def build_pool_for_pitcher(rows: pd.DataFrame) -> list[dict]:
    be_cache: dict[str, pd.DataFrame | None] = {}
    bp_cache: dict[str, pd.DataFrame | None] = {}
    pool = []

    for _, row in rows.iterrows():
        gs = row["game_string"]
        ppg = row["play_per_game"]

        if gs not in be_cache:
            path = _partition_path(gs, "ball-events")
            be_cache[gs] = pd.read_csv(path) if path and os.path.exists(path) else None
        if gs not in bp_cache:
            path = _partition_path(gs, "ball-positions")
            if path and os.path.exists(path):
                df = pd.read_csv(path)
                bp_cache[gs] = df[df["ball_position_z"].between(-1, 200)]
            else:
                bp_cache[gs] = None

        be, bp = be_cache[gs], bp_cache[gs]
        if be is None or bp is None:
            continue

        window = _pitch_window(be, ppg)
        if window is None:
            continue
        t_release, t_end = window

        play_bp = (
            bp[(bp["play_per_game"] == ppg) &
               (bp["timestamp"] >= t_release) &
               (bp["timestamp"] <= t_end)]
            .sort_values("timestamp")
        )
        if len(play_bp) < 3:
            continue

        frames = _downsample(
            play_bp[["ball_position_x", "ball_position_y",
                      "ball_position_z"]].values.tolist(),
            MAX_FRAMES,
        )
        pool.append({
            "game_string": gs,
            "play_per_game": int(ppg),
            "pitch_type_label": row["pitch_type_label"],
            "zone": row["zone"],
            "plate_x": round(float(row["plate_x"]), 4),
            "plate_z": round(float(row["plate_z"]), 4),
            "release_speed_mph": round(float(row["release_speed_mph"]), 1),
            "frames": [[round(v, 3) for v in f] for f in frames],
        })

    return pool


def main():
    pt = pd.read_csv(os.path.join(DERIVED, "pitch_type.csv"),
                     usecols=["game_string", "play_per_game", "pitcher",
                               "release_speed_mph", "pitch_type_label"])
    pz = pd.read_csv(os.path.join(DERIVED, "pitch_zone.csv"),
                     usecols=["game_string", "play_per_game",
                               "plate_x", "plate_z", "zone"])
    index = pt.merge(pz, on=["game_string", "play_per_game"], how="inner")

    totals = index.groupby("pitcher").size()
    eligible = totals[totals >= PITCHER_MIN_PITCHES].index
    index = index[index["pitcher"].isin(eligible)]

    print(f"eligible pitchers: {len(eligible)}  |  total pitches: {len(index)}")

    pools: dict[str, list] = {}
    for i, (pitcher_id, rows) in enumerate(index.groupby("pitcher"), 1):
        pool = build_pool_for_pitcher(rows)
        if len(pool) >= 6:
            pools[pitcher_id] = pool
        if i % 50 == 0:
            print(f"  {i}/{len(eligible)} pitchers processed …")

    os.makedirs(DERIVED, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(pools, f, separators=(",", ":"))

    import os as _os
    size_mb = _os.path.getsize(OUT_PATH) / 1e6
    print(f"\npitchers with usable pool: {len(pools)}")
    print(f"wrote {OUT_PATH}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
