"""
Builds the pitch pool consumed by the Streamlit "Swing or Take" app.

For each candidate pitch (one row of `Data/derived/pitch_zone.csv`) this:
  1. joins to `play_state.csv` to find which at-bat it belongs to,
  2. joins to `pa_value.csv` to get that at-bat's REAL run contribution
     (pa_runs_contribution -- see derive_run_expectancy.py),
  3. pulls the pitch's actual ~20Hz ball-position samples (its real flight,
     no parametric model) directly from the game's own partition file,
  4. writes one JSON pool of pitches for the app to sample from each round.

A random subset of games is read directly from
`Data/ball-positions/{home}/{away}/{year}/{day}/ball-positions.csv` (each
leaf holds exactly one game), which is much cheaper than scanning the full
2.28M-row partitioned dataset for a ~300-pitch pool.

At-bats with a negative pa_runs_contribution (17 of 19,148 -- a residual
data-quality artifact from corrupted/resumed half-innings documented in
derive_play_state.py) are excluded from the pool.
"""
import json
import os
import re
import random

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "Data")
OUT_PATH = os.path.join(DATA_DIR, "derived", "game_pitches.json")

POOL_SIZE = 300
MAX_FRAMES = 45  # downsample long pitch windows for smooth, lightweight playback
RNG_SEED = 42

GAME_STRING_RE = re.compile(r"^(y\d+)_(d[\d.]+)_([A-Za-z]+)_([A-Za-z]+)$")


def game_string_to_partition(game_string: str):
    m = GAME_STRING_RE.match(game_string)
    if not m:
        return None
    year, day, away, home = m.groups()
    return {"home_team": home, "away_team": away, "year": year, "day": day}


def build_pool() -> pd.DataFrame:
    pz = pd.read_csv(os.path.join(DATA_DIR, "derived", "pitch_zone.csv"))
    ps = pd.read_csv(os.path.join(DATA_DIR, "derived", "play_state.csv"))
    pa = pd.read_csv(os.path.join(DATA_DIR, "derived", "pa_value.csv"))

    merged = pz.merge(
        ps[["game_string", "play_per_game", "at_bat", "half_inning_id",
            "outs_inning", "base_state", "inning", "half_inning", "batter"]],
        on=["game_string", "play_per_game"], how="inner",
    )
    merged = merged.drop_duplicates(subset=["game_string", "play_per_game"])
    merged = merged.merge(
        pa[["game_string", "half_inning_id", "at_bat", "pa_runs_contribution"]],
        on=["game_string", "half_inning_id", "at_bat"], how="inner",
    )
    merged = merged[merged["pa_runs_contribution"] >= 0]

    rng = random.Random(RNG_SEED)
    idx = rng.sample(range(len(merged)), k=min(POOL_SIZE, len(merged)))
    return merged.iloc[idx].reset_index(drop=True)


def load_game_positions_cache(game_strings) -> dict:
    cache = {}
    for gs in sorted(set(game_strings)):
        parts = game_string_to_partition(gs)
        if parts is None:
            continue
        path = os.path.join(
            DATA_DIR, "ball-positions",
            parts["home_team"], parts["away_team"], parts["year"], parts["day"],
            "ball-positions.csv",
        )
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df[df["game_string"] == gs]
        cache[gs] = df
    return cache


def get_pitch_window(be_path_cache, gs, play_per_game):
    parts = game_string_to_partition(gs)
    if parts is None:
        return None
    key = gs
    if key not in be_path_cache:
        path = os.path.join(
            DATA_DIR, "ball-events",
            parts["home_team"], parts["away_team"], parts["year"], parts["day"],
            "ball_events.csv",
        )
        if not os.path.exists(path):
            be_path_cache[key] = None
        else:
            be_path_cache[key] = pd.read_csv(path)
    be = be_path_cache[key]
    if be is None:
        return None
    play = be[be["play_per_game"] == play_per_game]
    if play.empty:
        return None
    pitch_rows = play[play["ball_eventcode"] == 1]
    if pitch_rows.empty:
        return None
    t_release = pitch_rows["timestamp"].min()
    # Same rule as derive_zone.py: the pitch's flight ends at whichever
    # comes first after release -- Ball Acquired (2) or Ball Hit Into Play
    # (4) -- not the play's last event (pulls in the batted ball's own
    # flight) and not just "the next event of any code" (an intervening
    # Ball Deflection (9), e.g. off the catcher's glove on a pitch in the
    # dirt, can occur seconds before the real acquisition and truncate the
    # window mid-flight).
    end_events = play[(play["timestamp"] > t_release) & (play["ball_eventcode"].isin([2, 4]))]
    if end_events.empty:
        return None
    t_end = end_events["timestamp"].min()
    return t_release, t_end


def downsample(frames, max_frames):
    if len(frames) <= max_frames:
        return frames
    step = len(frames) / max_frames
    return [frames[int(i * step)] for i in range(max_frames)]


def main():
    pool = build_pool()
    bp_cache = load_game_positions_cache(pool["game_string"].unique())
    be_cache = {}

    pitches = []
    skipped = 0
    for _, row in pool.iterrows():
        gs, ppg = row["game_string"], row["play_per_game"]
        bp = bp_cache.get(gs)
        if bp is None:
            skipped += 1
            continue
        window = get_pitch_window(be_cache, gs, ppg)
        if window is None:
            skipped += 1
            continue
        t_release, t_end = window
        play_bp = bp[(bp["play_per_game"] == ppg)
                      & (bp["timestamp"] >= t_release)
                      & (bp["timestamp"] <= t_end)].sort_values("timestamp")
        if len(play_bp) < 3:
            skipped += 1
            continue
        frames = play_bp[["ball_position_x", "ball_position_y", "ball_position_z"]].to_numpy().tolist()
        frames = downsample(frames, MAX_FRAMES)

        pitches.append({
            "id": len(pitches),
            "game_string": gs,
            "play_per_game": int(ppg),
            "inning": int(row["inning"]),
            "half_inning": row["half_inning"],
            "outs_inning": int(row["outs_inning"]),
            "base_state": int(row["base_state"]),
            "plate_x": round(float(row["plate_x"]), 4),
            "plate_z": round(float(row["plate_z"]), 4),
            "zone": row["zone"],
            "pa_runs_contribution": round(float(row["pa_runs_contribution"]), 3),
            "frames": [[round(v, 3) for v in f] for f in frames],
        })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"pitches": pitches}, f)

    print(f"pool requested: {len(pool)}, built: {len(pitches)}, skipped: {skipped}")
    print(f"zone mix: {pd.Series([p['zone'] for p in pitches]).value_counts().to_dict()}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
