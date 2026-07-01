"""
Classifies each pitch as in-zone / out-of-zone using the actual tracked
ball flight in `ball-positions` -- no parametric trajectory model, just the
real ~20Hz samples.

Coordinates (official PDF): ball_position_y = 0 at the BACK of home plate,
y > 0 toward the pitcher/2nd base. Home plate is 17in (1.417ft) deep, so
the front edge -- where the rulebook measures the strike zone, and where
Statcast/Trackman's plate_x/plate_z are conventionally taken -- is at
y = 1.417, not y = 0.

For each pitch (a play_per_game with ball_eventcode 1 "Pitch" present),
take the window from the Pitch event (code 1) to whichever comes first:
Ball Acquired (code 2, a taken pitch) or Ball Hit Into Play (code 4,
contact) -- the two codes that actually end a pitch's flight per the
official glossary. Within that window, find the two consecutive ~50ms
samples where y crosses 1.417 going toward the plate, and linearly
interpolate x and z at that exact y. That (x, z) is the pitch's plate
location, analogous to Statcast plate_x/plate_z.

The strike zone box (height 1.5-3.5ft, width +/-0.708ft) is a fixed
generic rulebook zone -- this dataset has no batter stance/height data,
so it can't be batter-calibrated the way real Statcast sz_top/sz_bot are.
"""
import os
import pandas as pd
import pyarrow.dataset as pads

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "Data")
PLAY_STATE_PATH = os.path.join(DATA_DIR, "derived", "play_state.csv")
OUT_PATH = os.path.join(DATA_DIR, "derived", "pitch_zone.csv")

PLATE_FRONT_Y = 1.417  # ft; home plate is 17in deep, y=0 is the back tip
ZONE_HEIGHT = (1.5, 3.5)
ZONE_WIDTH = (-0.708, 0.708)


def load_pitch_windows(pitch_keys: pd.DataFrame) -> pd.DataFrame:
    ds = pads.dataset(
        os.path.join(DATA_DIR, "ball-events"),
        format="csv",
        partitioning=["home_team", "away_team", "year", "day"],
    )
    be = ds.to_table().to_pandas()
    be = be[be["game_string"] != "NA"].copy()
    be["play_per_game"] = be["play_per_game"].astype("Int64")
    be = be.merge(pitch_keys, on=["game_string", "play_per_game"], how="inner")

    be = be.sort_values(["game_string", "play_per_game", "timestamp"])
    pitch_t0 = (
        be[be["ball_eventcode"] == 1]
        .groupby(["game_string", "play_per_game"])["timestamp"]
        .min()
        .rename("t_release")
    )
    # Per the official glossary, a pitch's flight ends at whichever comes
    # first: ball_eventcode 2 (Ball Acquired -- caught) or 4 (Ball Hit Into
    # Play -- contact). The window must end there specifically, not at the
    # play's last event (an earlier bug: for a play with contact, the
    # batted ball's own much-longer flight downstream got pulled in, and the
    # crossing search locked onto THAT trajectory -- confirmed case
    # fabricated a ~170ft plate_z) and not at "whatever event comes next"
    # (a second bug: an intervening event with some other code -- e.g. a
    # Ball Deflection (9) off the catcher's glove on a pitch in the dirt --
    # can occur seconds before the ball is actually acquired, truncating the
    # window mid-flight and clipping the animation. Confirmed case: pitch ->
    # deflection at +550ms -> acquired at +3150ms; the old "next event of
    # any code" logic cut the window off at the deflection, 2.6s early).
    merged = be.merge(pitch_t0.reset_index(), on=["game_string", "play_per_game"])
    after_pitch = merged[merged["timestamp"] > merged["t_release"]]
    end_events = after_pitch[after_pitch["ball_eventcode"].isin([2, 4])]
    t_end = end_events.groupby(["game_string", "play_per_game"])["timestamp"].min().rename("t_end")
    windows = pd.concat([pitch_t0, t_end], axis=1).reset_index().dropna(subset=["t_release", "t_end"])
    return windows


def load_ball_positions(pitch_keys: pd.DataFrame) -> pd.DataFrame:
    ds = pads.dataset(
        os.path.join(DATA_DIR, "ball-positions"),
        format="csv",
        partitioning=["home_team", "away_team", "year", "day"],
    )
    bp = ds.to_table().to_pandas()
    bp = bp[bp["game_string"] != "NA"].copy()
    # Tracker failures (documented in the EDA notebook's ball-positions quality
    # section) put some samples at impossible heights -- as far as Z<-1 or
    # Z>200ft. Left in, a single corrupted sample can get selected as one of
    # the two points bracketing the plate crossing and produce a nonsensical
    # interpolated plate_x/plate_z (confirmed: pre-filter, ~1.9% of resolved
    # pitches had a plate_z outside [0,6]ft, some over 100ft).
    bp = bp[bp["ball_position_z"].between(-1, 200)].copy()
    bp["play_per_game"] = bp["play_per_game"].astype("Int64")
    bp = bp.merge(pitch_keys, on=["game_string", "play_per_game"], how="inner")
    return bp


def find_plate_crossing(group: pd.DataFrame):
    """Returns (plate_x, plate_z, t_cross) for the first y-crossing of
    PLATE_FRONT_Y, or None if the window never reaches the plate. t_cross is
    the interpolated timestamp at the crossing -- exposed (not just plate_x/
    plate_z) so derive_pitch_movement.py can reuse this exact crossing
    geometry/logic for time-of-flight instead of re-deriving it."""
    g = group.sort_values("timestamp")
    y = g["ball_position_y"].to_numpy()
    x = g["ball_position_x"].to_numpy()
    z = g["ball_position_z"].to_numpy()
    t = g["timestamp"].to_numpy()

    # first consecutive pair where y crosses PLATE_FRONT_Y moving toward the plate (decreasing)
    crosses = (y[:-1] >= PLATE_FRONT_Y) & (y[1:] < PLATE_FRONT_Y)
    idx = crosses.argmax() if crosses.any() else None
    if idx is None or not crosses.any():
        return None

    y0, y1 = y[idx], y[idx + 1]
    frac = (y0 - PLATE_FRONT_Y) / (y0 - y1)
    plate_x = x[idx] + frac * (x[idx + 1] - x[idx])
    plate_z = z[idx] + frac * (z[idx + 1] - z[idx])
    t_cross = t[idx] + frac * (t[idx + 1] - t[idx])
    return plate_x, plate_z, t_cross


def classify_zone(plate_x: float, plate_z: float) -> str:
    in_zone = (
        ZONE_WIDTH[0] <= plate_x <= ZONE_WIDTH[1]
        and ZONE_HEIGHT[0] <= plate_z <= ZONE_HEIGHT[1]
    )
    return "in_zone" if in_zone else "out_zone"


def main():
    play_state = pd.read_csv(PLAY_STATE_PATH)
    pitch_keys = play_state.loc[play_state["is_pitch"], ["game_string", "play_per_game"]].drop_duplicates()

    windows = load_pitch_windows(pitch_keys)
    bp = load_ball_positions(pitch_keys)

    # restrict each play's positions to its own pitch window (t_release..t_end)
    bp = bp.merge(windows, on=["game_string", "play_per_game"], how="inner")
    bp = bp[(bp["timestamp"] >= bp["t_release"]) & (bp["timestamp"] <= bp["t_end"])]

    results = []
    for (game_string, play_per_game), group in bp.groupby(["game_string", "play_per_game"]):
        crossing = find_plate_crossing(group)
        if crossing is None:
            continue
        plate_x, plate_z, _t_cross = crossing
        results.append(
            {
                "game_string": game_string,
                "play_per_game": play_per_game,
                "plate_x": plate_x,
                "plate_z": plate_z,
                "zone": classify_zone(plate_x, plate_z),
            }
        )

    pitch_zone = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    pitch_zone.to_csv(OUT_PATH, index=False)

    n_total = len(pitch_keys)
    n_found = len(pitch_zone)
    print(f"pitches with is_pitch=True: {n_total}")
    print(f"pitches with a resolvable plate crossing: {n_found} ({n_found/n_total:.1%})")
    print("zone distribution:")
    print(pitch_zone["zone"].value_counts())
    print(pitch_zone[["plate_x", "plate_z"]].describe())
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
