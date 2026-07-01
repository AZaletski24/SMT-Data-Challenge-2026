"""
Derives release speed, induced vertical break (IVB), and horizontal break
(HB) for each pitch from the real ~20Hz `ball-positions` samples -- no
spin-rate or parametric Statcast fields exist in this dataset (no
vx0/vy0/vz0/ax/ay/az), so both the trajectory and its "spin-free" baseline
are fit directly from the tracked ball flight, the same no-parametric-model
philosophy as `derive_zone.py`.

METHODOLOGY (mirrors how Trackman/Statcast itself derives these from raw
3D ball tracking, since this dataset gives us the same raw inputs they
start from):

For each pitch's flight window (release -> plate crossing, reusing
`derive_zone.load_pitch_windows` / `find_plate_crossing` so the windowing
rule lives in exactly one place -- see the Bug 4 lesson in
smt_processes.md Sec. 3/6 about two copies of the same rule drifting
apart), fit a constant-acceleration trajectory to each axis independently
via least squares over every sample in the window:

    x(t) = x0 + vx0*t + 0.5*ax*t^2      (t seconds since the Pitch event)
    y(t) = y0 + vy0*t + 0.5*ay*t^2
    z(t) = z0 + vz0*t + 0.5*az*t^2

This is the standard 9-parameter trajectory model -- ax/ay/az aren't
measured directly (no Statcast columns for them), they're recovered from
the position samples themselves.

Release speed = |v(t=0)| = sqrt(vx0^2+vy0^2+vz0^2), converted to mph.

IVB/HB compare the actual plate-crossing position to where the pitch
would have crossed if ax/az were exactly what a spin-free pitch would
have (ax=0 for horizontal: no spin-induced sideways force; az=-g for
vertical: gravity only, no Magnus lift). Since both trajectories share the
same release point and v0, the only difference at flight time T is the
extra 0.5*a*T^2 term contributed by each axis's *measured* acceleration:

    HB_in  = 0.5 * ax        * T^2 * 12   (sideways break vs. a no-spin path)
    IVB_in = 0.5 * (az + G)  * T^2 * 12   (vertical break vs. gravity alone)

This is algebraically identical to "actual plate position minus a
constant-velocity/gravity-only baseline projected from release" -- the
fitted model is just a numerically cleaner way to get there, since the
window's tracked samples are noisy and this regresses over all of them at
once instead of differencing a couple of points.

No handedness field exists anywhere in this dataset (checked lineups.csv,
game-info.csv, every ball-events/positions schema), so HB sign here is
raw world-frame (matches ball_position_x: left negative, right positive
from behind the plate) -- NOT mirrored to an arm-side/glove-side
convention. That normalization step happens later, per-pitcher, in the
clustering notebook (inferred from each pitcher's own dominant pitch,
since we have no ground-truth throwing-hand label to mirror against).
"""
import os
import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from derive_zone import (
    DATA_DIR,
    PLAY_STATE_PATH,
    load_pitch_windows,
    find_plate_crossing,
)

OUT_PATH = os.path.join(DATA_DIR, "derived", "pitch_movement.csv")

G = 32.174  # ft/s^2
FT_PER_S_TO_MPH = 0.681818
MIN_SAMPLES_FOR_FIT = 4  # need >3 (params per axis) with margin for noise

# Generous outer bounds vs. the real-world range of any known pitch (the
# nastiest MLB sweepers/knuckle-curves top out around +/-20-25in of
# break; legal flight time over the plate distance is ~0.40-0.55s). Rows
# outside these are fit artifacts -- confirmed by inspection: they have
# MORE samples than a typical clean pitch (12 avg vs 10), consistent with
# a bounced/deflected ball or a misclassified non-pitch play pulling extra,
# non-ballistic samples into the window despite the t<=t_cross truncation.
PLAUSIBLE_BOUNDS = {
    "release_speed_mph": (40, 105),
    "hb_in": (-30, 30),
    "ivb_in": (-30, 30),
    "flight_time_s": (0, 0.65),
}


def load_ball_positions(pitch_keys: pd.DataFrame) -> pd.DataFrame:
    ds = pads.dataset(
        os.path.join(DATA_DIR, "ball-positions"),
        format="csv",
        partitioning=["home_team", "away_team", "year", "day"],
    )
    bp = ds.to_table().to_pandas()
    bp = bp[bp["game_string"] != "NA"].copy()
    bp = bp[bp["ball_position_z"].between(-1, 200)].copy()  # tracker-failure filter, see derive_zone.py
    bp["play_per_game"] = bp["play_per_game"].astype("Int64")
    bp = bp.merge(pitch_keys, on=["game_string", "play_per_game"], how="inner")
    return bp


def load_pitchers() -> pd.DataFrame:
    lu = pd.read_csv(os.path.join(DATA_DIR, "lineups.csv"))
    lu = lu[lu["game_string"] != "NA"].copy()
    lu["play_per_game"] = lu["play_per_game"].astype("Int64")
    return lu[["game_string", "play_per_game", "pitcher", "batter"]].drop_duplicates(
        subset=["game_string", "play_per_game"]
    )


def fit_axis(t: np.ndarray, pos: np.ndarray):
    """Least-squares fit of pos(t) = p0 + v0*t + 0.5*a*t^2. Returns (p0, v0, a)."""
    design = np.column_stack([np.ones_like(t), t, 0.5 * t**2])
    coeffs, *_ = np.linalg.lstsq(design, pos, rcond=None)
    return coeffs  # p0, v0, a


def compute_movement(group: pd.DataFrame, t_release: float, t_cross: float):
    g = group.sort_values("timestamp")
    g = g[g["timestamp"] <= t_cross]  # fit only the in-flight samples, not post-catch/contact noise
    if len(g) < MIN_SAMPLES_FOR_FIT:
        return None

    t = (g["timestamp"].to_numpy() - t_release) / 1000.0  # seconds, t=0 at release
    T = (t_cross - t_release) / 1000.0
    if T <= 0:
        return None

    x0, vx0, ax = fit_axis(t, g["ball_position_x"].to_numpy())
    y0, vy0, ay = fit_axis(t, g["ball_position_y"].to_numpy())
    z0, vz0, az = fit_axis(t, g["ball_position_z"].to_numpy())

    release_speed_fts = np.sqrt(vx0**2 + vy0**2 + vz0**2)
    hb_in = 0.5 * ax * T**2 * 12.0
    ivb_in = 0.5 * (az + G) * T**2 * 12.0

    return {
        "release_speed_mph": release_speed_fts * FT_PER_S_TO_MPH,
        "hb_in": hb_in,
        "ivb_in": ivb_in,
        "flight_time_s": T,
        "n_samples_fit": len(g),
    }


def main():
    play_state = pd.read_csv(PLAY_STATE_PATH)
    pitch_keys = play_state.loc[play_state["is_pitch"], ["game_string", "play_per_game"]].drop_duplicates()

    windows = load_pitch_windows(pitch_keys)
    bp = load_ball_positions(pitch_keys)
    bp = bp.merge(windows, on=["game_string", "play_per_game"], how="inner")
    bp = bp[(bp["timestamp"] >= bp["t_release"]) & (bp["timestamp"] <= bp["t_end"])]

    results = []
    for (game_string, play_per_game), group in bp.groupby(["game_string", "play_per_game"]):
        t_release = group["t_release"].iloc[0]
        crossing = find_plate_crossing(group)
        if crossing is None:
            continue
        plate_x, plate_z, t_cross = crossing

        movement = compute_movement(group, t_release, t_cross)
        if movement is None:
            continue

        results.append(
            {
                "game_string": game_string,
                "play_per_game": play_per_game,
                "plate_x": plate_x,
                "plate_z": plate_z,
                **movement,
            }
        )

    pitch_movement = pd.DataFrame(results)
    pitchers = load_pitchers()
    pitch_movement = pitch_movement.merge(pitchers, on=["game_string", "play_per_game"], how="left")

    n_resolved = len(pitch_movement)
    plausible = pd.Series(True, index=pitch_movement.index)
    for col, (lo, hi) in PLAUSIBLE_BOUNDS.items():
        plausible &= pitch_movement[col].between(lo, hi)
    n_dropped = (~plausible).sum()
    pitch_movement = pitch_movement[plausible].reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    pitch_movement.to_csv(OUT_PATH, index=False)

    n_total = len(pitch_keys)
    n_found = len(pitch_movement)
    print(f"pitches with is_pitch=True: {n_total}")
    print(f"pitches with resolvable movement: {n_resolved} ({n_resolved/n_total:.1%})")
    print(f"dropped as implausible fit artifacts: {n_dropped} ({n_dropped/n_resolved:.1%})")
    print(f"final usable pitches: {n_found} ({n_found/n_total:.1%})")
    print(pitch_movement[["release_speed_mph", "hb_in", "ivb_in", "flight_time_s"]].describe())
    print(f"unique pitchers: {pitch_movement['pitcher'].nunique()}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
