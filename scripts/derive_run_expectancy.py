"""
Builds an empirical RE24-style run-expectancy table (8 base states x 3 out
states = 24 cells) entirely from `Data/derived/play_state.csv`, with no
external run-expectancy data.

Each at-bat's own run contribution is recovered the same way
`derive_play_state.py` recovers outs: lineups.csv reveals an at-bat's
outcome starting at the FIRST row of the NEXT at-bat (the cumulative
`runs_inning` counter jumps then). So for at-bat i:

    contribution_i = runs_inning_entering(i+1) - runs_inning_entering(i)

This works for every at-bat EXCEPT the half-inning's last one, since there
is no "next at-bat" within the half-inning to reveal it. For that one at-bat:
  - if it's a home run, the contribution is directly observable from
    ball-events (no lag needed): 1 (batter) + runners already on base.
  - otherwise (e.g. a sac fly or productive groundout scoring a runner
    while also making the 3rd out), the run truly can't be observed from
    this dataset, and is assumed to be 0.

KNOWN LIMITATION: this slightly undercounts runs scored on a half-inning's
final, non-home-run at-bat. It affects ~1 in ~4.5 at-bats (the last one of
each half-inning) and only when that at-bat actually drives in a run, so
it's a small downward bias concentrated in base-state/out cells that
frequently end half-innings (e.g. runner on 3rd, 2 outs).

VALIDATED CORRECTION: a single-pitch at-bat that itself ends in a home run
has is_home_run=True on its own (only) row in play_state.csv, so its
runs_inning value already includes its own result -- unlike every other
at-bat, where runs_inning at the first row only reflects PRIOR at-bats
(lineups.csv always lags baserunner state by one at-bat). Left
uncorrected, the lag subtraction below double-counts that self-contained
run away, silently zeroing out the contribution of 61 of 117 such
home runs (confirmed empirically pre-fix). Subtracting is_home_run from
runs_inning before lagging fixes this and is a no-op for every other
at-bat.

Once each at-bat has a `runs_remaining` value (its own contribution plus
everything that scores afterward, in the rest of the half-inning), the
RE24 table is just the average `runs_remaining` grouped by the state
ENTERING that at-bat (base_state, outs_inning).
"""
import os
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "Data")
PLAY_STATE_PATH = os.path.join(DATA_DIR, "derived", "play_state.csv")
RE24_OUT_PATH = os.path.join(DATA_DIR, "derived", "run_expectancy_24.csv")
PA_VALUE_OUT_PATH = os.path.join(DATA_DIR, "derived", "pa_value.csv")

GRP = ["game_string", "half_inning_id"]

BASE_STATE_LABELS = {
    0: "empty", 1: "1B", 2: "2B", 3: "1B_2B",
    4: "3B", 5: "1B_3B", 6: "2B_3B", 7: "loaded",
}


def build_at_bat_table(play_state: pd.DataFrame) -> pd.DataFrame:
    # one row per at-bat: take the FIRST pitch's snapshot, which carries the
    # state ENTERING that at-bat (see derive_play_state.py docstring).
    first_rows = play_state.loc[
        play_state.groupby(GRP + ["at_bat"])["play_per_game"].idxmin()
    ].copy()
    first_rows = first_rows.sort_values(GRP + ["at_bat"]).reset_index(drop=True)

    # is_home_run_first_pitch: this row's OWN flag, true only when the at-bat's
    # decisive (HR) pitch IS its first pitch (a single-pitch at-bat). Needed
    # separately from the at-bat-level "any" flag below for the
    # self-contamination fix -- using "any" there would wrongly also fire for
    # multi-pitch at-bats whose first pitch is clean but a later pitch homers.
    first_rows["is_home_run_first_pitch"] = first_rows["is_home_run"]

    # an at-bat is a home run if ANY of its pitches (the decisive one) was
    is_home_run_per_ab = (
        play_state.groupby(GRP + ["at_bat"])["is_home_run"].any().reset_index()
    )
    first_rows = first_rows.merge(is_home_run_per_ab, on=GRP + ["at_bat"], suffixes=("", "_any"))
    first_rows["is_home_run"] = first_rows["is_home_run_any"]
    first_rows = first_rows.drop(columns=["is_home_run_any"])

    first_rows["active_baserunners_entering"] = (
        (first_rows["base_state"] & 1 > 0).astype(int)
        + (first_rows["base_state"] & 2 > 0).astype(int)
        + (first_rows["base_state"] & 4 > 0).astype(int)
    )
    first_rows["is_final_pa"] = (
        first_rows["at_bat"] == first_rows.groupby(GRP)["at_bat"].transform("max")
    )

    # Same self-contamination fix as derive_play_state.py: a single-pitch
    # at-bat that itself is a home run has is_home_run=True on its own first
    # (only) row, so runs_inning there already includes its own result. No-op
    # for every other at-bat.
    runs_entering = first_rows["runs_inning"] - first_rows["is_home_run_first_pitch"].astype(int)
    next_runs_entering = runs_entering.groupby(
        [first_rows["game_string"], first_rows["half_inning_id"]]
    ).shift(-1)
    contribution_lagged = next_runs_entering - runs_entering
    contribution_hr_final = 1 + first_rows["active_baserunners_entering"]

    first_rows["pa_runs_contribution"] = contribution_lagged
    first_rows.loc[first_rows["is_final_pa"] & first_rows["is_home_run"], "pa_runs_contribution"] = (
        contribution_hr_final
    )
    first_rows.loc[first_rows["is_final_pa"] & ~first_rows["is_home_run"], "pa_runs_contribution"] = 0

    # reverse cumulative sum within each half-inning: runs scored from this
    # at-bat (inclusive) through the end of the half-inning
    first_rows["runs_remaining"] = (
        first_rows.groupby(GRP)["pa_runs_contribution"]
        .apply(lambda s: s[::-1].cumsum()[::-1])
        .reset_index(drop=True)
    )

    return first_rows


def build_re24_table(at_bats: pd.DataFrame) -> pd.DataFrame:
    re24 = (
        at_bats.groupby(["base_state", "outs_inning"])["runs_remaining"]
        .agg(re="mean", n="count")
        .reset_index()
    )
    return re24


def main():
    play_state = pd.read_csv(PLAY_STATE_PATH)
    at_bats = build_at_bat_table(play_state)
    re24 = build_re24_table(at_bats)

    os.makedirs(os.path.dirname(RE24_OUT_PATH), exist_ok=True)
    re24.to_csv(RE24_OUT_PATH, index=False)

    keep_cols = [
        "game_string", "half_inning_id", "at_bat", "base_state", "outs_inning",
        "is_final_pa", "is_home_run", "pa_runs_contribution", "runs_remaining",
    ]
    at_bats[keep_cols].to_csv(PA_VALUE_OUT_PATH, index=False)

    print(f"at-bats: {len(at_bats)}")
    print(f"final-PA non-HR at-bats (contribution assumed 0): {((at_bats['is_final_pa']) & (~at_bats['is_home_run'])).sum()}")
    print()
    print("RE24 table (rows=base state, cols=outs):")
    pivot = re24.copy()
    pivot["base_state"] = pivot["base_state"].map(BASE_STATE_LABELS)
    print(pivot.pivot(index="base_state", columns="outs_inning", values="re").reindex(BASE_STATE_LABELS.values()).round(3))
    print()
    print(f"wrote {RE24_OUT_PATH}")
    print(f"wrote {PA_VALUE_OUT_PATH}")


if __name__ == "__main__":
    main()
