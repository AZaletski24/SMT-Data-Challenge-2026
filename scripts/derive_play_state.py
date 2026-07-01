"""
Derives per-pitch base/out state using only columns the SMT Data Challenge
2026 dataset actually has. There is no `outs` or `runs_scored` column
anywhere in this dataset, so both are reconstructed from a counting
identity validated by a prior challenge team
(github.com/pranavrajaram/smt-2025-submission, master-notebook.ipynb):

    batter_num_inning  = cumulative distinct batters who've come up this half-inning
    active_baserunners = runners currently on base (on_1b/on_2b/on_3b not null)
    baserunners_inning = cumulative distinct new players who've ever reached base
                          this half-inning (a home run counts the batter as one,
                          even though they never appear in on_*b)
    runs_inning        = baserunners_inning - active_baserunners
                          (anyone who left the bases is assumed to have scored;
                          this over-counts the rare baserunning out, e.g. caught
                          stealing -- accepted noise, see README note below)
    outs_inning        = batter_num_inning - active_baserunners - runs_inning,
                          clamped to {0, 1, 2} (a half-inning can only have
                          0/1/2 outs before any in-progress play)

`lineups.csv` already gives the true pre-pitch baserunner state per
play_per_game, so (unlike Trackman-style pipelines) there's no need to
simulate base advancement.

Pickoff plays (is_pickoff=True) are kept in the chronological state walk
because they can change base occupancy (e.g. caught stealing), but they
don't start with ball_eventcode 1 ("Pitch") so they're flagged via
`is_pitch=False` and excluded downstream wherever a real pitch is required.

KNOWN LIMITATION: runs_inning conflates "scored" with "put out on the
bases" (e.g. caught stealing, picked off) since both look identical as
"a runner who is no longer on base." This is inherited from the validated
prior-team approach and is a small, accepted noise source.

VALIDATED CORRECTION vs. the prior-team formula: that team's literal code
computes outs_inning = batter_num_inning - active_baserunners - runs_inning
(no -1), evaluated row-by-row. Checked against the first at-bat of every
half-inning here (where outs must be 0), their literal formula gives 1
in 99% of cases -- it's off by one because batter_num_inning already
counts the in-progress at-bat. The "- 1" above is the fix, confirmed
empirically: with it, 4255/4302 half-innings (99%) correctly show 0 outs
entering their first at-bat.
"""
import os
import pandas as pd
import pyarrow.dataset as pads

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "Data")
OUT_PATH = os.path.join(DATA_DIR, "derived", "play_state.csv")

BASE_COLS = ["on_1b", "on_2b", "on_3b"]


def load_lineups():
    lu = pd.read_csv(os.path.join(DATA_DIR, "lineups.csv"))
    lu = lu[lu["game_string"] != "NA"].copy()
    return lu


def load_ball_event_flags():
    ds = pads.dataset(
        os.path.join(DATA_DIR, "ball-events"),
        format="csv",
        partitioning=["home_team", "away_team", "year", "day"],
    )
    be = ds.to_table().to_pandas()
    be = be[be["game_string"] != "NA"].copy()
    be["play_per_game"] = be["play_per_game"].astype("Int64")

    flags = (
        be.groupby(["game_string", "play_per_game"])["ball_eventcode"]
        .agg(
            is_pitch=lambda s: (s == 1).any(),
            has_contact=lambda s: (s == 4).any(),
            is_home_run=lambda s: (s == 11).any(),
        )
        .reset_index()
    )
    return flags


def compute_play_state(lu, flags):
    df = lu.merge(flags, on=["game_string", "play_per_game"], how="left")
    for col in ["is_pitch", "has_contact", "is_home_run"]:
        df[col] = df[col].astype("boolean").fillna(False).astype(bool)

    df = df.sort_values(["game_string", "play_per_game"]).reset_index(drop=True)

    # half-inning id: increments every time (inning, half_inning) changes within a game
    half_inning_key = df["inning"].astype(str) + "_" + df["half_inning"].astype(str)
    df["half_inning_id"] = (
        half_inning_key.ne(half_inning_key.groupby(df["game_string"]).shift())
        .groupby(df["game_string"])
        .cumsum()
    )

    grp = ["game_string", "half_inning_id"]

    # batter_num_inning: how many distinct batters have come up so far this half-inning
    df["batter_num_inning"] = (
        df.groupby(grp)["at_bat"]
        .transform(lambda s: s.ne(s.shift()).cumsum())
        .clip(upper=12)
    )

    df["active_baserunners"] = df[BASE_COLS].notna().sum(axis=1)

    # new_baserunner: players occupying a base now who weren't on any base last play,
    # plus 1 for a home run (the batter scores without ever appearing in on_*b)
    def new_baserunner_count(group):
        prev_sets = [set()] + [
            {v for v in row if pd.notna(v)}
            for row in group[BASE_COLS].values.tolist()[:-1]
        ]
        curr_sets = [{v for v in row if pd.notna(v)} for row in group[BASE_COLS].values.tolist()]
        new_counts = [len(c - p) for c, p in zip(curr_sets, prev_sets)]
        return pd.Series(new_counts, index=group.index)

    df["new_baserunner"] = (
        df.groupby(grp, group_keys=False)[BASE_COLS].apply(new_baserunner_count)
        + df["is_home_run"].astype(int)
    )

    df["baserunners_inning"] = df.groupby(grp)["new_baserunner"].cumsum().clip(upper=10)
    df["runs_inning"] = df["baserunners_inning"] - df["active_baserunners"]

    # `lineups.csv`'s on_1b/2b/3b for a play reflect the state ENTERING that
    # play, not the result of it -- an at-bat's outcome (new baserunner,
    # cleared bases, etc.) only becomes visible starting at the first row of
    # the *next* at-bat (verified directly against real games: on_2b stays
    # null through every pitch of the at-bat where a runner doubles, and
    # only appears starting at the following at-bat's first pitch).
    # batter_num_inning, however, already counts the CURRENT (still
    # in-progress) at-bat. So the out count entering an at-bat is
    # (batter_num_inning - 1) - active_baserunners - runs_inning, evaluated
    # at that at-bat's FIRST row (not its last -- a mid-at-bat pickoff can
    # change on-base state before the at-bat resolves, so don't assume the
    # at-bat is constant throughout).
    first_row_idx = df.groupby(grp + ["at_bat"])["play_per_game"].idxmin()
    # A single-pitch at-bat that itself ends in a home run has is_home_run=True
    # on its own (only) row, unlike baserunner occupancy -- which lineups.csv
    # always lags by one at-bat. Left uncorrected, runs_inning at that row
    # already reflects the at-bat's OWN result instead of the state truly
    # entering it. Subtracting is_home_run here is a no-op for every other
    # at-bat (the deciding pitch's play_per_game differs from the first row
    # whenever an at-bat spans more than one pitch).
    runs_entering = (
        df.loc[first_row_idx, "runs_inning"] - df.loc[first_row_idx, "is_home_run"].astype(int)
    )
    entering_outs = (
        df.loc[first_row_idx, "batter_num_inning"] - 1
        - df.loc[first_row_idx, "active_baserunners"]
        - runs_entering
    )
    pa_outs = df.loc[first_row_idx, grp + ["at_bat"]].copy()
    pa_outs["outs_entering_pa"] = entering_outs.values

    df = df.merge(pa_outs, on=grp + ["at_bat"], how="left")
    # Entering outs can only be 0/1/2. Negative values (rare data artifacts --
    # e.g. resumed/double-header games where the true first batter of a
    # half-inning isn't the first row captured) clamp to 0, not 2.
    df["outs_inning"] = df["outs_entering_pa"].clip(lower=0, upper=2)
    df = df.drop(columns=["outs_entering_pa"])

    # base_state as a 0-7 bitmask: 1B=1, 2B=2, 3B=4
    df["base_state"] = (
        df["on_1b"].notna().astype(int) * 1
        + df["on_2b"].notna().astype(int) * 2
        + df["on_3b"].notna().astype(int) * 4
    )

    keep_cols = [
        "game_string", "play_per_game", "inning", "half_inning", "half_inning_id",
        "at_bat", "batter", "is_pickoff", "is_pitch", "has_contact", "is_home_run",
        "base_state", "outs_inning", "runs_inning", "baserunners_inning",
    ]
    return df[keep_cols]


def main():
    lu = load_lineups()
    flags = load_ball_event_flags()
    play_state = compute_play_state(lu, flags)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    play_state.to_csv(OUT_PATH, index=False)

    print(f"rows: {len(play_state)}")
    print(f"games: {play_state['game_string'].nunique()}")
    print("outs_inning distribution:")
    print(play_state["outs_inning"].value_counts().sort_index())
    print("base_state distribution:")
    print(play_state["base_state"].value_counts().sort_index())
    print(f"pitch rows (is_pitch=True): {play_state['is_pitch'].sum()}")
    print(f"contact rate among pitches: {play_state.loc[play_state['is_pitch'], 'has_contact'].mean():.3f}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
