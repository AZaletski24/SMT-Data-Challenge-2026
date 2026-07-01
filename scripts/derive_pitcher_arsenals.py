"""
Aggregates pitch_type.csv into a per-pitcher arsenal table:
one row per (pitcher, pitch_type_label) with usage rate and
average movement metrics.

A pitch type is kept in a pitcher's arsenal only if it accounts
for >= 5% of their total pitches AND >= 5 individual pitches --
below either threshold the cluster assignment is too noisy to
present as a real pitch type the pitcher "throws."

Output: Data/derived/pitcher_arsenals.csv
"""
import os
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "Data")
IN_PATH = os.path.join(DATA_DIR, "derived", "pitch_type.csv")
OUT_PATH = os.path.join(DATA_DIR, "derived", "pitcher_arsenals.csv")

MIN_PITCH_TYPE_COUNT = 5
MIN_PITCH_TYPE_PCT = 0.05


def main():
    pt = pd.read_csv(IN_PATH)

    totals = pt.groupby("pitcher").size().rename("total_pitches")

    agg = (
        pt.groupby(["pitcher", "pitch_type_label"])
        .agg(
            count=("pitch_type_label", "size"),
            avg_release_speed_mph=("release_speed_mph", "mean"),
            avg_ivb_in=("ivb_in", "mean"),
            avg_hb_in=("hb_in", "mean"),
            avg_hb_in_mirrored=("hb_in_mirrored", "mean"),
        )
        .reset_index()
    )

    agg = agg.merge(totals, on="pitcher")
    agg["usage_pct"] = agg["count"] / agg["total_pitches"]

    agg = agg[
        (agg["count"] >= MIN_PITCH_TYPE_COUNT) &
        (agg["usage_pct"] >= MIN_PITCH_TYPE_PCT)
    ].copy()

    agg = agg.sort_values(["pitcher", "usage_pct"], ascending=[True, False])

    for col in ["avg_release_speed_mph", "avg_ivb_in", "avg_hb_in", "avg_hb_in_mirrored", "usage_pct"]:
        agg[col] = agg[col].round(3)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    agg.to_csv(OUT_PATH, index=False)

    n_pitchers = agg["pitcher"].nunique()
    avg_types = agg.groupby("pitcher").size().mean()
    print(f"pitchers with an arsenal: {n_pitchers}")
    print(f"avg pitch types per pitcher: {avg_types:.1f}")
    print(f"pitch type distribution across all arsenals:")
    print(agg["pitch_type_label"].value_counts().to_string())
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
