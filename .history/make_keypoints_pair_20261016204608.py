import pandas as pd
from pathlib import Path
# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
ORIGINAL_CSV = Path("./keypoints/original_keypoints.csv")
TRANSFORMED_CSV = Path("./keypoints/transformed_keypoints.csv")
OUTPUT_CSV = Path("./keypoints/keypoints_pairs.csv")
def load_keypoints(path):
    df = pd.read_csv(path)
    required = {"object", "point_id", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )
    df["object"] = df["object"].astype(str).str.strip().str.lower()
    df["point_id"] = df["point_id"].astype(str).str.strip()
    if df.duplicated(["object", "point_id"]).any():
        raise ValueError(
            f"Duplicate object/point_id pairs found in {path}"
        )
    return df
# --------------------------------------------------
# LOAD BOTH FILES
# --------------------------------------------------
original = load_keypoints(ORIGINAL_CSV)
transformed = load_keypoints(TRANSFORMED_CSV)
# Rename coordinate columns to distinguish source and target
original = original.rename(columns={
    "x": "src_x",
    "y": "src_y",
    "z": "src_z",
})
transformed = transformed.rename(columns={
    "x": "tgt_x",
    "y": "tgt_y",
    "z": "tgt_z",
})
# --------------------------------------------------
# MATCH CORRESPONDING KEYPOINTS
# --------------------------------------------------
pairs = original.merge(
    transformed,
    on=["object", "point_id"],
    how="outer",
    indicator=True,
    validate="one_to_one",
)
# Check that every original point has a transformed counterpart
unmatched = pairs[pairs["_merge"] != "both"]
if not unmatched.empty:
    raise ValueError(
        "Some keypoints do not have matching counterparts:\n"
        + unmatched[["object", "point_id", "_merge"]].to_string(index=False)
    )
pairs = pairs.drop(columns="_merge")
# Arrange columns in the desired order
pairs = pairs[
    [
        "object",
        "point_id",
        "src_x", "src_y", "src_z",
        "tgt_x", "tgt_y", "tgt_z",
    ]
]
# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------
pairs.to_csv(OUTPUT_CSV, index=False)
print(f"Saved keypoint pairs to: {OUTPUT_CSV.resolve()}")
print(f"Total keypoint pairs: {len(pairs)}")
print("\nPairs per object:")
print(pairs.groupby("object").size())
print("\nPreview:")
print(pairs.to_string(index=False))