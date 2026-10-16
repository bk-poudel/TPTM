import csv
from pathlib import Path
import numpy as np
# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
DEMO_CSV = Path(
    "/Users/bibek/TPTm/demonstrations/pick_place_20261016_160106.csv"
)
OUTPUT_CSV = Path("original_keypoints.csv")
def read_first_row(csv_path):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if row is None:
        raise ValueError(f"No data rows found in {csv_path}")
    return row
def read_keypoints(row, prefix, count=5):
    points = []
    for i in range(count):
        keys = [f"{prefix}_kp{i}_{axis}" for axis in "xyz"]
        missing = [key for key in keys if key not in row]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        points.append([float(row[key]) for key in keys])
    return np.asarray(points, dtype=float)
# --------------------------------------------------
# EXTRACT ORIGINAL KEYPOINTS
# --------------------------------------------------
row = read_first_row(DEMO_CSV)
# Original cylinder keypoints
cylinder_points = read_keypoints(row, "src")
# Original box keypoints
box_points = read_keypoints(row, "tgt")
# --------------------------------------------------
# SAVE TO CSV
# --------------------------------------------------
columns = ["object", "point_id", "x", "y", "z"]
with OUTPUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    for obj, points in [
        ("cylinder", cylinder_points),
        ("box", box_points),
    ]:
        for i, point in enumerate(points):
            writer.writerow({
                "object": obj,
                "point_id": f"kp{i}",
                "x": point[0],
                "y": point[1],
                "z": point[2],
            })
print(f"Saved original keypoints to: {OUTPUT_CSV.resolve()}")
print(f"Total keypoints: {len(cylinder_points) + len(box_points)}")
print("\nOriginal cylinder keypoints:")
print(cylinder_points)
print("\nOriginal box keypoints:")
print(box_points)