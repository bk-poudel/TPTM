import csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
DEMO_CSV = Path("/Users/bibek/TPTm/demonstrations/pick_place_20261016_160106.csv")
OUTPUT_CSV = Path("keypoints.csv")
# Desired cylinder translation relative to its original pose (metres).
CYLINDER_TRANSLATION = np.array([0.05, 0.03, 0.00])
# Desired cylinder tilt relative to its original orientation (degrees).
# Rotation is applied about the cylinder keypoints' centroid.
CYLINDER_TILT_DEG = np.array([15.0, 0.0, 20.0])  # roll, pitch, yaw
# Desired box translation relative to its original position (metres).
BOX_TRANSLATION = np.array([0.05, -0.04, 0.00])
# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def read_first_row(csv_path):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if row is None:
        raise ValueError(f"No data rows found in {csv_path}")
    return row
def read_keypoints(row, prefix, count=5):
    """Read kp0...kp4 XYZ coordinates from one CSV row."""
    points = []
    for i in range(count):
        keys = [f"{prefix}_kp{i}_{axis}" for axis in "xyz"]
        missing = [key for key in keys if key not in row]
        if missing:
            raise ValueError(
                f"Missing columns: {missing}. "
                f"Check that your demo CSV uses the expected headers."
            )
        points.append([float(row[key]) for key in keys])
    return np.asarray(points, dtype=float)
def transform_points(points, translation, euler_deg=None):
    """Rotate points about their centroid, then translate them."""
    center = points.mean(axis=0)
    if euler_deg is None:
        rotation = Rotation.identity()
    else:
        rotation = Rotation.from_euler(
            "xyz", euler_deg, degrees=True
        )
    return rotation.apply(points - center) + center + translation
# --------------------------------------------------
# BUILD SOURCE AND TARGET KEYPOINT PAIRS
# --------------------------------------------------
row = read_first_row(DEMO_CSV)
# Original cylinder keypoints from the first demonstration frame.
cylinder_source = read_keypoints(row, "src")
# The existing tgt_kp points are assumed to be the ORIGINAL box points.
box_source = read_keypoints(row, "tgt")
# Generate corresponding points in the modified scene.
cylinder_target = transform_points(
    cylinder_source,
    translation=CYLINDER_TRANSLATION,
    euler_deg=CYLINDER_TILT_DEG,
)
box_target = transform_points(
    box_source,
    translation=BOX_TRANSLATION,
)
# --------------------------------------------------
# SAVE CSV
# --------------------------------------------------
columns = [
    "object",
    "point_id",
    "src_x", "src_y", "src_z",
    "tgt_x", "tgt_y", "tgt_z",
]
with OUTPUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    for obj, source, target in [
        ("cylinder", cylinder_source, cylinder_target),
        ("box", box_source, box_target),
    ]:
        for i, (src, tgt) in enumerate(zip(source, target)):
            writer.writerow({
                "object": obj,
                "point_id": f"kp{i}",
                "src_x": src[0],
                "src_y": src[1],
                "src_z": src[2],
                "tgt_x": tgt[0],
                "tgt_y": tgt[1],
                "tgt_z": tgt[2],
            })
print(f"Saved: {OUTPUT_CSV.resolve()}")
print(f"Number of keypoint pairs: {len(cylinder_source) + len(box_source)}")
print("Cylinder translation:", CYLINDER_TRANSLATION)
print("Cylinder tilt (roll, pitch, yaw):", CYLINDER_TILT_DEG)
print("Box translation:", BOX_TRANSLATION)