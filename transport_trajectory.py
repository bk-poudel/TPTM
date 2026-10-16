import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
DEMO_CSV = Path("demonstrations/Original_demonstration.csv")
MODEL_FILE = Path("transport_maps.pkl")
OUTPUT_CSV = Path("demonstrations/Generalized_demonstration.csv")
# Set True only if target_x/y/z and target_roll/pitch/yaw
# represent the commanded end-effector pose in your CSV.
TRANSFORM_TARGET_COLUMNS = True
XYZ_COLUMNS = ["x", "y", "z"]
EULER_COLUMNS = ["roll", "pitch", "yaw"]
# --------------------------------------------------
# LOAD THE SAVED RIGID MAPS
# --------------------------------------------------
with MODEL_FILE.open("rb") as f:
    all_models = pickle.load(f)
cylinder_map = all_models["cylinder"]["rigid"]
box_map = all_models["box"]["rigid"]
def map_rotation(model):
    # Kabsch stores a row-vector rotation matrix.
    # SciPy Rotation uses column-vector conventions.
    return Rotation.from_matrix(model["R"].T)
def transform_points(points, model):
    rot = map_rotation(model)
    return (
        rot.apply(points - model["source_center"])
        + model["target_center"]
    )
def partial_transform(points, model, alpha):
    """Gradually apply a rigid map; alpha ranges from 0 to 1."""
    rot = map_rotation(model)
    rotvec = rot.as_rotvec()
    result = np.empty_like(points, dtype=float)
    for i, (p, a) in enumerate(zip(points, alpha)):
        partial_rot = Rotation.from_rotvec(a * rotvec)
        result[i] = (
            partial_rot.apply(p - model["source_center"])
            + model["source_center"]
            + a * (
                model["target_center"] - model["source_center"]
            )
        )
    return result
# --------------------------------------------------
# LOAD DEMONSTRATION
# --------------------------------------------------
df = pd.read_csv(DEMO_CSV)
required = [
    "segment", "ee_x", "ee_y", "ee_z",
    "ee_roll", "ee_pitch", "ee_yaw",
    "gripper_target",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")
# Work with a fresh sequential index.
df = df.reset_index(drop=True)
ee_cols = ["ee_x", "ee_y", "ee_z"]
orientation_cols = ["ee_roll", "ee_pitch", "ee_yaw"]
# Preserve original Cartesian poses.
original_xyz = df[ee_cols].to_numpy(float)
original_euler = df[orientation_cols].to_numpy(float)
if not np.isfinite(original_xyz).all():
    raise ValueError("Non-finite end-effector positions found.")
if not np.isfinite(original_euler).all():
    raise ValueError("Non-finite end-effector orientations found.")
generalized_xyz = original_xyz.copy()
generalized_rot = Rotation.from_euler("xyz", original_euler)
segments = df["segment"].astype(str).to_numpy()
# --------------------------------------------------
# PHASE 1: APPROACH CYLINDER
# Gradually adapt while keeping the first point fixed.
# --------------------------------------------------
mask = segments == "approach_cylinder"
idx = np.flatnonzero(mask)
if len(idx) == 0:
    raise ValueError("No approach_cylinder phase found.")
alpha = np.linspace(0.0, 1.0, len(idx))
generalized_xyz[idx] = partial_transform(
    original_xyz[idx], cylinder_map, alpha
)
cyl_rot = map_rotation(cylinder_map)
# Gradually adapt orientation during approach.
for i, a in zip(idx, alpha):
    partial_rot = Rotation.from_rotvec(a * cyl_rot.as_rotvec())
    generalized_rot[i] = partial_rot * Rotation.from_euler(
        "xyz", original_euler[i]
    )
# --------------------------------------------------
# PHASE 2: DESCEND, GRASP, LIFT
# Follow the transformed cylinder pose.
# --------------------------------------------------
cylinder_phases = [
    "descend_to_grasp",
    "close_gripper",
    "lift",
]
for phase in cylinder_phases:
    idx = np.flatnonzero(segments == phase)
    generalized_xyz[idx] = transform_points(
        original_xyz[idx], cylinder_map
    )
    generalized_rot[idx] = (
        cyl_rot * Rotation.from_euler("xyz", original_euler[idx])
    )
# --------------------------------------------------
# PHASE 3: TRANSPORT
# Blend cylinder-map and box-map positions smoothly.
# Preserve cylinder orientation while holding it.
# --------------------------------------------------
idx = np.flatnonzero(segments == "move_to_place")
if len(idx) == 0:
    raise ValueError("No move_to_place phase found.")
cylinder_path = transform_points(original_xyz[idx], cylinder_map)
box_path = transform_points(original_xyz[idx], box_map)
alpha = np.linspace(0.0, 1.0, len(idx))
generalized_xyz[idx] = (
    (1.0 - alpha[:, None]) * cylinder_path
    + alpha[:, None] * box_path
)
generalized_rot[idx] = (
    cyl_rot * Rotation.from_euler("xyz", original_euler[idx])
)
# --------------------------------------------------
# PHASE 4: LOWER AND RELEASE
# Move to the transformed box, keeping the held object's
# orientation until release.
# --------------------------------------------------
for phase in ["lower_to_place", "release"]:
    idx = np.flatnonzero(segments == phase)
    generalized_xyz[idx] = transform_points(
        original_xyz[idx], box_map
    )
    generalized_rot[idx] = (
        cyl_rot * Rotation.from_euler("xyz", original_euler[idx])
    )
# --------------------------------------------------
# PHASE 5: RETREAT
# Use box-transformed positions and smoothly restore
# the original end-effector orientation.
# --------------------------------------------------
idx = np.flatnonzero(segments == "retreat")
if len(idx) == 0:
    raise ValueError("No retreat phase found.")
generalized_xyz[idx] = transform_points(
    original_xyz[idx], box_map
)
alpha = np.linspace(0.0, 1.0, len(idx))
for i, a in zip(idx, alpha):
    transformed_orientation = (
        cyl_rot * Rotation.from_euler("xyz", original_euler[i])
    )
    # Slerp between transformed and original orientation.
    interpolator = Slerp(
        [0.0, 1.0],
        Rotation.concatenate([
            transformed_orientation,
            Rotation.from_euler("xyz", original_euler[i]),
        ]),
    )
    generalized_rot[i] = interpolator([a])[0]
# --------------------------------------------------
# SAVE GENERALIZED END-EFFECTOR POSES
# --------------------------------------------------
df[ee_cols] = generalized_xyz
# Convert back to XYZ Euler angles in radians.
df[orientation_cols] = generalized_rot.as_euler("xyz")
# Transform commanded EE targets only if these columns
# really describe commanded end-effector poses.
target_xyz_cols = ["target_x", "target_y", "target_z"]
target_euler_cols = [
    "target_roll", "target_pitch", "target_yaw"
]
if TRANSFORM_TARGET_COLUMNS and all(
    c in df.columns for c in target_xyz_cols
):
    original_targets = df[target_xyz_cols].to_numpy(float)
    generalized_targets = original_targets.copy()
    for phase in ["descend_to_grasp", "close_gripper", "lift"]:
        idx = np.flatnonzero(segments == phase)
        generalized_targets[idx] = transform_points(
            original_targets[idx], cylinder_map
        )
    idx = np.flatnonzero(segments == "approach_cylinder")
    generalized_targets[idx] = partial_transform(
        original_targets[idx],
        cylinder_map,
        np.linspace(0.0, 1.0, len(idx)),
    )
    idx = np.flatnonzero(segments == "move_to_place")
    cylinder_targets = transform_points(
        original_targets[idx], cylinder_map
    )
    box_targets = transform_points(
        original_targets[idx], box_map
    )
    a = np.linspace(0.0, 1.0, len(idx))
    generalized_targets[idx] = (
        (1.0 - a[:, None]) * cylinder_targets
        + a[:, None] * box_targets
    )
    for phase in ["lower_to_place", "release", "retreat"]:
        idx = np.flatnonzero(segments == phase)
        generalized_targets[idx] = transform_points(
            original_targets[idx], box_map
        )
    df[target_xyz_cols] = generalized_targets
# Keep gripper commands and original timing unchanged.
# Joint positions, velocities and torques are not recomputed:
# use a Cartesian controller to track the generalized pose.
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved generalized trajectory: {OUTPUT_CSV.resolve()}")
print(f"Samples: {len(df)}")
print("\nOutput phase counts:")
print(df["segment"].value_counts().to_string())
print("\nFirst and last end-effector positions by phase:")
for phase, group in df.groupby("segment", sort=False):
    print(
        f"{phase:20s} "
        f"start={group[ee_cols].iloc[0].to_numpy().round(4)} "
        f"end={group[ee_cols].iloc[-1].to_numpy().round(4)}"
    )
