import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
# ============================================================
# CONFIGURATION
# ============================================================
DEMO_CSV = Path("demonstrations/Original_demonstration.csv")
MODEL_FILE = Path("./output/transport_maps.pkl")
OUTPUT_CSV = Path("output/Generalized_trajectory.csv")
# Transform both commanded target positions and orientations.
TRANSFORM_TARGET_COLUMNS = True
XYZ_COLUMNS = ["x", "y", "z"]
EULER_COLUMNS = ["roll", "pitch", "yaw"]
EE_POS_COLS = ["ee_x", "ee_y", "ee_z"]
EE_EULER_COLS = ["ee_roll", "ee_pitch", "ee_yaw"]
TARGET_POS_COLS = ["target_x", "target_y", "target_z"]
TARGET_EULER_COLS = [
    "target_roll",
    "target_pitch",
    "target_yaw",
]
# ============================================================
# LOAD MAPS
# ============================================================
with MODEL_FILE.open("rb") as f:
    all_models = pickle.load(f)
cylinder_map = all_models["cylinder"]["rigid"]
box_map = all_models["box"]["rigid"]
def map_rotation(model):
    """
    Convert the Kabsch row-vector rotation matrix to a SciPy
    Rotation using the column-vector convention.
    """
    return Rotation.from_matrix(model["R"].T)
def transform_points(points, model):
    """Apply the complete fitted rigid map to 3D points."""
    points = np.asarray(points, dtype=float)
    rot = map_rotation(model)
    return (
        rot.apply(points - model["source_center"])
        + model["target_center"]
    )
def partial_transform(points, model, alpha):
    """
    Gradually transport positions from source to target.
    alpha = 0: original source position
    alpha = 1: fully transported position
    """
    points = np.asarray(points, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if len(points) != len(alpha):
        raise ValueError("points and alpha must have equal lengths.")
    rot = map_rotation(model)
    rotvec = rot.as_rotvec()
    result = np.empty_like(points, dtype=float)
    for i, (point, a) in enumerate(zip(points, alpha)):
        partial_rot = Rotation.from_rotvec(a * rotvec)
        result[i] = (
            partial_rot.apply(point - model["source_center"])
            + model["source_center"]
            + a * (
                model["target_center"] - model["source_center"]
            )
        )
    return result
# ============================================================
# ORIENTATION TRANSPORT
# ============================================================
def transport_orientations(euler_values, segments, cyl_rot):
    """
    Transport orientations according to the existing phase policy.
    approach_cylinder:
        Gradually apply the cylinder-map rotation.
    descend_to_grasp, close_gripper, lift, move_to_place,
    lower_to_place, release:
        Apply the cylinder-map rotation.
    retreat:
        Gradually restore the original orientation.
    Euler angles use XYZ order and radians.
    """
    euler_values = np.asarray(euler_values, dtype=float)
    if euler_values.ndim != 2 or euler_values.shape[1] != 3:
        raise ValueError("Expected an N x 3 array of Euler angles.")
    if not np.isfinite(euler_values).all():
        raise ValueError("Non-finite Euler angles found.")
    original = Rotation.from_euler("xyz", euler_values)
    count = len(euler_values)
    # Store individual rotations in a mutable Python list.
    result = [original[i] for i in range(count)]
    # --------------------------------------------------------
    # Phase 1: approach cylinder
    # --------------------------------------------------------
    idx = np.flatnonzero(segments == "approach_cylinder")
    alpha = np.linspace(0.0, 1.0, len(idx))
    for i, a in zip(idx, alpha):
        partial_rot = Rotation.from_rotvec(
            a * cyl_rot.as_rotvec()
        )
        result[i] = partial_rot * original[i]
    # --------------------------------------------------------
    # Phases 2-4: grasp, lift, transport, lower, release
    # --------------------------------------------------------
    full_rotation_phases = [
        "descend_to_grasp",
        "close_gripper",
        "lift",
        "move_to_place",
        "lower_to_place",
        "release",
    ]
    for phase in full_rotation_phases:
        idx = np.flatnonzero(segments == phase)
        for i in idx:
            result[i] = cyl_rot * original[i]
    # --------------------------------------------------------
    # Phase 5: retreat
    # --------------------------------------------------------
    idx = np.flatnonzero(segments == "retreat")
    alpha = np.linspace(0.0, 1.0, len(idx))
    for i, a in zip(idx, alpha):
        transported = cyl_rot * original[i]
        interpolator = Slerp(
            [0.0, 1.0],
            Rotation.concatenate([
                transported,
                original[i],
            ]),
        )
        result[i] = interpolator([a])[0]
    return Rotation.concatenate(result)
# ============================================================
# LOAD AND VALIDATE DEMONSTRATION
# ============================================================
if not DEMO_CSV.is_file():
    raise FileNotFoundError(f"Demonstration not found: {DEMO_CSV}")
if not MODEL_FILE.is_file():
    raise FileNotFoundError(f"Transport map file not found: {MODEL_FILE}")
df = pd.read_csv(DEMO_CSV).reset_index(drop=True)
required = [
    "segment",
    "sim_time",
    *EE_POS_COLS,
    *EE_EULER_COLS,
    "gripper_target",
]
if TRANSFORM_TARGET_COLUMNS:
    required += TARGET_POS_COLS + TARGET_EULER_COLS
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")
if df.empty:
    raise ValueError("The demonstration CSV is empty.")
segments = df["segment"].astype(str).to_numpy()
original_xyz = df[EE_POS_COLS].to_numpy(dtype=float)
original_euler = df[EE_EULER_COLS].to_numpy(dtype=float)
if not np.isfinite(original_xyz).all():
    raise ValueError("Non-finite end-effector positions found.")
if not np.isfinite(original_euler).all():
    raise ValueError("Non-finite end-effector orientations found.")
# Work on copies to preserve the original data.
generalized_xyz = original_xyz.copy()
# Rigid cylinder rotation used by the existing phase policy.
cyl_rot = map_rotation(cylinder_map)
# ============================================================
# PHASE 1: APPROACH CYLINDER
# ============================================================
idx = np.flatnonzero(segments == "approach_cylinder")
if len(idx) == 0:
    raise ValueError("No approach_cylinder phase found.")
alpha = np.linspace(0.0, 1.0, len(idx))
generalized_xyz[idx] = partial_transform(
    original_xyz[idx],
    cylinder_map,
    alpha,
)
# ============================================================
# PHASE 2: DESCEND, GRASP, LIFT
# ============================================================
cylinder_phases = [
    "descend_to_grasp",
    "close_gripper",
    "lift",
]
for phase in cylinder_phases:
    idx = np.flatnonzero(segments == phase)
    generalized_xyz[idx] = transform_points(
        original_xyz[idx],
        cylinder_map,
    )
# ============================================================
# PHASE 3: TRANSPORT
# ============================================================
idx = np.flatnonzero(segments == "move_to_place")
if len(idx) == 0:
    raise ValueError("No move_to_place phase found.")
cylinder_path = transform_points(
    original_xyz[idx],
    cylinder_map,
)
box_path = transform_points(
    original_xyz[idx],
    box_map,
)
alpha = np.linspace(0.0, 1.0, len(idx))
generalized_xyz[idx] = (
    (1.0 - alpha[:, None]) * cylinder_path
    + alpha[:, None] * box_path
)
# ============================================================
# PHASE 4: LOWER AND RELEASE
# ============================================================
for phase in ["lower_to_place", "release"]:
    idx = np.flatnonzero(segments == phase)
    generalized_xyz[idx] = transform_points(
        original_xyz[idx],
        box_map,
    )
# ============================================================
# PHASE 5: RETREAT
# ============================================================
idx = np.flatnonzero(segments == "retreat")
if len(idx) == 0:
    raise ValueError("No retreat phase found.")
generalized_xyz[idx] = transform_points(
    original_xyz[idx],
    box_map,
)
# ============================================================
# TRANSPORT END-EFFECTOR ORIENTATIONS
# ============================================================
generalized_rot = transport_orientations(
    original_euler,
    segments,
    cyl_rot,
)
# Save the generalized end-effector pose.
df[EE_POS_COLS] = generalized_xyz
df[EE_EULER_COLS] = generalized_rot.as_euler("xyz")
# ============================================================
# TRANSPORT COMMANDED TARGET POSES
# ============================================================
if TRANSFORM_TARGET_COLUMNS:
    original_target_xyz = df[TARGET_POS_COLS].to_numpy(
        dtype=float
    )
    original_target_euler = df[TARGET_EULER_COLS].to_numpy(
        dtype=float
    )
    if not np.isfinite(original_target_xyz).all():
        raise ValueError("Non-finite commanded target positions found.")
    if not np.isfinite(original_target_euler).all():
        raise ValueError("Non-finite commanded target orientations found.")
    generalized_target_xyz = original_target_xyz.copy()
    # Approach.
    idx = np.flatnonzero(segments == "approach_cylinder")
    alpha = np.linspace(0.0, 1.0, len(idx))
    generalized_target_xyz[idx] = partial_transform(
        original_target_xyz[idx],
        cylinder_map,
        alpha,
    )
    # Grasp and lift.
    for phase in cylinder_phases:
        idx = np.flatnonzero(segments == phase)
        generalized_target_xyz[idx] = transform_points(
            original_target_xyz[idx],
            cylinder_map,
        )
    # Transport: retain the existing position blending policy.
    idx = np.flatnonzero(segments == "move_to_place")
    cylinder_targets = transform_points(
        original_target_xyz[idx],
        cylinder_map,
    )
    box_targets = transform_points(
        original_target_xyz[idx],
        box_map,
    )
    alpha = np.linspace(0.0, 1.0, len(idx))
    generalized_target_xyz[idx] = (
        (1.0 - alpha[:, None]) * cylinder_targets
        + alpha[:, None] * box_targets
    )
    # Lower, release and retreat.
    for phase in ["lower_to_place", "release", "retreat"]:
        idx = np.flatnonzero(segments == phase)
        generalized_target_xyz[idx] = transform_points(
            original_target_xyz[idx],
            box_map,
        )
    # Apply the same phase-specific orientation policy.
    generalized_target_rot = transport_orientations(
        original_target_euler,
        segments,
        cyl_rot,
    )
    df[TARGET_POS_COLS] = generalized_target_xyz
    df[TARGET_EULER_COLS] = (
        generalized_target_rot.as_euler("xyz")
    )
# ============================================================
# SAVE GENERALIZED TRAJECTORY
# ============================================================
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print("=" * 65)
print("GENERALIZED TRAJECTORY GENERATED")
print("=" * 65)
print(f"Output: {OUTPUT_CSV.resolve()}")
print(f"Samples: {len(df)}")
print(f"Cylinder-map rotation (XYZ Euler, rad): "
      f"{cyl_rot.as_euler('xyz').round(6)}")
print("\nOutput phase counts:")
print(df["segment"].value_counts().to_string())
print("\nEnd-effector positions by phase:")
for phase, group in df.groupby("segment", sort=False):
    start = group[EE_POS_COLS].iloc[0].to_numpy()
    end = group[EE_POS_COLS].iloc[-1].to_numpy()
    print(
        f"{phase:20s} "
        f"start={start.round(4)} "
        f"end={end.round(4)}"
    )
if TRANSFORM_TARGET_COLUMNS:
    print("\nCommanded target orientations were also transformed.")
else:
    print("\nCommanded target transformation was disabled.")
print("\nNext: inspect the trajectory before replaying it.")
