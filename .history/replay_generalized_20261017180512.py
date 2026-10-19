import csv
import os
import time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
# =========================================================
# CONFIGURATION
# =========================================================
DT = 0.001
BASE_DIR = Path(__file__).resolve().parent
XML_PATH = BASE_DIR / "assets" / "fr3_on_table.xml"
TRAJECTORY_PATH = (
    BASE_DIR / "output" / "Generalized_trajectory_avoiding_obstacle.csv"
)
OUTPUT_DIR = BASE_DIR / "demonstrations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_DIR / (
    f"generalized_replay_{time.strftime('%Y%m%d_%H%M%S')}.csv"
)
EE_SITE = "center"
# Match the original demonstration controller.
KP_POS = 600.0
KD_POS = 80.0
KP_ROT = 40.0
KD_ROT = 5.0
MAX_TORQUE = 35.0
# Match the original demonstration's physics configuration.
ZERO_GRAVITY = True
# Slow down playback to approximately simulation time.
REALTIME = True
# Render updates are synchronized periodically.
RENDER_EVERY_STEPS = max(1, round(1.0 / (60.0 * DT)))
# =========================================================
# MODEL HELPERS
# =========================================================
def find_joint_ids(model):
    """Return FR3 arm joint IDs and their velocity DOF indices."""
    joint_ids = []
    for i in range(1, 8):
        name = f"joint{i}"
        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if jid < 0:
            raise ValueError(
                f"Joint '{name}' was not found in the XML."
            )
        joint_ids.append(jid)
    qpos_ids = np.array(
        [model.jnt_qposadr[jid] for jid in joint_ids],
        dtype=int,
    )
    dof_ids = np.array(
        [model.jnt_dofadr[jid] for jid in joint_ids],
        dtype=int,
    )
    return qpos_ids, dof_ids
def require_site(model, site_name):
    sid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, site_name
    )
    if sid < 0:
        available = [
            model.site(i).name for i in range(model.nsite)
        ]
        raise ValueError(
            f"Site '{site_name}' was not found.\n"
            f"Available sites: {available}"
        )
    return sid
def get_cartesian_state(model, data, ee_site_id, dof_ids):
    """Return EE pose, arm Jacobian, and EE spatial velocity."""
    position = data.site_xpos[ee_site_id].copy()
    rotation = data.site_xmat[ee_site_id].reshape(3, 3).copy()
    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))
    mujoco.mj_jacSite(
        model, data, jac_pos, jac_rot, ee_site_id
    )
    J = np.vstack([
        jac_pos[:, dof_ids],
        jac_rot[:, dof_ids],
    ])
    dq = data.qvel[dof_ids].copy()
    velocity = J @ dq
    return position, rotation, J, velocity
# =========================================================
# CARTESIAN IMPEDANCE CONTROLLER
# =========================================================
def compute_torque(
    model, data, ee_site_id, dof_ids,
    target_pos, target_rot,
):
    position, rotation, J, velocity = get_cartesian_state(
        model, data, ee_site_id, dof_ids
    )
    position_error = target_pos - position
    rotation_error = Rotation.from_matrix(
        target_rot @ rotation.T
    ).as_rotvec()
    force = (
        KP_POS * position_error
        - KD_POS * velocity[:3]
    )
    moment = (
        KP_ROT * rotation_error
        - KD_ROT * velocity[3:]
    )
    wrench = np.concatenate([force, moment])
    # Same bias/gravity compensation structure as your demo.
    bias = data.qfrc_bias[dof_ids].copy()
    tau = J.T @ wrench + bias
    return np.clip(tau, -MAX_TORQUE, MAX_TORQUE)
# =========================================================
# LOAD AND VALIDATE TRAJECTORY
# =========================================================
def load_trajectory():
    if not XML_PATH.is_file():
        raise FileNotFoundError(
            f"MuJoCo XML not found: {XML_PATH}"
        )
    if not TRAJECTORY_PATH.is_file():
        raise FileNotFoundError(
            f"Generalized trajectory not found: "
            f"{TRAJECTORY_PATH}"
        )
    df = pd.read_csv(TRAJECTORY_PATH)
    required = [
        "segment",
        "sim_time",
        "target_x", "target_y", "target_z",
        "target_roll", "target_pitch", "target_yaw",
        "gripper_target",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Trajectory is missing columns: {missing}"
        )
    if df.empty:
        raise ValueError("The trajectory CSV is empty.")
    if df[required].isna().any().any():
        raise ValueError(
            "Trajectory contains missing target values."
        )
    times = df["sim_time"].to_numpy(dtype=float)
    if not np.all(np.isfinite(times)):
        raise ValueError("Trajectory timestamps are not finite.")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError(
            "sim_time must be strictly increasing."
        )
    targets = df[
        ["target_x", "target_y", "target_z"]
    ].to_numpy(dtype=float)
    euler = df[
        ["target_roll", "target_pitch", "target_yaw"]
    ].to_numpy(dtype=float)
    if not np.all(np.isfinite(targets)):
        raise ValueError("Target positions contain invalid values.")
    if not np.all(np.isfinite(euler)):
        raise ValueError("Target orientations contain invalid values.")
    rotations = Rotation.from_euler(
        "xyz", euler
    ).as_matrix()
    gripper = df["gripper_target"].to_numpy(dtype=float)
    return df, times, targets, rotations, gripper
# =========================================================
# REPLAY
# =========================================================
def main():
    df, times, targets, rotations, gripper = load_trajectory()
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    model.opt.timestep = DT
    if ZERO_GRAVITY:
        model.opt.gravity[:] = [0.0, 0.0, 0.0]
    data = mujoco.MjData(model)
    qpos_ids, dof_ids = find_joint_ids(model)
    ee_site_id = require_site(model, EE_SITE)
    if model.nu < 7:
        raise ValueError(
            f"Expected at least 7 actuators, found {model.nu}."
        )
    if model.nu < 8:
        print(
            "WARNING: fewer than 8 actuators found; "
            "the script cannot command the gripper actuator."
        )
    mujoco.mj_forward(model, data)
    # Check the initial pose against the first requested target.
    initial_pos, _, _, _ = get_cartesian_state(
        model, data, ee_site_id, dof_ids
    )
    initial_gap = np.linalg.norm(targets[0] - initial_pos)
    print("=" * 60)
    print("GENERALIZED TRAJECTORY REPLAY")
    print("=" * 60)
    print("Scene:", XML_PATH)
    print("Trajectory:", TRAJECTORY_PATH)
    print("Samples:", len(df))
    print("Phases:", list(df["segment"].drop_duplicates()))
    print(f"Simulation timestep: {DT:.4f} s")
    print(f"Initial target distance: {initial_gap:.4f} m")
    print("Gravity:", "zero" if ZERO_GRAVITY else "enabled")
    print("Execution log:", LOG_PATH)
    if initial_gap > 0.05:
        print(
            "\nWARNING: The initial end-effector position is more "
            "than 5 cm from the first target. Check the start pose "
            "before proceeding."
        )
    print(
        "\nThe robot will begin replay when the viewer opens. "
        "Watch the motion and stop if it behaves unexpectedly."
    )
    log_fields = [
        "sample", "source_sim_time", "replay_sim_time", "segment",
        "target_x", "target_y", "target_z",
        "ee_x", "ee_y", "ee_z", "position_error_m",
        "target_roll", "target_pitch", "target_yaw",
        "ee_roll", "ee_pitch", "ee_yaw",
        "gripper_target",
        *[f"q{i}" for i in range(1, 8)],
        *[f"dq{i}" for i in range(1, 8)],
        *[f"tau{i}" for i in range(1, 8)],
    ]
    wall_start = time.monotonic()
    replay_start_time = float(times[0])
    previous_phase = None
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.sync()
            with open(LOG_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=log_fields)
                writer.writeheader()
                for i in range(len(df)):
                    if not viewer.is_running():
                        print("Viewer closed; stopping replay.")
                        break
                    phase = str(df["segment"].iloc[i])
                    if phase != previous_phase:
                        print(f"\nPhase: {phase}", flush=True)
                        previous_phase = phase
                    target_pos = targets[i]
                    target_rot = rotations[i]
                    finger_target = float(gripper[i])
                    # Current state, before the next simulation step.
                    mujoco.mj_forward(model, data)
                    tau = compute_torque(
                        model, data, ee_site_id, dof_ids,
                        target_pos, target_rot,
                    )
                    data.ctrl[:7] = tau
                    if model.nu >= 8:
                        data.ctrl[7] = finger_target
                    # Advance one physics step.
                    mujoco.mj_step(model, data)
                    # Read the resulting state.
                    mujoco.mj_forward(model, data)
                    ee_pos, ee_rot, _, _ = get_cartesian_state(
                        model, data, ee_site_id, dof_ids
                    )
                    q = data.qpos[qpos_ids].copy()
                    dq = data.qvel[dof_ids].copy()
                    error = float(
                        np.linalg.norm(target_pos - ee_pos)
                    )
                    ee_euler = Rotation.from_matrix(
                        ee_rot
                    ).as_euler("xyz")
                    target_euler = Rotation.from_matrix(
                        target_rot
                    ).as_euler("xyz")
                    row = {
                        "sample": i,
                        "source_sim_time": times[i],
                        "replay_sim_time": data.time,
                        "segment": phase,
                        "target_x": target_pos[0],
                        "target_y": target_pos[1],
                        "target_z": target_pos[2],
                        "ee_x": ee_pos[0],
                        "ee_y": ee_pos[1],
                        "ee_z": ee_pos[2],
                        "position_error_m": error,
                        "target_roll": target_euler[0],
                        "target_pitch": target_euler[1],
                        "target_yaw": target_euler[2],
                        "ee_roll": ee_euler[0],
                        "ee_pitch": ee_euler[1],
                        "ee_yaw": ee_euler[2],
                        "gripper_target": finger_target,
                    }
                    row.update({
                        f"q{j + 1}": q[j] for j in range(7)
                    })
                    row.update({
                        f"dq{j + 1}": dq[j] for j in range(7)
                    })
                    row.update({
                        f"tau{j + 1}": tau[j] for j in range(7)
                    })
                    writer.writerow(row)
                    if i % RENDER_EVERY_STEPS == 0:
                        viewer.sync()
                        f.flush()
                        print(
                            f"sample {i + 1}/{len(df)} | "
                            f"t={data.time:.2f}s | "
                            f"phase={phase} | "
                            f"position error={error:.4f} m",
                            flush=True,
                        )
                    if REALTIME:
                        # Pace by source timestamps. The simulation
                        # itself advances exactly one DT per sample.
                        elapsed = times[i] - replay_start_time
                        remaining = (
                            wall_start + elapsed - time.monotonic()
                        )
                        if remaining > 0:
                            time.sleep(remaining)
            print("\nReplay loop finished.")
            print("Execution log saved to:", LOG_PATH)
            print("Close the viewer when you are finished.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")
if __name__ == "__main__":
    main()
