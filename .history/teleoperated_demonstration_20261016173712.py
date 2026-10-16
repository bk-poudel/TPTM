import csv
import os
import time
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
# =========================================================
# CONFIGURATION
# =========================================================
DT = 0.001
# XML file path
XML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "fr3_on_table_original_demonstration.xml",
)
# Multi-keypoint site definitions (5 keypoints per object for 3D pose mapping)
# Multi-keypoint site definitions (5 keypoints per object for 3D pose mapping)
CYLINDER_SITES = [
    "cylinder_center",
    "cylinder_top",
    "cylinder_bottom",
    "cylinder_side_x",
    "cylinder_side_y",
]
PLACE_SITES = [
    "box_center",
    "box_corner_front_left",
    "box_corner_front_right",
    "box_corner_back_left",
    "box_corner_back_right",
]
# End-effector site name
EE_SITE = "center"
# Pick-and-place geometry
APPROACH_HEIGHT = 0.12
LIFT_HEIGHT = 0.17
GRASP_OFFSET = np.array([0.0, 0.0, -0.025])
PLACE_OFFSET = np.array([0.0, 0.0, 0.06])
# Gripper configuration
OPEN_GRIPPER = 0.035
CLOSED_GRIPPER = 0.0
# Cartesian impedance gains
KP_POS = 600.0
KD_POS = 80.0
KP_ROT = 40.0
KD_ROT = 5.0
MAX_TORQUE = 35.0
REALTIME = True
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "demonstrations",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(
    OUTPUT_DIR,
    f"pick_place_{time.strftime('%Y%m%d_%H%M%S')}.csv",
)
# =========================================================
# MODEL INITIALIZATION & KEYPOINT HELPERS
# =========================================================
def find_joint_ids(model):
    """Find the seven FR3 arm joints by their standard names."""
    joint_ids = []
    for i in range(1, 8):
        name = f"joint{i}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(
                f"Could not find joint '{name}' in the XML. "
                "Check your model's arm joint names."
            )
        joint_ids.append(jid)
    qpos_ids = np.array([model.jnt_qposadr[jid] for jid in joint_ids])
    dof_ids = np.array([model.jnt_dofadr[jid] for jid in joint_ids])
    return joint_ids, qpos_ids, dof_ids
def require_site(model, site_name):
    """Ensure a site exists and return its ID."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        available = [model.site(i).name for i in range(model.nsite)]
        raise ValueError(
            f"Site '{site_name}' was not found.\n"
            f"Available sites: {available}\n"
            f"Update the corresponding site name in the script."
        )
    return sid
def require_sites(model, site_names):
    """Retrieve and validate site IDs for a list of site names."""
    return [require_site(model, name) for name in site_names]
def get_site_pose(model, data, site_id):
    """Return a copy of the site's world position and rotation."""
    position = data.site_xpos[site_id].copy()
    rotation = data.site_xmat[site_id].reshape(3, 3).copy()
    return position, rotation
def get_keypoints_array(model, data, site_ids):
    """Extract flattened 3D coordinates (x, y, z) for all keypoints."""
    coords = []
    for sid in site_ids:
        pos, _ = get_site_pose(model, data, sid)
        coords.extend(pos)
    return coords
# =========================================================
# MUJOCO CARTESIAN JACOBIAN & STATE
# =========================================================
def get_cartesian_state(model, data, ee_site_id, dof_ids):
    """Return end-effector position, orientation, Jacobian, and spatial velocity."""
    position, rotation = get_site_pose(model, data, ee_site_id)
    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac_pos, jac_rot, ee_site_id)
    J_pos = jac_pos[:, dof_ids]
    J_rot = jac_rot[:, dof_ids]
    J = np.vstack([J_pos, J_rot])
    dq = data.qvel[dof_ids].copy()
    velocity = J @ dq
    return position, rotation, J, velocity
# =========================================================
# TRAJECTORY & CONTROLLER
# =========================================================
def interpolate_pose(start_pos, end_pos, start_rot, end_rot, steps):
    """Quintic position interpolation and SLERP orientation."""
    fractions = np.linspace(0.0, 1.0, steps + 1)
    s = 10.0 * fractions**3 - 15.0 * fractions**4 + 6.0 * fractions**5
    positions = start_pos[None, :] + s[:, None] * (end_pos - start_pos)[None, :]
    rotations = Slerp(
        [0.0, 1.0],
        Rotation.from_matrix(np.stack([start_rot, end_rot])),
    )(fractions).as_matrix()
    return positions, rotations
def compute_torque(model, data, ee_site_id, dof_ids, target_pos, target_rot):
    """Standalone Cartesian impedance controller."""
    position, rotation, J, velocity = get_cartesian_state(
        model, data, ee_site_id, dof_ids
    )
    position_error = target_pos - position
    rotation_error = Rotation.from_matrix(target_rot @ rotation.T).as_rotvec()
    linear_velocity = velocity[:3]
    angular_velocity = velocity[3:]
    force = KP_POS * position_error - KD_POS * linear_velocity
    moment = KP_ROT * rotation_error - KD_ROT * angular_velocity
    wrench = np.concatenate([force, moment])
    bias = data.qfrc_bias[dof_ids].copy()
    tau = J.T @ wrench + bias
    return np.clip(tau, -MAX_TORQUE, MAX_TORQUE)
# =========================================================
# EXECUTE AND RECORD TRAJECTORY SEGMENT
# =========================================================
def execute_segment(
    model,
    data,
    viewer,
    ee_site_id,
    cylinder_site_ids,
    place_site_ids,
    dof_ids,
    writer,
    segment_name,
    target_pos,
    target_rot,
    duration,
    gripper_target,
):
    mujoco.mj_forward(model, data)
    start_pos, start_rot, _, _ = get_cartesian_state(
        model, data, ee_site_id, dof_ids
    )
    steps = max(1, int(round(duration / DT)))
    positions, rotations = interpolate_pose(
        start_pos, target_pos, start_rot, target_rot, steps
    )
    print(
        f"{segment_name}: {np.round(start_pos, 3)} -> {np.round(target_pos, 3)}",
        flush=True,
    )
    for i in range(1, steps + 1):
        if viewer is not None and not viewer.is_running():
            raise RuntimeError("MuJoCo viewer was closed.")
        mujoco.mj_forward(model, data)
        desired_pos = positions[i]
        desired_rot = rotations[i]
        tau = compute_torque(
            model, data, ee_site_id, dof_ids, desired_pos, desired_rot
        )
        data.ctrl[:7] = tau
        if model.nu >= 8:
            data.ctrl[7] = gripper_target
        mujoco.mj_step(model, data)
        if viewer is not None:
            if i % max(1, round(1.0 / (60.0 * DT))) == 0:
                viewer.sync()
        # Record states and multi-keypoints
        q = data.qpos[[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j}")] for j in range(1, 8)]].copy()
        dq = data.qvel[dof_ids].copy()
        mujoco.mj_forward(model, data)
        ee_pos, ee_rot, _, _ = get_cartesian_state(model, data, ee_site_id, dof_ids)
        src_keypoints = get_keypoints_array(model, data, cylinder_site_ids)
        tgt_keypoints = get_keypoints_array(model, data, place_site_ids)
        writer.writerow([
            segment_name,
            data.time,
            *ee_pos,
            *Rotation.from_matrix(ee_rot).as_euler("xyz"),
            *desired_pos,
            *Rotation.from_matrix(desired_rot).as_euler("xyz"),
            *src_keypoints,
            *tgt_keypoints,
            *q,
            *dq,
            *tau,
            gripper_target,
        ])
        if REALTIME:
            time.sleep(DT)
# =========================================================
# MAIN ROUTINE
# =========================================================
def main():
    if not os.path.isfile(XML_PATH):
        raise FileNotFoundError(
            f"MuJoCo XML file not found:\n{XML_PATH}\n"
            "Update XML_PATH to point to your FR3 scene XML."
        )
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    model.opt.gravity[:] = [0.0, 0.0, 0.0]
    model.opt.timestep = DT
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print("Loaded XML:", XML_PATH)
    _, _, dof_ids = find_joint_ids(model)
    # Confirm keypoint and end-effector sites
    ee_site_id = require_site(model, EE_SITE)
    cylinder_site_ids = require_sites(model, CYLINDER_SITES)
    place_site_ids = require_sites(model, PLACE_SITES)
    if model.nu < 7:
        raise ValueError(f"Expected at least 7 arm actuators, found {model.nu}.")
    mujoco.mj_forward(model, data)
    # Target positions based on primary center sites (index 0)
    cylinder_center_pos, _ = get_site_pose(model, data, cylinder_site_ids[0])
    place_center_pos, _ = get_site_pose(model, data, place_site_ids[0])
    grasp_pos = cylinder_center_pos + GRASP_OFFSET
    approach_pos = grasp_pos + np.array([0.0, 0.0, APPROACH_HEIGHT])
    lift_pos = grasp_pos + np.array([0.0, 0.0, LIFT_HEIGHT])
    place_pos = place_center_pos + PLACE_OFFSET
    place_approach_pos = place_pos + np.array([0.0, 0.0, APPROACH_HEIGHT])
    ee_pos, fixed_rot, _, _ = get_cartesian_state(model, data, ee_site_id, dof_ids)
    viewer = None
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.sync()
            with open(CSV_PATH, "w", newline="") as f:
                writer = csv.writer(f)
                # Build CSV headers dynamically for keypoints
                source_kp_headers = [f"src_kp{i}_{axis}" for i in range(len(CYLINDER_SITES)) for axis in ("x", "y", "z")]
                target_kp_headers = [f"tgt_kp{i}_{axis}" for i in range(len(PLACE_SITES)) for axis in ("x", "y", "z")]
                writer.writerow([
                    "segment", "sim_time",
                    "ee_x", "ee_y", "ee_z",
                    "ee_roll", "ee_pitch", "ee_yaw",
                    "target_x", "target_y", "target_z",
                    "target_roll", "target_pitch", "target_yaw",
                    *source_kp_headers,
                    *target_kp_headers,
                    "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                    "dq1", "dq2", "dq3", "dq4", "dq5", "dq6", "dq7",
                    "tau1", "tau2", "tau3", "tau4", "tau5", "tau6", "tau7",
                    "gripper_target",
                ])
                sequence = [
                    ("approach_cylinder", approach_pos, 5.0, OPEN_GRIPPER),
                    ("descend_to_grasp", grasp_pos, 3.0, OPEN_GRIPPER),
                    ("close_gripper", grasp_pos, 1.0, CLOSED_GRIPPER),
                    ("lift", lift_pos, 2.0, CLOSED_GRIPPER),
                    ("move_to_place", place_approach_pos, 5.0, CLOSED_GRIPPER),
                    ("lower_to_place", place_pos, 2.0, CLOSED_GRIPPER),
                    ("release", place_pos, 1.0, OPEN_GRIPPER),
                    ("retreat", place_approach_pos, 2.0, OPEN_GRIPPER),
                ]
                for name, target, duration, grip in sequence:
                    execute_segment(
                        model=model,
                        data=data,
                        viewer=viewer,
                        ee_site_id=ee_site_id,
                        cylinder_site_ids=cylinder_site_ids,
                        place_site_ids=place_site_ids,
                        dof_ids=dof_ids,
                        writer=writer,
                        segment_name=name,
                        target_pos=target,
                        target_rot=fixed_rot,
                        duration=duration,
                        gripper_target=grip,
                    )
            print("\nPick-and-place demonstration finished successfully.")
            print("CSV saved to:", CSV_PATH)
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("Stopped by user.")
if __name__ == "__main__":
    main()