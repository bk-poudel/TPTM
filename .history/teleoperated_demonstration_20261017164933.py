#!/usr/bin/env python3
import argparse
import csv
import time
from pathlib import Path
import glfw
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
# =========================================================
# ARGUMENTS
# =========================================================
parser = argparse.ArgumentParser(
    description="Teleoperate the FR3 around an obstacle and record demonstrations."
)
parser.add_argument(
    "--scene_file",
    type=str,
    required=True,
    help="XML path or XML filename inside assets/",
)
parser.add_argument(
    "--output_file",
    type=Path,
    default=Path("demonstrations/obstacle_avoidance_demo.csv"),
    help="Output CSV path, relative to the current directory or absolute.",
)
parser.add_argument(
    "--record_every",
    type=int,
    default=10,
    help="Record one sample every N simulation steps.",
)
parser.add_argument(
    "--speed",
    type=float,
    default=0.10,
    help="Cartesian movement speed in metres/second.",
)
parser.add_argument(
    "--rotation_speed",
    type=float,
    default=0.6,
    help="Rotation speed in radians/second.",
)
args = parser.parse_args()
if args.record_every < 1:
    parser.error("--record_every must be >= 1")
PROJECT_DIR = Path(__file__).resolve().parent
scene_arg = Path(args.scene_file)
if scene_arg.is_absolute():
    XML_PATH = scene_arg
elif scene_arg.is_file():
    XML_PATH = scene_arg.resolve()
else:
    XML_PATH = PROJECT_DIR / "assets" / scene_arg
OUTPUT_FILE = args.output_file
if not OUTPUT_FILE.is_absolute():
    OUTPUT_FILE = Path.cwd() / OUTPUT_FILE
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
DT = 0.002
MOVE_SPEED = args.speed
ROT_SPEED = args.rotation_speed
# Cartesian impedance controller
KP_POS = 600.0
KD_POS = 80.0
KP_ROT = 40.0
KD_ROT = 5.0
MAX_TORQUE = 35.0
OPEN_GRIPPER = 0.035
CLOSED_GRIPPER = 0.0
EE_SITE = "center"
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
# =========================================================
# MUJOCO HELPERS
# =========================================================
def require_site(model, name):
    sid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, name
    )
    if sid < 0:
        raise ValueError(f"Required site not found: {name}")
    return sid
def require_sites(model, names):
    return [require_site(model, name) for name in names]
def find_joint_ids(model):
    joint_ids = []
    for i in range(1, 8):
        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}"
        )
        if jid < 0:
            raise ValueError(f"Joint joint{i} not found")
        joint_ids.append(jid)
    qpos_ids = np.array(
        [model.jnt_qposadr[jid] for jid in joint_ids]
    )
    dof_ids = np.array(
        [model.jnt_dofadr[jid] for jid in joint_ids]
    )
    return qpos_ids, dof_ids
def get_site_pose(data, site_id):
    pos = data.site_xpos[site_id].copy()
    rot = data.site_xmat[site_id].reshape(3, 3).copy()
    return pos, rot
def get_cartesian_state(model, data, site_id, dof_ids):
    pos, rot = get_site_pose(data, site_id)
    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))
    mujoco.mj_jacSite(
        model, data, jac_pos, jac_rot, site_id
    )
    J = np.vstack([
        jac_pos[:, dof_ids],
        jac_rot[:, dof_ids],
    ])
    velocity = J @ data.qvel[dof_ids]
    return pos, rot, J, velocity
def compute_torque(
    model, data, site_id, dof_ids, target_pos, target_rot
):
    pos, rot, J, velocity = get_cartesian_state(
        model, data, site_id, dof_ids
    )
    position_error = target_pos - pos
    rotation_error = Rotation.from_matrix(
        target_rot @ rot.T
    ).as_rotvec()
    wrench = np.concatenate([
        KP_POS * position_error - KD_POS * velocity[:3],
        KP_ROT * rotation_error - KD_ROT * velocity[3:],
    ])
    bias = data.qfrc_bias[dof_ids].copy()
    torque = J.T @ wrench + bias
    return np.clip(torque, -MAX_TORQUE, MAX_TORQUE)
# =========================================================
# TELEOPERATION AND RECORDING
# =========================================================
def main():
    if not XML_PATH.is_file():
        raise FileNotFoundError(
            f"XML not found: {XML_PATH}\n"
            "Pass the scene filename inside assets/ or a valid XML path."
        )
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    model.opt.timestep = DT
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if model.nu < 7:
        raise ValueError(
            f"Expected at least 7 arm actuators; found {model.nu}"
        )
    qpos_ids, dof_ids = find_joint_ids(model)
    ee_site_id = require_site(model, EE_SITE)
    cylinder_ids = require_sites(model, CYLINDER_SITES)
    place_ids = require_sites(model, PLACE_SITES)
    # Collect every site belonging to the cuboid.
    obstacle_names = sorted(
        model.site(i).name
        for i in range(model.nsite)
        if (model.site(i).name or "").startswith("obstacle_")
    )
    if not obstacle_names:
        raise ValueError(
            "No obstacle_* sites found. Check your cuboid XML."
        )
    obstacle_ids = [
        require_site(model, name) for name in obstacle_names
    ]
    ee_pos, target_rot, _, _ = get_cartesian_state(
        model, data, ee_site_id, dof_ids
    )
    target_pos = ee_pos.copy()
    keys_down = set()
    gripper_target = OPEN_GRIPPER
    segment_name = "free_teleoperation"
    paused = False
    quit_requested = False
    segment_keys = {
        glfw.KEY_1: "approach_obstacle",
        glfw.KEY_2: "side_approach_cylinder",
        glfw.KEY_3: "side_grasp",
        glfw.KEY_4: "lift",
        glfw.KEY_5: "transfer_to_target",
        glfw.KEY_6: "place",
        glfw.KEY_7: "retreat",
    }
    def key_callback(key, scancode, action, mods):
        nonlocal gripper_target, segment_name
        nonlocal paused, quit_requested
        if action == glfw.RELEASE:
            keys_down.discard(key)
            return
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            quit_requested = True
            return
        if key == glfw.KEY_SPACE and action == glfw.PRESS:
            paused = not paused
            return
        if key == glfw.KEY_O and action == glfw.PRESS:
            gripper_target = OPEN_GRIPPER
            print("Gripper: OPEN")
            return
        if key == glfw.KEY_C and action == glfw.PRESS:
            gripper_target = CLOSED_GRIPPER
            print("Gripper: CLOSE")
            return
        if key in segment_keys and action == glfw.PRESS:
            segment_name = segment_keys[key]
            print(f"Recording phase: {segment_name}")
            return
        keys_down.add(key)
    # CSV records are captured after physics steps.
    sample_count = 0
    step_count = 0
    last_print_time = time.monotonic()
    print("\n========== FR3 OBSTACLE TELEOP ==========")
    print("Cartesian movement:")
    print("  W / S : +X / -X")
    print("  A / D : +Y / -Y")
    print("  R / F : +Z / -Z")
    print("  Q / E : rotate around world Z")
    print("  O     : open gripper")
    print("  C     : close gripper")
    print("  1-7   : label demonstration phase")
    print("  SPACE : pause/resume motion")
    print("  ESC   : stop and save")
    print("\nMove slowly around the cuboid, approach the")
    print("cylinder from the side, close the gripper,")
    print("and lift only after confirming the grasp.")
    print("=========================================\n")
    with open(OUTPUT_FILE, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        headers = [
            "segment",
            "sim_time",
            "ee_x", "ee_y", "ee_z",
            "ee_roll", "ee_pitch", "ee_yaw",
            "target_x", "target_y", "target_z",
            "target_roll", "target_pitch", "target_yaw",
        ]
        for name in CYLINDER_SITES:
            headers.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
        for name in PLACE_SITES:
            headers.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
        for name in obstacle_names:
            headers.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
        headers.extend([f"q{i}" for i in range(1, 8)])
        headers.extend([f"dq{i}" for i in range(1, 8)])
        headers.extend([f"tau{i}" for i in range(1, 8)])
        headers.append("gripper_target")
        writer.writerow(headers)
        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback
        ) as viewer:
            viewer.sync()
            previous_wall_time = time.monotonic()
            while viewer.is_running() and not quit_requested:
                loop_start = time.monotonic()
                if not paused:
                    # Translation is in WORLD coordinates.
                    direction = np.zeros(3)
                    if glfw.KEY_W in keys_down:
                        direction[0] += 1
                    if glfw.KEY_S in keys_down:
                        direction[0] -= 1
                    if glfw.KEY_A in keys_down:
                        direction[1] += 1
                    if glfw.KEY_D in keys_down:
                        direction[1] -= 1
                    if glfw.KEY_R in keys_down:
                        direction[2] += 1
                    if glfw.KEY_F in keys_down:
                        direction[2] -= 1
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction /= norm
                        target_pos += direction * MOVE_SPEED * DT
                    # Rotate desired EE orientation around world Z.
                    rotation_sign = 0
                    if glfw.KEY_Q in keys_down:
                        rotation_sign += 1
                    if glfw.KEY_E in keys_down:
                        rotation_sign -= 1
                    if rotation_sign:
                        delta_rot = Rotation.from_rotvec(
                            [0, 0, rotation_sign * ROT_SPEED * DT]
                        )
                        target_rot = (
                            delta_rot
                            * Rotation.from_matrix(target_rot)
                        ).as_matrix()
                    torque = compute_torque(
                        model,
                        data,
                        ee_site_id,
                        dof_ids,
                        target_pos,
                        target_rot,
                    )
                    data.ctrl[:7] = torque
                    if model.nu >= 8:
                        data.ctrl[7] = gripper_target
                    mujoco.mj_step(model, data)
                    step_count += 1
                    if step_count % args.record_every == 0:
                        mujoco.mj_forward(model, data)
                        ee_pos, ee_rot, _, _ = get_cartesian_state(
                            model, data, ee_site_id, dof_ids
                        )
                        ee_euler = Rotation.from_matrix(
                            ee_rot
                        ).as_euler("xyz")
                        target_euler = Rotation.from_matrix(
                            target_rot
                        ).as_euler("xyz")
                        row = [
                            segment_name,
                            float(data.time),
                            *ee_pos,
                            *ee_euler,
                            *target_pos,
                            *target_euler,
                        ]
                        for sid in cylinder_ids:
                            row.extend(data.site_xpos[sid].copy())
                        for sid in place_ids:
                            row.extend(data.site_xpos[sid].copy())
                        for sid in obstacle_ids:
                            row.extend(data.site_xpos[sid].copy())
                        row.extend(data.qpos[qpos_ids].copy())
                        row.extend(data.qvel[dof_ids].copy())
                        row.extend(torque.copy())
                        row.append(gripper_target)
                        writer.writerow(row)
                        sample_count += 1
                        # Flush periodically so data survives interruptions.
                        if sample_count % 100 == 0:
                            csv_file.flush()
                else:
                    # Hold the robot in its current desired pose while paused.
                    torque = compute_torque(
                        model,
                        data,
                        ee_site_id,
                        dof_ids,
                        target_pos,
                        target_rot,
                    )
                    data.ctrl[:7] = torque
                    if model.nu >= 8:
                        data.ctrl[7] = gripper_target
                    mujoco.mj_step(model, data)
                now = time.monotonic()
                if now - last_print_time >= 2.0:
                    print(
                        f"\rPhase: {segment_name:24s} "
                        f"EE: {np.round(data.site_xpos[ee_site_id], 3)} "
                        f"Gripper: {gripper_target:.3f} "
                        f"Samples: {sample_count}",
                        end="",
                        flush=True,
                    )
                    last_print_time = now
                viewer.sync()
                # Maintain approximate real-time execution.
                elapsed = time.monotonic() - loop_start
                if elapsed < DT:
                    time.sleep(DT - elapsed)
                previous_wall_time = now
        csv_file.flush()
    print("\n\nTeleoperation stopped.")
    print(f"Recorded samples: {sample_count}")
    print(f"Saved CSV: {OUTPUT_FILE.resolve()}")
if __name__ == "__main__":
    main()
