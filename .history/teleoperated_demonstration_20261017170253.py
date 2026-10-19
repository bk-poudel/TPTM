#!/usr/bin/env python3
"""
FR3 Cartesian keyboard teleoperation for MuJoCo.
Controls
--------
W / S : move EE in +X / -X
A / D : move EE in +Y / -Y
R / F : move EE in +Z / -Z
Q / E : rotate EE around world Z
O     : open gripper
C     : close gripper
1-7   : label demonstration phase
SPACE : pause/resume Cartesian target movement
ESC   : stop and save demonstration
Assumptions
-----------
- First 7 actuators are torque-controlled arm joints.
- Actuator 8 is a position-controlled gripper.
- The EE site is named "center".
- The gripper opens at 0.035 and closes at 0.0.
- Arm joint DOFs occupy the first 7 generalized velocity entries.
Run using mjpython on macOS.
"""
import argparse
import csv
import os
import time
import threading
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from pynput import keyboard
# ============================================================
# CONFIGURATION
# ============================================================
DT = 0.002
LINEAR_SPEED = 0.04       # m/s; start slowly for safety
ANGULAR_SPEED = 0.20      # rad/s
KP_POS = 400.0
KD_POS = 50.0
KP_ROT = 30.0
KD_ROT = 4.0
MAX_TORQUE = 35.0
OPEN_GRIPPER = 0.035
CLOSED_GRIPPER = 0.0
EE_SITE = "center"
PHASES = {
    "1": "approach_obstacle",
    "2": "side_approach_cylinder",
    "3": "side_grasp",
    "4": "lift",
    "5": "transfer_to_target",
    "6": "place",
    "7": "retreat",
}
# ============================================================
# ARGUMENTS AND PATHS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="FR3 obstacle-avoidance teleoperation"
    )
    parser.add_argument(
        "--scene_file",
        type=str,
        required=True,
        help="Path to the MuJoCo scene XML",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="demonstrations/obstacle_avoidance_demo.csv",
        help="CSV path for the recorded demonstration",
    )
    parser.add_argument(
        "--record_every",
        type=int,
        default=10,
        help="Record one sample every N simulation steps",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=LINEAR_SPEED,
        help="Cartesian translation speed in m/s",
    )
    parser.add_argument(
        "--rotation_speed",
        type=float,
        default=ANGULAR_SPEED,
        help="World-Z rotation speed in rad/s",
    )
    parser.add_argument(
        "--ee_site",
        type=str,
        default=EE_SITE,
        help="Name of the end-effector site in the XML",
    )
    return parser.parse_args()
def resolve_scene_path(scene_file):
    """Resolve an absolute path, current-directory path, or assets path."""
    supplied = Path(scene_file).expanduser()
    candidates = [
        supplied,
        Path.cwd() / supplied,
        Path.cwd() / "assets" / supplied.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot find scene XML: {scene_file}\n"
        f"Current directory: {Path.cwd()}"
    )
# ============================================================
# KEYBOARD LISTENER
# ============================================================
class KeyboardState:
    def __init__(self):
        self.lock = threading.Lock()
        self.pressed = set()
        self.paused = False
        self.quit_requested = False
        self.phase = "free_teleoperation"
        self.gripper_target = OPEN_GRIPPER
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        )
    @staticmethod
    def normalize_key(key):
        if key == keyboard.Key.space:
            return "space"
        if key == keyboard.Key.esc:
            return "esc"
        try:
            return key.char.lower() if key.char else None
        except AttributeError:
            return None
    def on_press(self, key):
        name = self.normalize_key(key)
        if name is None:
            return
        with self.lock:
            # Ignore key-repeat events.
            if name in self.pressed:
                return
            self.pressed.add(name)
            if name == "esc":
                self.quit_requested = True
            elif name == "space":
                self.paused = not self.paused
            elif name == "o":
                self.gripper_target = OPEN_GRIPPER
                print("Gripper command: OPEN")
            elif name == "c":
                self.gripper_target = CLOSED_GRIPPER
                print("Gripper command: CLOSE")
            elif name in PHASES:
                self.phase = PHASES[name]
                print(f"Phase: {self.phase}")
    def on_release(self, key):
        name = self.normalize_key(key)
        if name is not None:
            with self.lock:
                self.pressed.discard(name)
    def snapshot(self):
        with self.lock:
            return (
                set(self.pressed),
                self.paused,
                self.quit_requested,
                self.phase,
                self.gripper_target,
            )
    def start(self):
        self.listener.start()
    def stop(self):
        if self.listener.is_alive():
            self.listener.stop()
            self.listener.join(timeout=1.0)
# ============================================================
# MUJOCO HELPERS
# ============================================================
def find_site_id(model, name):
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, name
    )
    if site_id < 0:
        available = []
        for i in range(model.nsite):
            available.append(
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_SITE, i
                )
            )
        raise ValueError(
            f"Site '{name}' was not found in the XML.\n"
            f"Available sites: {available}"
        )
    return site_id
def get_site_rotation(data, site_id):
    return data.site_xmat[site_id].reshape(3, 3).copy()
def calculate_ee_jacobian(model, data, site_id):
    """
    Return position and rotation Jacobians.
    This implementation uses all generalized velocity columns.
    The first seven columns are assumed to correspond to the arm.
    """
    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))
    mujoco.mj_jacSite(
        model, data, jac_pos, jac_rot, site_id
    )
    return jac_pos, jac_rot
def apply_cartesian_controller(
    model,
    data,
    site_id,
    target_pos,
    target_rot,
):
    """
    Cartesian impedance controller with bias-force compensation.
    Returns the commanded seven arm torques.
    """
    jac_pos, jac_rot = calculate_ee_jacobian(
        model, data, site_id
    )
    # Current EE pose.
    current_pos = data.site_xpos[site_id].copy()
    current_rot = get_site_rotation(data, site_id)
    # The arm is assumed to occupy the first seven DOFs.
    jac_pos_arm = jac_pos[:, :7]
    jac_rot_arm = jac_rot[:, :7]
    arm_velocity = data.qvel[:7]
    # Cartesian velocity from the Jacobian.
    linear_velocity = jac_pos_arm @ arm_velocity
    angular_velocity = jac_rot_arm @ arm_velocity
    # Position error.
    position_error = target_pos - current_pos
    # World-frame orientation error.
    rotation_error = Rotation.from_matrix(
        target_rot @ current_rot.T
    ).as_rotvec()
    # Desired Cartesian wrench.
    force = (
        KP_POS * position_error
        - KD_POS * linear_velocity
    )
    moment = (
        KP_ROT * rotation_error
        - KD_ROT * angular_velocity
    )
    wrench = np.concatenate([force, moment])
    jacobian = np.vstack([
        jac_pos_arm,
        jac_rot_arm,
    ])
    # Map Cartesian wrench into joint torque.
    torque_task = jacobian.T @ wrench
    # Compensate gravity and other bias forces.
    # Assumes the first seven generalized DOFs are the arm.
    torque_bias = data.qfrc_bias[:7].copy()
    torque = torque_task + torque_bias
    return np.clip(torque, -MAX_TORQUE, MAX_TORQUE)
# ============================================================
# CSV RECORDING
# ============================================================
def make_csv_header(model):
    header = [
        "sim_time",
        "wall_time",
        "phase",
        "gripper_target",
        "ee_x",
        "ee_y",
        "ee_z",
        "target_x",
        "target_y",
        "target_z",
        "target_roll",
        "target_pitch",
        "target_yaw",
    ]
    for i in range(model.nq):
        header.append(f"qpos_{i}")
    for i in range(model.nv):
        header.append(f"qvel_{i}")
    for i in range(7):
        header.append(f"tau_{i + 1}")
    # Record the world position of every site.
    # This includes cylinder, target box, and obstacle sites.
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_SITE, site_id
        )
        name = name or f"site_{site_id}"
        header.extend([
            f"site_{name}_x",
            f"site_{name}_y",
            f"site_{name}_z",
        ])
    return header
def make_csv_row(
    model,
    data,
    site_id,
    target_pos,
    target_rot,
    phase,
    gripper_target,
    torque,
    wall_time,
):
    target_rpy = Rotation.from_matrix(
        target_rot
    ).as_euler("xyz", degrees=False)
    ee_pos = data.site_xpos[site_id].copy()
    row = [
        float(data.time),
        float(wall_time),
        phase,
        float(gripper_target),
        *ee_pos.tolist(),
        *target_pos.tolist(),
        *target_rpy.tolist(),
    ]
    row.extend(data.qpos.tolist())
    row.extend(data.qvel.tolist())
    row.extend(torque.tolist())
    for i in range(model.nsite):
        row.extend(data.site_xpos[i].tolist())
    return row
# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    if args.record_every < 1:
        raise ValueError("--record_every must be at least 1")
    scene_path = resolve_scene_path(args.scene_file)
    output_path = Path(args.output_file).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading scene: {scene_path}")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    # Use the model's timestep unless you explicitly configure it.
    dt = float(model.opt.timestep)
    if dt <= 0:
        raise ValueError("MuJoCo timestep must be positive")
    if model.nu < 8:
        raise ValueError(
            f"Expected at least 8 actuators (7 arm + 1 gripper), "
            f"but this model has {model.nu}."
        )
    if model.nv < 7:
        raise ValueError("Model has fewer than seven velocity DOFs")
    ee_site_id = find_site_id(model, args.ee_site)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    # Initialize the Cartesian target to the actual current pose.
    target_pos = data.site_xpos[ee_site_id].copy()
    target_rot = get_site_rotation(data, ee_site_id)
    # Start with the gripper open.
    gripper_target = OPEN_GRIPPER
    data.ctrl[7] = gripper_target
    keys = KeyboardState()
    header = make_csv_header(model)
    samples = 0
    simulation_steps = 0
    start_wall_time = time.time()
    last_status_time = start_wall_time
    last_torque = np.zeros(7)
    print("\n========== FR3 OBSTACLE TELEOP ==========")
    print("W / S : +X / -X")
    print("A / D : +Y / -Y")
    print("R / F : +Z / -Z")
    print("Q / E : rotate around world Z")
    print("O     : open gripper")
    print("C     : close gripper")
    print("1-7   : label demonstration phase")
    print("SPACE : pause/resume Cartesian movement")
    print("ESC   : stop and save")
    print(f"Scene timestep: {dt:.6f} s")
    print(f"End-effector site: {args.ee_site}")
    print(f"Output CSV: {output_path}")
    print("=========================================\n")
    # Use a context manager so the viewer closes cleanly.
    try:
        with open(output_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            keys.start()
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running():
                    loop_start = time.perf_counter()
                    (
                        pressed,
                        paused,
                        quit_requested,
                        phase,
                        gripper_target,
                    ) = keys.snapshot()
                    if quit_requested:
                        break
                    # Apply movement only while keys are held.
                    if not paused:
                        direction = np.zeros(3)
                        if "w" in pressed:
                            direction[0] += 1.0
                        if "s" in pressed:
                            direction[0] -= 1.0
                        if "a" in pressed:
                            direction[1] += 1.0
                        if "d" in pressed:
                            direction[1] -= 1.0
                        if "r" in pressed:
                            direction[2] += 1.0
                        if "f" in pressed:
                            direction[2] -= 1.0
                        # Normalize diagonal movement to avoid
                        # moving faster when multiple keys are held.
                        norm = np.linalg.norm(direction)
                        if norm > 0:
                            direction /= norm
                            target_pos += (
                                direction * args.speed * dt
                            )
                        # Rotate the desired orientation about
                        # the world Z axis.
                        yaw_increment = 0.0
                        if "q" in pressed:
                            yaw_increment += (
                                args.rotation_speed * dt
                            )
                        if "e" in pressed:
                            yaw_increment -= (
                                args.rotation_speed * dt
                            )
                        if yaw_increment != 0.0:
                            world_rotation = Rotation.from_rotvec(
                                [0.0, 0.0, yaw_increment]
                            ).as_matrix()
                            target_rot = (
                                world_rotation @ target_rot
                            )
                    # Recompute dynamics before calculating torque.
                    mujoco.mj_forward(model, data)
                    last_torque = apply_cartesian_controller(
                        model,
                        data,
                        ee_site_id,
                        target_pos,
                        target_rot,
                    )
                    # First seven actuators: arm torques.
                    data.ctrl[:7] = last_torque
                    # Eighth actuator: gripper position target.
                    data.ctrl[7] = gripper_target
                    # Step physics.
                    mujoco.mj_step(model, data)
                    simulation_steps += 1
                    # Record measured state after the step.
                    if simulation_steps % args.record_every == 0:
                        row = make_csv_row(
                            model=model,
                            data=data,
                            site_id=ee_site_id,
                            target_pos=target_pos,
                            target_rot=target_rot,
                            phase=phase,
                            gripper_target=gripper_target,
                            torque=last_torque,
                            wall_time=time.time() - start_wall_time,
                        )
                        writer.writerow(row)
                        samples += 1
                    # Update viewer.
                    viewer.sync()
                    # Periodic status output.
                    now = time.time()
                    if now - last_status_time >= 1.0:
                        ee_pos = data.site_xpos[
                            ee_site_id
                        ].copy()
                        print(
                            f"\rPhase: {phase:<25} "
                            f"EE: {np.round(ee_pos, 3)} "
                            f"Gripper: {gripper_target:.3f} "
                            f"Samples: {samples}",
                            end="",
                            flush=True,
                        )
                        last_status_time = now
                    # Prevent an unnecessarily fast busy loop.
                    elapsed = time.perf_counter() - loop_start
                    remaining = dt - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
                    # Periodically flush data so a long recording
                    # is less likely to be lost if interrupted.
                    if simulation_steps % 500 == 0:
                        csv_file.flush()
    except KeyboardInterrupt:
        print("\nInterrupted. Saving recorded samples...")
    finally:
        keys.stop()
    print("\n\n========== RECORDING FINISHED ==========")
    print(f"Samples saved: {samples}")
    print(f"Output file: {output_path}")
    print("========================================")
if __name__ == "__main__":
    main()
