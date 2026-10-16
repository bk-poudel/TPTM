"""
Cartesian keyboard teleoperation + 1 kHz simulated-state recording.
Requires:
    pip install mujoco pin pynput scipy numpy
Assumes:
    MujocoSim.py contains FR3Sim
    FR3Sim.get_pose(q) returns a 4x4 end-effector transform
    FR3Sim.get_jacobian(q) returns [linear; angular] Jacobian (6x7)
    FR3Sim.get_gravity(q) returns a 7-element gravity compensation vector
Controls:
    W / S : +X / -X
    D / A : +Y / -Y
    R / F : +Z / -Z
    U / O : +Roll / -Roll
    I / K : +Pitch / -Pitch
    J / L : +Yaw / -Yaw
    SPACE : pause/resume motion
    ENTER : start/stop recording
    ESC   : quit and save CSV
"""
import csv
import os
import time
import threading
import queue
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation
from pynput import keyboard
from MujocoSim import FR3Sim
# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DT = 0.001                    # 1 kHz simulation/control timestep
TRANSLATION_SPEED = 0.08      # m/s
ROTATION_SPEED = 0.5          # rad/s
# Cartesian impedance gains
KP_POS = 300.0                # N/m
KD_POS = 35.0                 # N/(m/s)
KP_ROT = 20.0                 # Nm/rad
KD_ROT = 3.0                  # Nm/(rad/s)
MAX_TORQUE = 35.0             # Conservative initial torque clamp (Nm)
# World-frame Cartesian workspace limits (adjust for your scene)
X_LIMITS = (-0.65, 0.65)
Y_LIMITS = (-0.80, 0.50)
Z_LIMITS = (0.05, 1.15)
# Each CSV row corresponds to one simulated 1 ms step.
OUTPUT_DIR = "demonstrations"
# ------------------------------------------------------------
# Keyboard state
# ------------------------------------------------------------
pressed_keys = set()
key_lock = threading.Lock()
running = True
motion_enabled = True
recording = False
def key_name(key):
    try:
        return key.char.lower()
    except AttributeError:
        if key == keyboard.Key.space:
            return "space"
        if key == keyboard.Key.enter:
            return "enter"
        if key == keyboard.Key.esc:
            return "esc"
        return None
def on_press(key):
    global running, motion_enabled, recording
    name = key_name(key)
    with key_lock:
        if name in ("w", "s", "a", "d", "r", "f",
                    "u", "o", "i", "k", "j", "l"):
            pressed_keys.add(name)
    if name == "space":
        motion_enabled = not motion_enabled
        print("Motion:", "ENABLED" if motion_enabled else "PAUSED")
    elif name == "enter":
        recording = not recording
        print("Recording:", "ON" if recording else "OFF")
    elif name == "esc":
        running = False
        return False
def on_release(key):
    name = key_name(key)
    with key_lock:
        pressed_keys.discard(name)
# ------------------------------------------------------------
# CSV writer runs independently of the control loop
# ------------------------------------------------------------
class CSVLogger:
    def __init__(self, filename):
        self.filename = filename
        self.data_queue = queue.Queue(maxsize=200000)
        self.thread = threading.Thread(
            target=self._writer, daemon=True
        )
        self.thread.start()
    def log(self, row):
        # Keep file I/O out of the control loop.
        # A dropped row is counted rather than silently blocking control.
        try:
            self.data_queue.put_nowait(row)
        except queue.Full:
            self.dropped_rows += 1
    def start(self):
        self.dropped_rows = 0
    def _writer(self):
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sim_time",
                "wall_time",
                "recording_time",
                "x", "y", "z",
                "roll", "pitch", "yaw",
                "vx", "vy", "vz",
                "wx", "wy", "wz",
                "ax", "ay", "az",
                "alphax", "alphay", "alphaz",
                "target_x", "target_y", "target_z",
                "target_roll", "target_pitch", "target_yaw",
                "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "dq1", "dq2", "dq3", "dq4", "dq5", "dq6", "dq7",
                "is_recording"
            ])
            while True:
                row = self.data_queue.get()
                try:
                    if row is None:
                        break
                    writer.writerow(row)
                finally:
                    self.data_queue.task_done()
            f.flush()
    def close(self):
        self.data_queue.put(None)
        self.data_queue.join()
        self.thread.join()
# ------------------------------------------------------------
# Cartesian controller
# ------------------------------------------------------------
def cartesian_impedance_torque(sim, q, dq, target_position,
                               target_rotation):
    """
    Return joint torques from a Cartesian impedance controller.
    Jacobian convention:
        J[:3] = linear Jacobian
        J[3:] = angular Jacobian
    Rotation error and angular velocity are expressed in world
    coordinates, assuming the simulator Jacobian uses world-aligned
    Cartesian components.
    """
    T = sim.get_pose(q)
    current_position = T[:3, 3]
    current_rotation = T[:3, :3]
    J = sim.get_jacobian(q)
    velocity = J @ dq
    linear_velocity = velocity[:3]
    angular_velocity = velocity[3:]
    # Position error, expressed in world coordinates.
    position_error = target_position - current_position
    # World-frame orientation error as a rotation vector.
    rotation_error = Rotation.from_matrix(
        target_rotation @ current_rotation.T
    ).as_rotvec()
    force_command = (
        KP_POS * position_error - KD_POS * linear_velocity
    )
    moment_command = (
        KP_ROT * rotation_error - KD_ROT * angular_velocity
    )
    wrench = np.concatenate([force_command, moment_command])
    tau = J.T @ wrench + sim.get_gravity(q)
    # Limit commanded torques as a basic safety precaution.
    tau = np.clip(tau, -MAX_TORQUE, MAX_TORQUE)
    return tau, T, velocity
# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("A: Starting script", flush=True)
    global running
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(
        OUTPUT_DIR, f"fr3_demo_{timestamp}.csv"
    )
    print("B: Creating FR3Sim", flush=True)
    sim = FR3Sim(
        interface_type="torque",
        render=True,
        dt=DT,
        gravity=True,
    )
    print("C: FR3Sim created", flush=True)
    print("D: Creating CSV logger", flush=True)
    logger = CSVLogger(csv_path)
    print("E: Logger created", flush=True)
    logger.start()
    print("F: Starting keyboard listener", flush=True)
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release,
    )
    listener.start()
    print("G: Keyboard listener started", flush=True)
    try:
        print("H: Reading initial pose", flush=True)
        q, dq = sim.get_state()
        q = np.asarray(q).copy()
        dq = np.asarray(dq).copy()
        T = sim.get_pose(q)
        target_position = T[:3, 3].copy()
        target_rotation = T[:3, :3].copy()
        print("\nCartesian FR3 teleoperation started.")
        print("Move with W/S, A/D, R/F.")
        print("Rotate with U/O, I/K, J/L.")
        print("ENTER toggles recording; ESC exits.")
        print("CSV:", csv_path)
        previous_velocity = np.zeros(6)
        sim_time = 0.0
        recording_start = None
        while running and sim.viewer.is_running():
            loop_start = time.perf_counter()
            # Read the keyboard state safely.
            with key_lock:
                keys = set(pressed_keys)
            if motion_enabled:
                # Integrate translational target velocity.
                direction = np.array([
                    float("w" in keys) - float("s" in keys),
                    float("d" in keys) - float("a" in keys),
                    float("r" in keys) - float("f" in keys),
                ])
                target_position += (
                    TRANSLATION_SPEED * DT * direction
                )
                target_position[0] = np.clip(
                    target_position[0], *X_LIMITS
                )
                target_position[1] = np.clip(
                    target_position[1], *Y_LIMITS
                )
                target_position[2] = np.clip(
                    target_position[2], *Z_LIMITS
                )
                # Apply incremental rotations about world axes.
                angular_direction = np.array([
                    float("u" in keys) - float("o" in keys),
                    float("i" in keys) - float("k" in keys),
                    float("j" in keys) - float("l" in keys),
                ])
                delta_rotation = Rotation.from_rotvec(
                    ROTATION_SPEED * DT * angular_direction
                ).as_matrix()
                target_rotation = (
                    delta_rotation @ target_rotation
                )
            # Read the measured joint state.
            q, dq = sim.get_state()
            q = np.asarray(q).copy()
            dq = np.asarray(dq).copy()
            # Compute torque from Cartesian impedance.
            tau, T, velocity = cartesian_impedance_torque(
                sim,
                q,
                dq,
                target_position,
                target_rotation,
            )
            # Advance the simulator by one 1 ms step.
            sim.send_joint_torque(
                tau,
                finger_pos=0.035,
            )
            sim_time += DT
            # Re-read the state after stepping so the logged pose
            # corresponds to the updated simulation state.
            q, dq = sim.get_state()
            q = np.asarray(q).copy()
            dq = np.asarray(dq).copy()
            T = sim.get_pose(q)
            J = sim.get_jacobian(q)
            position = T[:3, 3].copy()
            rotation = T[:3, :3].copy()
            # Use XYZ Euler angles for roll, pitch, yaw.
            # Euler angles are singular at certain orientations.
            rpy = Rotation.from_matrix(rotation).as_euler("xyz")
            # Spatial velocity, calculated from the Jacobian.
            velocity = J @ dq
            # Finite-difference acceleration at the simulation rate.
            acceleration = (
                velocity - previous_velocity
            ) / DT
            previous_velocity = velocity.copy()
            if recording:
                if recording_start is None:
                    recording_start = sim_time
                recording_time = sim_time - recording_start
                row = [
                    sim_time,
                    time.time(),
                    recording_time,
                    *position,
                    *rpy,
                    *velocity[:3],
                    *velocity[3:],
                    *acceleration[:3],
                    *acceleration[3:],
                    *target_position,
                    *Rotation.from_matrix(
                        target_rotation
                    ).as_euler("xyz"),
                    *q,
                    *dq,
                    1,
                ]
                logger.log(row)
            else:
                recording_start = None
            # Optional console status at a lower frequency.
            if int(sim_time * 10) != int((sim_time - DT) * 10):
                print(
                    f"\rt={sim_time:8.2f}s | "
                    f"EE={position.round(3)} | "
                    f"recording={recording}",
                    end="",
                    flush=True,
                )
            # No sleep: advance simulation as fast as possible.
            # Each iteration advances exactly DT simulated seconds.
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        listener.stop()
        logger.close()
        sim.close()
        print("\nTeleoperation stopped.")
        print("CSV saved to:", csv_path)
        print("Dropped CSV rows:", logger.dropped_rows)
if __name__ == "__main__":
    main()
