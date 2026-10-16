"""
Keyboard Cartesian teleoperation for the FR3Sim class.
Assumptions:
- Your existing simulator class is saved in fr3_sim.py.
- FR3Sim.step(finger_pos=...) writes the 8th actuator control as a gripper position target.
- The first seven actuators are torque-controlled arm joints.
- Pinocchio frame ID and Jacobian ordering in FR3Sim.get_jacobian() are correct for your model.
Install keyboard listener dependency:
    python -m pip install pynput
Run:
    python fr3_keyboard_teleop.py
The simulation advances in fixed 1 ms MuJoCo steps. CSV rows are buffered in
memory and written at shutdown to avoid disk I/O on every simulation step.
A Python loop is best-effort real-time; it does not guarantee 1 kHz wall-clock
execution on macOS.
"""
import csv
import time
from pathlib import Path
from threading import Lock
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from pynput import keyboard
from fr3_sim import FR3Sim
DT = 0.001
CSV_PATH = Path("fr3_teleop_data.csv")
# Keyboard Cartesian speeds
LINEAR_SPEED = 0.10       # metres/second
ANGULAR_SPEED = 0.50      # radians/second
GRIPPER_SPEED = 0.04      # metres/second; target is clipped below
# Controller gains (start conservatively; tune for your model)
KP_CART_POS = 3.0
KP_CART_ROT = 2.0
KP_JOINT = 80.0
KD_JOINT = 18.0
DAMPING = 0.05
GRIPPER_OPEN = 0.035
GRIPPER_CLOSED = 0.0
# Cartesian command axes: x/y/z and roll/pitch/yaw.
KEY_TO_AXIS = {
    "w": (0, +1), "s": (0, -1),
    "d": (1, +1), "a": (1, -1),
    "r": (2, +1), "f": (2, -1),
    "u": (3, +1), "j": (3, -1),
    "i": (4, +1), "k": (4, -1),
    "o": (5, +1), "l": (5, -1),
}
def key_name(key):
    try:
        return key.char.lower() if key.char else None
    except AttributeError:
        return None
def pose_from_pinocchio(sim, q):
    """Compute the current EE transform using the simulator's Pinocchio model."""
    model = sim.pin_model.model
    data = sim.pin_model.data
    q_pin = np.append(np.asarray(q, dtype=float), [0.0, 0.0])
    mujoco_pose = None
    try:
        import pinocchio as pin
        pin.forwardKinematics(model, data, q_pin)
        pin.updateFramePlacements(model, data)
        return np.array(data.oMf[19].homogeneous, copy=True)
    except Exception:
        # Fallback to the helper supplied by FR3Sim; this may depend on the
        # Pinocchio version and frame-placement update behavior.
        mujoco_pose = np.asarray(sim.get_pose(q), dtype=float)
        return mujoco_pose
def damped_pseudoinverse(J, damping=DAMPING):
    """Numerically stable damped least-squares inverse for a 6x7 Jacobian."""
    return J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(6), np.eye(6))
def main():
    sim = FR3Sim(interface_type="torque", render=True, dt=DT)
    if abs(float(sim.model.opt.timestep) - DT) > 1e-12:
        raise RuntimeError("MuJoCo timestep is not 0.001 s")
    state_lock = Lock()
    held_keys = set()
    stop_requested = False
    gripper_target = GRIPPER_OPEN
    csv_rows = []
    def on_press(key):
        nonlocal stop_requested, gripper_target
        name = key_name(key)
        with state_lock:
            if name:
                held_keys.add(name)
                if name == "g":
                    gripper_target = GRIPPER_OPEN
                elif name == "h":
                    gripper_target = GRIPPER_CLOSED
            elif key == keyboard.Key.space:
                # Clear motion keys; target pose is retained.
                held_keys.clear()
            elif key == keyboard.Key.esc:
                stop_requested = True
                return False
    def on_release(key):
        name = key_name(key)
        if name:
            with state_lock:
                held_keys.discard(name)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)