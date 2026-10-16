import csv
import os
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp
from MujocoSim import FR3Sim
# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
DT = 0.001
CYLINDER_SITE = "cylinder_center"
PLACE_SITE = "box_center"
APPROACH_HEIGHT = 0.12
LIFT_HEIGHT = 0.15
# Adjust these offsets to match your object and gripper.
# If cylinder_top_center is at the cylinder's top, move down
# toward its grasp height using GRASP_OFFSET.
GRASP_OFFSET = np.array([0.0, 0.0, -0.025])
PLACE_OFFSET = np.array([0.0, 0.0, 0.04])
OPEN_GRIPPER = 0.035
CLOSED_GRIPPER = 0.0
KP_POS = 300.0
KD_POS = 35.0
KP_ROT = 20.0
KD_ROT = 3.0
MAX_TORQUE = 35.0
OUTPUT_DIR = "demonstrations"
os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_path = os.path.join(
    OUTPUT_DIR,
    f"pick_place_{time.strftime('%Y%m%d_%H%M%S')}.csv"
)
# ---------------------------------------------------------
# Smooth Cartesian trajectory segment
# ---------------------------------------------------------
def interpolate_pose(start_pos, end_pos,
                     start_rot, end_rot, steps):
    """Quintic position interpolation and SLERP orientation."""
    fractions = np.linspace(0.0, 1.0, steps + 1)
    # Quintic time scaling: zero velocity and acceleration
    # at both ends of the segment.
    s = 10 * fractions**3 - 15 * fractions**4 + 6 * fractions**5
    positions = (
        start_pos[None, :]
        + s[:, None] * (end_pos - start_pos)[None, :]
    )
    rotations = Slerp(
        [0.0, 1.0],
        Rotation.from_matrix(np.stack([start_rot, end_rot]))
    )(fractions)
    return positions, rotations.as_matrix()
# ---------------------------------------------------------
# Cartesian impedance controller
# ---------------------------------------------------------
def compute_torque(sim, q, dq, target_pos, target_rot):
    T = sim.get_pose(q)
    current_pos = T[:3, 3]
    current_rot = T[:3, :3]
    J = sim.get_jacobian(q)
    ee_velocity = J @ dq
    pos_error = target_pos - current_pos
    rot_error = Rotation.from_matrix(
        target_rot @ current_rot.T
    ).as_rotvec()
    force = KP_POS * pos_error - KD_POS * ee_velocity[:3]
    moment = KP_ROT * rot_error - KD_ROT * ee_velocity[3:]
    wrench = np.concatenate([force, moment])
    tau = J.T @ wrench + sim.get_gravity(q)
    tau = np.clip(tau, -MAX_TORQUE, MAX_TORQUE)
    return tau
# ---------------------------------------------------------
# Execute one trajectory segment and record the execution
# ---------------------------------------------------------
def execute_segment(
    sim, writer, segment_name,
    target_pos, target_rot, duration, finger_pos
):
    q, dq = sim.get_state()
    current_T = sim.get_pose(q)
    start_pos = current_T[:3, 3].copy()
    start_rot = current_T[:3, :3].copy()
    steps = max(1, int(round(duration / DT)))
    positions, rotations = interpolate_pose(
        start_pos, target_pos,
        start_rot, target_rot,
        steps
    )
    print(
        f"{segment_name}: "
        f"{start_pos.round(3)} -> {target_pos.round(3)}",
        flush=True
    )
    for i in range(1, steps + 1):
        if sim.render and not sim.viewer.is_running():
            raise RuntimeError("MuJoCo viewer was closed.")
        desired_pos = positions[i]
        desired_rot = rotations[i]
        q, dq = sim.get_state()
        q = np.asarray(q).copy()
        dq = np.asarray(dq).copy()
        tau = compute_torque(
            sim, q, dq, desired_pos, desired_rot
        )
        # Advance the simulator by exactly one DT step.
        sim.send_joint_torque(
            tau, finger_pos=finger_pos
        )
        # Record the resulting state after the step.
        q, dq = sim.get_state()
        q = np.asarray(q).copy()
        dq = np.asarray(dq).copy()
        T = sim.get_pose(q)
        J = sim.get_jacobian(q)
        ee_pos = T[:3, 3]
        ee_rot = Rotation.from_matrix(
            T[:3, :3]
        ).as_euler("xyz")
        ee_velocity = J @ dq
        writer.writerow([
            segment_name,
            sim.data.time,
            *ee_pos,
            *ee_rot,
            *desired_pos,
            *Rotation.from_matrix(
                desired_rot
            ).as_euler("xyz"),
            *q,
            *dq,
            *tau,
            finger_pos,
        ])
# ---------------------------------------------------------
# Main pick-and-place sequence
# ---------------------------------------------------------
def main():
    sim = None
    try:
        sim = FR3Sim(
            interface_type="torque",
            render=True,
            dt=DT,
            gravity=True,
        )
        # Refresh MuJoCo's derived positions before reading sites.
        mujoco.mj_forward(sim.model, sim.data)
        available_sites = [
            sim.model.site(i).name
            for i in range(sim.model.nsite)
        ]
        print("Available MuJoCo sites:", available_sites)
        if CYLINDER_SITE not in available_sites:
            raise ValueError(
                f"Missing site '{CYLINDER_SITE}'. "
                "Set CYLINDER_SITE to a name from the list above."
            )
        if PLACE_SITE not in available_sites:
            raise ValueError(
                f"Missing site '{PLACE_SITE}'. "
                "Set PLACE_SITE to a name from the list above."
            )
        # Get the sites' world-frame positions.
        cylinder_site_pos = np.array(
            sim.get_site_pose(CYLINDER_SITE)
        ).copy()
        place_site_pos = np.array(
            sim.get_site_pose(PLACE_SITE)
        ).copy()
        grasp_pos = cylinder_site_pos + GRASP_OFFSET
        approach_pos = grasp_pos + np.array(
            [0.0, 0.0, APPROACH_HEIGHT]
        )
        lift_pos = grasp_pos + np.array(
            [0.0, 0.0, LIFT_HEIGHT]
        )
        place_pos = place_site_pos + PLACE_OFFSET
        place_approach_pos = place_pos + np.array(
            [0.0, 0.0, APPROACH_HEIGHT]
        )
        print("Cylinder site:", cylinder_site_pos)
        print("Grasp target:", grasp_pos)
        print("Place site:", place_site_pos)
        print("Place target:", place_pos)
        # Keep the initial end-effector orientation throughout
        # this basic pick-and-place demonstration.
        q, dq = sim.get_state()
        initial_T = sim.get_pose(np.asarray(q).copy())
        fixed_rotation = initial_T[:3, :3].copy()
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "segment", "sim_time",
                "ee_x", "ee_y", "ee_z",
                "ee_roll", "ee_pitch", "ee_yaw",
                "target_x", "target_y", "target_z",
                "target_roll", "target_pitch", "target_yaw",
                "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "dq1", "dq2", "dq3", "dq4", "dq5", "dq6", "dq7",
                "tau1", "tau2", "tau3", "tau4",
                "tau5", "tau6", "tau7",
                "gripper_target",
            ])
            # 1. Approach the cylinder from above.
            execute_segment(
                sim, writer, "approach_cylinder",
                approach_pos, fixed_rotation, 2.0,
                OPEN_GRIPPER
            )
            # 2. Descend to the grasp location.
            execute_segment(
                sim, writer, "descend_to_grasp",
                grasp_pos, fixed_rotation, 1.5,
                OPEN_GRIPPER
            )
            # 3. Close gripper while maintaining the grasp pose.
            execute_segment(
                sim, writer, "close_gripper",
                grasp_pos, fixed_rotation, 0.8,
                CLOSED_GRIPPER
            )
            # 4. Lift the cylinder.
            execute_segment(
                sim, writer, "lift",
                lift_pos, fixed_rotation, 1.5,
                CLOSED_GRIPPER
            )
            # 5. Move above the placement destination.
            execute_segment(
                sim, writer, "move_to_place",
                place_approach_pos, fixed_rotation, 2.0,
                CLOSED_GRIPPER
            )
            # 6. Lower to the placement location.
            execute_segment(
                sim, writer, "lower_to_place",
                place_pos, fixed_rotation, 1.5,
                CLOSED_GRIPPER
            )
            # 7. Release the object.
            execute_segment(
                sim, writer, "release",
                place_pos, fixed_rotation, 0.8,
                OPEN_GRIPPER
            )
            # 8. Retreat upward.
            execute_segment(
                sim, writer, "retreat",
                place_approach_pos, fixed_rotation, 1.5,
                OPEN_GRIPPER
            )
        print("\nPick-and-place trajectory finished.")
        print("Demonstration saved to:", csv_path)
    finally:
        if sim is not None:
            sim.close()
if __name__ == "__main__":
    main()
