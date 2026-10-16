import time
import numpy as np
import mujoco
import mujoco.viewer
# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = "/Users/bibek/TPTm/assets/fr3_on_table.xml"
# End-effector site defined in your MuJoCo XML.
EE_SITE_NAME = "center"
# Seven FR3 arm joints and actuators.
ARM_JOINT_NAMES = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]
GRIPPER_ACTUATOR_NAME = "gripper"
# Cartesian movement speed in meters per second.
LINEAR_SPEED = 0.15
# Simulation rate.
SIM_DT = 0.002
# Damped least-squares inverse kinematics.
DAMPING = 0.05
# ============================================================
# LOAD MODEL
# ============================================================
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
ee_site_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME
)
if ee_site_id < 0:
    raise ValueError(
        f"End-effector site '{EE_SITE_NAME}' not found in XML."
    )
arm_joint_ids = []
arm_qpos_ids = []
arm_dof_ids = []
for name in ARM_JOINT_NAMES:
    jid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name
    )
    if jid < 0:
        raise ValueError(f"Joint '{name}' not found in XML.")
    arm_joint_ids.append(jid)
    arm_qpos_ids.append(model.jnt_qposadr[jid])
    arm_dof_ids.append(model.jnt_dofadr[jid])
gripper_actuator_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    GRIPPER_ACTUATOR_NAME,
)
if gripper_actuator_id < 0:
    raise ValueError(
        f"Gripper actuator '{GRIPPER_ACTUATOR_NAME}' not found."
    )
arm_qpos_ids = np.array(arm_qpos_ids)
arm_dof_ids = np.array(arm_dof_ids)
# ============================================================
# CONTROLLER
# ============================================================
def get_ee_position():
    return data.site_xpos[ee_site_id].copy()
def set_gripper(open_gripper):
    actuator = model.actuator_ctrlrange[gripper_actuator_id]
    if model.actuator_ctrllimited[gripper_actuator_id]:
        low, high = actuator
    else:
        low, high = 0.0, 0.04
    data.ctrl[gripper_actuator_id] = (
        high if open_gripper else low
    )
def move_ee(target_position):
    current_position = get_ee_position()
    position_error = target_position - current_position
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(
        model, data, jacp, jacr, ee_site_id
    )
    J = jacp[:, arm_dof_ids]
    # Damped least-squares inverse kinematics.
    dq = J.T @ np.linalg.solve(
        J @ J.T + DAMPING**2 * np.eye(3),
        position_error,
    )
    # Bound joint increments for stability.
    dq = np.clip(dq, -0.02, 0.02)
    # Compute target joint positions.
    q_target = data.qpos[arm_qpos_ids] + dq
    # Position actuators are assumed here.
    for i, joint_id in enumerate(arm_joint_ids):
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            ARM_JOINT_NAMES[i],
        )
        if actuator_id < 0:
            continue
        data.ctrl[actuator_id] = q_target[i]
# ============================================================
# KEYBOARD TELEOPERATION
# ============================================================
print("""
FR3 MuJoCo Keyboard Teleoperation
Cartesian movement:
    W / S : +X / -X
    A / D : +Y / -Y
    R / F : +Z / -Z
Gripper:
    O     : Open
    P     : Close
Other:
    ESC   : Exit
Click the viewer and use the keyboard controls.
""")
target_position = None
gripper_open = True
def key_callback(keycode):
    global target_position, gripper_open
    # GLFW key codes.
    import glfw
    if target_position is None:
        return
    step = LINEAR_SPEED * SIM_DT
    if keycode == glfw.KEY_W:
        target_position[0] += step
    elif keycode == glfw.KEY_S:
        target_position[0] -= step
    elif keycode == glfw.KEY_A:
        target_position[1] += step
    elif keycode == glfw.KEY_D:
        target_position[1] -= step
    elif keycode == glfw.KEY_R:
        target_position[2] += step
    elif keycode == glfw.KEY_F:
        target_position[2] -= step
    elif keycode == glfw.KEY_O:
        gripper_open = True
        set_gripper(True)
    elif keycode == glfw.KEY_P:
        gripper_open = False
        set_gripper(False)
# ============================================================
# SIMULATION AND RENDERER
# ============================================================
try:
    import glfw
    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mj_forward(model, data)
        target_position = get_ee_position()
        while viewer.is_running():
            start = time.time()
            move_ee(target_position)
            mujoco.mj_step(model, data)
            viewer.sync()
            elapsed = time.time() - start
            time.sleep(max(0.0, SIM_DT - elapsed))
except KeyboardInterrupt:
    print("Teleoperation stopped.")
print("Simulation closed.")