import mujoco
import mujoco.viewer
from MujocoSim import FR3Sim
print("Creating simulator...", flush=True)
sim = FR3Sim(render=True, dt=0.001)
print("Simulator created. Viewer should be visible.", flush=True)
try:
    while sim.viewer.is_running():
        mujoco.mj_step(sim.model, sim.data)
        sim.send_joint_torque(sim.get_gravity)
        sim.viewer.sync()
finally:
    sim.close()