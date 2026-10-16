import time
import mujoco
import mujoco.viewer
from MujocoSim import FR3Sim
def main():
    # Instantiate the robot simulation with viewer rendering enabled
    robot = FR3Sim(render=True, dt=0.001)
    print("Visualizing Franka FR3 in initial home pose...")
    print("Press Ctrl+C in the terminal to stop.")
    try:
        while True:
            # Retrieve joint positions and velocities
            q, dq = robot.get_state()
            # Apply gravity compensation torques to hold the robot stationary in its home pose
            tau_gravity = robot.get_gravity(q)
            robot.send_joint_torque(tau_gravity)
            # Sleep to maintain real-time step rate
            time.sleep(robot.dt)
    except KeyboardInterrupt:
        print("\nClosing viewer...")
        robot.close()
if __name__ == "__main__":
    main()