import time
import mujoco
import mujoco.viewer
from pynput import keyboard
print("1. Imports successful", flush=True)
model = mujoco.MjModel.from_xml_path(
    "assets/fr3_on_table.xml"
)
data = mujoco.MjData(model)
print("2. Model loaded", flush=True)
viewer = mujoco.viewer.launch_passive(model, data)
print("3. Viewer opened", flush=True)
def on_press(key):
    print(f"Pressed: {key}", flush=True)
listener = keyboard.Listener(on_press=on_press)
listener.start()
print("4. Keyboard listener started", flush=True)
try:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)
finally:
    listener.stop()
    viewer.close()
    print("5. Clean shutdown", flush=True)