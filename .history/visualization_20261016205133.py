from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
original_path = Path("demonstrations/Original_demonstration.csv")
generalized_path = Path("output/Generalized_demonstration.csv")
original = pd.read_csv(original_path)
generalized = pd.read_csv(generalized_path)
xyz = ["ee_x", "ee_y", "ee_z"]
required = ["segment", "sim_time"] + xyz
for name, df in [("Original", original), ("Generalized", generalized)]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} CSV is missing columns: {missing}")
    if df[xyz].isna().any().any():
        raise ValueError(f"{name} trajectory contains missing XYZ values.")
    points = df[xyz].to_numpy()
    step_distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    print(f"\n{name} trajectory")
    print("Rows:", len(df))
    print("Phases:", df["segment"].unique())
    print("Start XYZ:", points[0])
    print("End XYZ:", points[-1])
    print("Maximum consecutive position change (m):",
          step_distances.max())
    print("\nPhase endpoints:")
    print(
        df.groupby("segment", sort=False)
          .agg(
              start_time=("sim_time", "first"),
              end_time=("sim_time", "last"),
              start_x=("ee_x", "first"),
              start_y=("ee_y", "first"),
              start_z=("ee_z", "first"),
              end_x=("ee_x", "last"),
              end_y=("ee_y", "last"),
              end_z=("ee_z", "last"),
          ).round(4)
    )
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(
    original["ee_x"], original["ee_y"], original["ee_z"],
    label="Original", alpha=0.75
)
ax.plot(
    generalized["ee_x"], generalized["ee_y"], generalized["ee_z"],
    label="Generalized", linewidth=2
)
ax.scatter(*original[xyz].iloc[0], marker="o", label="Original start")
ax.scatter(*generalized[xyz].iloc[-1], marker="^",
           label="Generalized end")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Original vs Generalized End-Effector Trajectory")
ax.legend()
plt.tight_layout()
plt.savefig("trajectory_comparison.png", dpi=200)
plt.show()