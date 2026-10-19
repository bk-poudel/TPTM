import argparse
import csv
from pathlib import Path
import mujoco
import numpy as np
# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
parser = argparse.ArgumentParser(
    description="Extract cuboid obstacle keypoints from a MuJoCo scene."
)
parser.add_argument(
    "--xml",
    type=Path,
    required=True,
    help="Path to the MuJoCo scene XML file",
)
parser.add_argument(
    "--output_csv",
    type=Path,
    default=Path("keypoints/obstacle_keypoints.csv"),
    help="Path to save the extracted obstacle keypoints CSV",
)
args = parser.parse_args()
XML_PATH = args.xml
OUTPUT_CSV = args.output_csv
# --------------------------------------------------
# OBSTACLE KEYPOINT SITE NAMES
# Must match the names in your MuJoCo XML.
# --------------------------------------------------
OBSTACLE_SITES = [
    "obstacle_center",
    "obstacle_pos_x_center",
    "obstacle_neg_x_center",
    "obstacle_pos_y_center",
    "obstacle_neg_y_center",
    "obstacle_pos_z_center",
    "obstacle_neg_z_center",
    "obstacle_corner_1",
    "obstacle_corner_2",
    "obstacle_corner_3",
    "obstacle_corner_4",
    "obstacle_corner_5",
    "obstacle_corner_6",
    "obstacle_corner_7",
    "obstacle_corner_8",
]
# --------------------------------------------------
# EXTRACT OBSTACLE KEYPOINTS
# --------------------------------------------------
def extract_obstacle_keypoints(model, data, site_names):
    """Extract named obstacle sites in world coordinates."""
    mujoco.mj_forward(model, data)
    points = []
    for site_name in site_names:
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        if site_id == -1:
            raise ValueError(
                f"Site '{site_name}' was not found in the MuJoCo XML."
            )
        position = data.site_xpos[site_id].copy()
        if not np.isfinite(position).all():
            raise ValueError(
                f"Invalid coordinates for site '{site_name}'."
            )
        points.append({
            "object": "obstacle",
            "point_id": site_name.removeprefix("obstacle_"),
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
        })
    return points
# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    if not XML_PATH.is_file():
        raise FileNotFoundError(
            f"MuJoCo XML not found: {XML_PATH.resolve()}"
        )
    xml_path = XML_PATH.expanduser().resolve()
    output_path = OUTPUT_CSV.expanduser().resolve()
    print(f"Loading XML: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    obstacle_points = extract_obstacle_keypoints(
        model=model,
        data=data,
        site_names=OBSTACLE_SITES,
    )
    # --------------------------------------------------
    # SAVE TO CSV
    # --------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["object", "point_id", "x", "y", "z"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(obstacle_points)
    print(f"\nSaved obstacle keypoints to: {output_path}")
    print(f"Total obstacle keypoints: {len(obstacle_points)}")
    print("\nExtracted obstacle keypoints (world coordinates):")
    for point in obstacle_points:
        print(
            f"{point['point_id']:25s} "
            f"x={point['x']: .6f}, "
            f"y={point['y']: .6f}, "
            f"z={point['z']: .6f}"
        )
if __name__ == "__main__":
    main()
