import argparse
from pathlib import Path
import mujoco
import pandas as pd
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
def extract_obstacle_keypoints(model, data, output_path):
    """Extract the cuboid obstacle's 15 sites in world coordinates."""
    mujoco.mj_forward(model, data)
    rows = []
    for site_name in OBSTACLE_SITES:
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        if site_id == -1:
            raise ValueError(
                f"Site '{site_name}' was not found in the MuJoCo XML."
            )
        x, y, z = data.site_xpos[site_id].copy()
        rows.append({
            "object": "obstacle",
            "point_id": site_name.removeprefix("obstacle_"),
            "x": float(x),
            "y": float(y),
            "z": float(z),
        })
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        rows,
        columns=["object", "point_id", "x", "y", "z"],
    )
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} obstacle keypoints to:")
    print(output_path)
    print("\nExtracted keypoints:")
    print(df.to_string(index=False))
    return df
def main():
    parser = argparse.ArgumentParser(
        description="Extract cuboid obstacle keypoints from a MuJoCo scene."
    )
    parser.add_argument(
        "--xml",
        type=Path,
        required=True,
        help="Path to the MuJoCo scene XML file.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("./keypoints/obstacle_keypoints.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()
    xml_path = args.xml.expanduser().resolve()
    if not xml_path.is_file():
        parser.error(f"XML file does not exist: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    extract_obstacle_keypoints(
        model=model,
        data=data,
        output_path=args.output_csv,
    )
if __name__ == "__main__":
    main()
