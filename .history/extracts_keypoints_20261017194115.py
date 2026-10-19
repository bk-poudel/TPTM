import argparse
from pathlib import Path
import mujoco
import pandas as pd
def get_site_position(model, data, site_name):
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, site_name
    )
    if site_id == -1:
        raise ValueError(
            f"Site '{site_name}' not found in the XML scene."
        )
    return data.site_xpos[site_id].copy()
def main():
    parser = argparse.ArgumentParser(
        description="Extract object keypoints from a MuJoCo XML scene."
    )
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()
    if not args.xml.is_file():
        raise FileNotFoundError(f"XML file not found: {args.xml}")
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Five corresponding sites for the cylinder.
    site_groups = {
        "cylinder": [
            ("kp0", "cylinder_center"),
            ("kp1", "cylinder_top"),
            ("kp2", "cylinder_bottom"),
            ("kp3", "cylinder_side_x"),
            ("kp4", "cylinder_side_y"),
        ],
        # Five corresponding sites for the target box.
        "box": [
            ("kp0", "box_center"),
            ("kp1", "box_corner_front_left"),
            ("kp2", "box_corner_front_right"),
            ("kp3", "box_corner_back_left"),
            ("kp4", "box_corner_back_right"),
        ],
    }
    rows = []
    for obj_name, sites in site_groups.items():
        for point_id, site_name in sites:
            xyz = get_site_position(model, data, site_name)
            rows.append({
                "object": obj_name,
                "point_id": point_id,
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
            })
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Scene: {args.xml}")
    print(f"Saved keypoints: {args.output_csv}")
    print(pd.DataFrame(rows).to_string(index=False))
if __name__ == "__main__":
    main()