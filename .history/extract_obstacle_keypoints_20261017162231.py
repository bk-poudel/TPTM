import argparse
from pathlib import Path
import mujoco
import pandas as pd
def extract_obstacle_keypoints(model, data, output_path):
    """Extract the seven cuboid obstacle sites and save their world coordinates."""
    site_names = [
        "obstacle_center",
        "obstacle_face_pos_x",
        "obstacle_face_neg_x",
        "obstacle_face_pos_y",
        "obstacle_face_neg_y",
        "obstacle_face_pos_z",
        "obstacle_face_neg_z",
    ]
    # Compute world-coordinate positions of all sites.
    mujoco.mj_forward(model, data)
    rows = []
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
        x, y, z = data.site_xpos[site_id].copy()
        rows.append({
            "object": "obstacle",
            "point_id": site_name.replace("obstacle_", ""),
            "x": float(x),
            "y": float(y),
            "z": float(z),
        })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        rows,
        columns=["object", "point_id", "x", "y", "z"],
    )
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} obstacle keypoints to:")
    print(output_path.resolve())
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
    if not args.xml.is_file():
        parser.error(f"XML file does not exist: {args.xml}")
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    extract_obstacle_keypoints(
        model=model,
        data=data,
        output_path=args.output_csv,
    )
if __name__ == "__main__":
    main()