import argparse
from pathlib import Path
import mujoco
import numpy as np
import pandas as pd
def closest_box_surface_point(point_local, half_sizes):
    """Find the closest point on a box surface and its outward normal."""
    point_local = np.asarray(point_local, dtype=float)
    half_sizes = np.asarray(half_sizes, dtype=float)
    # For points outside the box, clipping gives the nearest surface point.
    surface = np.clip(point_local, -half_sizes, half_sizes)
    delta = point_local - surface
    distance = np.linalg.norm(delta)
    if distance > 1e-10:
        return surface, delta / distance, float(distance)
    # For points inside the box, select the nearest face.
    face_distances = half_sizes - np.abs(point_local)
    axis = int(np.argmin(face_distances))
    normal = np.zeros(3)
    normal[axis] = 1.0 if point_local[axis] >= 0 else -1.0
    surface = point_local.copy()
    surface[axis] = normal[axis] * half_sizes[axis]
    return surface, normal, float(face_distances[axis])
def measure_clearance(points, box_position, box_rotation, half_sizes):
    """Measure signed distance to the cuboid surface; negative means inside."""
    distances = []
    for point in points:
        local = box_rotation.T @ (point - box_position)
        _, _, distance = closest_box_surface_point(local, half_sizes)
        # Distinguish points inside the box from points outside it.
        inside = np.all(np.abs(local) < half_sizes - 1e-10)
        if inside:
            distance = -distance
        distances.append(distance)
    return np.asarray(distances)
def apply_gaussian_field(
    trajectory,
    box_position,
    box_rotation,
    half_sizes,
    clearance=0.10,
    sigma=0.20,
    amplitude=0.05,
    influence_distance=0.50,
):
    """
    Apply Gaussian repulsion to EE waypoints.
    Points within the required clearance are projected outward.
    Points farther away receive a Gaussian-weighted displacement.
    """
    corrected = trajectory.copy()
    for i, point_world in enumerate(trajectory):
        local = box_rotation.T @ (point_world - box_position)
        surface, normal, distance = closest_box_surface_point(
            local, half_sizes
        )
        # Detect points inside the obstacle.
        inside = np.all(np.abs(local) < half_sizes - 1e-10)
        if inside:
            # Move from the nearest face outward by the required clearance.
            corrected_local = surface + normal * clearance
        elif distance < clearance:
            # Hard clearance constraint.
            corrected_local = surface + normal * clearance
        elif distance < influence_distance:
            # Gaussian influence decays with distance to the surface.
            weight = np.exp(-(distance ** 2) / (2.0 * sigma ** 2))
            corrected_local = local + amplitude * weight * normal
            # Ensure the displacement did not leave the point too close.
            new_surface, new_normal, new_distance = (
                closest_box_surface_point(corrected_local, half_sizes)
            )
            if new_distance < clearance:
                corrected_local = new_surface + new_normal * clearance
        else:
            corrected_local = local
        corrected[i] = (
            box_position + box_rotation @ corrected_local
        )
    return corrected
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply a Gaussian obstacle-avoidance field to a transformed "
            "EE trajectory."
        )
    )
    parser.add_argument(
        "--xml",
        type=Path,
        required=True,
        help="MuJoCo scene XML.",
    )
    parser.add_argument(
        "--trajectory_csv",
        type=Path,
        required=True,
        help="Transformed EE trajectory CSV.",
    )
    parser.add_argument(
        "--keypoints_csv",
        type=Path,
        default=Path("./keypoints/obstacle_keypoints.csv"),
        help="Extracted obstacle keypoints CSV.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("./output/transformed_trajectory_avoiding_obstacle.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--geom_name",
        default="cuboid_obstacle_geom",
        help="Name of the cuboid collision geom in the XML.",
    )
    parser.add_argument("--x_col", default="x")
    parser.add_argument("--y_col", default="y")
    parser.add_argument("--z_col", default="z")
    parser.add_argument("--clearance", type=float, default=0.10)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument("--influence_distance", type=float, default=0.50)
    args = parser.parse_args()
    xml_path = args.xml.expanduser().resolve()
    trajectory_path = args.trajectory_csv.expanduser().resolve()
    keypoints_path = args.keypoints_csv.expanduser().resolve()
    output_path = args.output_csv.expanduser().resolve()
    for path, description in [
        (xml_path, "XML"),
        (trajectory_path, "Trajectory CSV"),
        (keypoints_path, "Obstacle keypoints CSV"),
    ]:
        if not path.is_file():
            parser.error(f"{description} not found: {path}")
    if args.clearance <= 0 or args.sigma <= 0:
        parser.error("Clearance and sigma must be positive.")
    if args.amplitude < 0 or args.influence_distance <= 0:
        parser.error("Amplitude must be nonnegative and influence distance positive.")
    # Load the obstacle keypoints.
    keypoints_df = pd.read_csv(keypoints_path)
    required_keypoint_columns = {"x", "y", "z"}
    if not required_keypoint_columns.issubset(keypoints_df.columns):
        parser.error("Keypoints CSV must contain x, y, and z columns.")
    if "object" in keypoints_df.columns:
        keypoints_df = keypoints_df[
            keypoints_df["object"] == "obstacle"
        ]
    if keypoints_df.empty:
        parser.error("No obstacle keypoints were found.")
    keypoints = keypoints_df[["x", "y", "z"]].to_numpy(dtype=float)
    if not np.isfinite(keypoints).all():
        parser.error("Obstacle keypoints contain invalid coordinates.")
    print(f"Loaded {len(keypoints)} obstacle keypoints from {keypoints_path}")
    # Load the MuJoCo scene.
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        args.geom_name,
    )
    if geom_id == -1:
        parser.error(f"Obstacle geom '{args.geom_name}' not found in {xml_path}.")
    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
        parser.error(f"Geom '{args.geom_name}' must be a box.")
    box_position = data.geom_xpos[geom_id].copy()
    box_rotation = data.geom_xmat[geom_id].reshape(3, 3).copy()
    half_sizes = model.geom_size[geom_id, :3].copy()
    # Sanity check that the keypoints are reasonably consistent with this geom.
    local_keypoints = (keypoints - box_position) @ box_rotation
    keypoint_extent = np.max(np.abs(local_keypoints), axis=0)
    if np.any(keypoint_extent > half_sizes + 0.02):
        print(
            "WARNING: Some keypoints do not appear to match the selected "
            "cuboid geom. Check the XML, geom name, and CSV."
        )
    # Load the transformed trajectory.
    df = pd.read_csv(trajectory_path)
    columns = [args.x_col, args.y_col, args.z_col]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        parser.error(
            f"Trajectory is missing columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    trajectory = df[columns].to_numpy(dtype=float)
    if not np.isfinite(trajectory).all():
        parser.error("Trajectory contains NaN or infinite coordinates.")
    before = measure_clearance(
        trajectory, box_position, box_rotation, half_sizes
    )
    corrected = apply_gaussian_field(
        trajectory=trajectory,
        box_position=box_position,
        box_rotation=box_rotation,
        half_sizes=half_sizes,
        clearance=args.clearance,
        sigma=args.sigma,
        amplitude=args.amplitude,
        influence_distance=args.influence_distance,
    )
    after = measure_clearance(
        corrected, box_position, box_rotation, half_sizes
    )
    # Preserve timestamps, joint columns, and all other data.
    df[columns] = corrected
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved obstacle-avoiding trajectory to:\n{output_path}")
    print(f"Waypoints: {len(trajectory)}")
    print(f"Minimum waypoint clearance before: {before.min():.4f} m")
    print(f"Minimum waypoint clearance after:  {after.min():.4f} m")
    print(f"Target clearance:                  {args.clearance:.4f} m")
    print(f"Gaussian sigma:                    {args.sigma:.4f} m")
    print(f"Gaussian amplitude:                {args.amplitude:.4f} m")
    if after.min() < args.clearance - 1e-7:
        print("WARNING: The minimum waypoint clearance was not achieved.")
    else:
        print("All waypoints satisfy the requested clearance.")
    print(
        "\nNote: This checks EE waypoints only. Validate the interpolated "
        "trajectory and full robot geometry in MuJoCo before execution."
    )
if __name__ == "__main__":
    main()
