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
import argparse
import csv
from pathlib import Path
import numpy as np
import pandas as pd
def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply a Gaussian repulsive field using extracted obstacle keypoints."
    )
    parser.add_argument("--trajectory_csv", type=Path, required=True)
    parser.add_argument(
        "--keypoints_csv",
        type=Path,
        default=Path("keypoints/obstacle_keypoints.csv"),
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("output/transformed_trajectory_avoiding_obstacle.csv"),
    )
    parser.add_argument("--x_col", default="x")
    parser.add_argument("--y_col", default="y")
    parser.add_argument("--z_col", default="z")
    parser.add_argument("--clearance", type=float, default=0.10)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument("--influence_distance", type=float, default=0.50)
    return parser.parse_args()
def load_obstacle(keypoints_csv):
    """Reconstruct an oriented cuboid from its extracted MuJoCo keypoints."""
    df = pd.read_csv(keypoints_csv)
    required = {"point_id", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Keypoints CSV is missing columns: {sorted(missing)}")
    if "object" in df.columns:
        df = df[df["object"] == "obstacle"]
    points = {
        str(row["point_id"]): np.array(
            [row["x"], row["y"], row["z"]], dtype=float
        )
        for _, row in df.iterrows()
    }
    required_points = {
        "center",
        "pos_x_center", "neg_x_center",
        "pos_y_center", "neg_y_center",
        "pos_z_center", "neg_z_center",
    }
    missing_points = required_points - points.keys()
    if missing_points:
        raise ValueError(
            "Missing obstacle keypoints: "
            + ", ".join(sorted(missing_points))
        )
    center = points["center"]
    axes = []
    half_sizes = []
    for positive, negative in [
        ("pos_x_center", "neg_x_center"),
        ("pos_y_center", "neg_y_center"),
        ("pos_z_center", "neg_z_center"),
    ]:
        vector = points[positive] - points[negative]
        full_size = np.linalg.norm(vector)
        if full_size < 1e-8:
            raise ValueError(f"Invalid face-center pair: {positive}, {negative}")
        axes.append(vector / full_size)
        half_sizes.append(full_size / 2.0)
    # Each column of R is a cuboid axis expressed in world coordinates.
    R = np.column_stack(axes)
    half_sizes = np.asarray(half_sizes)
    # Verify that the reconstructed axes form an orthonormal frame.
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3):
        raise ValueError(
            "Face-center keypoints do not define perpendicular cuboid axes."
        )
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    # Validate the reconstruction against the eight corner keypoints.
    corner_names = [f"corner_{i}" for i in range(1, 9)]
    available_corners = [name for name in corner_names if name in points]
    if available_corners:
        errors = []
        for name in available_corners:
            local = R.T @ (points[name] - center)
            errors.append(
                np.max(np.abs(np.abs(local) - half_sizes))
            )
        max_error = max(errors)
        if max_error > 1e-3:
            raise ValueError(
                f"Corners disagree with reconstructed cuboid "
                f"(maximum error {max_error:.6f} m)."
            )
    return center, R, half_sizes
def closest_surface(point, center, R, half_sizes):
    """
    Return closest surface point, outward normal, and signed distance.
    Signed distance is positive outside the cuboid and negative inside.
    """
    local = R.T @ (point - center)
    excess = np.abs(local) - half_sizes
    outside = np.maximum(excess, 0.0)
    if np.any(outside > 1e-12):
        # Closest point on the box for an exterior point.
        surface_local = np.clip(local, -half_sizes, half_sizes)
        delta_local = local - surface_local
        distance = np.linalg.norm(delta_local)
        normal_local = delta_local / distance
        signed_distance = distance
    else:
        # For an interior point, choose its nearest face.
        face_distances = half_sizes - np.abs(local)
        axis = int(np.argmin(face_distances))
        sign = 1.0 if local[axis] >= 0 else -1.0
        surface_local = local.copy()
        surface_local[axis] = sign * half_sizes[axis]
        normal_local = np.zeros(3)
        normal_local[axis] = sign
        signed_distance = -face_distances[axis]
    surface_world = center + R @ surface_local
    normal_world = R @ normal_local
    normal_world /= np.linalg.norm(normal_world)
    return surface_world, normal_world, signed_distance
def apply_gaussian_field(
    point, center, R, half_sizes,
    clearance, sigma, amplitude, influence_distance,
):
    surface, normal, distance = closest_surface(
        point, center, R, half_sizes
    )
    # Guarantee the required clearance for this trajectory waypoint.
    # This also moves points out of the cuboid if they start inside it.
    if distance < clearance:
        return surface + clearance * normal
    # Apply the Gaussian repulsive displacement near the obstacle.
    if distance < influence_distance:
        displacement = amplitude * np.exp(
            -(distance ** 2) / (2.0 * sigma ** 2)
        )
        point = point + displacement * normal
        # Recheck clearance after the displacement.
        surface, normal, distance = closest_surface(
            point, center, R, half_sizes
        )
        if distance < clearance:
            point = surface + clearance * normal
    return point
def main():
    args = parse_args()
    if args.clearance < 0:
        raise ValueError("--clearance must be non-negative.")
    if args.sigma <= 0:
        raise ValueError("--sigma must be positive.")
    if args.amplitude < 0:
        raise ValueError("--amplitude must be non-negative.")
    if args.influence_distance <= 0:
        raise ValueError("--influence_distance must be positive.")
    if not args.trajectory_csv.is_file():
        raise FileNotFoundError(
            f"Trajectory CSV not found: {args.trajectory_csv.resolve()}"
        )
    if not args.keypoints_csv.is_file():
        raise FileNotFoundError(
            f"Keypoints CSV not found: {args.keypoints_csv.resolve()}"
        )
    trajectory = pd.read_csv(args.trajectory_csv)
    columns = [args.x_col, args.y_col, args.z_col]
    missing = set(columns) - set(trajectory.columns)
    if missing:
        raise ValueError(
            f"Trajectory CSV is missing coordinate columns: {sorted(missing)}"
        )
    xyz = trajectory[columns].to_numpy(dtype=float)
    if not np.isfinite(xyz).all():
        raise ValueError("Trajectory contains NaN or infinite coordinates.")
    center, R, half_sizes = load_obstacle(args.keypoints_csv)
    print("Reconstructed obstacle from keypoints:")
    print(f"  Center:     {center}")
    print(f"  Half-sizes: {half_sizes}")
    print(f"  Clearance:  {args.clearance:.3f} m")
    output_xyz = np.empty_like(xyz)
    for i, point in enumerate(xyz):
        output_xyz[i] = apply_gaussian_field(
            point=point.copy(),
            center=center,
            R=R,
            half_sizes=half_sizes,
            clearance=args.clearance,
            sigma=args.sigma,
            amplitude=args.amplitude,
            influence_distance=args.influence_distance,
        )
    # Keep timestamps, gripper values, and all other columns unchanged.
    trajectory[columns] = output_xyz
    # Check waypoint clearances in the resulting trajectory.
    signed_distances = np.array([
        closest_surface(p, center, R, half_sizes)[2]
        for p in output_xyz
    ])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(args.output_csv, index=False)
    print(f"\nSaved trajectory: {args.output_csv.resolve()}")
    print(f"Waypoints processed: {len(output_xyz)}")
    print(f"Minimum waypoint clearance: {signed_distances.min():.4f} m")
    print(
        "Waypoints below requested clearance: "
        f"{np.sum(signed_distances < args.clearance - 1e-6)}"
    )
if __name__ == "__main__":
    main()
