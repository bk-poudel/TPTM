import argparse
from pathlib import Path
import mujoco
import numpy as np
import pandas as pd
def closest_box_surface_point(point_local, half_sizes):
    """Return the closest surface point and outward normal in box-local coordinates."""
    point_local = np.asarray(point_local, dtype=float)
    half_sizes = np.asarray(half_sizes, dtype=float)
    # Closest point on the box, including when the point is outside.
    surface_point = np.clip(point_local, -half_sizes, half_sizes)
    delta = point_local - surface_point
    distance = np.linalg.norm(delta)
    if distance > 1e-10:
        # Point is outside the box.
        normal = delta / distance
        return surface_point, normal, distance
    # Point is inside or on the box: select the nearest face.
    face_distances = half_sizes - np.abs(point_local)
    axis = int(np.argmin(face_distances))
    normal = np.zeros(3)
    normal[axis] = 1.0 if point_local[axis] >= 0 else -1.0
    surface_point = point_local.copy()
    surface_point[axis] = normal[axis] * half_sizes[axis]
    distance = float(face_distances[axis])
    return surface_point, normal, distance
def apply_gaussian_field(
    trajectory,
    geom_position,
    geom_rotation,
    half_sizes,
    clearance=0.10,
    sigma=0.20,
    amplitude=0.05,
    influence_distance=0.50,
):
    """
    Apply a proximity-dependent Gaussian repulsive field.
    All distances are in meters.
    The hard projection enforces clearance at every processed waypoint.
    """
    corrected = trajectory.copy()
    for i, point_world in enumerate(trajectory):
        # Transform world point into box-local coordinates.
        point_local = geom_rotation.T @ (point_world - geom_position)
        surface_local, normal_local, distance = (
            closest_box_surface_point(point_local, half_sizes)
        )
        if distance < clearance:
            # Hard constraint: move to at least the required clearance.
            corrected_local = surface_local + normal_local * clearance
        elif distance < influence_distance:
            # Gaussian repulsion decreases with distance.
            weight = np.exp(
                -(distance ** 2) / (2.0 * sigma ** 2)
            )
            displacement = amplitude * weight
            corrected_local = point_local + normal_local * displacement
            # Numerical safety check after applying the field.
            new_surface, new_normal, new_distance = (
                closest_box_surface_point(corrected_local, half_sizes)
            )
            if new_distance < clearance:
                corrected_local = (
                    new_surface + new_normal * clearance
                )
        else:
            corrected_local = point_local
        # Transform corrected point back into world coordinates.
        corrected[i] = (
            geom_position + geom_rotation @ corrected_local
        )
    return corrected
def measure_clearance(
    points,
    geom_position,
    geom_rotation,
    half_sizes,
):
    """Measure distance from every point to the box surface."""
    distances = []
    for point in points:
        local = geom_rotation.T @ (point - geom_position)
        _, _, distance = closest_box_surface_point(local, half_sizes)
        distances.append(distance)
    return np.asarray(distances)
def main():
    parser = argparse.ArgumentParser(
        description="Apply a Gaussian repulsive field to an EE trajectory."
    )
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--trajectory_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--geom_name", default="cuboid_obstacle_geom")
    parser.add_argument("--x_col", default="x")
    parser.add_argument("--y_col", default="y")
    parser.add_argument("--z_col", default="z")
    parser.add_argument("--clearance", type=float, default=0.10)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument("--influence_distance", type=float, default=0.50)
    args = parser.parse_args()
    if not args.xml.is_file():
        parser.error(f"XML not found: {args.xml}")
    if not args.trajectory_csv.is_file():
        parser.error(f"Trajectory CSV not found: {args.trajectory_csv}")
    if args.clearance <= 0 or args.sigma <= 0:
        parser.error("Clearance and sigma must be positive.")
    if args.amplitude < 0 or args.influence_distance <= 0:
        parser.error("Amplitude must be nonnegative and influence distance positive.")
    # Load the MuJoCo scene and locate the cuboid geom.
    model = mujoco.MjModel.from_xml_path(
        str(args.xml.expanduser().resolve())
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        args.geom_name,
    )
    if geom_id == -1:
        raise ValueError(
            f"Geom '{args.geom_name}' not found in the XML."
        )
    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError(
            f"Geom '{args.geom_name}' is not a box."
        )
    geom_position = data.geom_xpos[geom_id].copy()
    geom_rotation = data.geom_xmat[geom_id].reshape(3, 3).copy()
    half_sizes = model.geom_size[geom_id, :3].copy()
    # Read the original trajectory.
    df = pd.read_csv(args.trajectory_csv)
    coordinate_columns = [args.x_col, args.y_col, args.z_col]
    missing = [c for c in coordinate_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Coordinate columns not found: {missing}. "
            f"Available columns: {list(df.columns)}. "
            "Pass --x_col, --y_col, and --z_col to match your CSV."
        )
    original = df[coordinate_columns].to_numpy(dtype=float)
    if not np.isfinite(original).all():
        raise ValueError("Trajectory coordinates contain NaN or infinite values.")
    original_clearance = measure_clearance(
        original, geom_position, geom_rotation, half_sizes
    )
    corrected = apply_gaussian_field(
        trajectory=original,
        geom_position=geom_position,
        geom_rotation=geom_rotation,
        half_sizes=half_sizes,
        clearance=args.clearance,
        sigma=args.sigma,
        amplitude=args.amplitude,
        influence_distance=args.influence_distance,
    )
    corrected_clearance = measure_clearance(
        corrected, geom_position, geom_rotation, half_sizes
    )
    df[coordinate_columns] = corrected
    output_path = args.output_csv.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved corrected trajectory to:\n{output_path}")
    print(f"Waypoints processed: {len(original)}")
    print(f"Original minimum clearance: {original_clearance.min():.4f} m")
    print(f"Corrected minimum waypoint clearance: {corrected_clearance.min():.4f} m")
    print(f"Target clearance: {args.clearance:.4f} m")
    print(f"Sigma: {args.sigma:.3f} m")
    print(f"Maximum Gaussian displacement parameter: {args.amplitude:.3f} m")
    if corrected_clearance.min() < args.clearance - 1e-7:
        print("WARNING: Some waypoints remain below target clearance.")
    else:
        print("All processed waypoints meet the clearance constraint.")
    print(
        "\nNote: This validates waypoint clearance, not the full path "
        "between consecutive waypoints or the robot's entire body."
    )
if __name__ == "__main__":
    main()
