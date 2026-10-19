#!/usr/bin/env python3
"""
Keypoint-based obstacle avoidance for a generalized FR3 trajectory.
Inputs:
    --trajectory_csv : generalized trajectory CSV
    --keypoints_csv  : obstacle keypoints CSV
Output:
    A modified trajectory that keeps the end-effector target outside
    an axis-aligned obstacle bounding box expanded by the requested
    clearance.
No MuJoCo XML or geometry-name lookup is required.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an obstacle-avoiding trajectory using keypoints."
    )
    parser.add_argument(
        "--trajectory_csv",
        type=Path,
        required=True,
        help="Input generalized trajectory CSV.",
    )
    parser.add_argument(
        "--keypoints_csv",
        type=Path,
        required=True,
        help="Obstacle keypoints CSV with object, point_id, x, y, z.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path(
            "output/Generalized_trajectory_avoiding_obstacle.csv"
        ),
    )
    parser.add_argument("--x_col", default="target_x")
    parser.add_argument("--y_col", default="target_y")
    parser.add_argument("--z_col", default="target_z")
    parser.add_argument(
        "--clearance",
        type=float,
        default=0.10,
        help="Desired minimum distance from the estimated box surface (m).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.20,
        help="Gaussian smoothing sigma in waypoint units.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.05,
        help="Maximum additional smoothing displacement (m).",
    )
    parser.add_argument(
        "--influence_distance",
        type=float,
        default=0.50,
        help="Distance around the obstacle where avoidance may be applied (m).",
    )
    return parser.parse_args()
def load_obstacle_keypoints(path):
    if not path.is_file():
        raise FileNotFoundError(f"Keypoints CSV not found: {path}")
    points = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"x", "y", "z"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Obstacle keypoints CSV must contain x, y, and z columns."
            )
        for row in reader:
            if "object" in row and row["object"] != "obstacle":
                continue
            xyz = np.array(
                [float(row["x"]), float(row["y"]), float(row["z"])],
                dtype=float,
            )
            if not np.isfinite(xyz).all():
                raise ValueError("Non-finite obstacle keypoint detected.")
            points.append(xyz)
    if len(points) < 4:
        raise ValueError(
            f"Expected multiple obstacle keypoints; found {len(points)}."
        )
    return np.asarray(points)
def load_trajectory(path, columns):
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory CSV not found: {path}")
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("Trajectory CSV is empty or has no header.")
        missing = [c for c in columns if c not in fieldnames]
        if missing:
            raise ValueError(
                f"Trajectory CSV is missing columns: {missing}. "
                f"Available columns: {fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("Trajectory CSV contains no waypoints.")
    trajectory = np.array(
        [[float(row[c]) for c in columns] for row in rows],
        dtype=float,
    )
    if not np.isfinite(trajectory).all():
        raise ValueError("Trajectory contains non-finite coordinates.")
    return fieldnames, rows, trajectory
def estimate_box(keypoints):
    """
    Estimate the axis-aligned bounding box from the obstacle keypoints.
    The method is appropriate for the axis-aligned cuboid in the supplied
    scene. For a rotated cuboid, an oriented bounding box is needed.
    """
    lower = np.min(keypoints, axis=0)
    upper = np.max(keypoints, axis=0)
    if np.any(upper <= lower):
        raise ValueError(
            "Invalid obstacle bounds. Check that the keypoints span "
            "all three dimensions of the obstacle."
        )
    return lower, upper
def closest_box_surface(point, lower, upper):
    """
    Return the nearest point on the box surface and its outward direction.
    For points outside the box, use the nearest point on the box.
    For points inside the box, choose the nearest face to exit through.
    """
    closest = np.clip(point, lower, upper)
    delta = point - closest
    distance = np.linalg.norm(delta)
    if distance > 1e-12:
        return closest, delta / distance, distance
    # The point is inside the box or on its surface.
    distances_to_faces = np.concatenate(
        [point - lower, upper - point]
    )
    face_index = int(np.argmin(distances_to_faces))
    direction = np.zeros(3)
    if face_index < 3:
        axis = face_index
        direction[axis] = -1.0
    else:
        axis = face_index - 3
        direction[axis] = 1.0
    surface = point.copy()
    surface[axis] = (
        lower[axis] if face_index < 3 else upper[axis]
    )
    return surface, direction, 0.0
def project_outside_expanded_box(point, lower, upper, clearance):
    """
    Project a point outside the box expanded by clearance on each axis.
    This expanded-box constraint is conservative around corners.
    """
    expanded_lower = lower - clearance
    expanded_upper = upper + clearance
    if np.all(point >= expanded_lower) and np.all(point <= expanded_upper):
        distances_to_faces = np.concatenate(
            [
                point - expanded_lower,
                expanded_upper - point,
            ]
        )
        face_index = int(np.argmin(distances_to_faces))
        result = point.copy()
        if face_index < 3:
            result[face_index] = expanded_lower[face_index] - 1e-6
        else:
            axis = face_index - 3
            result[axis] = expanded_upper[axis] + 1e-6
        return result
    return point.copy()
def minimum_box_surface_distance(trajectory, lower, upper):
    distances = []
    for point in trajectory:
        closest = np.clip(point, lower, upper)
        distances.append(np.linalg.norm(point - closest))
    return float(np.min(distances))
def smooth_displacements(displacements, sigma):
    """
    Gaussian-smooth displacement vectors without requiring SciPy.
    Sigma is measured in waypoint indices.
    """
    if sigma <= 0 or len(displacements) < 2:
        return displacements.copy()
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    result = np.zeros_like(displacements)
    for axis in range(displacements.shape[1]):
        padded = np.pad(
            displacements[:, axis],
            (radius, radius),
            mode="edge",
        )
        result[:, axis] = np.convolve(
            padded, kernel, mode="valid"
        )
    return result
def main():
    args = parse_args()
    if args.clearance < 0:
        raise ValueError("--clearance must be non-negative.")
    if args.sigma < 0:
        raise ValueError("--sigma must be non-negative.")
    if args.amplitude < 0:
        raise ValueError("--amplitude must be non-negative.")
    if args.influence_distance < 0:
        raise ValueError("--influence_distance must be non-negative.")
    columns = [args.x_col, args.y_col, args.z_col]
    print(f"Loading trajectory: {args.trajectory_csv}")
    fieldnames, rows, trajectory = load_trajectory(
        args.trajectory_csv, columns
    )
    print(f"Loading obstacle keypoints: {args.keypoints_csv}")
    keypoints = load_obstacle_keypoints(args.keypoints_csv)
    lower, upper = estimate_box(keypoints)
    print(f"Loaded {len(keypoints)} obstacle keypoints.")
    print(f"Estimated obstacle lower bounds: {lower}")
    print(f"Estimated obstacle upper bounds: {upper}")
    before = minimum_box_surface_distance(
        trajectory, lower, upper
    )
    # First, find the minimum correction needed to satisfy the
    # clearance constraint at each waypoint.
    projected = np.vstack(
        [
            project_outside_expanded_box(
                point, lower, upper, args.clearance
            )
            for point in trajectory
        ]
    )
    raw_displacements = projected - trajectory
    # Smooth the correction so consecutive waypoints change more
    # gradually. The smoothed trajectory is projected again below
    # to retain the requested conservative box clearance.
    smoothed = smooth_displacements(
        raw_displacements, args.sigma
    )
    # Limit the extra smoothing displacement, not the mandatory
    # clearance correction.
    extra = smoothed - raw_displacements
    extra_norms = np.linalg.norm(extra, axis=1)
    too_large = extra_norms > args.amplitude
    if args.amplitude == 0:
        extra[:] = 0.0
    elif np.any(too_large):
        extra[too_large] *= (
            args.amplitude / extra_norms[too_large]
        )[:, None]
    candidate = trajectory + raw_displacements + extra
    # Only allow additional smoothing corrections near the obstacle.
    # The direct projection remains active wherever needed.
    if args.influence_distance > 0:
        for i, point in enumerate(candidate):
            closest = np.clip(point, lower, upper)
            distance = np.linalg.norm(point - closest)
            if distance > args.clearance + args.influence_distance:
                candidate[i] = projected[i]
    # Re-project to preserve the clearance constraint after smoothing.
    final_trajectory = np.vstack(
        [
            project_outside_expanded_box(
                point, lower, upper, args.clearance
            )
            for point in candidate
        ]
    )
    after = minimum_box_surface_distance(
        final_trajectory, lower, upper
    )
    # Preserve every original column and its ordering.
    for i, row in enumerate(rows):
        for j, col in enumerate(columns):
            row[col] = f"{final_trajectory[i, j]:.9f}"
    output_path = args.output_csv.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"Waypoints processed: {len(final_trajectory)}")
    print(f"Minimum surface distance before: {before:.4f} m")
    print(f"Minimum surface distance after:  {after:.4f} m")
    print(f"Requested clearance:             {args.clearance:.4f} m")
    print(f"Saved trajectory: {output_path}")
    if after + 1e-5 < args.clearance:
        print(
            "WARNING: The measured clearance is below the requested "
            "value. Inspect the obstacle bounds and output trajectory."
        )
    else:
        print("The end-effector waypoints satisfy the estimated box clearance.")
    print(
        "NOTE: This checks the end-effector target points only. "
        "It does not certify full-arm collision avoidance or dynamic "
        "feasibility."
    )
if __name__ == "__main__":
    main()
