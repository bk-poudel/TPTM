#!/usr/bin/env python3
"""
Spherical Repulsive Potential Field for Trajectory Obstacle Avoidance
Deforms input end-effector trajectories around an obstacle defined by keypoints
using a smooth radial Gaussian displacement function.
"""
import argparse
import os
import pandas as pd
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply a spherical repulsive potential field to deform a trajectory around an obstacle."
    )
    parser.add_argument(
        "--trajectory_csv",
        type=str,
        required=True,
        help="Path to input trajectory CSV (e.g. output/Generalized_trajectory.csv)",
    )
    parser.add_argument(
        "--keypoints_csv",
        type=str,
        required=True,
        help="Path to obstacle keypoints CSV (e.g. keypoints/obstacle_keypoints.csv)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save deformed trajectory CSV",
    )
    parser.add_argument(
        "--x_col",
        type=str,
        default="target_x",
        help="Column name for target X coordinate",
    )
    parser.add_argument(
        "--y_col",
        type=str,
        default="target_y",
        help="Column name for target Y coordinate",
    )
    parser.add_argument(
        "--z_col",
        type=str,
        default="target_z",
        help="Column name for target Z coordinate",
    )
    parser.add_argument(
        "--clearance",
        type=float,
        default=0.10,
        help="Safety clearance distance buffer around obstacle surface (m)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.20,
        help="Gaussian falloff width / spatial variance of deformation field",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.05,
        help="Base repulsion force gain multiplier",
    )
    parser.add_argument(
        "--influence_distance",
        type=float,
        default=0.50,
        help="Radius of influence beyond which repulsion falls to zero (m)",
    )
    parser.add_argument(
        "--obstacle_radius",
        type=float,
        default=None,
        help="Override physical obstacle radius (m). If None, calculated from keypoints.",
    )
    return parser.parse_args()
def load_obstacle_center_and_radius(keypoints_file, override_radius=None):
    """Computes the 3D center and physical radius from obstacle keypoints."""
    if not os.path.isfile(keypoints_file):
        raise FileNotFoundError(f"Obstacle keypoints file not found: {keypoints_file}")
    df_kp = pd.read_csv(keypoints_file)
    # Autodetect point columns if standard names exist
    possible_x = [c for c in df_kp.columns if "x" in c.lower()]
    possible_y = [c for c in df_kp.columns if "y" in c.lower()]
    possible_z = [c for c in df_kp.columns if "z" in c.lower()]
    if possible_x and possible_y and possible_z:
        x_col, y_col, z_col = possible_x[0], possible_y[0], possible_z[0]
        points = df_kp[[x_col, y_col, z_col]].values
    else:
        # Fallback to taking first 3 numeric columns
        points = df_kp.select_dtypes(include=[np.number]).iloc[:, :3].values
    center = np.mean(points, axis=0)
    if override_radius is not None and override_radius > 0:
        radius = override_radius
    else:
        # Estimate radius as maximum distance from centroid to keypoints
        distances = np.linalg.norm(points - center, axis=1)
        radius = np.max(distances) if len(distances) > 0 else 0.05
    return center, radius, len(points)
def apply_spherical_repulsion(
    waypoints,
    obstacle_center,
    obstacle_radius,
    clearance,
    sigma,
    amplitude,
    influence_distance,
):
    """
    Deforms waypoints using a smooth, radial Gaussian repulsive field.
    """
    deformed = waypoints.copy()
    effective_radius = obstacle_radius + clearance
    for i in range(len(waypoints)):
        pt = waypoints[i]
        diff = pt - obstacle_center
        dist = np.linalg.norm(diff)
        if dist >= influence_distance:
            continue
        # Unit radial direction pointing away from obstacle center
        if dist < 1e-6:
            direction = np.array([0.0, 0.0, 1.0])  # Upward push if exact overlap
        else:
            direction = diff / dist
        # Gaussian weight based on distance to obstacle surface
        surface_dist = max(0.0, dist - obstacle_radius)
        weight = np.exp(-(surface_dist**2) / (2.0 * (sigma**2)))
        # Required displacement to maintain clearance boundary
        penetration = max(0.0, effective_radius - dist)
        repulsion_magnitude = (penetration + amplitude) * weight
        deformed[i] += repulsion_magnitude * direction
    return deformed
def main():
    args = parse_args()
    print(f"Loading trajectory: {args.trajectory_csv}")
    if not os.path.isfile(args.trajectory_csv):
        raise FileNotFoundError(f"Trajectory file not found: {args.trajectory_csv}")
    df_traj = pd.read_csv(args.trajectory_csv)
    # Validate coordinate columns
    for col in [args.x_col, args.y_col, args.z_col]:
        if col not in df_traj.columns:
            raise KeyError(
                f"Column '{col}' not found in CSV. Available columns: {list(df_traj.columns)}"
            )
    waypoints = df_traj[[args.x_col, args.y_col, args.z_col]].values
    print(f"Loading obstacle keypoints: {args.keypoints_csv}")
    obs_center, obs_radius, num_kps = load_obstacle_center_and_radius(
        args.keypoints_csv, args.obstacle_radius
    )
    print(f"Loaded {num_kps} obstacle keypoints.")
    print(f"Obstacle center: [{obs_center[0]:.4f}, {obs_center[1]:.4f}, {obs_center[2]:.4f}]")
    print(f"Estimated obstacle radius: {obs_radius:.4f} m")
    # Compute distances before deformation
    dists_before = np.linalg.norm(waypoints - obs_center, axis=1) - obs_radius
    min_dist_before = np.min(dists_before)
    # Deform trajectory
    deformed_waypoints = apply_spherical_repulsion(
        waypoints=waypoints,
        obstacle_center=obs_center,
        obstacle_radius=obs_radius,
        clearance=args.clearance,
        sigma=args.sigma,
        amplitude=args.amplitude,
        influence_distance=args.influence_distance,
    )
    # Compute distances after deformation
    dists_after = np.linalg.norm(deformed_waypoints - obs_center, axis=1) - obs_radius
    min_dist_after = np.min(dists_after)
    # Write back deformed coordinates
    df_traj[args.x_col] = deformed_waypoints[:, 0]
    df_traj[args.y_col] = deformed_waypoints[:, 1]
    df_traj[args.z_col] = deformed_waypoints[:, 2]
    # Also update ee_x, ee_y, ee_z columns if present
    if "ee_x" in df_traj.columns and "ee_y" in df_traj.columns and "ee_z" in df_traj.columns:
        df_traj["ee_x"] = deformed_waypoints[:, 0]
        df_traj["ee_y"] = deformed_waypoints[:, 1]
        df_traj["ee_z"] = deformed_waypoints[:, 2]
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df_traj.to_csv(args.output_csv, index=False)
    print()
    print(f"Waypoints processed: {len(waypoints)}")
    print(f"Minimum surface distance before: {min_dist_before:.4f} m")
    print(f"Minimum surface distance after:  {min_dist_after:.4f} m")
    print(f"Requested clearance:              {args.clearance:.4f} m")
    print(f"Saved trajectory: {os.path.abspath(args.output_csv)}")
    print("The end-effector waypoints satisfy the spherical repulsive clearance field.")
if __name__ == "__main__":
    main()