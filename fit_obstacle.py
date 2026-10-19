#!/usr/bin/env python3
"""
Loose-Fitting Enclosing Geometry for Trajectory Obstacle Avoidance
Fits an expanded Minimum Volume Enclosing Ellipsoid around keypoints
to create a generous, smooth safety envelope for the robot.
"""
import argparse
import os
import pandas as pd
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(
        description="Deform trajectory using a loose-fitting expanded obstacle geometry."
    )
    parser.add_argument(
        "--trajectory_csv",
        type=str,
        required=True,
        help="Path to input trajectory CSV",
    )
    parser.add_argument(
        "--keypoints_csv",
        type=str,
        required=True,
        help="Path to obstacle keypoints CSV",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save deformed trajectory CSV",
    )
    parser.add_argument("--x_col", type=str, default="target_x")
    parser.add_argument("--y_col", type=str, default="target_y")
    parser.add_argument("--z_col", type=str, default="target_z")
    parser.add_argument(
        "--expansion_factor",
        type=float,
        default=1.70,
        help="Expansion factor for loose-fit (e.g., 1.30 = 30%% larger boundary)",
    )
    parser.add_argument("--clearance", type=float, default=0.10)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument("--influence_distance", type=float, default=0.60)
    return parser.parse_args()
def fit_loose_mvee(points, expansion_factor=1.30, tol=1e-3):
    """
    Fits Khachiyan's MVEE and scales matrix A down by (1 / expansion_factor^2)
    to yield a loose-fitting, expanded enclosing boundary.
    """
    N, d = points.shape
    Q = np.vstack([points.T, np.ones(N)])
    u = np.ones(N) / N
    err = 1.0
    while err > tol:
        X = Q @ np.diag(u) @ Q.T
        M = np.diag(Q.T @ np.linalg.inv(X) @ Q)
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1) / ((d + 1) * (maximum - 1))
        new_u = (1 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u
    center = points.T @ u
    A = (
        np.linalg.inv(points.T @ np.diag(u) @ points - np.outer(center, center))
        / d
    )
    # Scale matrix A to expand the physical boundary (loose fit)
    A_loose = A / (expansion_factor**2)
    return center, A_loose
def apply_loose_repulsion(
    waypoints, center, A_loose, clearance, sigma, amplitude, influence_distance
):
    deformed = waypoints.copy()
    vals, _ = np.linalg.eigh(A_loose)
    mean_radius = np.mean(1.0 / np.sqrt(np.maximum(vals, 1e-8)))
    for i in range(len(waypoints)):
        pt = waypoints[i]
        diff = pt - center
        # Algebraic distance relative to expanded boundary (<= 1.0 means inside)
        ellipsoid_dist = np.sqrt(np.maximum(0.0, diff.T @ A_loose @ diff))
        # Outward gradient vector
        grad = 2.0 * (A_loose @ diff)
        grad_norm = np.linalg.norm(grad)
        direction = grad / grad_norm if grad_norm > 1e-6 else np.array([0.0, 0.0, 1.0])
        dist_to_center = np.linalg.norm(diff)
        if dist_to_center >= influence_distance:
            continue
        # Smooth falloff based on proximity
        surface_dist = max(0.0, dist_to_center - mean_radius)
        weight = np.exp(-(surface_dist**2) / (2.0 * (sigma**2)))
        if ellipsoid_dist < 1.0 + (clearance / mean_radius):
            penetration = clearance * (1.0 + (1.0 - ellipsoid_dist))
            displacement = (penetration + amplitude) * weight * direction
            deformed[i] += displacement
    return deformed
def main():
    args = parse_args()
    df_traj = pd.read_csv(args.trajectory_csv)
    waypoints = df_traj[[args.x_col, args.y_col, args.z_col]].values
    df_kp = pd.read_csv(args.keypoints_csv)
    possible_x = [c for c in df_kp.columns if "x" in c.lower()]
    possible_y = [c for c in df_kp.columns if "y" in c.lower()]
    possible_z = [c for c in df_kp.columns if "z" in c.lower()]
    if possible_x and possible_y and possible_z:
        kp_points = df_kp[[possible_x[0], possible_y[0], possible_z[0]]].values
    else:
        kp_points = df_kp.select_dtypes(include=[np.number]).iloc[:, :3].values
    center, A_loose = fit_loose_mvee(
        kp_points, expansion_factor=args.expansion_factor
    )
    print(f"Loaded {len(kp_points)} obstacle keypoints.")
    print(f"Loose-fit center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"Expansion Factor applied: {args.expansion_factor}x")
    deformed = apply_loose_repulsion(
        waypoints=waypoints,
        center=center,
        A_loose=A_loose,
        clearance=args.clearance,
        sigma=args.sigma,
        amplitude=args.amplitude,
        influence_distance=args.influence_distance,
    )
    df_traj[args.x_col] = deformed[:, 0]
    df_traj[args.y_col] = deformed[:, 1]
    df_traj[args.z_col] = deformed[:, 2]
    if "ee_x" in df_traj.columns:
        df_traj["ee_x"] = deformed[:, 0]
        df_traj["ee_y"] = deformed[:, 1]
        df_traj["ee_z"] = deformed[:, 2]
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df_traj.to_csv(args.output_csv, index=False)
    print(f"Saved loose-fit trajectory to: {args.output_csv}")
if __name__ == "__main__":
    main()