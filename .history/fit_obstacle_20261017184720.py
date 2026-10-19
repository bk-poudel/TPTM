#!/usr/bin/env python3
"""
Minimum Volume Enclosing Geometry for Trajectory Obstacle Avoidance
Fits a tight Minimum Volume Ellipsoid/Sphere around obstacle keypoints 
and deforms the trajectory using a smooth distance gradient field.
"""
import argparse
import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
def parse_args():
    parser = argparse.ArgumentParser(
        description="Deform trajectory using a Minimum Volume Enclosing obstacle model."
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
    parser.add_argument("--clearance", type=float, default=0.10)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument("--influence_distance", type=float, default=0.50)
    return parser.parse_args()
def fit_minimum_bounding_sphere(points):
    """Computes minimum volume bounding sphere (center and radius)."""
    center = np.mean(points, axis=0)
    # Solve for optimal center that minimizes max distance to points
    def objective(c):
        return np.max(np.linalg.norm(points - c, axis=1))
    res = minimize(objective, center, method="Nelder-Mead")
    optimal_center = res.x
    radius = float(res.fun)
    return optimal_center, radius
def fit_minimum_volume_ellipsoid(points, tol=1e-3):
    """
    Finds Khachiyan's Minimum Volume Enclosing Ellipsoid (MVEE) parameters:
    (x - c)^T A (x - c) <= 1
    Returns center 'c' and principal radii axes matrix.
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
    return center, A
def apply_mvee_repulsion(
    waypoints, center, A, clearance, sigma, amplitude, influence_distance
):
    """
    Applies repulsive field based on algebraic distance to the MVEE boundary.
    Ellipsoid surface equation: (x - c)^T A (x - c) = 1
    """
    deformed = waypoints.copy()
    # Eigendecomposition to extract principal axes radii
    vals, vecs = np.linalg.eigh(A)
    radii = 1.0 / np.sqrt(np.maximum(vals, 1e-8))
    for i in range(len(waypoints)):
        pt = waypoints[i]
        diff = pt - center
        # Normalized ellipsoidal radial distance (1.0 = on surface)
        ellipsoid_dist = np.sqrt(np.maximum(0.0, diff.T @ A @ diff))
        # Physical gradient direction (outward normal from ellipsoid surface)
        grad = 2.0 * (A @ diff)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-6:
            direction = np.array([0.0, 0.0, 1.0])
        else:
            direction = grad / grad_norm
        # Approximate physical distance from center along current direction
        est_physical_dist = np.linalg.norm(diff)
        if est_physical_dist >= influence_distance:
            continue
        # Gaussian smoothing weight
        surface_dist = max(0.0, est_physical_dist - np.min(radii))
        weight = np.exp(-(surface_dist**2) / (2.0 * (sigma**2)))
        # Required outward displacement
        if ellipsoid_dist < 1.0 + (clearance / np.mean(radii)):
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
    # Fit Minimum Volume Enclosing Geometry
    center, A = fit_minimum_volume_ellipsoid(kp_points)
    mbs_center, mbs_radius = fit_minimum_bounding_sphere(kp_points)
    print(f"Loaded {len(kp_points)} obstacle keypoints.")
    print(f"MVEE Center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"Minimum Bounding Sphere Radius: {mbs_radius:.4f} m")
    deformed_waypoints = apply_mvee_repulsion(
        waypoints=waypoints,
        center=center,
        A=A,
        clearance=args.clearance,
        sigma=args.sigma,
        amplitude=args.amplitude,
        influence_distance=args.influence_distance,
    )
    df_traj[args.x_col] = deformed_waypoints[:, 0]
    df_traj[args.y_col] = deformed_waypoints[:, 1]
    df_traj[args.z_col] = deformed_waypoints[:, 2]
    if "ee_x" in df_traj.columns:
        df_traj["ee_x"] = deformed_waypoints[:, 0]
        df_traj["ee_y"] = deformed_waypoints[:, 1]
        df_traj["ee_z"] = deformed_waypoints[:, 2]
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df_traj.to_csv(args.output_csv, index=False)
    print(f"Successfully saved minimum volume deformed path to: {args.output_csv}")
if __name__ == "__main__":
    main()