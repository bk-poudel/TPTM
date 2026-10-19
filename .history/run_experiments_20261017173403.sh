#!/bin/bash
set -euo pipefail
# Navigate to the project directory
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
ENV_NAME="mujoco_policy_trans"
DEMO_CSV="demonstrations/Original_demonstration.csv"
CHANGED_DEMO_CSV="demonstrations/Changed_environment_surface_points.csv"
ORIGINAL_KEYPOINTS_CSV="keypoints/original_keypoints.csv"
TRANSFORMED_KEYPOINTS_CSV="keypoints/transformed_keypoints.csv"
OBSTACLE_KEYPOINTS_CSV="keypoints/obstacle_keypoints.csv"
# Update this path if transport_trajectory.py saves to a different location.
TRANSPORTED_TRAJECTORY_CSV="output/transformed_trajectory.csv"
OBSTACLE_AVOIDING_CSV="output/transformed_trajectory_avoiding_obstacle.csv"
# --------------------------------------------------
# ACTIVATE CONDA ENVIRONMENT
# --------------------------------------------------
source "/opt/anaconda3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
# --------------------------------------------------
# RUN EXPERIMENT
# --------------------------------------------------
echo "1) Making Changes for the Scene File"
mjpython teleoperated_demonstration.py \
    --scene_file "fr3_on_table.xml" \
    --output_file "$CHANGED_DEMO_CSV"
echo "2) Extract Keypoints from Demonstrations"
python3 extracts_keypoints.py \
    --demo_csv "$DEMO_CSV" \
    --output_csv "$ORIGINAL_KEYPOINTS_CSV"
python3 extracts_keypoints.py \
    --demo_csv "$CHANGED_DEMO_CSV" \
    --output_csv "$TRANSFORMED_KEYPOINTS_CSV"
echo "3) Extract Obstacle Keypoints from MuJoCo XML"
python3 extract_obstacle_keypoints.py \
    --xml "assets/fr3_on_table.xml" \
    --output_csv "$OBSTACLE_KEYPOINTS_CSV"
echo "4) Make Keypoint Pairs"
python3 make_keypoints_pair.py
echo "5) Fit Transform and RBF"
python3 fit_transport_maps.py
echo "6) Transform the Original Trajectory"
python3 transport_trajectory.py
echo "7) Apply Gaussian Obstacle Avoidance"
python3 fit_obstacle.py \
    --trajectory_csv "$TRANSPORTED_TRAJECTORY_CSV" \
    --keypoints_csv "$OBSTACLE_KEYPOINTS_CSV" \
    --output_csv "$OBSTACLE_AVOIDING_CSV" \
    --clearance 0.10 \
    --sigma 0.20 \
    --amplitude 0.05 \
    --influence_distance 0.50
echo "8) Visualize the Results"
python3 visualization.py
echo "9) Visualize in MuJoCo"
# Configure replay_generalized.py to load OBSTACLE_AVOIDING_CSV
# if you want to replay the obstacle-avoiding trajectory.
mjpython replay_generalized.py
echo "Pipeline completed."
