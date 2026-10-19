#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ENV_NAME="mujoco_policy_trans"
SCENE_XML="fr3_on_table.xml"
DEMO_CSV="demonstrations/Original_demonstration.csv"
CHANGED_DEMO_CSV="demonstrations/Changed_environment_surface_points.csv"
ORIGINAL_KEYPOINTS_CSV="keypoints/original_keypoints.csv"
TRANSFORMED_KEYPOINTS_CSV="keypoints/transformed_keypoints.csv"
OBSTACLE_KEYPOINTS_CSV="keypoints/obstacle_keypoints.csv"
TRANSPORTED_TRAJECTORY_CSV="output/Generalized_trajectory.csv"
AVOIDING_TRAJECTORY_CSV="output/Generalized_trajectory_avoiding_obstacle.csv"
source "/opt/anaconda3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
mkdir -p demonstrations keypoints output
echo "1) Collect changed-environment demonstration"
mjpython teleoperated_demonstration.py \
    --scene_file "$SCENE_XML" \
    --output_file "Changed_environment_surface_points.csv"
echo "2) Extract demonstration keypoints"
python3 extracts_keypoints.py \
    --demo_csv "$DEMO_CSV" \
    --output_csv "$ORIGINAL_KEYPOINTS_CSV"
python3 extracts_keypoints.py \
    --demo_csv "$CHANGED_DEMO_CSV" \
    --output_csv "$TRANSFORMED_KEYPOINTS_CSV"
echo "3) Extract obstacle keypoints"
python3 extract_obstacle_keypoints.py \
    --xml "assets/fr3_on_table.xml" \
    --output_csv "$OBSTACLE_KEYPOINTS_CSV"
echo "4) Make keypoint pairs"
python3 make_keypoints_pair.py
echo "5) Fit transport maps"
python3 fit_transport_maps.py
echo "6) Transform original trajectory"
python3 transport_trajectory.py
echo "7) Apply obstacle avoidance"
python3 fit_obstacle.py \
    --xml "assets/fr3_on_table.xml" \
    --trajectory_csv "$TRANSPORTED_TRAJECTORY_CSV" \
    --keypoints_csv "$OBSTACLE_KEYPOINTS_CSV" \
    --output_csv "$AVOIDING_TRAJECTORY_CSV" \
    --x_col target_x \
    --y_col target_y \
    --z_col target_z \
    --clearance 0.10 \
    --sigma 0.20 \
    --amplitude 0.05 \
    --influence_distance 0.50
echo "8) Visualize results"
python3 visualization.py
echo "9) Replay obstacle-avoiding trajectory"
mjpython replay_generalized.py
echo "Pipeline completed."