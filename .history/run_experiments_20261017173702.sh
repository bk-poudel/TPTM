#!/bin/bash
set -euo pipefail
# =========================================================
# PROJECT DIRECTORY
# =========================================================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# =========================================================
# CONFIGURATION
# =========================================================
ENV_NAME="mujoco_policy_trans"
SCENE_XML="assets/fr3_on_table.xml"
DEMO_CSV="demonstrations/Original_demonstration.csv"
CHANGED_DEMO_CSV="demonstrations/Changed_environment_surface_points.csv"
ORIGINAL_KEYPOINTS_CSV="keypoints/original_keypoints.csv"
TRANSFORMED_KEYPOINTS_CSV="keypoints/transformed_keypoints.csv"
OBSTACLE_KEYPOINTS_CSV="keypoints/obstacle_keypoints.csv"
TRANSPORTED_TRAJECTORY_CSV="output/Generalized_trajectory.csv"
AVOIDING_TRAJECTORY_CSV="output/Generalized_trajectory_avoiding_obstacle.csv"
# Obstacle avoidance settings (metres)
CLEARANCE="0.10"
SIGMA="0.20"
AMPLITUDE="0.05"
INFLUENCE_DISTANCE="0.50"
# =========================================================
# ACTIVATE CONDA ENVIRONMENT
# =========================================================
source "/opt/anaconda3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
# =========================================================
# CREATE OUTPUT DIRECTORIES
# =========================================================
mkdir -p demonstrations keypoints output
# =========================================================
# 1. COLLECT CHANGED-ENVIRONMENT KEYPOINT DEMONSTRATION
# =========================================================
echo
echo "========================================================="
echo "1) Making Changes for the Scene File"
echo "========================================================="
mjpython teleoperated_demonstration.py \
    --scene_file "$SCENE_XML" \
    --output_file "Changed_environment_surface_points.csv"
# Verify that the demonstration was created.
if [[ ! -f "$CHANGED_DEMO_CSV" ]]; then
    echo "ERROR: Changed-environment demonstration was not created:"
    echo "  $CHANGED_DEMO_CSV"
    echo "Check how teleoperated_demonstration.py constructs CSV_PATH."
    exit 1
fi
# =========================================================
# 2. EXTRACT DEMONSTRATION KEYPOINTS
# =========================================================
echo
echo "========================================================="
echo "2) Extracting Demonstration Keypoints"
echo "========================================================="
python3 extracts_keypoints.py \
    --demo_csv "$DEMO_CSV" \
    --output_csv "$ORIGINAL_KEYPOINTS_CSV"
python3 extracts_keypoints.py \
    --demo_csv "$CHANGED_DEMO_CSV" \
    --output_csv "$TRANSFORMED_KEYPOINTS_CSV"
# =========================================================
# 3. EXTRACT OBSTACLE KEYPOINTS FROM M U J O C O XML
# =========================================================
echo
echo "========================================================="
echo "3) Extracting Obstacle Keypoints"
echo "========================================================="
python3 extract_obstacle_keypoints.py \
    --xml "$SCENE_XML" \
    --output_csv "$OBSTACLE_KEYPOINTS_CSV"
# =========================================================
# 4. MAKE KEYPOINT PAIRS
# =========================================================
echo
echo "========================================================="
echo "4) Making Keypoint Pairs"
echo "========================================================="
python3 make_keypoints_pair.py
# =========================================================
# 5. FIT TRANSFORM AND RBF
# =========================================================
echo
echo "========================================================="
echo "5) Fitting Transform and RBF"
echo "========================================================="
python3 fit_transport_maps.py
# =========================================================
# 6. TRANSFORM ORIGINAL TRAJECTORY
# =========================================================
echo
echo "========================================================="
echo "6) Transforming Original Trajectory"
echo "========================================================="
python3 transport_trajectory.py
if [[ ! -f "$TRANSPORTED_TRAJECTORY_CSV" ]]; then
    echo "ERROR: Transported trajectory was not created:"
    echo "  $TRANSPORTED_TRAJECTORY_CSV"
    exit 1
fi
# =========================================================
# 7. APPLY GAUSSIAN OBSTACLE AVOIDANCE
# =========================================================
echo
echo "========================================================="
echo "7) Applying Gaussian Obstacle Avoidance"
echo "========================================================="
python3 fit_obstacle.py \
    --xml "$SCENE_XML" \
    --trajectory_csv "$TRANSPORTED_TRAJECTORY_CSV" \
    --keypoints_csv "$OBSTACLE_KEYPOINTS_CSV" \
    --output_csv "$AVOIDING_TRAJECTORY_CSV" \
    --x_col target_x \
    --y_col target_y \
    --z_col target_z \
    --clearance "$CLEARANCE" \
    --sigma "$SIGMA" \
    --amplitude "$AMPLITUDE" \
    --influence_distance "$INFLUENCE_DISTANCE"
if [[ ! -f "$AVOIDING_TRAJECTORY_CSV" ]]; then
    echo "ERROR: Obstacle-avoiding trajectory was not created:"
    echo "  $AVOIDING_TRAJECTORY_CSV"
    exit 1
fi
# =========================================================
# 8. VISUALIZE RESULTS
# =========================================================
echo
echo "========================================================="
echo "8) Visualizing Results"
echo "========================================================="
python3 visualization.py
# =========================================================
# 9. REPLAY OBSTACLE-AVOIDING TRAJECTORY IN MUJOCO
# =========================================================
echo
echo "========================================================="
echo "9) Replaying Obstacle-Avoiding Trajectory"
echo "========================================================="
mjpython replay_generalized.py
echo
echo "========================================================="
echo "Pipeline completed."
echo "Obstacle-avoiding trajectory:"
echo "  $AVOIDING_TRAJECTORY_CSV"
echo "========================================================="
