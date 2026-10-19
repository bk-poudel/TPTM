#!/usr/bin/env bash
set -Eeuo pipefail
# ============================================================
# FR3 MUJOCO: DEMONSTRATION -> KEYPOINTS -> GENERALIZATION
# -> OBSTACLE AVOIDANCE -> REPLAY
# ============================================================
# Always run from the directory containing this script.
cd "$(dirname "${BASH_SOURCE[0]}")"
BASE_DIR="$(pwd)"
# ----------------------------
# Configuration
# ----------------------------
ENV_NAME="mujoco_policy_trans"
ORIGINAL_XML="assets/fr3_on_table_original_demonstration.xml"
CHANGED_XML="assets/fr3_on_table.xml"
ORIGINAL_DEMO="demonstrations/Original_demonstration.csv"
CHANGED_DEMO="demonstrations/Changed_environment_surface_points.csv"
ORIGINAL_KEYPOINTS="keypoints/original_keypoints.csv"
CHANGED_KEYPOINTS="keypoints/transformed_keypoints.csv"
OBSTACLE_KEYPOINTS="keypoints/obstacle_keypoints.csv"
KEYPOINT_PAIRS="keypoints/keypoints_pairs.csv"
GENERALIZED_TRAJECTORY="output/Generalized_trajectory.csv"
OBSTACLE_TRAJECTORY="output/Generalized_trajectory_avoiding_obstacle.csv"
# Obstacle-avoidance parameters
CLEARANCE="0.20"
SIGMA="0.20"
AMPLITUDE="0.05"
INFLUENCE_DISTANCE="0.50"
# ----------------------------
# Helpers
# ----------------------------
step() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}
require_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: Required file not found: $1" >&2
        exit 1
    fi
}
# ----------------------------
# Check environment
# ----------------------------
step "Checking Conda environment"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
    echo "ERROR: Conda environment '$ENV_NAME' is not active."
    echo
    echo "Activate it first:"
    echo "  conda activate $ENV_NAME"
    echo "Then run:"
    echo "  bash run_experiments.sh"
    exit 1
fi
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Project directory: $BASE_DIR"
# ----------------------------
# Check project files
# ----------------------------
step "Checking required scripts and scene files"
for file in \
    "$ORIGINAL_XML" \
    "$CHANGED_XML" \
    "teleoperated_demonstration.py" \
    "extracts_keypoints.py" \
    "make_keypoints_pair.py" \
    "fit_transport_maps.py" \
    "transport_trajectory.py" \
    "extract_obstacle_keypoints.py" \
    "fit_obstacle.py" \
    "replay_generalized.py"
do
    require_file "$file"
done
mkdir -p demonstrations keypoints output
# ============================================================
# STEP 1: COLLECT ORIGINAL DEMONSTRATION
# ============================================================
step "STEP 1: Original demonstration"
echo "Launching teleoperation using the original scene."
echo "Save the demonstration to: $ORIGINAL_DEMO"
echo
python3 teleoperated_demonstration.py \
    --scene_file "$ORIGINAL_XML" \
    --output_file "$ORIGINAL_DEMO"
require_file "$ORIGINAL_DEMO"
# ============================================================
# STEP 2: COLLECT CHANGED-ENVIRONMENT DEMONSTRATION
# ============================================================
step "STEP 2: Changed-environment demonstration"
if [[ ! -f "$CHANGED_DEMO" ]]; then
    echo "Launching teleoperation using the changed scene."
    echo "Save the demonstration to: $CHANGED_DEMO"
    echo
    echo "IMPORTANT: Ensure the teleoperation script accepts"
    echo "the appropriate XML and output CSV arguments."
    python3 teleoperated_demonstration.py \
        --xml "$CHANGED_XML" \
        --output_csv "$CHANGED_DEMO"
else
    echo "Using existing changed-environment demonstration:"
    echo "$CHANGED_DEMO"
    echo "Delete it manually if you want to recollect it."
fi
require_file "$CHANGED_DEMO"
# ============================================================
# STEP 3: EXTRACT KEYPOINTS FROM SCENE XML FILES
# ============================================================
step "STEP 3: Extract original scene keypoints"
python3 extracts_keypoints.py \
    --xml "$ORIGINAL_XML" \
    --output_csv "$ORIGINAL_KEYPOINTS"
require_file "$ORIGINAL_KEYPOINTS"
step "STEP 4: Extract changed scene keypoints"
python3 extracts_keypoints.py \
    --xml "$CHANGED_XML" \
    --output_csv "$CHANGED_KEYPOINTS"
require_file "$CHANGED_KEYPOINTS"
# ============================================================
# STEP 5: PAIR CORRESPONDING KEYPOINTS
# ============================================================
step "STEP 5: Create source-target keypoint pairs"
python3 make_keypoints_pair.py
require_file "$KEYPOINT_PAIRS"
# ============================================================
# STEP 6: FIT TRANSPORT MAPS
# ============================================================
step "STEP 6: Fit transport maps"
python3 fit_transport_maps.py
# ============================================================
# STEP 7: GENERALIZE THE TRAJECTORY
# ============================================================
step "STEP 7: Generate generalized trajectory"
python3 transport_trajectory.py
require_file "$GENERALIZED_TRAJECTORY"
# ============================================================
# STEP 8: EXTRACT OBSTACLE KEYPOINTS
# ============================================================
step "STEP 8: Extract obstacle keypoints"
echo "Make sure extract_obstacle_keypoints.py uses the changed"
echo "scene XML and writes the CSV schema expected by fit_obstacle.py."
python3 extract_obstacle_keypoints.py \
    --xml "$CHANGED_XML" \
    --output_csv "$OBSTACLE_KEYPOINTS"
require_file "$OBSTACLE_KEYPOINTS"
# ============================================================
# STEP 9: FIT OBSTACLE AVOIDANCE
# ============================================================
# STEP 9: Generate obstacle-avoiding trajectory
step "STEP 9: Generate obstacle-avoiding trajectory"
python3 fit_obstacle.py \
    --trajectory_csv output/Generalized_trajectory.csv \
    --keypoints_csv keypoints/obstacle_keypoints.csv \
    --output_csv output/Generalized_trajectory_avoiding_obstacle.csv \
    --x_col target_x \
    --y_col target_y \
    --z_col target_z \
    --clearance 0.25 \
    --sigma 0.20 \
    --amplitude 0.05 \
    --influence_distance 0.50
require_file "$OBSTACLE_TRAJECTORY"
# ============================================================
# STEP 10: REPLAY THE GENERATED TRAJECTORY
# ============================================================
step "STEP 10: Replay obstacle-avoiding trajectory"
echo "Before continuing, replay_generalized.py must point to:"
echo "$CHANGED_XML"
echo "$OBSTACLE_TRAJECTORY"
echo
echo "If the script uses hardcoded paths, update XML_PATH and"
echo "TRAJECTORY_PATH in replay_generalized.py before this step."
mjpython replay_generalized.py
# ============================================================
# FINISHED
# ============================================================
step "PIPELINE FINISHED"
echo "Generated files:"
echo "  Original keypoints:       $ORIGINAL_KEYPOINTS"
echo "  Changed keypoints:        $CHANGED_KEYPOINTS"
echo "  Keypoint pairs:           $KEYPOINT_PAIRS"
echo "  Generalized trajectory:   $GENERALIZED_TRAJECTORY"
echo "  Obstacle keypoints:       $OBSTACLE_KEYPOINTS"
echo "  Avoiding trajectory:      $OBSTACLE_TRAJECTORY"
