#!/bin/bash
set -Eeuo pipefail
# ============================================================
# FR3 MuJoCo EXPERIMENT PIPELINE
# ============================================================
PROJECT_DIR="/Users/bibek/TPTm"
cd "$PROJECT_DIR"
# Do not activate Conda inside this script.
# Activate mujoco_policy_trans in your terminal before running.
if [[ "${CONDA_DEFAULT_ENV:-}" != "mujoco_policy_trans" ]]; then
    echo "ERROR: The mujoco_policy_trans environment is not active."
    echo "Run: conda activate mujoco_policy_trans"
    exit 1
fi
echo "============================================================"
echo "FR3 MuJoCo experiment pipeline"
echo "Project: $PROJECT_DIR"
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python3)"
echo "============================================================"
# ============================================================
# CONFIGURATION
# ============================================================
DEMO_SCENE="fr3_on_table_original_demonstration.xml"
OBSTACLE_SCENE="assets/fr3_on_table.xml"
ORIGINAL_DEMO="demonstrations/Original_demonstration.csv"
CHANGED_DEMO="demonstrations/Changed_environment_surface_points.csv"
ORIGINAL_KEYPOINTS="keypoints/original_keypoints.csv"
TRANSFORMED_KEYPOINTS="keypoints/transformed_keypoints.csv"
OBSTACLE_KEYPOINTS="keypoints/obstacle_keypoints.csv"
GENERALIZED_TRAJECTORY="output/Generalized_trajectory.csv"
AVOIDING_TRAJECTORY="output/Generalized_trajectory_avoiding_obstacle.csv"
CLEARANCE="0.10"
SIGMA="0.20"
AMPLITUDE="0.05"
INFLUENCE_DISTANCE="0.50"
mkdir -p demonstrations keypoints output
# Print the step that fails.
CURRENT_STEP="initialization"
trap 'status=$?; echo ""; echo "ERROR: ${CURRENT_STEP} failed (exit code ${status})."; exit "$status"' ERR
# ============================================================
# STEP 1: COLLECT TELEOPERATED DEMONSTRATION
# ============================================================
CURRENT_STEP="Step 1 - Teleoperated demonstration"
echo ""
echo "============================================================"
echo "$CURRENT_STEP"
echo "============================================================"
mjpython teleoperated_demonstration.py \
    --scene_file "$DEMO_SCENE" \
    --output_file "Changed_environment_surface_points.csv"
if [[ ! -f "$CHANGED_DEMO" ]]; then
    echo "ERROR: Demonstration CSV not found:"
    echo "$CHANGED_DEMO"
    echo "Check the actual output path printed by the teleoperation script."
    exit 1
fi
echo "Demonstration found: $CHANGED_DEMO"
# ============================================================
# STEP 2: EXTRACT ORIGINAL AND TRANSFORMED KEYPOINTS
# ============================================================
CURRENT_STEP="Step 2 - Demonstration keypoint extraction"
echo ""
echo "============================================================"
echo "$CURRENT_STEP"
echo "============================================================"
if [[ ! -f "$ORIGINAL_DEMO" ]]; then
    echo "ERROR: Original demonstration is missing: $ORIGINAL_DEMO"
    echo "Place your original demonstration CSV at this path."
    exit 1
fi
python3 extracts_keypoints.py \
    --demo_csv "$ORIGINAL_DEMO" \
    --output_csv "$ORIGINAL_KEYPOINTS"
python3 extracts_keypoints.py \
    --demo_csv "$CHANGED_DEMO" \
    --output_csv "$TRANSFORMED_KEYPOINTS"
# ============================================================
# STEP 3: EXTRACT OBSTACLE KEYPOINTS
# ============================================================
CURRENT_STEP="Step 3 - Obstacle keypoint extraction"
echo ""
echo "============================================================"
echo "$CURRENT_STEP"
echo "============================================================"
python3 extract_obstacle_keypoints.py
if [[ ! -f "$OBSTACLE_KEYPOINTS" ]]; then
    echo "ERROR: Obstacle keypoints were not created:"
    echo "$OBSTACLE_KEYPOINTS"
    exit 1
fi
# ============================================================
# STEP 4: CREATE KEYPOINT PAIRS
# ============================================================
CURRENT_STEP="Step 4 - Keypoint pairing"
echo ""
echo "============================================================"
echo "$CURRENT_STEP"
echo "============================================================"
python3 make_keypoints_pair.py
# ============================================================
# STEP 5: FIT TRANSPORT MAPS
# ============================================================
CURRENT_STEP="Step 5 - Fit transport maps"
echo ""
echo "============================================================"
echo "$CURRENT_STEP"
echo "============================================================"
python3 fit_transport_maps.py
# ============================================================
# STEP 6: GENERATE GENERALIZED TRAJECTORY
# ============================================================
CURRENT_STEP="Step 6 - Generate generalized trajectory"
echo ""
echo "============================================================"
echo "$CURRENT_STEP"
echo "============================================================"
python3 transport_trajectory.py
if [[ ! -f "$GENERALIZED_TRAJECTORY" ]]; then
    echo "ERROR: Generalized trajectory was not created:"
    echo "$GENERALIZED_TRAJECTORY"
    exit 1
fi
# ============================================================
# STEP 7: GENERATE OBSTACLE-AVOIDING TRAJECTORY
# Uses the XML-based fit_obstacle.py interface.
# ============================================================
CURRENT_STEP="Step 7 - Obstacle avoidance"
echo ""
echo "============================================================"
echo "$CURRENT_STEP"
echo "============================================================"
python3 fit_obstacle.py \
    --xml "$OBSTACLE_SCENE" \
    --trajectory_csv "$GENERALIZED_TRAJECTORY" \
    --keypoints_csv "$OBSTACLE_KEYPOINTS" \
    --output_csv "$AVOIDING_TRAJECTORY" \
    --geom_name "cuboid_obstacle_geom" \
    --x_col target_x \
    --y_col target_y \
    --z_col target_z \
    --clearance "$CLEARANCE" \
    --sigma "$SIGMA" \
    --amplitude "$AMPLITUDE" \
    --influence_distance "$INFLUENCE_DISTANCE"
if [[ ! -f "$AVOIDING_TRAJECTORY" ]]; then
    echo "ERROR: Obstacle-avoiding trajectory was not created."
    exit 1
fi
# ============================================================
# STEP 8: VISUALIZE RESULTS
# ============================================================
CURRENT_STEP="Step 8 - Visualization"
echo ""
echo "============================================================"
echo "$CURRENT_STEP"
echo "============================================================"
python3 visualization.py
# ============================================================
# STEP 9: REPLAY OBSTACLE-AVOIDING TRAJECTORY
# ============================================================
CURRENT_STEP="Step 9 - Replay obstacle-avoiding trajectory"
echo ""
echo "============================================================"
echo "$CURRENT_STEP"
echo "============================================================"
echo "Ensure replay_generalized.py loads:"
echo "$AVOIDING_TRAJECTORY"
mjpython replay_generalized.py
# ============================================================
# COMPLETE
# ============================================================
echo ""
echo "============================================================"
echo "PIPELINE FINISHED"
echo "============================================================"
echo "Original keypoints:      $ORIGINAL_KEYPOINTS"
echo "Transformed keypoints:   $TRANSFORMED_KEYPOINTS"
echo "Obstacle keypoints:      $OBSTACLE_KEYPOINTS"
echo "Generalized trajectory:  $GENERALIZED_TRAJECTORY"
echo "Avoiding trajectory:     $AVOIDING_TRAJECTORY"
echo ""
