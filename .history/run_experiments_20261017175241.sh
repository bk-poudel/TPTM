#!/bin/bash
set -Eeuo pipefail
# ============================================================
# FR3 MuJoCo: Demonstration -> Keypoints -> Transport Maps
# -> Generalized Trajectory -> Obstacle Avoidance -> Replay
# ============================================================
PROJECT_DIR="/Users/bibek/TPTm"
cd "$PROJECT_DIR"
# Activate the project environment.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mujoco_policy_trans
# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DEMO_SCENE="fr3_on_table_original_demonstration.xml"
OBSTACLE_SCENE="assets/fr3_on_table.xml"
ORIGINAL_DEMO="demonstrations/Original_demonstration.csv"
CHANGED_DEMO="demonstrations/Changed_environment_surface_points.csv"
ORIGINAL_KEYPOINTS="keypoints/original_keypoints.csv"
TRANSFORMED_KEYPOINTS="keypoints/transformed_keypoints.csv"
OBSTACLE_KEYPOINTS="keypoints/obstacle_keypoints.csv"
KEYPOINT_PAIRS="keypoints/keypoints_pairs.csv"
TRANSPORT_MAPS="output/transport_maps.pkl"
GENERALIZED_TRAJECTORY="output/Generalized_trajectory.csv"
AVOIDING_TRAJECTORY="output/Generalized_trajectory_avoiding_obstacle.csv"
CLEARANCE="0.10"
SIGMA="0.20"
AMPLITUDE="0.05"
INFLUENCE_DISTANCE="0.50"
mkdir -p demonstrations keypoints output
# Stop immediately if a command fails.
trap 'echo ""; echo "ERROR: Pipeline failed at line $LINENO."; exit 1' ERR
echo "============================================================"
echo "FR3 MuJoCo experiment pipeline"
echo "Project: $PROJECT_DIR"
echo "Environment: mujoco_policy_trans"
echo "============================================================"
# ------------------------------------------------------------
# STEP 1: Collect a teleoperated demonstration
# ------------------------------------------------------------
echo ""
echo "STEP 1: Collect teleoperated demonstration"
echo "Scene: $DEMO_SCENE"
echo "IMPORTANT: Start teleoperation and save the demonstration."
echo "The output should be: $CHANGED_DEMO"
mjpython teleoperated_demonstration.py \
    --scene_file "$DEMO_SCENE" \
    --output_file "Changed_environment_surface_points.csv"
# The teleoperation script has previously saved the CSV under
# demonstrations/. Stop if the expected file is not present.
if [[ ! -f "$CHANGED_DEMO" ]]; then
    echo "ERROR: Expected demonstration not found: $CHANGED_DEMO"
    echo "Check where teleoperated_demonstration.py saved the CSV."
    exit 1
fi
# ------------------------------------------------------------
# STEP 2: Extract keypoints from demonstrations
# ------------------------------------------------------------
echo ""
echo "STEP 2: Extract demonstration keypoints"
python3 extracts_keypoints.py
# ------------------------------------------------------------
# STEP 3: Extract obstacle keypoints from the obstacle scene
# ------------------------------------------------------------
echo ""
echo "STEP 3: Extract obstacle keypoints"
python3 extract_obstacle_keypoints.py
# ------------------------------------------------------------
# STEP 4: Create source-target keypoint pairs
# ------------------------------------------------------------
echo ""
echo "STEP 4: Build keypoint pairs"
python3 make_keypoints_pair.py
# ------------------------------------------------------------
# STEP 5: Fit transport maps
# ------------------------------------------------------------
echo ""
echo "STEP 5: Fit transport maps"
python3 fit_transport_maps.py
# ------------------------------------------------------------
# STEP 6: Generate the generalized trajectory
# ------------------------------------------------------------
echo ""
echo "STEP 6: Generate generalized trajectory"
python3 transport_trajectory.py
# ------------------------------------------------------------
# STEP 7: Fit obstacle avoidance to the generalized trajectory
# NOTE: This matches the XML-based fit_obstacle.py CLI.
# ------------------------------------------------------------
echo ""
echo "STEP 7: Generate obstacle-avoiding trajectory"
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
# ------------------------------------------------------------
# STEP 8: Visualize results
# ------------------------------------------------------------
echo ""
echo "STEP 8: Visualize results"
python3 visualization.py
# ------------------------------------------------------------
# STEP 9: Replay the obstacle-avoiding trajectory
# ------------------------------------------------------------
echo ""
echo "STEP 9: Replay obstacle-avoiding trajectory"
echo "Trajectory: $AVOIDING_TRAJECTORY"
echo "Make sure replay_generalized.py loads:"
echo 'output/Generalized_trajectory_avoiding_obstacle.csv'
mjpython replay_generalized.py
echo ""
echo "============================================================"
echo "Experiment pipeline finished."
echo "============================================================"
