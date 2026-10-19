
#!/bin/bash
set -euo pipefail

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="mujoco_policy_trans"

# Scene used to collect the demonstration
DEMONSTRATION_SCENE="assets/fr3_on_table_original_demonstration.xml"

# Scene containing the obstacle geometry
OBSTACLE_SCENE="assets/fr3_on_table.xml"

# Demonstration CSV files
ORIGINAL_DEMO="demonstrations/Original_demonstration.csv"
CHANGED_DEMO="demonstrations/Changed_environment_surface_points.csv"

# Keypoint files
ORIGINAL_KEYPOINTS="keypoints/original_keypoints.csv"
TRANSFORMED_KEYPOINTS="keypoints/transformed_keypoints.csv"
OBSTACLE_KEYPOINTS="keypoints/obstacle_keypoints.csv"

# Trajectory files
TRANSPORTED_TRAJECTORY="output/Generalized_trajectory.csv"
AVOIDING_TRAJECTORY="output/Generalized_trajectory_avoiding_obstacle.csv"

# Obstacle avoidance parameters
CLEARANCE="0.10"
SIGMA="0.20"
AMPLITUDE="0.05"
INFLUENCE_DISTANCE="0.50"

# ============================================================
# ACTIVATE CONDA ENVIRONMENT
# ============================================================

if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda.sh."
    echo "Please update the Conda initialization path in this script."
    exit 1
fi

conda activate "$ENV_NAME"

# ============================================================
# CHECK INPUT FILES
# ============================================================

mkdir -p demonstrations keypoints output

if [ ! -f "$DEMONSTRATION_SCENE" ]; then
    echo "ERROR: Demonstration scene not found: $DEMONSTRATION_SCENE"
    exit 1
fi

if [ ! -f "$OBSTACLE_SCENE" ]; then
    echo "ERROR: Obstacle scene not found: $OBSTACLE_SCENE"
    exit 1
fi

if [ ! -f "$ORIGINAL_DEMO" ]; then
    echo "ERROR: Original demonstration not found: $ORIGINAL_DEMO"
    exit 1
fi

echo "============================================================"
echo "MuJoCo trajectory generalization and obstacle avoidance"
echo "Project directory: $SCRIPT_DIR"
echo "Demonstration scene: $DEMONSTRATION_SCENE"
echo "Obstacle scene: $OBSTACLE_SCENE"
echo "============================================================"

# ============================================================
# STEP 1: COLLECT DEMONSTRATION
# ============================================================

echo ""
echo "STEP 1: Collect demonstration in the original demonstration scene"
echo "Output: $CHANGED_DEMO"

# The script is expected to resolve the scene filename inside assets/
# and the output filename inside demonstrations/.
mjpython teleoperated_demonstration.py \
    --scene_file "fr3_on_table_original_demonstration.xml" \
    --output_file "Changed_environment_surface_points.csv"

if [ ! -f "$CHANGED_DEMO" ]; then
    echo "ERROR: Demonstration CSV was not created: $CHANGED_DEMO"
    exit 1
fi

# ============================================================
# STEP 2: EXTRACT DEMONSTRATION KEYPOINTS
# ============================================================

echo ""
echo "STEP 2: Extract original demonstration keypoints"

python3 extracts_keypoints.py \
    --demo_csv "$ORIGINAL_DEMO" \
    --output_csv "$ORIGINAL_KEYPOINTS"

echo ""
echo "Extract keypoints from the newly collected demonstration"

python3 extracts_keypoints.py \
    --demo_csv "$CHANGED_DEMO" \
    --output_csv "$TRANSFORMED_KEYPOINTS"

# ============================================================
# STEP 3: EXTRACT OBSTACLE KEYPOINTS
# ============================================================

echo ""
echo "STEP 3: Extract obstacle keypoints from the obstacle scene"

python3 extract_obstacle_keypoints.py \
    --xml "$OBSTACLE_SCENE" \
    --output_csv "$OBSTACLE_KEYPOINTS"

# ============================================================
# STEP 4: CREATE KEYPOINT PAIRS
# ============================================================

echo ""
echo "STEP 4: Create keypoint pairs"

python3 make_keypoints_pair.py

# ============================================================
# STEP 5: FIT TRANSPORT MAPS
# ============================================================

echo ""
echo "STEP 5: Fit transport maps"

python3 fit_transport_maps.py

# ============================================================
# STEP 6: TRANSFORM THE ORIGINAL TRAJECTORY
# ============================================================

echo ""
echo "STEP 6: Generate generalized trajectory"

python3 transport_trajectory.py

if [ ! -f "$TRANSPORTED_TRAJECTORY" ]; then
    echo "ERROR: Generalized trajectory was not created: $TRANSPORTED_TRAJECTORY"
    exit 1
fi

# ============================================================
# STEP 7: APPLY OBSTACLE AVOIDANCE
# ============================================================

echo ""
echo "STEP 7: Generate obstacle-avoiding trajectory"

python3 fit_obstacle.py \
    --xml "$OBSTACLE_SCENE" \
    --trajectory_csv "$TRANSPORTED_TRAJECTORY" \
    --keypoints_csv "$OBSTACLE_KEYPOINTS" \
    --output_csv "$AVOIDING_TRAJECTORY" \
    --x_col target_x \
    --y_col target_y \
    --z_col target_z \
    --clearance "$CLEARANCE" \
    --sigma "$SIGMA" \
    --amplitude "$AMPLITUDE" \
    --influence_distance "$INFLUENCE_DISTANCE"

if [ ! -f "$AVOIDING_TRAJECTORY" ]; then
    echo "ERROR: Obstacle-avoiding trajectory was not created: $AVOIDING_TRAJECTORY"
    exit 1
fi

# ============================================================
# STEP 8: VISUALIZE RESULTS
# ============================================================

echo ""
echo "STEP 8: Visualize trajectories"

python3 visualization.py

# ============================================================
# STEP 9: REPLAY OBSTACLE-AVOIDING TRAJECTORY
# ============================================================

echo ""
echo "STEP 9: Replay obstacle-avoiding trajectory in MuJoCo"
echo "Make sure replay_generalized.py uses:"
echo "output/Generalized_trajectory_avoiding_obstacle.csv"

mjpython replay_generalized.py

# ============================================================
# COMPLETE
# ============================================================

echo ""
echo "============================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "Generated trajectory: $TRANSPORTED_TRAJECTORY"
echo "Obstacle-avoiding trajectory: $AVOIDING_TRAJECTORY"
echo "============================================================"
