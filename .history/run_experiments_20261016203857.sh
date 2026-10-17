#!/bin/bash
set -e
set -euo pipefail
# Navigate to the project directory
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
ENV_NAME="mujoco_policy_trans"
DEMO_CSV="demonstrations/Original_demonstration.csv"
KEYPOINTS_CSV="demonstrations/transformed_keypoints.csv"
# --------------------------------------------------
# ACTIVATE CONDA ENVIRONMENT
# --------------------------------------------------
source "/opt/anaconda3/etc/profile.d/conda.sh"
conda activate mujoco_policy_trans
# --------------------------------------------------
# RUN EXPERIMENT
# --------------------------------------------------
echo "1) Running Demonstration"
mjpython teleoperated_demonstration.py --scene_file "fr3_on_table_original_demonstration.xml" --output_file "Original_demonstration.csv"
echo "2) Making Changes for the Scene File"
mjpython teleoperated_demonstration.py --scene_file "fr3_on_table_original.xml" --output_file "Changed_environment_keypoints.csv"
echo "3) Extract Keypoints from demonstrations and changed environment"
python3 extracts_keypoints.py --demo_csv "demonstrations/Original_demonstration.csv" --output_csv "./keypoints/original_keypoints.csv"
python3 python3 extracts_keypoints.py --demo_csv "demonstrations/Changed_environment_keypoints.csv" --output_csv "./keypoints/transformed_keypoints.csv"
echo "4) Make Keyoints Pair"
python3 make_keypoints_pair.py
echo "5) Fit Transform and RBF"
python3 fit_transport_maps.py
echo "6) Transform the Original Trajectory"
python3 transport_trajectory.py
echo "7) Visualize the Results"
python3 visualization.py
echo "8) Visualize in Mujoco"
python3 replay_generalized.py
