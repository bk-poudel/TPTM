#!/bin/bash
set -e
# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
ENV_NAME="mujoco_policy_trans"
DEMO_CSV="demonstrations/Original_demonstration.csv"
KEYPOINTS_CSV="demonstrations/transformed_keypoints.csv"
# --------------------------------------------------
# ACTIVATE CONDA ENVIRONMENT
# --------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
# --------------------------------------------------
# RUN EXPERIMENT
# --------------------------------------------------
echo "1) Running Demonstration"
mjpython teleopetaed_demonstration.py --scene_file "fr3_on_table_original_demonstration.xml"
