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
echo "Extracting keypoints from demonstration..."
python extract_keypoints.py \
    --demo_csv "$DEMO_CSV" \
    --output_csv "$KEYPOINTS_CSV"
echo "Experiment completed successfully."
echo "Output saved to: $KEYPOINTS_CSV"
