#!/bin/bash
#SBATCH --job-name=vital_recon
#SBATCH --output=vital_vis/vital_recon_%j.out
#SBATCH --error=vital_vis/vital_recon_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=debug

# Initialize conda for bash shell
source /opt/anaconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate vital

export MKL_THREADING_LAYER=GNU

# Print job information
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Architecture: $ARCH"
echo "Target Layer: $TARGET_LAYER"
echo "Channel: $CHANNEL"
echo "Started: $(date)"
echo "=================================================="

# Run the reconstruction script
python reverse_engineer_neurons.py \
    --arch $ARCH \
    --target_layer $TARGET_LAYER \
    --chs $CHANNEL

# Print completion
echo "=================================================="
echo "Completed: $(date)"
echo "=================================================="
