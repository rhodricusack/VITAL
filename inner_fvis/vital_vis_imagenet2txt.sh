#!/bin/bash
#SBATCH --job-name=vital_vis
#SBATCH --output=vital_vis/vital_vis_%j.out
#SBATCH --error=vital_vis/vital_vis_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=debug

# Initialize conda for bash shell
source /opt/anaconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate vital

# Run the Python script
python imagenet2txt_blurry.py --arch $ARCH --target_layer ${TARGET_LAYER}
    
