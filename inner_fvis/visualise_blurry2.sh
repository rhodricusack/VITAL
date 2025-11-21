#!/bin/bash
TARGET_LAYER=layer2.2
conda activate vital

# List of architectures from modeldict keys in txt2patch.py
ARCHS=(

    "resnet50_BV_g6For60_E60"  
    "resnet50_BV_g3For60_E60" 
    "resnet50_BV_g0pt5For60_E60"
    "resnet50_BV_g1For60_E60"
    "resnet50_BV_g1pt5For60_E60"
    "resnet50_BV_g2For60_E60"
    "resnet50_BV_g4For60_E60"
    "resnet50_BV_g0For60_E60"
)

for ARCH in "${ARCHS[@]}"; do
    echo "Processing architecture: $ARCH"
    
    # Submit imagenet2txt job and capture job ID
    JOB_OUTPUT=$(sbatch --parsable --export=TARGET_LAYER=$TARGET_LAYER,ARCH=$ARCH vital_vis_imagenet2txt.sh)
    IMAGENET2TXT_JOB_ID=$JOB_OUTPUT
    echo "  Submitted imagenet2txt job: $IMAGENET2TXT_JOB_ID"
    
    # Submit txt2patch jobs that depend on imagenet2txt completion
    for CHANNEL in {0..1024..16}; do
        echo "  Processing channel $CHANNEL (depends on job $IMAGENET2TXT_JOB_ID)"
        sbatch --dependency=afterok:$IMAGENET2TXT_JOB_ID --export=TARGET_LAYER=$TARGET_LAYER,ARCH=$ARCH,CHANNEL=$CHANNEL vital_vis.sh
    done
    #python reverse_engineer_neurons.py --arch $ARCH --target_layer ${TARGET_LAYER}
done


