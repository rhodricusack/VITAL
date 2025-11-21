#!/bin/bash
ARCH=resnet50
TARGET_LAYER=layer4.2
export CUDA_VISIBLE_DEVICES=3
conda activate vital
python imagenet2txt_blurry.py       --arch $ARCH --target_layer ${TARGET_LAYER}
for CHANNEL in {0..15}; do
    echo "Processing channel $CHANNEL"
    python txt2patch.py  --arch $ARCH --target_layer ${TARGET_LAYER} --channel ${CHANNEL}
done

python reverse_engineer_neurons.py  --arch $ARCH --target_layer ${TARGET_LAYER}      


