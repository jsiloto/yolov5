#!/bin/bash

PARAMETERS_BASE="--img 640 --epochs 15 --batch-size 64 --weights yolov5s.pt --exist-ok"
PARAMETERS_HYP0="${PARAMETERS_BASE} --data bdd100k.yaml"

for P in "02" "04" "06" "10" "20"
do
    python train.py --cfg models/split/yolov5s_split.${P}.yaml ${PARAMETERS_HYP0} \
    --project hyp0 --name ${P} 2>&1 
done
