#!/bin/bash

PARAMETERS_BASE="--img 640 --epochs 30 --batch-size 64 --weights yolov5s.pt --exist-ok"
PARAMETERS_HYP1="${PARAMETERS_BASE} --data bdd100khyp1.yaml"

for P in "02" "04" "06" "10"
do
    python train.py --cfg models/split/yolov5s_split.${P}.yaml ${PARAMETERS_HYP1} \
    --project hyp1 --name ${P} 2>&1
done