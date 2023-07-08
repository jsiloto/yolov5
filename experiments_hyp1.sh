#!/bin/bash

PARAMETERS_BASE="--img 640 --epochs 30 --batch-size 64 --exist-ok"
PARAMETERS_HYP1="${PARAMETERS_BASE} --data bdd100khyp1.yaml"

for P in "yolov5n.pt" "yolov5s.pt" "yolov5m.pt" "yolov5l.pt"
do
    python train.py --weights ${P} ${PARAMETERS_HYP1} \
    --project hyp1 --name ${P} 2>&1
done
