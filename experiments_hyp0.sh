#!/bin/bash

PARAMETERS_BASE="--img 640 --epochs 30 --batch-size 64 --exist-ok"
PARAMETERS_HYP0="${PARAMETERS_BASE} --data bdd100k.yaml"

for P in "yolov5n.pt" "yolov5s.pt" "yolov5m.pt" "yolov5l.pt"
do
    python train.py --weights ${P} ${PARAMETERS_HYP0} \
    --project hyp0 --name ${P} 2>&1 
done
