#!/bin/bash

PARAMETERS_BASE="--img 640 --epochs 15 --batch-size 64 --weights yolov5s.pt --exist-ok"
PARAMETERS_HYP2="${PARAMETERS_BASE} --data bdd100k.yaml"

for P in "02" "04" "06" "10" "20"
do
    python train.py --cfg models/split/yolov5s_split.${P}.yaml ${PARAMETERS_HYP2} \
    --project hyp2 --name ${P} 2>&1
done



PARAMETERS_BASE="--img 640 --epochs 30 --batch-size 64 --exist-ok"
PARAMETERS_HYP2="${PARAMETERS_BASE} --data bdd100khyp2.yaml"

for P in "yolov5n.pt" "yolov5s.pt" "yolov5m.pt" "yolov5l.pt"
do
    python train.py --weights ${P} ${PARAMETERS_HYP2} \
    --project hyp2 --name ${P} 2>&1
done
