#!/bin/bash

PARAMETERS_BASE="--img 640 --epochs 15 --batch-size 64 --weights yolov5s.pt --exist-ok"
PARAMETERS_HYP0="${PARAMETERS_BASE} --data bdd100k.yaml"
PARAMETERS_HYP1="${PARAMETERS_BASE} --data bdd100kno9.yaml"


for P in 0 1 2 3 4 5
do
    python train.py --cfg models/split/yolov5s_split.${P}.yaml ${PARAMETERS_HYP0} \
    --project hyp0 --name ${P} 2>&1 | tee -a hyp0/log${P}.txt
done

for P in 0 1 2 3 4 5
do
    python train.py --cfg models/split/yolov5s_split.${P}.yaml ${PARAMETERS_HYP1} \
    --project hyp1 --name ${P} 2>&1 | tee -a hyp1/log${P}.txt
done