#!/bin/bash

PARAMETERS_BASE="--img 640 --epochs 30 --batch-size 64 --weights yolov5s.pt --exist-ok --data bdd100k_no.yaml"

python train.py ${PARAMETERS_BASE} --project hyp_no --name regular 2>&1

for P in "02" "04" "06" "10"; do
  python train.py --cfg models/split/yolov5s_split.${P}.yaml ${PARAMETERS_BASE} \
    --project hyp_no --name ${P} 2>&1
done
