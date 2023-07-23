#!/bin/bash

PARAMETERS_BASE="--img 640 --epochs 50 --batch-size 64 --exist-ok --weights yolov5s.pt --hyp data/hyps/hyp.fine-bdd.yaml"

for D in "baseline" "clear" "daytime"  "night" "partly_cloudy" \
  "residential" "city_street" "dawn_dusk" "highway" "overcast" "rainy" "snowy"
do
    python train.py --data data/domains/${D}.yaml ${PARAMETERS_BASE} \
    --project domain --name ${D} 2>&1
done
