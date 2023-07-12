#!/bin/bash

PARAMETERS_BASE="--img 640"

for model in "baseline" "clear" "daytime" "night" "partly_cloudy" \
  "residential" "city_street" "dawn_dusk" "highway" "overcast" "rainy" "snowy"; do

  for data in "baseline" "clear" "daytime" "night" "partly_cloudy" \
    "residential" "city_street" "dawn_dusk" "highway" "overcast" "rainy" "snowy"; do

    python val.py --data data/domains/${data}.yaml ${PARAMETERS_BASE} --weights domain/${model}/weights/best.pt \
      --project domain_val --name ${model}.${data} 2>&1

  done

done

for model in "baseline" "clear" "daytime" "night" "partly_cloudy" \
  "residential" "city_street" "dawn_dusk" "highway" "overcast" "rainy" "snowy"; do

  for data in "baseline" "clear" "daytime" "night" "partly_cloudy" \
    "residential" "city_street" "dawn_dusk" "highway" "overcast" "rainy" "snowy"; do

    python val.py --data data/domains/${data}.yaml ${PARAMETERS_BASE} --weights split/${model}/weights/best.pt \
      --project domain_split_val --name ${model}.${data} 2>&1

  done

done
