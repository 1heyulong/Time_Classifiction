#!/bin/bash

for K in 2 3 4; do
  for L in 0.3 0.5 0.7; do
    for M in cos xcorr; do
      python /Time_Classifiction/Classification/0825TSLANet.py \
        --data_path /hy-tmp/0716_realdata/ \
        --name "K${K}_L${L}_${M}" \
        --ICB True --ASB False \
        --use_shapelet_head True \
        --shapelet_per_class ${K} \
        --shapelet_len_tokens 7 \
        --shapelet_metric ${M} \
        --num_epochs 1000 \
        --shapelet_fuse_lambda ${L}
    done
  done
done
