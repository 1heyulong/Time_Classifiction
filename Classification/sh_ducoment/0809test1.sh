#!/bin/bash

python /Time_Classifiction/Classification/0809TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name A1_基线 --patch_size 60 --num_epochs 1000

# A2_时域增强
python /Time_Classifiction/Classification/0809TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name A2_时域增强 --patch_size 60 --num_epochs 1000 --use_temporal_aug True --temporal_shift 0.2

# A3_频域增强
python /Time_Classifiction/Classification/0809TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name A3_频域增强 --patch_size 60 --num_epochs 1000 --use_freq_aug True --freq_jitter_strength 0.2

# A4_MixUp增强
python /Time_Classifiction/Classification/0809TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name A4_MixUp增强 --patch_size 60 --num_epochs 1000 --use_mixup True --mixup_alpha 0.3

# A5_对比学习
python /Time_Classifiction/Classification/0809TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name A5_对比学习 --patch_size 60 --num_epochs 1000 --use_contrastive True --contrastive_weight 0.2 --contrastive_temp 0.2
