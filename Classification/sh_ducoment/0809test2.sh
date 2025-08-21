#!/bin/bash

# B1_基线结构
python /Time_Classifiction/Classification/0809TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name B1_基线结构 --patch_size 90 --num_epochs 1000

# B2_AttnPooling
python /Time_Classifiction/Classification/0809TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name B2_AttnPooling --patch_size 90 --num_epochs 1000 --use_attn_pool True

# B3_高Dropout
python /Time_Classifiction/Classification/0809TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name B3_高Dropout --patch_size 90 --num_epochs 1000 --dropout 0.3

# B4_小Depth多尺度
python /Time_Classifiction/Classification/0809TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name B4_小Depth多尺度 --patch_size 90 --num_epochs 1000 --depth 3 --multi_scale True
