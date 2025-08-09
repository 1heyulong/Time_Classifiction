#!/bin/bash

python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0805实验4_1_1 --patch_size 60 --depth 4 --num_epochs 1000

python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0805实验4_1_2 --patch_size 90 --num_epochs 1000

python /Time_Classifiction/Classification/TSLANetshiyan.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0805实验4_1_3 --patch_size 60 --num_epochs 1000