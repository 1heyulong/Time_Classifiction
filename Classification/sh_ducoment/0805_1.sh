#!/bin/bash

python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0805实验1_1 --patch_size 60 --num_epochs 1000 --depth 4

python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0805实验1_2 --patch_size 60 --num_epochs 1000

python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0805实验1_3 --patch_size 90 --num_epochs 1000