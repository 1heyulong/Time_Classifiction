#!/bin/bash

# C1_plateau
python /Time_Classifiction/Classification/0809TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name C1_plateau --patch_size 90 --num_epochs 1000 --scheduler plateau

# C2_cosine
python /Time_Classifiction/Classification/0809TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name C2_cosine --patch_size 90 --num_epochs 1000 --scheduler cosine

# C3_onecycle
python /Time_Classifiction/Classification/0809TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name C3_onecycle --patch_size 90 --num_epochs 1000 --scheduler onecycle --max_lr 1e-3

# C4_focal_loss
python /Time_Classifiction/Classification/0809TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name C4_focal_loss --patch_size 90 --num_epochs 1000 --use_focal True --focal_gamma 2.0
