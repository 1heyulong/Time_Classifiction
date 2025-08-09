#!/bin/bash

# 设置Python脚本路径
PYTHON_SCRIPT="/Time_Classifiction/Classification/shiyanTSLANet_0703_1.py"
DATA_PATH="/hy-tmp/0712_realdata/"
EPOCHS=500


python $PYTHON_SCRIPT --dropout_rate 0.3 --depth 1 --emb_dim 96 --patch_size 75 --loss_rate 0.15 --name 0714_best_guess


python $PYTHON_SCRIPT --dropout_rate 0.5 --depth 2 --emb_dim 96 --patch_size 90 --name 0714_exp1a

python $PYTHON_SCRIPT --dropout_rate 0.2 --depth 1 --emb_dim 64 --patch_size 90 --name 0714_exp1b

#python /Time_Classifiction/Classification/shiyanTSLANet_0703_1.py --dropout_rate 0.2 --depth 1 --emb_dim 64 --patch_size 90 --name 0714_exp1b

# python /Time_Classifiction/Classification/shiyanTSLANet_0703_1.py --dropout_rate 0.15 --depth 4 --data_path /hy-tmp/0712_realdata/ --emb_dim 128 --patch_size 60 --num_epochs 300 --name 0714_exp1b_0730
# python /Time_Classifiction/Classification/shiyanTSLANet_0703_1.py --dropout_rate 0.15 --depth 4 --emb_dim 64 --patch_size 90 --num_epochs 300 --data_path /hy-tmp/0712_realdata/ --name 0714_exp1b_0730_1

# python /Time_Classifiction/Classification/shiyanTSLANet_0703_1.py --dropout_rate 0.3 --depth 4 --emb_dim 128 --patch_size 60 --num_epochs 300 --data_path /hy-tmp/0712_realdata/ --name 0714_exp1b_0730_2

# python /Time_Classifiction/Classification/0801TSLANet.py --dropout_rate 0.15 --depth 4 --emb_dim 128 --patch_size 60 --num_epochs 300 --data_path /hy-tmp/0712_realdata/ --name 0714_exp1b_0801_1

# python /Time_Classifiction/Classification/0801TSLANet.py --dropout_rate 0.15 --depth 4 --emb_dim 128 --patch_size 90 --num_epochs 300 --data_path /hy-tmp/0712_realdata/ --name 0714_exp1b_0801_2

# python /Time_Classifiction/Classification/0801TSLANet.py --dropout_rate 0.15 --depth 2 --emb_dim 128 --patch_size 60 --num_epochs 300 --data_path /hy-tmp/0712_realdata/ --name 0714_exp1b_0801_3
