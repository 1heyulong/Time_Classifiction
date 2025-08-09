#!/bin/bash

# 设置Python脚本路径
PYTHON_SCRIPT="/Time_Classifiction/Classification/shiyanTSLANet_0703_1.py"
DATA_PATH="/hy-tmp/0712_realdata/"
EPOCHS=500


python $PYTHON_SCRIPT --dropout_rate 0.15 --depth 2 --patch_size 60 --masking_ratio 0.5 --name 0714_exp2a

python $PYTHON_SCRIPT --dropout_rate 0.15 --depth 2 --patch_size 30 --mstb True --name 0714_exp2b

python $PYTHON_SCRIPT --dropout_rate 0.3 --depth 1 --emb_dim 96 --patch_size 90 --loss_rate 0.15 --name 0714_exp3a


#!/bin/bash
for dropout in 0.2 0.3 0.4; do
  for depth in 1 2 3; do
    python $PYTHON_SCRIPT --dropout_rate $dropout --depth $depth --name "0714_tune_d${depth}_do${dropout}"
  done
done

python /Time_Classifiction/Classification/shiyanTSLANet_0703_1.py --dropout_rate 0.3 --depth 1 --emb_dim 96 --patch_size 90 --loss_rate 0.15 --name 0714_exp3a_1000 --num_epochs 1000