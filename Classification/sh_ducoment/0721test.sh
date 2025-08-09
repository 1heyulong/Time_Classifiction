#!/bin/bash

# 基础参数设置
PYTHON_SCRIPT="/Time_Classifiction/Classification/shiyanTSLANet_0715_0703.py"
DATA_PATH="/hy-tmp/0712_realdata/"
EPOCHS=500

# 实验1：基础模型结构对照
echo "Running baseline architecture experiments..."
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB False --ASB False --name exp1_baseline \
    --dropout_rate 0.15 --depth 2 --patch_size 60

python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB True --ASB False --name exp1_ICB_only \
    --dropout_rate 0.15 --depth 2 --patch_size 60

python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB False --ASB True --name exp1_ASB_only \
    --dropout_rate 0.15 --depth 2 --patch_size 60

python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB True --ASB True --name exp1_full_model \
    --dropout_rate 0.15 --depth 2 --patch_size 60

# 实验2：多尺度时间块对照
echo "Running multi-scale temporal block experiments..."
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB True --ASB True --mstb False --name exp2_original_ICB \
    --dropout_rate 0.15 --depth 2 --patch_size 60

python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB True --ASB True --mstb True --name exp2_multi_scale \
    --dropout_rate 0.15 --depth 2 --patch_size 60





