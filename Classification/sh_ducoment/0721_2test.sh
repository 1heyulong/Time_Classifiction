#!/bin/bash

# 基础参数设置
PYTHON_SCRIPT="/Time_Classifiction/Classification/shiyanTSLANet_0715_0703.py"
DATA_PATH="/hy-tmp/0712_realdata/"
EPOCHS=500


# 实验6：模型深度对照
echo "Running model depth experiments..."
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB True --ASB True --depth 1 --name exp6_depth1 \
    --dropout_rate 0.15 --patch_size 60

python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB True --ASB True --depth 2 --name exp6_depth2 \
    --dropout_rate 0.15 --patch_size 60

python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB True --ASB True --depth 3 --name exp6_depth3 \
    --dropout_rate 0.15 --patch_size 60

# 实验7：patch size对照
echo "Running patch size experiments..."
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB True --ASB True --patch_size 30 --name exp7_patch30 \
    --dropout_rate 0.15 --depth 2

python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB True --ASB True --patch_size 60 --name exp7_patch60 \
    --dropout_rate 0.15 --depth 2

python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB True --ASB True --patch_size 90 --name exp7_patch90 \
    --dropout_rate 0.15 --depth 2

# 实验8：预训练策略对照
echo "Running pretraining experiments..."
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB True --ASB True --load_from_pretrained False --name exp8_no_pretrain \
    --dropout_rate 0.15 --depth 2 --patch_size 60

python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB True --ASB True --load_from_pretrained True --name exp8_with_pretrain \
    --dropout_rate 0.15 --depth 2 --patch_size 60 \
    --pretrain_epochs 50

echo "All experiments completed!"