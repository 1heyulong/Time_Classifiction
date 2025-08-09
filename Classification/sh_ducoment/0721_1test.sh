#!/bin/bash

# 基础参数设置
PYTHON_SCRIPT="/Time_Classifiction/Classification/shiyanTSLANet_0715_0703.py"
DATA_PATH="/hy-tmp/0712_realdata/"
EPOCHS=500



# 实验3：池化策略对照
echo "Running pooling strategy experiments..."
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB True --ASB True --topk_pool False --name exp3_attention_pool \
    --dropout_rate 0.15 --depth 2 --patch_size 60

python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB True --ASB True --topk_pool True --name exp3_topk_pool \
    --dropout_rate 0.15 --depth 2 --patch_size 60

# 实验4：数据增强策略对照
echo "Running data augmentation experiments..."
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB True --ASB True --use_simulated_anomaly False --name exp4_no_aug \
    --dropout_rate 0.15 --depth 2 --patch_size 60

python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB True --ASB True --use_simulated_anomaly True --name exp4_with_aug \
    --dropout_rate 0.15 --depth 2 --patch_size 60 \
    --min_anomaly_length 7 --max_anomaly_length 90

# 实验5：特征融合策略对照
echo "Running feature fusion experiments..."
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB True --ASB True --use_mp_features False --name exp5_no_mp \
    --dropout_rate 0.15 --depth 2 --patch_size 60

python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --ICB True --ASB True --use_mp_features True --name exp5_with_mp \
    --dropout_rate 0.15 --depth 2 --patch_size 60 \
    --mp_window_size 30
