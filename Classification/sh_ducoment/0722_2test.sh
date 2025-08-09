#!/bin/bash

# 基础参数设置
PYTHON_SCRIPT="/Time_Classifiction/Classification/0722_TSLA.py"
DATA_PATH="/hy-tmp/0712_realdata/"
EPOCHS=500


##############################################
# 实验5：模块消融实验
##############################################
echo "Running module ablation 20722experiments..."

# 5.1 仅ICB模块
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp5_only_icb \
    --ICB True --ASB False --depth 4 --emb_dim 128 --patch_size 60

# 5.2 仅ASB模块
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp5_only_asb \
    --ICB False --ASB True --depth 4 --emb_dim 128 --patch_size 60

# 5.3 完整模型（ICB+ASB）
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp5_full_model \
    --ICB True --ASB True --depth 4 --emb_dim 128 --patch_size 60

##############################################
# 实验6：学习率策略实验
##############################################
echo "Running learning rate 20722experiments..."

# 6.1 高学习率
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp6_high_lr \
    --train_lr 1e-2 --depth 4 --emb_dim 128 --patch_size 60

# 6.2 低学习率
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp6_low_lr \
    --train_lr 1e-4 --depth 4 --emb_dim 128 --patch_size 60

# 6.3 学习率预热
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp6_lr_warmup \
    --use_lr_warmup True --depth 4 --emb_dim 128 --patch_size 60

##############################################
# 实验7：高级配置组合
##############################################
echo "Running advanced configuration 20722experiments..."

# 7.1 最优组合1（深层+高维）
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp7_deep_highdim \
    --depth 8 --emb_dim 256 --patch_size 60 \
    --dropout_rate 0.25

# 7.2 最优组合2（自适应patch）
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp7_adaptive_patch \
    --depth 6 --emb_dim 192 --patch_size "adaptive" \
    --adaptive_patch True

# 7.3 完整最优配置
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp7_full_optimal \
    --depth 6 --emb_dim 192 --patch_size 70 \
    --ICB True --ASB True --adaptive_filter True \
    --dropout_rate 0.2 --weight_decay 1e-4

echo "All 20722experiments completed!"