#!/bin/bash

# 基础参数设置
PYTHON_SCRIPT="/Time_Classifiction/Classification/0722_TSLA.py"
DATA_PATH="/hy-tmp/0712_realdata/"
EPOCHS=500
LOG_DIR="/tf_logs"

##############################################
# 实验1：正则化策略对照实验
##############################################
echo "Running regularization 20722experiments..."

# python /Time_Classifiction/Classification/0722_TSLA.py --data_path /hy-tmp/0712_realdata/ --num_epochs 500 --name 20722exp1 --dropout_rate 0.15 --depth 4 --emb_dim 128 --patch_size 90
# python /Time_Classifiction/Classification/0722_TSLA.py --data_path /hy-tmp/0712_realdata/ --num_epochs 500 --name 20722exp1_depth2_ --dropout_rate 0.15 --depth 2 --emb_dim 64 --patch_size 90
# python /Time_Classifiction/Classification/0722_TSLA.py --data_path /hy-tmp/0712_realdata/ --num_epochs 500 --name 20722exp1_depth4_ --dropout_rate 0.15 --depth 4 --emb_dim 64 --patch_size 90
# 目前看来--depth 4 --emb_dim 128 --patch_size 60是最好的参数

# python /Time_Classifiction/Classification/shiyanTSLANet.py --data_path /hy-tmp/0712_realdata/ --num_epochs 500 --name 10722exp1_128_ --dropout_rate 0.15 --depth 4 --emb_dim 128 --patch_size 60
# python /Time_Classifiction/Classification/shiyanTSLANet.py --data_path /hy-tmp/0712_realdata/ --num_epochs 500 --name 10722exp1_64_ --dropout_rate 0.15 --depth 4 --emb_dim 64 --patch_size 60
# 1.1 基础模型（低正则化）
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp1_baseline_low_reg \
    --dropout_rate 0.15 \
    --depth 4 --emb_dim 128 --patch_size 60

# 1.2 高正则化（增加dropout）
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp1_high_dropout \
    --dropout_rate 0.4 \
    --depth 4 --emb_dim 128 --patch_size 60

# 1.3 权重衰减增强
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp1_high_weight_decay \
    --weight_decay 1e-3 \
    --depth 4 --emb_dim 128 --patch_size 60

# 1.4 标签平滑增强
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp1_label_smoothing \
    --depth 4 --emb_dim 128 --patch_size 60 \
    --label_smoothing 0.2  # 假设脚本支持此参数

##############################################
# 实验2：模型深度与复杂度实验
##############################################
echo "Running model depth 20722experiments..."

# 2.1 浅层模型（2层）
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp2_shallow \
    --depth 2 --emb_dim 128 --patch_size 60

# 2.2 深层模型（8层）
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp2_deep \
    --depth 8 --emb_dim 128 --patch_size 60

# 2.3 超深层模型（16层）
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp2_very_deep \
    --depth 16 --emb_dim 128 --patch_size 60

##############################################
# 实验3：嵌入维度实验
##############################################
echo "Running embedding dimension 20722experiments..."

# 3.1 低维度（64）
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp3_low_dim \
    --emb_dim 64 --depth 4 --patch_size 60

# 3.2 中维度（128）- 基线
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp3_med_dim \
    --emb_dim 128 --depth 4 --patch_size 60

# 3.3 高维度（256）
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp3_high_dim \
    --emb_dim 256 --depth 4 --patch_size 60

# 3.4 超高维度（512）
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp3_very_high_dim \
    --emb_dim 512 --depth 4 --patch_size 60

##############################################
# 实验4：时间序列处理实验
##############################################
echo "Running time-series processing 20722experiments..."

# 4.1 小patch size（捕捉局部特征）
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp4_small_patch \
    --patch_size 20 --depth 4 --emb_dim 128

# 4.2 大patch size（捕捉全局特征）
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp4_large_patch \
    --patch_size 100 --depth 4 --emb_dim 128

# 4.3 自适应滤波开启
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp4_adaptive_filter \
    --adaptive_filter True --depth 4 --emb_dim 128 --patch_size 60

# 4.4 自适应滤波关闭
python $PYTHON_SCRIPT --data_path $DATA_PATH --num_epochs $EPOCHS \
    --name 20722exp4_no_adaptive_filter \
    --adaptive_filter False --depth 4 --emb_dim 128 --patch_size 60

