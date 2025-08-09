#!/bin/bash

# 设置Python脚本路径
PYTHON_SCRIPT="/Time_Classifiction/Classification/shiyanTSLANet_0715_0703.py"
DATA_PATH="/hy-tmp/0712_realdata/"
EPOCHS=1

# 基础参数配置
BASE_ARGS="--data_path $DATA_PATH --num_epochs $EPOCHS --batch_size 32 --dropout_rate 0.15 --depth 2 --patch_size 60"

# --------------------------
# 实验组1：基线模型 (原始TSLANet)
# --------------------------
echo "Running Baseline Model..."
python $PYTHON_SCRIPT $BASE_ARGS \
    --name "0715baseline" \
    --model_id "exp1_baseline"

# --------------------------
# 实验组2：仅数据增强
# --------------------------
echo "Running Data Augmentation Only..."
python $PYTHON_SCRIPT $BASE_ARGS \
    --use_simulated_anomaly True \
    --min_anomaly_length 7 \
    --max_anomaly_length 90 \
    --name "0715aug_only"

# --------------------------
# 实验组3：仅特征重构
# --------------------------
echo "Running MP Features Only..."
python $PYTHON_SCRIPT $BASE_ARGS \
    --use_mp_features True \
    --mp_window_size 30 \
    --name "0715mp_only" \
    --model_id "exp3_mp_only"

# --------------------------
# 实验组4：数据增强+特征重构
# --------------------------
echo "Running Full Enhanced Model..."
python $PYTHON_SCRIPT $BASE_ARGS \
    --use_simulated_anomaly True \
    --min_anomaly_length 7 \
    --max_anomaly_length 90 \
    --use_mp_features True \
    --mp_window_size 30 \
    --name "0715full_enhanced" \
    --model_id "exp4_full_enhanced"

# --------------------------
# 实验组5：消融研究 (移除关键组件)
# --------------------------
echo "Running Ablation Study..."
python $PYTHON_SCRIPT $BASE_ARGS \
    --ICB False \
    --ASB False \
    --name "0715ablation_no_icb_asb" \
    --model_id "exp5_ablation"

# --------------------------
# 实验组6：参数调优组 (最优配置候选)
# --------------------------
echo "Running Tuned Configuration..."
python $PYTHON_SCRIPT $BASE_ARGS \
    --use_simulated_anomaly True \
    --min_anomaly_length 14 \
    --max_anomaly_length 120 \
    --use_mp_features True \
    --mp_window_size 45 \
    --loss_rate 0.2 \
    --name "0715tuned_config" \
    --model_id "exp6_tuned"