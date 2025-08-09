#!/bin/bash

# 实验1：基础模型 (仅保留核心组件)
python Time_Classification/Classification/0726TSLANet.py \
--data_path /hy-tmp/0712_realdata/ \
--name 0726exp1_baseline \
--ICB False \
--ASB False \
--adaptive_filter False \
--depth 2 \
--emb_dim 64

# 实验2：仅ICB模块
python Time_Classification/Classification/0726TSLANet.py \
--data_path /hy-tmp/0712_realdata/ \
--name 0726exp2_ICB_only \
--ICB True \
--ASB False \
--adaptive_filter False

# 实验3：仅ASB模块
python Time_Classification/Classification/0726TSLANet.py \
--data_path /hy-tmp/0712_realdata/ \
--name 0726exp3_ASB_only \
--ICB False \
--ASB True \
--adaptive_filter True

# 实验4：完整模型 (您原始命令)
python Time_Classification/Classification/0726TSLANet.py \
--data_path /hy-tmp/0712_realdata/ \
--name 0726exp4_full_model \
--ICB True \
--ASB True \
--adaptive_filter True