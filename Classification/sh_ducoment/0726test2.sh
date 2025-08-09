#!/bin/bash

# 实验5：不同损失函数对比
python Time_Classification/Classification/0726TSLANet.py \
--data_path /hy-tmp/0712_realdata/ \
--name 0726exp5_focal_loss \
--focal_loss True \
--loss_rate 0.2

# 实验6：不使用对比学习
python Time_Classification/Classification/0726TSLANet.py \
--data_path /hy-tmp/0712_realdata/ \
--name 0726exp6_no_CL \
--loss_rate 0.0

# 实验7：分层学习率优化
python Time_Classification/Classification/0726TSLANet.py \
--data_path /hy-tmp/0712_realdata/ \
--name 0726exp7_layerwise_lr \
--train_lr 2e-3

# 实验8：仅基础增强
python Time_Classification/Classification/0726TSLANet.py \
--data_path /hy-tmp/0712_realdata/ \
--name 0726exp8_basic_aug \
--patch_size 30  # 更小的patch增强局部特征

# 实验9：强季节增强
python Time_Classification/Classification/0726TSLANet.py \
--data_path /hy-tmp/0712_realdata/ \
--name 0726exp9_seasonal_aug \
--patch_size 90 \
--masking_ratio 0.6  # 更高掩码率