#!/bin/bash


# 实验10：不同深度对比
for depth in 2 4 6 8; do
python Time_Classification/Classification/0726TSLANet.py \
--data_path /hy-tmp/0712_realdata/ \
--name 0726exp10_depth_${depth} \
--depth $depth
done

# 实验11：不同embedding维度
for dim in 64 128 256 512; do
python Time_Classification/Classification/0726TSLANet.py \
--data_path /hy-tmp/0712_realdata/ \
--name 0726exp11_dim_${dim} \
--emb_dim $dim
done

# 实验12：对比InceptionTime
python Time_Classification/Classification/baselines/InceptionTime.py \
--data_path /hy-tmp/0712_realdata/ \
--name 0726exp12_InceptionTime

# 实验13：对比ROCKET
python Time_Classification/Classification/baselines/ROCKET.py \
--data_path /hy-tmp/0712_realdata/ \
--name 0726exp13_ROCKET