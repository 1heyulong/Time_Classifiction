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




python /Time_Classifiction/Classification/0904.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name ShapeletReplace_len30_K10_ftTrue \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --shapelet_len 30 --K_anomaly 10 \
  --num_epochs 1000 --batch_size 32 \
  --train_lr 1e-3 --weight_decay 5e-4 \
  --finetune_shapelets True

python /Time_Classifiction/Classification/0904.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name ShapeletReplace_len30_K20_ftTrue \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --shapelet_len 30 --K_anomaly 20 \
  --num_epochs 1000 --batch_size 32 \
  --train_lr 1e-3 --weight_decay 5e-4 \
  --finetune_shapelets True

python /Time_Classifiction/Classification/0904.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name ShapeletReplace_len60_K10_ftTrue \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --shapelet_len 60 --K_anomaly 10 \
  --num_epochs 1000 --batch_size 32 \
  --train_lr 1e-3 --weight_decay 5e-4 \
  --finetune_shapelets True

python /Time_Classifiction/Classification/0904.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name ShapeletReplace_len30_K10_ftFalse \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --shapelet_len 30 --K_anomaly 10 \
  --num_epochs 1000 --batch_size 32 \
  --train_lr 1e-3 --weight_decay 5e-4 \
  --finetune_shapelets False

python /Time_Classifiction/Classification/0904.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name ShapeletReplace_len30_K5_ftTrue \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --shapelet_len 30 --K_anomaly 5 \
  --num_epochs 1000 --batch_size 32 \
  --train_lr 1e-3 --weight_decay 5e-4 \
  --finetune_shapelets True


python /Time_Classifiction/Classification/0904.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name ShapeletReplace_len60_K20_ftTrue \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --shapelet_len 60 --K_anomaly 20 \
  --num_epochs 1000 --batch_size 32 \
  --train_lr 1e-3 --weight_decay 5e-4 \
  --finetune_shapelets True



python /Time_Classifiction/Classification/0907TSLANet.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name Fusion_Baseline --num_epochs 1000 \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --shapelet_len 30 --K_anomaly 10 \
  --finetune_shapelets True

python /Time_Classifiction/Classification/0907TSLANet.py \
  --data_path /hy-tmp/0716_realdata/ --num_epochs 1000 \
  --name Fusion_len30_K20 \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --shapelet_len 30 --K_anomaly 20 \
  --finetune_shapelets True

python /Time_Classifiction/Classification/0907TSLANet.py \
  --data_path /hy-tmp/0716_realdata/ --num_epochs 1000 \
  --name Fusion_len60_K10 \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --shapelet_len 60 --K_anomaly 10 \
  --finetune_shapelets True


python /Time_Classifiction/Classification/0909model.py \
  --data_path /hy-tmp/0716_realdata/ --num_epochs 1000 \
  --name MIL_max --mil_topk 1 --pseudo_anomaly False


python /Time_Classifiction/Classification/0909model.py \
  --data_path /hy-tmp/0716_realdata/ --num_epochs 1000 \
  --name MIL_topk3 --mil_topk 3 --pseudo_anomaly False

python /Time_Classifiction/Classification/0909model.py \
  --data_path /hy-tmp/0716_realdata/ --num_epochs 1000 \
  --name MIL_topk5_focal --mil_topk 5 --focal_gamma 2.0 --pseudo_anomaly False

python /Time_Classifiction/Classification/0909model.py \
  --data_path /hy-tmp/0716_realdata/ --num_epochs 1000 \
  --name MIL_topk3_focal_pseudo --mil_topk 3 --focal_gamma 2.0 --pseudo_anomaly True



python /Time_Classifiction/Classification/TSL_Statis.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name fusion_gate_nn \
  --module TSLA_Statis \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --fusion_type gate_nn \
  --num_epochs 1000 --batch_size 32 --train_lr 1e-3 --weight_decay 5e-4 \
  --pseudo_anomaly False

