#!/bin/bash
# A1: Baseline ICB-only
python /Time_Classifiction/Classification/0903.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name A1_ICB_only \
  --use_shapelet False \
 --patch_size 60 --emb_dim 128 --depth 2 \
  --num_epochs 500 --batch_size 32 --train_lr 1e-3 --weight_decay 5e-4

# A2: ICB + RawShapelet (Gate, finetune)
python /Time_Classifiction/Classification/0903.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name A2_ICB_RawShapelet_FT \
  --use_shapelet True --finetune_shapelets True \
  --shapelet_len 30 --K_anomaly 10 --shapelet_init kmeans_anomaly_only \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --num_epochs 500 --batch_size 32 --train_lr 1e-3 --weight_decay 5e-4

# A3: ICB + RawShapelet (Gate, frozen)
python /Time_Classifiction/Classification/0903.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name A3_ICB_RawShapelet_Frozen \
  --use_shapelet True --finetune_shapelets False \
  --shapelet_len 30 --K_anomaly 10 --shapelet_init kmeans_anomaly_only \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --num_epochs 500 --batch_size 32 --train_lr 1e-3 --weight_decay 5e-4

# 对比试验B
python /Time_Classifiction/Classification/0903.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name 实验B扩大K_anomaly值 \
  --use_shapelet True --finetune_shapelets False \
  --shapelet_len 30 --K_anomaly 20 --shapelet_init kmeans_anomaly_only \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --num_epochs 500 --batch_size 32 --train_lr 1e-3 --weight_decay 5e-4


python /Time_Classifiction/Classification/0903.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name 实验B缩小K_anomaly值 \
  --use_shapelet True --finetune_shapelets False \
  --shapelet_len 30 --K_anomaly 5 --shapelet_init kmeans_anomaly_only \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --num_epochs 500 --batch_size 32 --train_lr 1e-3 --weight_decay 5e-4

python /Time_Classifiction/Classification/0903.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name 实验B扩大shape长度 \
  --use_shapelet True --finetune_shapelets False \
  --shapelet_len 60 --K_anomaly 10 --shapelet_init kmeans_anomaly_only \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --num_epochs 500 --batch_size 32 --train_lr 1e-3 --weight_decay 5e-4

python /Time_Classifiction/Classification/0903.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name 实验B减少shape长度 \
  --use_shapelet True --finetune_shapelets False \
  --shapelet_len 15 --K_anomaly 10 --shapelet_init kmeans_anomaly_only \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --num_epochs 500 --batch_size 32 --train_lr 1e-3 --weight_decay 5e-4

# 实验C

python /Time_Classifiction/Classification/0903.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name 实验C正常开启 \
  --use_shapelet True --finetune_shapelets True \
  --shapelet_len 30 --K_anomaly 10 --shapelet_init kmeans_anomaly_only \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --num_epochs 500 --batch_size 32 --train_lr 1e-3 --weight_decay 5e-4



python /Time_Classifiction/Classification/0904.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name 实验_Shapelet_only \
  --use_shapelet True --shapelet_only True \
  --shapelet_len 30 --K_anomaly 10 --shapelet_init kmeans_anomaly_only \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --num_epochs 500 --batch_size 32 --train_lr 1e-3 --weight_decay 5e-4
