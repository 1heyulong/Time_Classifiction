#!/bin/bash
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

python /Time_Classifiction/Classification/0903.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name 实验C正常开启 \
  --use_shapelet True --finetune_shapelets True \
  --shapelet_len 30 --K_anomaly 10 --shapelet_init kmeans_anomaly_only \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --num_epochs 500 --batch_size 32 --train_lr 1e-3 --weight_decay 5e-4
