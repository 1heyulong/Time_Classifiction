# 时间0816


python /Time_Classifiction/Classification/0807TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0816实验4_1_2 --patch_size 90 --num_epochs 2000

python /Time_Classifiction/Classification/0807TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0816实验4_1_3 --patch_size 60 --num_epochs 2000


python /Time_Classifiction/Classification/0807TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0816实验4_1_4 --patch_size 60 --num_epochs 2000



python /Time_Classifiction/Classification/0818TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0818实验4_1_2 --patch_size 60 --num_epochs 1000
# 以下实验是为了调整0818TSLANet.py，看看哪个环节影响的收敛情况
python /Time_Classifiction/Classification/0818TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0818实验ICB --patch_size 60 --num_epochs 1000 --ASB False  
python /Time_Classifiction/Classification/0818TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0818实验ASB --patch_size 60 --num_epochs 1000 --ICB False
python /Time_Classifiction/Classification/0818TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0818实验none --patch_size 60 --num_epochs 1000 --ASB False --ICB False

python /Time_Classifiction/Classification/0818_2TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0818实验4_2_2 --patch_size 60 --num_epochs 1000

python /Time_Classifiction/Classification/0818TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0818实验ASB_nonea_f --patch_size 60 --num_epochs 1000 --ICB False --adaptive_filter False

python /Time_Classifiction/Classification/0818TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0818实验ICB_2 --patch_size 60 --num_epochs 1000 --ASB False

# 实验1、2、5是比较稳定的，其中1和5的F1分数相近（0.66），2的F1分数低一些（0.6）
# 实验1（0.65-0.66）
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0805实验4_1_3ICB --patch_size 60 --num_epochs 1000 --ASB False
# 实验2（0.6246）
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0805实验4_1_3ASB --patch_size 60 --num_epochs 1000 --ICB False
# 实验3（0.6036）
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0805实验4_1_3none --patch_size 60 --num_epochs 1000 --ICB False --ASB False
# 实验4（0.597）
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0805实验4_1_3ASB_noad --patch_size 60 --num_epochs 1000 --ICB False --adaptive_filter False
# 实验5（0.6777）
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0805实验4_1_3noad --patch_size 60 --num_epochs 1000 --adaptive_filter False



# 实验1、2、5是比较稳定的，其中1和5的F1分数相近（0.66），2的F1分数低一些（0.6）
# （0.6346）
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0806实验4_1_3ICB --patch_size 60 --num_epochs 1000 --ASB False


python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0806实验4_1_3ICB5 --patch_size 60 --num_epochs 1000 --ASB False --dropout_rate 0.4

python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0806实验4_1_3ICB3 --patch_size 60 --num_epochs 1000 --ASB False
# 我记得这个实验是修改了ICB模块中的dropout_rate层    
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0806实验4_1_3ICB2 --patch_size 60 --num_epochs 1000 --ASB False



# （0.6280）
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0806实验4_1_3ASB --patch_size 60 --num_epochs 1000 --ICB False

python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0806实验4_1_3none --patch_size 60 --num_epochs 1000 --ICB False --ASB False

python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0806实验4_1_3ASB_noad --patch_size 60 --num_epochs 1000 --ICB False --adaptive_filter False
# （0.6163）
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0806实验4_1_3noad --patch_size 60 --num_epochs 1000 --adaptive_filter False




# 基线（与当前最优非常接近）
python /Time_Classifiction/Classification/0823TSLA.py --data_path /hy-tmp/0716_realdata/ --name 0806实验4_1_3ICB --patch_size 60 --num_epochs 1000 --ASB False --pooling mean --use_se False --use_layerscale False --use_ema False --noise_std 0.0


# 建议增强版（参数量很小，稳健提升优先）
python /Time_Classifiction/Classification/0823TSLA.py --data_path /hy-tmp/0716_realdata/ --name 0806实验4_1_3ICB1 --patch_size 60 --num_epochs 1000 --ASB False --pooling attn --use_se True --use_layerscale True --use_ema True --drop_path_max 0.2 --noise_std 0.05 --use_bn_in_icb True


# 0825增加shapelet模块内容,来增强模型的少类识别能力。

# 维持你当前最佳主干（ICB+DropPath+LabelSmoothing），开启二分类 shapelet 头
python /Time_Classifiction/Classification/0825TSLANet.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name "ICB_LS_ShapeletBinary" \
  --ICB True --ASB False \
  --emb_dim 128 --depth 2 --patch_size 7 \
  --num_epochs 1000 --batch_size 32 \
  --train_lr 1e-3 --weight_decay 5e-4 \
  --use_shapelet_head True \
  --shapelet_per_class 3 \
  --shapelet_len_tokens 7 \
  --shapelet_metric cos \
  --shapelet_fuse_lambda 0.5 \
  --shapelet_tau 10.0

python /Time_Classifiction/Classification/0825TSLANet.py \
  --data_path /hy-tmp/dataset_rate_0605_realy/ \
  --name "ICB_LS_ShapeletBinary0605data" \
  --ICB True --ASB False \
  --emb_dim 128 --depth 2 --patch_size 7 \
  --num_epochs 1000 --batch_size 32 \
  --train_lr 1e-3 --weight_decay 5e-4 \
  --use_shapelet_head True \
  --shapelet_per_class 3 \
  --shapelet_len_tokens 7 \
  --shapelet_metric cos \
  --shapelet_fuse_lambda 0.5 \
  --shapelet_tau 10.0

# 哈哈我发现把shapelet_len_tokens调大到60，效果会更好一些，现在尝试设置为30会不会更好（反正一开始的7是效果很差很差）
python /Time_Classifiction/Classification/0825TSLANet.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name "ICB_LS_ShapeletBinary60" \
  --ICB True --ASB False \
  --emb_dim 128 --depth 2 --patch_size 7 \
  --num_epochs 1000 --batch_size 32 \
  --train_lr 1e-3 --weight_decay 5e-4 \
  --use_shapelet_head True \
  --shapelet_per_class 5 \
  --shapelet_len_tokens 60 \
  --shapelet_metric cos \
  --shapelet_fuse_lambda 0.5 \
  --shapelet_tau 10.0

python /Time_Classifiction/Classification/0825TSLANet.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name "ICB_LS_ShapeletBinary30" \
  --ICB True --ASB False \
  --emb_dim 128 --depth 2 --patch_size 60 \
  --num_epochs 1000 --batch_size 32 \
  --train_lr 1e-3 --weight_decay 5e-4 \
  --use_shapelet_head True \
  --shapelet_per_class 5 \
  --shapelet_len_tokens 30 \
  --shapelet_metric cos \
  --shapelet_fuse_lambda 0.5 \
  --shapelet_tau 10.0

python /Time_Classifiction/Classification/0825TSLANet.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name "ICB_LS_ShapeletBinary30_10" \
  --ICB True --ASB False \
  --emb_dim 128 --depth 2 --patch_size 60 \
  --num_epochs 1000 --batch_size 32 \
  --train_lr 1e-3 --weight_decay 5e-4 \
  --use_shapelet_head True \
  --shapelet_per_class 10 \
  --shapelet_len_tokens 30 \
  --shapelet_metric cos \
  --shapelet_fuse_lambda 0.5 \
  --shapelet_tau 10.0



python /Time_Classifiction/Classification/0825TSLANet.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name "ICB_LS_ShapeletBinary7_10" \
  --ICB True --ASB False \
  --emb_dim 128 --depth 2 --patch_size 60 \
  --num_epochs 1000 --batch_size 32 \
  --train_lr 1e-3 --weight_decay 5e-4 \
  --use_shapelet_head True \
  --shapelet_per_class 10 \
  --shapelet_len_tokens 7 \
  --shapelet_metric cos \
  --shapelet_fuse_lambda 0.5 \
  --shapelet_tau 10.0

python /Time_Classifiction/Classification/0825TSLANet.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name "ICB_LS_ShapeletBinary60xcos" \
  --ICB True --ASB False \
  --emb_dim 128 --depth 2 --patch_size 7 \
  --num_epochs 1000 --batch_size 32 \
  --train_lr 1e-3 --weight_decay 5e-4 \
  --use_shapelet_head True \
  --shapelet_per_class 5 \
  --shapelet_len_tokens 60 \
  --shapelet_metric xcorr \
  --shapelet_fuse_lambda 0.5 \
  --shapelet_tau 10.0



python /Time_Classifiction/Classification/0831TSLANet.py \
  --data_path /hy-tmp/0716_realdata/ \
  --patch_size 60 --num_epochs 1000 \
  --name ICB_AuxGateShapelet_joint \
  --ICB True --ASB False \
  --use_shapelet_head True \
  --shapelet_per_class 10 \
  --shapelet_len_tokens 30 \
  --shapelet_aux_weight 0.2 \
  --shapelet_pos_margin 0.35 \
  --shapelet_neg_margin 0.25

python /Time_Classifiction/Classification/0831_1TSLANet.py --data_path /hy-tmp/0716_realdata/ --name "ICB_ShapeAttn" --use_shapelet_attn True --shapelet_K 6 --shapelet_L 7 --patch_size 60 --num_epochs 1000


python /Time_Classifiction/Classification/0831_1TSLANet.py \
    --name "Baseline_ICB_only" \
    --data_path /hy-tmp/0716_realdata/ \
    --num_epochs 1000 \
    --batch_size 64 \
    --train_lr 5e-4 \
    --weight_decay 1e-4 \
    --emb_dim 256 \
    --depth 4 \
    --dropout_rate 0.2 \
    --patch_size 128 \
    --ICB True \
    --ASB False \
    --use_shapelet_attn False \
    --adaptive_filter False

python /Time_Classifiction/Classification/0831_1TSLANet.py \
    --name "TSLANet_Shapelet_Attention_Main" \
    --data_path /hy-tmp/0716_realdata/ \
    --num_epochs 1000 \
    --batch_size 64 \
    --train_lr 5e-4 \
    --weight_decay 1e-4 \
    --emb_dim 256 \
    --depth 4 \
    --dropout_rate 0.2 \
    --patch_size 128 \
    --use_shapelet_attn True \
    --shapelet_K 8 \
    --shapelet_L 5 \
    --shapelet_agg mean \
    --shapelet_fuse concat \
    --shapelet_gate False \
    --adaptive_filter False


python /Time_Classifiction/Classification/0831_1TSLANet.py \
    --name "TSLANet_Shapelet_Attention_Main120/30" \
    --data_path /hy-tmp/0716_realdata/ \
    --num_epochs 1000 \
    --batch_size 64 \
    --train_lr 5e-4 \
    --weight_decay 1e-4 \
    --emb_dim 256 \
    --depth 4 \
    --dropout_rate 0.2 \
    --patch_size 120 \
    --use_shapelet_attn True \
    --shapelet_K 8 \
    --shapelet_L 30 \
    --shapelet_agg mean \
    --shapelet_fuse concat \
    --shapelet_gate False \
    --adaptive_filter False


python /Time_Classifiction/Classification/0901TSLANet.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name ICB_ShapeletMinor \
  --ICB True --ASB False \
  --use_shapelet_head True \
  --shapelet_len 30 --K_normal 4 --K_anomaly 6 \
  --shapelet_init kmeans_anomaly_only \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --num_epochs 300 --batch_size 32 \
  --train_lr 1e-3 --weight_decay 5e-4

python /Time_Classifiction/Classification/0901TSLANet.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name ICB_ShapeletMinor11 \
  --ICB True --ASB False \
  --use_shapelet_head True \
  --shapelet_len 30 --K_normal 10 --K_anomaly 10 \
  --shapelet_init kmeans_anomaly_only \
  --patch_size 60 --emb_dim 128 --depth 4 \
  --num_epochs 1000 --batch_size 32 \
  --train_lr 1e-3 --weight_decay 5e-4

python /Time_Classifiction/Classification/0901TSLANet.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name ICB_ShapeletMinor22 \
  --ICB True --ASB False \
  --use_shapelet_head True \
  --shapelet_len 30 --K_normal 3 --K_anomaly 10 \
  --shapelet_init kmeans_anomaly_only \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --num_epochs 1000 --batch_size 32 \
  --train_lr 1e-3 --weight_decay 5e-4 --shapelet_contrast_weight 1 





