# 时间:0802
python /Time_Classifiction/Classification/0722_TSLA.py --data_path /hy-tmp/0802_realdata/ --num_epochs 500 --name 0802_exp1_depth4 --dropout_rate 0.15 --depth 4 --emb_dim 128 --patch_size 60

python /Time_Classifiction/Classification/0722_TSLA.py --data_path /hy-tmp/0802_realdata/ --num_epochs 500 --name 0802_exp1_depth4 --dropout_rate 0.3 --depth 4 --emb_dim 128 --patch_size 60

# 这个实验采取了两类损失函数分权的样式
python /Time_Classifiction/Classification/0722_TSLA_loss.py --data_path /hy-tmp/0716_realdata/ --num_epochs 500 --name 0802_exp1_depth4_loss --dropout_rate 0.15 --depth 4 --emb_dim 128 --patch_size 60


python /Time_Classifiction/Classification/TSLANetshiyan.py --data_path /hy-tmp/0716_realdata/ --name 实验1 --patch_size 60

python /Time_Classifiction/Classification/TSLANetshiyan.py --data_path /hy-tmp/0716_realdata/ --name 实验1_1

python /Time_Classifiction/Classification/TSLANetshiyan.py --data_path /hy-tmp/0712_realdata/ --name 实验2

python /Time_Classifiction/Classification/TSLANetshiyan.py --data_path /hy-tmp/0624datarealy/ --name 实验3

python /Time_Classifiction/Classification/TSLANetshiyan.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 实验4 

python /Time_Classifiction/Classification/TSLANetshiyan.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 实验4_1 --patch_size 60 --num_epochs 1000


# 时间:0805，目前看来实验1和实验4——1的效果最好，因此尝试修改部分参数，找的较好的收敛模型

# 实验1
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0805实验1_1 --patch_size 60 --num_epochs 1000 --depth 4

python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0805实验1_2 --patch_size 60 --num_epochs 1000

python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/0716_realdata/ --name 0805实验1_3 --patch_size 90 --num_epochs 1000

# 实验4_1
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0805实验4_1_1 --patch_size 60 --depth 4 --num_epochs 1000

python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0805实验4_1_2 --patch_size 90 --num_epochs 1000

python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0805实验4_1_3 --patch_size 60 --num_epochs 1000


python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0805实验4_1_2 --patch_size 90 --num_epochs 1000 --load_from_pretrained True

python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0805实验4_1_3 --patch_size 60 --num_epochs 1000 --load_from_pretrained True

# 时间:0807，本次主要实验对象/Time_Classifiction/Classification/0807TSLANet.py，该文件的创新点是损失函数的创新，本次的优化方向是准确率


python /Time_Classifiction/Classification/0807TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0807实验4_1_2 --patch_size 90 --num_epochs 1000

python /Time_Classifiction/Classification/0807TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0807实验4_1_3 --patch_size 60 --num_epochs 1000











# 时间:0809，本次实验对象/Time_Classifiction/Classification/0809TSLANet.py，本次主要操作对比实验来调参使用

# 对比实验1
# A1_基线
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name A1_基线 --patch_size 90 --num_epochs 1000

# A2_时域增强
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name A2_时域增强 --patch_size 90 --num_epochs 1000 --use_temporal_aug True --temporal_shift 0.2

# A3_频域增强
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name A3_频域增强 --patch_size 90 --num_epochs 1000 --use_freq_aug True --freq_jitter_strength 0.2

# A4_MixUp增强
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name A4_MixUp增强 --patch_size 90 --num_epochs 1000 --use_mixup True --mixup_alpha 0.3

# A5_对比学习
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name A5_对比学习 --patch_size 90 --num_epochs 1000 --use_contrastive True --contrastive_weight 0.2 --contrastive_temp 0.2

# 对比实验2
# B1_基线结构
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name B1_基线结构 --patch_size 90 --num_epochs 1000

# B2_AttnPooling
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name B2_AttnPooling --patch_size 90 --num_epochs 1000 --use_attn_pool True

# B3_高Dropout
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name B3_高Dropout --patch_size 90 --num_epochs 1000 --dropout 0.3

# B4_小Depth多尺度
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name B4_小Depth多尺度 --patch_size 90 --num_epochs 1000 --depth 3 --multi_scale True

# 对比实验3
# C1_plateau
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name C1_plateau --patch_size 90 --num_epochs 1000 --scheduler plateau

# C2_cosine
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name C2_cosine --patch_size 90 --num_epochs 1000 --scheduler cosine

# C3_onecycle
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name C3_onecycle --patch_size 90 --num_epochs 1000 --scheduler onecycle --max_lr 1e-3

# C4_focal_loss
python /Time_Classifiction/Classification/0805TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name C4_focal_loss --patch_size 90 --num_epochs 1000 --use_focal True --focal_gamma 2.0
