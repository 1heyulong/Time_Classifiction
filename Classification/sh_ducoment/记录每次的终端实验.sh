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

# 时间:0807，本次主要实验对象/Time_Classifiction/Classification/0807TSLANet.py，该文件的创新点是损失函数的创新，本次的优化方向是准确率


python /Time_Classifiction/Classification/0807TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0807实验4_1_2 --patch_size 90 --num_epochs 1000

python /Time_Classifiction/Classification/0807TSLANet.py --data_path /hy-tmp/dataset_rate_0605_realy/ --name 0807实验4_1_3 --patch_size 60 --num_epochs 1000
