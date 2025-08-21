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
