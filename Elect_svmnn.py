# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 07:57:20 2019

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Studied and Created on  2018/3/1

@author: batch William
"""
import datetime

starttime = datetime.datetime.now()

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import csv
import numpy as np
import scipy.io as sio
import pandas as pd
from keras.optimizers import SGD
from numpy.random import shuffle  # 引入随机函数
from sklearn import svm

import os, sys

os.getcwd()
os.chdir("C:/Users/Administrator/Desktop/electricity-theft-detection-master/Elect_dat/08_elect_theft/")
outputfile = "val_dat5.csv"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


ttl_dat = pd.read_csv("Ele_data.csv")  # header=None
# ttl_dat = pd.read_excel('F:/08_论文写作/02_窃漏电预测/03_案例数据/GBDT/备份/train_res5.xls','rf_res')
train_v = ttl_dat.values[:700, :]  # 取5000样本作为训练集
train_t = ttl_dat.values[700:, :]  # 剩余样本为测试集


train_label = train_v[:, -1]  # 获取真实训练集标签
train_data = train_v[:, :-1]  # 训练集数据
v_label = train_t[:, -1]  # 验证集标签
v_data = train_t[:, :-1]  # 验证机数据

test = v_data
cls_num = 2


from keras.utils import to_categorical  # 编码，将训练集标签转换成one-hot编码

categorical_labels = to_categorical(
    train_label, num_classes=cls_num
)  # 编码，将训练集标签转换成1-2维（1-2类小说）
# categorical_labels = np.delete(categorical_labels, 0, axis=1)
_labels = to_categorical(
    v_label, num_classes=cls_num
)  # 编码，将验证集标签转换成12维（12类小说）
# _labels = np.delete(_labels, 0, axis=1)

input_shape = (1035,)

# ----构建ANN神经网络模型------------------------------------------------------------
from keras.models import Sequential  # 导入神经网络初始化函数
from keras.layers import Dense, Activation  # 导入神经网络层函数、激活函数

netfile = "obj_reco/net1.model"  # 构建的神经网络模型存储路径
net = Sequential()  # 建立神经网络
net.add(Dense(units=10, input_shape=(1035,))
)  # 添加输入层（3节点）到隐藏层（10节点）的连接
net.add(Activation("relu"))  # 隐藏层使用relu激活函数
net.add(Dense(units=cls_num)
)  # 添加隐藏层（10节点）到输出层（1节点）的连接 one-hot后2维
net.add(Activation("softmax"))  # 输出层使用sigmoid激活函数
# 编译模型，使用adam方法求解
net.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
train_data = train_data.astype("float32")
v_data = v_data.astype("float32")
categorical_labels = categorical_labels.astype("float32")
net.fit(train_data, categorical_labels, epochs=30, batch_size=128)  # 训练模型，循环30次
netfile = "obj_reco/net.model.weights1.h5"
# net.save_weights(netfile)  # 保存模型
predictions = net.predict(train_data)
# 获取每个样本最可能的类别索引
pre_trn = np.argmax(predictions, axis=1)

# keras用predict给出预测概率，predict_classes才是给出预测类别
predictions_v = net.predict(v_data)
# 获取每个样本最可能的类别索引
pre_v = np.argmax(predictions_v, axis=1)


#
trn_acc = sum(pre_trn == train_label)/len(pre_trn)  # 计算训练集准确率
print(trn_acc)
acc = sum(pre == v_label)/len(pre)  # 计算验证集准确率
print(acc)
#


# ----统计指标-1，训练集---------------------------------------------------------------------
trn_acc = accuracy_score(list(train_label), list(pre_trn))
print("trn_acc=" + str(trn_acc))

trn_pre = precision_score(list(train_label), list(pre_trn), average="macro")
print("trn_pre=" + str(trn_pre))

trn_recall = recall_score(list(train_label), list(pre_trn), average="micro")  #'macro'
print("trn_recall=" + str(trn_recall))

trn_f1_score = f1_score(list(train_label), list(pre_trn), average="weighted")
print("trn_f1_score=" + str(trn_f1_score))


# ----统计指标-2，测试集---------------------------------------------------------------------
tst_acc = accuracy_score(list(v_label), list(pre_v))
print("tst_acc=" + str(tst_acc))

tst_pre = precision_score(list(v_label), list(pre_v), average="macro")
print("tst_pre=" + str(tst_pre))

tst_recall = recall_score(list(v_label), list(pre_v), average="micro")  #'macro'
print("tst_recall=" + str(tst_recall))

tst_f1_score = f1_score(list(v_label), list(pre_v), average="weighted")
print("tst_f1_score=" + str(tst_f1_score))


"""
#---训练集结果写入csv----------------------------------------------------------------------
trn_arr=np.column_stack((pre_trn,train_v))  # 训练集narray结果合并
trn_dat=pd.DataFrame(trn_arr)
trn_dat.to_csv('trn_dat.csv', header=None, index=False) #输出结果


#---验证集结果写入csv----------------------------------------------------------------------
val_arr=np.column_stack((pre,train_t))  # 验证集narray结果合并
val_dat=pd.DataFrame(val_arr)
val_dat.to_csv(outputfile, header=None, index=False) #输出结果
"""


"""
#导入输出相关的库，生成混淆矩阵
from sklearn import metrics
cm_train = metrics.confusion_matrix(y_train, model.predict(x_train)) #训练样本的混淆矩阵
cm_test = metrics.confusion_matrix(y_test, model.predict(x_test)) #测试样本的混淆矩阵

#保存结果
pd.DataFrame(cm_train, index = range(1, 6), columns = range(1, 6)).to_excel(outputfile1)
pd.DataFrame(cm_test, index = range(1, 6), columns = range(1, 6)).to_excel(outputfile2)
"""
