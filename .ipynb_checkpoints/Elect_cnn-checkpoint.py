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

import os, sys

os.getcwd()
os.chdir(
    "C:/Users/Administrator/Desktop/electricity-theft-detection-master/Elect_dat/08_elect_theft/"
)
outputfile = "Ele_data.csv"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"






ttl_dat = pd.read_csv("Ele_data.csv")  # header=None此处的读取excel有问题，和上面的输出重合了，所以要重新读取
train_v = ttl_dat.values[:300, :]  # 取800样本作为训练集
train_t = ttl_dat.values[300:, :]  # 取200样本为测试集


train_label = train_v[:, -1]  # 获取训练集标签
train_data = train_v[:, :-1]  # 训练集数据
v_label = train_t[:, -1]  # 测试集标签
v_data = train_t[:, :-1]  # 测试集数据


test = v_data

cls_num = 2  # 只有两类


from keras.utils import to_categorical  # 编码，将训练集标签转换成one-hot编码

categorical_labels = to_categorical(train_label, num_classes=cls_num)  # 编码，将训练集标签转换成1-2维
# categorical_labels = np.delete(categorical_labels, 0, axis=1)
_labels = to_categorical(v_label, num_classes=cls_num)  # 编码，将验证集标签转换成1-2维
# _labels = np.delete(_labels, 0, axis=1)#将前面的结果第一项删除

num_clas1 = len([x for x in train_label if x == 0])
num_clas2 = len([x for x in train_label if x == 1])


# 多标签情况下的损失函数
def focal_loss(classes_num, gamma=2, alpha=0.3, e=1e-6):
    # classes_num：每个标签包含的样本个数
    def focal_loss_fixed(target_tensor, prediction_tensor):
        import tensorflow as tf
        from tensorflow.python.ops import array_ops
        from keras import backend as K

        # 1# 得到没有平衡权重的损失function (4)
        zeros = array_ops.zeros_like(
            prediction_tensor, dtype=prediction_tensor.dtype
        )  # 类型相似的全零张量
        one_minus_p = array_ops.where(
            tf.greater(target_tensor, zeros), target_tensor - prediction_tensor, zeros
        )
        # one_minus_p = array_ops.where(条件, X, Y)满足条件是X，不满足是Y
        FT = (
            -1
            * (one_minus_p**2)
            * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))
        )
        # tf.clip_by_value(X, 1e-8, 1.0)，判定X的值，X值大于1，返回1，小于1e-8返回1e-8，在区间内返回正常值

        # 2# get 平衡权重的alpha
        classes_weight = array_ops.zeros_like(
            prediction_tensor, dtype=prediction_tensor.dtype
        )

        total_num = float(sum(classes_num))
        classes_w_t1 = [total_num / ff for ff in classes_num]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ff / sum_ for ff in classes_w_t1]  # scale
        classes_w_tensor = tf.convert_to_tensor(
            classes_w_t2, dtype=prediction_tensor.dtype
        )
        # 将w_t2转换为与prediction_tensor同类型
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

        # 3# 平衡权重后的损失
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_mean(balanced_fl)

        # 4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1 - e) * balanced_fl + e * K.categorical_crossentropy(
            K.ones_like(prediction_tensor) / nb_classes, prediction_tensor
        )

        return fianal_loss

    return focal_loss_fixed


from keras.layers import (
    Input,
    Dense,
    Conv1D,
    MaxPooling1D,
    AveragePooling1D,
    GlobalAveragePooling1D,
)
from keras.layers import BatchNormalization
from keras import backend as K
from keras.models import Model
from keras.layers import Flatten


# 扩充为三个维度，使数据集满足一维卷积的形式
x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
y_train = v_data.reshape(v_data.shape[0], v_data.shape[1], 1)
test = test.reshape(test.shape[0], test.shape[1], 1)

# 一维卷积输入层，神经元个数为5（特征数）
input_img = Input(shape=(train_data.shape[1], 1))

import numpy as np
from keras.models import Model, save_model, load_model
from keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    LeakyReLU,
    concatenate,
)
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D


def DenseLayer(x, nb_filter, bn_size=4, alpha=0.0, drop_rate=0.2):
    # Bottleneck layers瓶颈层，提高计算效率
    x = BatchNormalization()(x)
    # LeakyReLU是一个改良的激活函数
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv1D(bn_size * nb_filter, 1, padding="same")(x)
    # Composite function复合函数
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv1D(nb_filter, 3, padding="same")(x)

    if drop_rate:
        x = Dropout(drop_rate)(x)

    return x


# 生成新的特征密度表示
def DenseBlock(x, nb_layers, growth_rate, drop_rate=0.2):
    for i in range(nb_layers):
        conv = DenseLayer(x, nb_filter=growth_rate, drop_rate=drop_rate)
        x = concatenate([x, conv], axis=2)  # 卷积特征和原本x特征的融合
    return x


# 池化的操作
def TransitionLayer(x, compression=0.2, alpha=0.0, is_max=0):
    nb_filter = int(x.shape[-1] * compression)  # 计算化简后的滤波器数目
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv1D(nb_filter, 1, padding="same")(x)
    if is_max != 0:
        x = MaxPooling1D(2, padding="same")(x)
    else:
        x = AveragePooling1D(2, padding="same")(x)
    return x

growth_rate = 16

inpt = Input(shape=(train_data.shape[1], 1))  # 一维卷积输入层，神经元个数为5（特征数）


x = Conv1D(growth_rate * 2, 3, padding="same")(inpt)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)

x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
x = TransitionLayer(x)

x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
x = TransitionLayer(x)
x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)

x = BatchNormalization()(x)  # 数据标准化处理
x = GlobalAveragePooling1D()(x)  #
# x = MaxPooling1D(2, padding='same')(x)  # 全连接层
# x = Flatten()(x)
# x = Dense(1024,activation='relu')(x)


x = Dense(cls_num, activation="softmax")(x)
autoencoder = Model(inpt, x)
# autoencoder.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
autoencoder.compile(
    optimizer="adam", loss=[focal_loss([num_clas1, num_clas2])], metrics=["accuracy"]
)

MODEL_PATH = "obj_reco/tst_model3.h5"
autoencoder.save(MODEL_PATH)


if x_train.dtype != "float32":
    x_train = x_train.astype("float32")

a_trn = autoencoder.predict(x_train)  # 训练集预测结果
pre_trn = np.argmax(a_trn, axis=1)  # 训练集预测结果格式转换，形成向量

if y_train.dtype != "float32":
    y_train = y_train.astype("float32")
ax = autoencoder.predict(y_train)  # 验证集预测结果
pre = np.argmax(ax, axis=1)  # 验证集预测结果格式转换，形成向量

if test.dtype != "float32":
    test = test.astype("float32")
pr = autoencoder.predict(test)  # 测试集预测结果
predict = np.argmax(pr, axis=1)


# trn_acc = sum(pre_trn == train_label)/len(pre_trn)  # 计算训练集准确率
# print(trn_acc)

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 计算准确率
acc = sum(pre == v_label) / len(pre)
print(f"验证集准确率: {acc}")

# ---- 计算训练集指标 ----
trn_acc = accuracy_score(train_label, pre_trn)
trn_pre = precision_score(train_label, pre_trn, average="macro")
trn_recall = recall_score(train_label, pre_trn, average="micro")
trn_f1_score = f1_score(train_label, pre_trn, average="weighted")

# ---- 计算测试集指标 ----
tst_acc = accuracy_score(v_label, pre)
tst_pre = precision_score(v_label, pre, average="macro")
tst_recall = recall_score(v_label, pre, average="micro")
tst_f1_score = f1_score(v_label, pre, average="weighted")

# 将结果汇总为表格
metrics_df = pd.DataFrame({
    "Dataset": ["Training Set", "Test Set"],
    "Accuracy": [trn_acc, tst_acc],
    "Precision": [trn_pre, tst_pre],
    "Recall": [trn_recall, tst_recall],
    "F1 Score": [trn_f1_score, tst_f1_score]
})

# 调整显示选项，确保结果完整显示
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)     # 显示所有行
pd.set_option('display.width', 1000)        # 设置显示宽度
pd.set_option('display.colheader_justify', 'center')  # 列标题居中

# 输出表格
print(metrics_df)




#---训练集结果写入csv----------------------------------------------------------------------
trn_arr=np.column_stack((pre_trn,train_v))  # 训练集narray结果合并
trn_dat=pd.DataFrame(trn_arr)
trn_dat.to_csv('trn_dat.csv', header=None, index=False) #输出结果

#---验证集结果写入csv----------------------------------------------------------------------
val_arr=np.column_stack((pre,train_t))  # 验证集narray结果合并
val_dat=pd.DataFrame(val_arr)
val_dat.to_csv(outputfile, header=None, index=False) #输出结果

