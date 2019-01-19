#!usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 2018年9月16日

@author: yrh
'''
from src.Tf_Graph import *
from src.MnistDemo import *
import tensorflow as tf


'''
=====================1、Placeholder占位符==========
'''
x = tf.placeholder(tf.float32, shape=[None, img_flat], name='x')          # 原始输入
x_image = tf.reshape(x, [-1, img_size, img_size, input_channels])                # 转换为2维图像
y_true = tf.placeholder(tf.float32, shape=[None, output_size], name='y_true')  # 原始输出
y_true_cls = tf.argmax(y_true, dimension=1)                  # 转换为真实类别，与之前的使用placeholder不同


'''
卷积层 1
'''

layer_conv1, weights_conv1 = \
    new_con_layer(input_d=x_image,                    # 输入图像
                  input_shape=input_channels,  # 输入通道数
                  filter_size=filter_size1,          # 卷积核尺寸
                  filter_num=num_filters1,          # 卷积核数目
                  use_pool=True)
print(layer_conv1)

'''
卷积层 2
'''
layer_conv2, weights_conv2 = \
    new_con_layer(input_d=layer_conv1,
                   input_shape=num_filters1,
                   filter_size=filter_size2,
                   filter_num=num_filters2,
                   use_pool=True)
print(layer_conv2)

'''
展平层
展平层将第二个卷积层展平为二维tensor。
'''
layer_flat, num_features = flatten_layer(layer_conv2)
print(layer_flat)

'''
全连接层 1
'''
layer_fc1 = new_fc_layer(input_d=layer_flat,   # 展平层输出
                         input_shape=num_features,   # 输入特征维度
                         out_shape=fc_size,       # 输出特征维度
                         use_relu=True)
print(layer_fc1)

'''
全连接层 2
'''
layer_fc2 = new_fc_layer(input_d=layer_fc1,           # 上一全连接层
                         input_shape=fc_size,        # 输入特征维度
                         out_shape=output_size,   # 输出类别数
                         use_relu=False)
print(layer_fc2)

'''
预测类别
第二个全连接层估计输入的图像属于某一类别的程度，这个估计有些粗糙，需要添加一个softmax层归一化为概率表示。
'''
y_pred = tf.nn.softmax(layer_fc2)              # softmax归一化
y_pred_cls = tf.argmax(y_pred, dimension=1)         # 真实类别

'''
代价函数
概率交叉熵
'''
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)

'''
优化方法
'''
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

'''
性能度量
'''

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
