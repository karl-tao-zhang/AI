#!usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 2018年9月16日

@author: yrh
'''

#导入
import tensorflow as tf
import numpy as np

'''
=========================1、创建变量=======================
'''

'''
权重参数初始化
零均值，0.01标准差正态分布的随机数
new_weights
'''
def new_weights(fileshape):
    return tf.Variable(tf.truncated_normal(shape=fileshape,stddev=0.01))






'''
偏置项初始化
常量值
new_biases
'''
def new_biases(filenum):
    return tf.constant(value=0.5,shape=[filenum])






'''
=========================2、创建卷积层=======================
'''

'''
卷积层
def new_conv_layer
args：
    前一层输入数据.
    前一层通道数
    卷积核尺寸
    卷积核数目
    使用 2x2 max-pooling.
    
return:结果层和权重

'''
def new_con_layer(input_d,input_shape,filter_size,filter_num,use_pool=True):
    # 卷积核形状定义:宽 高  数量  颜色通道
    shape=[filter_size,filter_size,input_shape,filter_num]
    weights=new_weights(shape)
    bias=new_biases(filter_num)

    layer=tf.nn.conv2d(input=input_d,  #从上一层输入的数据
                 filter=weights,  # 权重参数
                 strides=[1,1,1,1], # 移动的步长
                 padding='SAME')   # 填充类型,等长卷积
    layer+=bias   # 添加阈值
    if use_pool:   # 添加池化层
        layer=tf.nn.max_pool(layer,
                             ksize=[1,2,2,1],
                             strides=[1,2,2,1],
                             padding='SAME')

    layer=tf.nn.relu(layer)  #添加relu激活函数,对线性结果进行标准化
    return layer,weights







    # 卷积核权重的形状，由TensorFlow API决定

    

    # 根据给定形状创建权重
    

    # 创建新的偏置，每个卷积核一个偏置
    

    # 1、创建卷积层。注意stride全设置为1。
    # 第1个和第4个必须是1，因为第1个是图像的数目，第4个是图像的通道。
    # 第2和第3指定和左右、上下的步长。
    # padding设置为'SAME' 意味着给图像补零，以保证前后像素相同。
   

    # 2、给卷积层的输出添加一个偏置，每个卷积通道一个偏置值
   

    # 3、是否使用pooling
    
        # 这是 2x2 max-pooling, 表明使用 2x2 的窗口，选择每一窗口的最大值作为该窗口的像素，
        # 然后移动2格到下一窗口。
        

    # 4、 Linear Unit (ReLU).
    # 对每个输入像素x，计算 max(x, 0)，把负数的像素值变为0.
    # 这一步为原输出添加了一定的非线性特性，允许我们学习更加复杂的函数。
   

    # 注意 relu 通常在pooling前执行，但是由于 relu(max_pool(x)) == max_pool(relu(x))，
    # 我们可以通过先max_pooling再relu省去75%的计算。

    # 返回结果层和权重，结果层用于下一层输入，权重用于显式输出
    

'''
展评操作
flatten_layer

args： layer——上一层输入
return：展平层、特征值维度
'''
'''
   卷积核数量   高   宽    颜色通道
   卷积核数量   高*宽*颜色通道  
'''

def  flatten_layer(layer):
    layer_shape=layer.get_shape()
    fcs=layer_shape[1:4].num_elements()
    layer=tf.reshape(layer,shape=[-1,fcs])
    return layer,fcs




    # 获取输入层的形状，
    # layer_shape == [num_images, img_height, img_width, num_channels]
    

    # 特征数量: img_height * img_width * num_channels
    # 可以使用TensorFlow内建操作计算.
   

    # 将形状重塑为 [num_images, num_features].
    # 注意只设定了第二个维度的尺寸为num_filters，第一个维度为-1，保证第一个维度num_images不变
    # 展平后的层的形状为:
    # [num_images, img_height * img_width * num_channels]
   

    #返回展平层、特征值维度

'''
全连接层
new_fc_layer
args：
    前一层输入
    前一层输入维度
    输出维度
    是否使用relu
    
return：全连接层

'''
def new_fc_layer(input_d,input_shape,out_shape,use_relu=True):
    weights=new_weights(fileshape=[input_shape,out_shape])
    bias=new_biases(out_shape)
    layer=tf.matmul(input_d,weights)+bias
    if use_relu:
        layer=tf.nn.relu(layer)
    return layer




    # 权重和偏置
    

    # 计算 y = wx + b
   

    # 是否使用RELU
    

    #返回层
