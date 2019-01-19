#!usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 2018年9月16日

@author: yrh
数据预处理
'''

'''
=======================1、导入需要的包=======================

'''
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

'''
=======================2、卷积神经网络配置=======================

'''
# 卷积层 1
#filter_size1 :   5 x 5 卷积核
#num_filters1 :  共 16 个卷积核
filter_size1=5
num_filters1=16

# 卷积层 2
#filter_size2 : 5 x 5 卷积核
#num_filters2 : 共 36 个卷积核
filter_size2=5
num_filters2=36

img_size=28  #图片大小
img_flat=img_size*img_size #图片扁平化后大小
img_shape=(img_size,img_size)  #图片维度
# 全连接层
#fc_size：128个神经元
fc_size=128

input_channels=1 #颜色通道数
output_size=10 #输出类别数量

'''
=======================3、载入数据 =========================

TensorFlow在样例教程中已经做了下载并导入
     MNIST数字手写体识别数据集的实现，可以直接使用
     
以下代码会将MNIST数据集下载到data/MNIST2目录下，
    将标签保存为one-hot编码
load_datas(filename)    
========================================================
'''
def loadData(filename):
    mnist=input_data.read_data_sets(filename,one_hot=True)
    mnist.train.cls=np.argmax(mnist.train.labels,axis=1)
    mnist.test.cls = np.argmax(mnist.test.labels, axis=1)
    mnist.validation.cls = np.argmax(mnist.validation.labels, axis=1)
    return mnist







'''
MNIST数据集总共有70000张手写数字图片，
数据集被分为训练集、测试集和验证集三部分
Size of: 
-Training-set: 55000 
-Test-set: 10000 
-Validation-set: 5000

'''


 






 
'''
=======================4、数据维度=========================
'''
# 图片大小 img_size

 
# 图片扁平化  img_size_flat

 
# 图像维度  img_shape

 
#颜色通道  num_channels 

 
#标签类别  num_classes

 
'''
=======================图片数据可视化=======================
在3x3的栅格中显示9张图像
plot_images(images, cls_true, cls_pred=None)
'''
def viewData(imgs,cls,pred=None):
    assert  len(imgs)==len(cls)==9
    fig,axs=plt.subplots(3,3)
    for i,ax in enumerate(axs.flat):
        ax.imshow(imgs[i].reshape(img_shape),cmap='binary')
        lbl='' #文字
        if pred is None:  # 如果没有预测结果
            lbl='cls:{0}'.format(cls[i]) #只显示正确标签
        else: #否则同时显示正确标签和预测结果
            lbl='cls:{0};pred:{1}'.format(cls[i],pred[i])
        ax.set_xlabel(lbl)
        ax.set_xticks([])  # 取消刻度
        ax.set_yticks([])
    plt.show()



mnist=loadData('../dataset/mnist')
viewData(mnist.train.images[0:9],mnist.train.cls[0:9])










     
# --------------代码测试-----------
# 测试数据

# 测试数据标签

# 对测试数据和标签进行可视化

      
    
