#!usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 2018年7月28日

@author: yrh
'''
import tensorflow as tf
from src.MnistDemo import *
from src.Tf_Graph import *
from src.Placeholder import *
import math
import time
from datetime import timedelta
from sklearn.metrics import confusion_matrix 
import os
'''
==========================================
        保存训练模型
====================================================
'''
# 为了保存神经网络的变量，我们创建一个称为Saver-object的对象，
# 它用来保存及恢复TensorFlow图的所有变量
# 保存操作在后面的optimize()函数中完成

saver=tf.train.Saver()
save_dir='checkpoints/'   # 保存路径
if not os.path.exists(save_dir):  #如果路径不存在
    os.makedirs(save_dir)   # 就创建它
    
save_path=os.path.join(save_dir,'best_validation')  #模型保存路径




'''
==========================================
        运行Placeholder占位符
====================================================
'''

'''
1、创建session
'''
session=tf.Session()
session.run(tf.global_variables_initializer())

'''
2、执行优化函数
'''
train_batch_size=64

#当前迭代次数
total_iterations=0


def optimize(num_iterations):
    #初始化变量
    global total_iterations
    
    # 最优验证准确率.
    best_validation_accuracy = 0.0
    
    # 最后更新验证准确率时的迭代次数.
    last_improvement = 0
    
    # 停止更新优化.
    require_improvement = 1000
    
    
    # 用来输出用时.
    start_time = time.time()
    for i in range(total_iterations,total_iterations+num_iterations):
         # 获取一批数据，放入dict
        x_batch, y_true_batch = mnist.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch,
                          y_true: y_true_batch}
        # 运行优化器
        session.run(optimizer, feed_dict=feed_dict_train)

        # 每100轮迭代输出状态
        if i % 100 == 0:
            # 计算训练集准确率.
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "迭代轮次: {0:>6}, 训练准确率: {1:>6.1%}"
            print(msg.format(i + 1, acc))
            
            acc_validation, _ = validation_accuracy()
            if acc_validation>best_validation_accuracy:
                best_validation_accuracy=acc_validation
                last_improvement=total_iterations
                 
                saver.save(sess=session, save_path=save_path)

               
                improved_str = '*'
            else:
                
                improved_str = ''
            
            
            msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Validation Acc: {2:>6.1%} {3}"

            # 打印.
            print(msg.format(i + 1, acc, acc_validation, improved_str))

        # 如果超过1000次验证准确率没有更新.
        if total_iterations - last_improvement > require_improvement:
            print("No improvement found in a while, stopping optimization.")

            # 停止优化.
            break
            
    total_iterations += num_iterations

    end_time = time.time()
    time_dif = end_time - start_time

    # 输出用时.
    print("用时: " + str(timedelta(seconds=int(round(time_dif)))))
    


'''
3、输出部分错误样例与混淆矩阵
'''
def plot_example_errors(cls_pred, correct):
    # 计算错误情况
    incorrect = (correct == False)
    images = mnist.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = mnist.test.cls[incorrect]

    # 随机挑选9个
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    indices = indices[:9]

    viewData(images[indices], cls_true[indices], cls_pred[indices])

def plot_confusion_matrix(cls_pred):
    cls_true = mnist.test.cls  # 真实类别  

    # 使用scikit-learn的confusion_matrix来计算混淆矩阵
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)

    # 打印混淆矩阵
    print(cm)

    # 将混淆矩阵输出为图像
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # 调整图像
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(output_size)
    plt.xticks(tick_marks, range(output_size))
    plt.yticks(tick_marks, range(output_size))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

'''
4、测试
'''

# 将测试集分成更小的批次
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    # 测试集图像数量.
    num_test = len(mnist.test.images)

    # 为预测结果申请一个数组.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # 数据集的起始id为0
    i = 0
    while i < num_test:
        # j为下一批次的截止id
        j = min(i + test_batch_size, num_test)

        # 获取i，j之间的图像
        images = mnist.test.images[i:j, :]

        # 获取相应标签.
        labels = mnist.test.labels[i:j, :]

        # 创建feed_dict
        feed_dict = {x: images,
                    y_true: labels}

        # 计算预测结果
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # 设定为下一批次起始值.
        i = j

    cls_true = mnist.test.cls
    # 正确的分类
    correct = (cls_true == cls_pred)
    # 正确分类的数量
    correct_sum = correct.sum()
    # 分类准确率
    acc = float(correct_sum) / num_test

    # 打印准确率.
    msg = "测试集准确率: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # 打印部分错误样例.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # 打印混淆矩阵.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)    
        
    return correct,cls_pred
 
# 预测类别.
batch_size = 256

def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred 
        
# 计算测试集上的预测类别
def predict_cls_test():
    return predict_cls(images = mnist.test.images,
                       labels = mnist.test.labels,
                       cls_true = mnist.test.cls)

#计算验证集上的预测类别
def predict_cls_validation():
    return predict_cls(images = mnist.validation.images,
                       labels = mnist.validation.labels,
                       cls_true = mnist.validation.cls)
    
    
#分类准确率的帮助函数
#这个函数计算了给定布尔数组的分类准确率，布尔数组表示每张图像是否被正确分类
def cls_accuracy(correct):
    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / len(correct)

    return acc, correct_sum

#计算验证集上的分类准确率
def validation_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the validation-set.
    # The function returns two values but we only need the first.
    correct, _ = predict_cls_validation()
    
    # Calculate the classification accuracy and return it.
    return cls_accuracy(correct)

'''    
#优化前的性能测试
print_test_accuracy()

#执行一轮优化后的性能
optimize(num_iterations=1)
print_test_accuracy()
    
#100轮优化后的性能
optimize(num_iterations=99)
print_test_accuracy()

#1000轮优化后性能
optimize(num_iterations=999)
print_test_accuracy(show_example_errors=True)
'''
#10000轮次优化后的性能
optimize(num_iterations=3000)
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)



'''
==========================================
        权重和层的可视化
====================================================
'''
#卷积权重可视化
def plot_conv_weights(weights, input_channel=0):
    # weights_conv1 or weights_conv2.

    # 运行weights以获得权重
    w = session.run(weights)

    # 获取权重最小值最大值，这将用户纠正整个图像的颜色密集度，来进行对比
    w_min = np.min(w)
    w_max = np.max(w)

    # 卷积核
    num_filters = w.shape[3]

    # 需要输出的卷积核
    num_grids = math.ceil(math.sqrt(num_filters))

    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        # 只输出有用的子图.
        if i<num_filters:
            # 获得第i个卷积核在特定输入通道上的权重
            img = w[:, :, input_channel, i]

            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # 移除坐标.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
    
#卷积层输出可视化
def plot_conv_layer(layer, image):
    # layer_conv1 or layer_conv2.

    # feed_dict只需要x，标签信息在此不需要.
    feed_dict = {x: [image]}

    # 获取该层的输出结果
    values = session.run(layer, feed_dict=feed_dict)

    # 卷积核
    num_filters = values.shape[3]

    # 每行需要输出的卷积核网格数
    num_grids = math.ceil(math.sqrt(num_filters))

    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        # 只输出有用的子图.
        if i<num_filters:
            # 获取第i个卷积核的输出
            img = values[0, :, :, i]

            ax.imshow(img, interpolation='nearest', cmap='binary')

        # 移除坐标.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
    
#打印输入图像
def plot_image(image):
    plt.imshow(image.reshape(img_shape),
              interpolation='nearest',
              cmap='binary')

    plt.show()
    
    
#打印第一张，第二张图像：
image1 = mnist.test.images[0]
plot_image(image1)

image2 = mnist.test.images[13]
plot_image(image2)

#卷积层 1
#权重
plot_conv_weights(weights=weights_conv1)
#输出
plot_conv_layer(layer=layer_conv1, image=image1)
plot_conv_layer(layer=layer_conv1, image=image2)

#卷积层 2

#第1个通道的权重
plot_conv_weights(weights=weights_conv2, input_channel=0)
#第2个通道的权重
plot_conv_weights(weights=weights_conv2, input_channel=1)
#images1输出
plot_conv_layer(layer=layer_conv1, image=image2)
#images2输出
plot_conv_layer(layer=layer_conv1, image=image2)


#关闭session
session.close()