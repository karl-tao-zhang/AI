import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

print('读取数据')
mnist=input_data.read_data_sets('../../datas/mnist/',one_hot=True)
