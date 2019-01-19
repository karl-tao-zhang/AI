"""
 使用单层神经网络实现AND运算
"""
"""
 ============导入
"""
import tensorflow as tf

"""
 1 定义训练样本和样本标签
"""
datas=[[1,0],[0,0],[0,1],[1,1],[1,1],[1,1]]
lbls=[[1],[0],[1],[1],[1],[1]]

"""
 2 定义训练样本和样本标签占位符
 只指定第二维大小,第一维样本数量为None,表示由系统自动计算
 
"""
X=tf.placeholder(dtype=tf.float32,shape=[None,2])
Y=tf.placeholder(dtype=tf.float32,shape=[None,1])
"""
 3 定义变量:权重参数 阈值,0均值0.01标准差的正态分布随机数
"""
w=tf.Variable(tf.random_normal([2,1]),name='w')
b=tf.Variable(tf.random_normal([1]),name='b')
"""
 4 前向传播进行预测
"""
hx=tf.sigmoid((tf.matmul(X,w)+b))

"""
 5 计算预测误差
 (y*log(hx))+((1-y)*log(1-hx))
"""
cost=-tf.reduce_mean( Y*tf.log(hx)+(1-Y)*tf.log(1-hx) )
train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 最终预测结果
predict=tf.cast(hx>0.5,dtype=tf.float32)


sess=tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(5000):
    cost_vel,_=sess.run([cost,train],feed_dict={X:datas,Y:lbls})
    if i%200==0:
        print('i=',i,'误差:',cost_vel)

h,p=sess.run([hx,predict],feed_dict={X:datas,Y:lbls})
print('预测结果:',p)












