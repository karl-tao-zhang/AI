import numpy as np
import tensorflow as tf

# 使用不同方式定义二维序列,并输出类型
def fun01():
    #list
    m1=[[1.0,2.0],[3.0,4.0]]
    m2 =np.array([[1.0, 2.0], [3.0, 4.0]])
    # 使用tf中的constant
    m3=tf.constant([[1.0,2.0],[3.0,4.0]],dtype=tf.float32)

    # 把m1,m2转换为tensor类型 t1,t2
    t1=tf.convert_to_tensor(m1,dtype=tf.float32)
    t2=tf.convert_to_tensor(m2,dtype=tf.float32)

    # 对t1进行取负运算
    t1_neg=tf.negative(t1)

    # 对t1,t2进行加运算
    t_add=tf.add(t1 ,t2)
    # 输出运算结果
    print(t1_neg)
    print(t_add)



    print(type(m1))
    print(type(m2))
    print(type(m3))


def fun02():
    p1=tf.placeholder(dtype=tf.float32,name='p1')
    p2=tf.placeholder(dtype=tf.float32,name='p2')
    res=tf.add(p1,p2)
    print(res)
    sess=tf.Session()
    res2=sess.run(res,feed_dict={p1:3.0,p2:4.0})
    print(res2)
    sess.close()



fun02()


