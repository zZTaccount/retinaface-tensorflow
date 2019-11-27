'''
loss.py
用于定义loss函数
'''
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import slim
from rcnn.config import config
import pdb
os.environ['CUDA_VISIBLE_DEVICES']='1'

def smoothL1(diff, sigma):
    diff = tf.abs(diff)
    f = tf.cast(1/tf.pow(sigma,2), tf.float32)
    # f = tf.Print(f, [f], 'INFO: f =', first_n=3, summarize=20)
    position = tf.to_float(tf.less(diff, f))
    position = tf.cast(position, tf.float32)
    # position = tf.Print(position, [tf.shape(position),position], 'INFO: position =', first_n=3, summarize=20)
    # 实现公式中的条件分支
    loss = tf.pow(sigma*diff, 2) * (0.5) * position + (diff - 0.5/tf.cast(tf.pow(sigma,2), tf.float32)) * (1. - position)
    # loss = tf.Print(loss, [tf.shape(loss),loss], 'INFO: loss =', first_n=3, summarize=20)
    loss = tf.reduce_mean(tf.reduce_sum(loss,axis=[1,2]))
    return loss

def  softmax_loss(logits, labels, name="softmax"):
    '''
    :param logits: (batch_size, w*h*anchornum, 2)
    :param labels: (batch_size, w*h*anchornum)
    :return: cls_loss
    '''
    # cls_loss = slim.losses.sparse_softmax_cross_entropy(logits, label, weights=label_weight, scope=name)
    cls_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name=name)
    return cls_losses

def focal_loss(logits,labels,alpha=0.25,gama=2):
    '''
    :param logits: (batch_size, w*h*anchornum, 2)
    :param labels: (batch_size, w*h*anchornum)
    :param alpha: alpha越大，正样本越重要
    :param gama: gama越大，难样本越重要
    :return:
    '''
    logits = tf.Print(logits, [tf.shape(logits),logits], '\nINFO: logits=',summarize=15)
    prob = tf.nn.softmax(logits)[:,:,1]  #对正样本的打分做归一化
    prob = tf.Print(prob, [tf.shape(prob),prob], 'INFO: softmax >prob=',summarize=15)
    prob = tf.cast(prob, tf.float32)
    position = tf.to_float(tf.equal(labels, 1))
    # position = tf.Print(position, [tf.shape(position), position], 'INFO: >>position=', summarize=15)
    pt = prob*position + (1-prob)*(1-position)
    alphat = alpha*position + (1-alpha)*(1-position)
    loss = -alphat * tf.pow(1-pt , gama) * tf.log(pt+1e-5)
    loss = tf.reduce_sum(loss)
    # return prob, pt, loss
    return loss


def test_focalloss():
    a = np.array((((1.,200.),(2.,300.),(3.,300.)), ((4., 500.),(5.,500.),(6.,3.))))
    b = np.array(((1,1,1),(1,1,0)))
    prob, pt, loss = focal_loss(a,b,0.5,1)
    with tf.Session() as sess:
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        pr, ptt,lo = sess.run((prob,pt,loss))
    print(pr,'\n', ptt,'\n',lo)

def test_smothL1():
    a = np.array(((1.,2.,3.),(4., 5., 6.)))
    b = np.array(((1.2,2.1,5.),(4.1, 4.9, 7.1)))
    loss = smoothL1(a,b)
    with tf.Session() as sess:
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        loss = sess.run(loss)
    print('smoothL1 loss = ',loss)

def test_softmax():
    tf.reset_default_graph()
    labels = [[1, 1], [1, 0]]

    logits = [[[1, 2],
              [1, 10]],
              [[10.1, 1],
              [20, 1]]]
    logits = tf.constant(logits)
    # logits = -logits

    loss = softmax_loss(logits, labels)
    with tf.Session() as sess:
        print(sess.run(loss))

# test_smothL1()
# test_softmax()
# test_focalloss()

