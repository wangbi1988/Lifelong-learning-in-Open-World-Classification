# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 09:53:58 2018

@author: 王碧
"""

from baseNet import NetWork
import numpy as np
import tensorflow as tf

class KBNets(): # 分两种，一种是存的，一种是学的。学的用DQL。
    """
    知识库，最好可以定义一个自学习的方法，用多线程，类似于任务计划;
    主要是将NN当作存储设备来使用;
    这样可以使得我们抛开原有的信息，例如下标，指针等。
    假设成一个完全未知的agent来学习。
    """
    """
    现在还没有解决关于存储的问题，所以不会讨论如何增量存储的问题；
    但有想法是否可以通过“沉淀”来降低或者阻断新增知识对以往知识的梯度更新；
    或者，利用weight-mask来点亮对应区域内的weight？
    """
    """
    关键是要记住，最好就是过拟合。泛化能力不是通过控制拟合获得，而是通过逻辑推理获得
    """
    """
    params = {n_features: 10, n_lay:3, type_1:fc, hidd_1: 64, type_2:fc, hid_2: 32, type_3:out, hid_3: 2}
    """
    def __init__(self, params):
        n_lay = params['n_lay'];
        self.net = NetWork(None);
        self.x_ = tf.placeholder(tf.float32, shape = [None, params['n_features']]);
        self.out = None;
        h_hat = params['n_features'];
        for i in range(n_lay):
            t = params['type_{}'.format(i)];
            h = params['hidd_{}'.format(i)];
            if t == 'fc':
                self.out = self.net.fc_block(self.out, shape = [h_hat, h]);
                
            h_hat = h;
            
        self.y_ = tf.placeholder(tf.float32, shape = [None, h_hat]);
        
        self.loss = tf.reduce_sum(tf.squared_difference(self.out, self.y_));
        
        lr = params['lr'];
        if lr is None:
            lr = 1e-3;
        self.opt = tf.train.AdamOptimizer(lr).minimize(self.loss);
        return;
        
    def pred(self, infos, sess):
        o = sess.run([self.out], {self.x_ : infos});
        return o;
        
    """
    ngram
    """
    def train(self, infos, sess):
        i = 0;
        x_ = infos['x_'];
        y_ = infos['y_'];
        while i < 100000:
            _, l = sess.run([self.opt, self.loss], {self.x_: x_, self.y_: y_});
            if (l - 0) < 1e-10:
                break;
        return;

        
class Agent():
    LEARNED = 0;
    UNKNOW = 1;
    
    """
    params = {n_features: 10}
    """
    def __init__(self, params):
        self.info_idx = KBNets(); # 二分类网络，用来记录是否学习过，强过拟合
        self.infos = KBNets(); # 信息存储网络，通过输入的信息，对n-gram进行自组合训练，用同一个网络输出
        return;
    
    def learn(self, infos, ngram, sess): # 暂不支持增量学习，这是个问题。
        self.info_idx.train(infos, sess);
        self.infos.train(infos, sess, ngram);
        return;
    
    """
    用二分类进行判别。两个网络1-w和w。在保证sum f = 1的时候，sum(1-w)f=1-sum(wf)=1-t
    """
    def isLearn(self, infos, sess):
        rs = self.info_idx.pred(infos, sess);
        return np.argmax(rs);
        
    
    def asked(self, infos, sess):
        return self.infos.pred(infos, sess);
    