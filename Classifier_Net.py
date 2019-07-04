# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:28:23 2018

@author: bb
"""


import tensorflow as tf
import numpy as np;

from baseNet import NetWork

class Classifier_Net():
    def __init__(self, x, y, hidden_units = 1024, alpha = 0.2, delta = 1e-3):
        self.x = x;
        self.y = y;
        self.out = self.x;
        self.iput_units = self.x.shape[1].value;
        self.out_units = self.y.shape[1].value;
        self.hidden_units = hidden_units;
        self.net = NetWork(None);
        self.alpha = alpha;
        self.delta = delta;
        self.step2 = 1;
        self.build();
    '''
    这里可以考虑使用分类算法了
    但进来的y 不是one-hot，所以还是用回归算了
    '''
    def build(self):
        self.w1 = self.net._norm_variable('lay1', [self.iput_units, self.hidden_units]);
        self.w1_mask = self.net._norm_variable('lay1_mask', self.w1.shape);
        # 重要数据，每一次训练完之后需要在上面减去w_mask_bi，下次用的时候再加上。是个全局变量
        self.identify_mask_1 = tf.Variable(np.ones([self.iput_units, self.hidden_units]), 
                                           dtype = tf.float32, trainable = False); 
        
        self.w2 = self.net._norm_variable('lay2', [self.hidden_units, self.out_units]);
        self.w2_mask = self.net._norm_variable('lay2_mask', self.w2.shape);
        self.identify_mask_2 = tf.Variable(np.ones([self.hidden_units, self.out_units]),
                                           dtype = tf.float32, trainable = False);
        
        # 这个是要存到每一个identifier里面的
        self.w1_mask_bi = tf.nn.sigmoid(self.w1_mask); 
        self.w1_mask_bi = tf.multiply(self.w1_mask_bi, self.identify_mask_1); 
        self.w2_mask_bi = tf.nn.sigmoid(self.w2_mask);
        self.w2_mask_bi = tf.multiply(self.w2_mask_bi, self.identify_mask_2);
        
#        self.w1_mask_bi = tf.nn.sigmoid(self.w1_mask); 
#        self.w1_mask_bi = tf.multiply(self.w1_mask_bi, self.identify_mask_1); # 这两步应该是没有同步mask修改。所以存在一定的问题
        w11 = tf.multiply(self.w1, self.w1_mask_bi);
        self.out = tf.nn.tanh(tf.matmul(self.out, w11));
#        self.reg = tf.nn.l2_loss(w1);
#        self.reg = self.alpha * self.reg;
        self.reg_mask = tf.reduce_sum(tf.abs(self.w1_mask_bi));

        
#        self.w2_mask_bi = tf.nn.sigmoid(self.w2_mask);
#        self.w2_mask_bi = tf.multiply(self.w2_mask_bi, self.identify_mask_2);
        w22 = tf.multiply(self.w2, self.w2_mask_bi);
        
        self.out = tf.nn.softmax(tf.matmul(self.out, w22));
#        self.reg = self.reg + tf.nn.l2_loss(w2);
#        self.reg = self.alpha * self.reg;
        self.reg_mask += tf.reduce_sum(tf.abs(self.w2_mask_bi));
        self.loss1 = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.y, logits= self.out));
        self.global_step_classifier = tf.Variable(
                name = 'global_step_classifier', initial_value = 1);
        
    def loss(self, reg = True):
        return self.loss1 + self.alpha * self.reg_mask;
        
    def optimizer(self):
        self.opt_classifier = tf.train.AdamOptimizer(1e-2).minimize(
                self.loss(), global_step = self.global_step_classifier);
        return self.opt_classifier;
    
    def renewmask(self, sess):
        # 对现有变量进行分配值;
        sess.run([tf.assign(self.w1_mask, tf.random_normal(self.w1_mask.shape, stddev = 0.05)),
                  tf.assign(self.w2_mask, tf.random_normal(self.w2_mask.shape, stddev = 0.05))]);
        self.step2 = 1;
        
    
    def mask(self, sess, end = False):
        mask = {'lay1': self.w1_mask_bi.eval(session = sess).copy(),
                'lay2': self.w2_mask_bi.eval(session = sess).copy()};
        
        if self.step2 == 1:
            self.step2 = 1 - self.step2;
            ass1 = tf.assign(self.w1_mask, 
                         tf.where(self.w1_mask_bi >= self.delta, 
                                  tf.ones_like(self.w1_mask) * 1000, 
                                  tf.ones_like(self.w1_mask) * -1000));
            ass2 = tf.assign(self.w2_mask, 
                         tf.where(self.w2_mask_bi >= self.delta, 
                                  tf.ones_like(self.w2_mask) * 1000, 
                                  tf.ones_like(self.w2_mask) * -1000));
            sess.run([ass1, ass2]);
            
        if end: #将可用的位置不断减少
            mask = {'lay1': self.w1_mask_bi.eval(session = sess).copy(),
                    'lay2': self.w2_mask_bi.eval(session = sess).copy()};
            sess.run(tf.assign(self.identify_mask_1, self.identify_mask_1 - mask['lay1']));
            sess.run(tf.assign(self.identify_mask_2, self.identify_mask_2 - mask['lay2']));
            self.renewmask(sess);
        return mask;
    
    def pred(self, sess = None, mask = None):
        opts = None;
        if mask is not None:
            idx_mask_1_bak = self.identify_mask_1.eval(session = sess);
            idx_mask_2_bak = self.identify_mask_2.eval(session = sess);
#            sess.run(tf.assign(self.identify_mask_1, self.identify_mask_1 + mask['lay1']));
#            sess.run(tf.assign(self.identify_mask_2, self.identify_mask_2 + mask['lay2']));
            sess.run(tf.assign(self.identify_mask_1, mask['lay1']));
            sess.run(tf.assign(self.identify_mask_2, mask['lay2']));
            self.step2 = 1;
            ass1 = tf.assign(self.w1_mask, 
                         tf.where(mask['lay1'] >= self.delta, 
                                  tf.ones_like(self.w1_mask) * 1000, 
                                  tf.ones_like(self.w1_mask) * -1000));
            ass2 = tf.assign(self.w2_mask, 
                         tf.where(mask['lay2'] >= self.delta, 
                                  tf.ones_like(self.w2_mask) * 1000, 
                                  tf.ones_like(self.w2_mask) * -1000));
            sess.run([ass1, ass2]);
            
#            opts = [tf.assign(self.identify_mask_1, self.identify_mask_1 - mask['lay1']),
#                    tf.assign(self.identify_mask_2, self.identify_mask_2 - mask['lay2'])];
            opts = [tf.assign(self.identify_mask_1, idx_mask_1_bak),
                    tf.assign(self.identify_mask_2, idx_mask_2_bak)];
        return self.out, opts;