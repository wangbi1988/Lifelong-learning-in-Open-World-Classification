# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 14:11:16 2018

@author: bb
"""
import tensorflow as tf

from baseNet import NetWork

class Identify_Net():
    def __init__(self, x, noisy, index = 1, alpha = 0.3, delta = 1e-5, hidden_units = 10):
        self.x = x;
        self.y = 1; # 学过的都标记为1
        self.iput_units = self.x.shape[1].value;
        self.hidden_units = hidden_units;
        self.out_units = 2;
        self.out = self.x;
        self.net = NetWork(None);
        self.alpha = alpha;
        self.delta = delta;
        self.index = index;
        self.masked = None;
        self.random_noisy = tf.random_normal([self.iput_units, self.iput_units]);
        self.noisy = noisy;
        self.build();
        self.v = 1;
        return;
        
    def build(self):
        with tf.variable_scope('Identify_{}'.format(self.index)):
            w = self.net._cons_variable('lay1', 
                                        [self.iput_units, self.hidden_units]);
            b = self.net._cons_variable('lay1b', 
                                        [1, self.hidden_units])
            self.out = tf.nn.softplus(tf.matmul(self.out, w) + b) # ? * 10
            self.reg = tf.nn.l2_loss(w);
            
            w2 = self.net._cons_variable('lay11', 
                                        [self.hidden_units, 5]);
            b2 = self.net._cons_variable('lay11b', 
                                        [1, 5])
            self.out = tf.nn.softplus(tf.matmul(self.out, w2) + b2);
            self.reg = tf.nn.l2_loss(w2);
            
            #decoder
            w2 = self.net._cons_variable('dlay11', 
                                        [5, self.hidden_units]);
            b2 = self.net._cons_variable('dlay11b', 
                                        [1, self.hidden_units])
            self.out = tf.nn.softplus(tf.matmul(self.out, w2) + b2);
            self.reg = tf.nn.l2_loss(w2);
            
            w = self.net._cons_variable('dlay1', 
                                        [self.hidden_units, self.iput_units]);
            b = self.net._cons_variable('dlay1b', 
                                        [1, self.iput_units])
            self.out = tf.nn.softplus(tf.matmul(self.out, w) + b) # ? * 10
            self.reg = tf.nn.l2_loss(w);
#            
            self.loss1 = tf.reduce_mean(tf.reduce_mean(tf.abs(self.x - self.out), 1))\
            
            self.learn = 1 - tf.reduce_mean(tf.abs(self.x - self.out), 1);
            self.global_step_identify = tf.Variable(
                        name = 'global_step_identify', initial_value = 1);
                    
    def build_dis(self):
        with tf.variable_scope('Identify_{}'.format(self.index)):
#            self.noisy = 1.05 - self.out;
            
            self.w = self.net._cons_variable('lay1', 
                                        [self.iput_units, self.hidden_units]);
            b = self.net._cons_variable('lay1b', 
                                        [1, self.hidden_units])
#            self.out = tf.pow(tf.matmul(self.out, self.w), 3);
            self.out = (tf.matmul(self.out, self.w))
            self.noisy = tf.nn.softplus(tf.matmul(self.noisy, self.w) + b);
            self.reg = tf.nn.l2_loss(self.w);
            
#            self.w2 = self.net._cons_variable('lay11', 
#                                        [self.hidden_units, self.hidden_units]);
#            b2 = self.net._cons_variable('lay11b', 
#                                        [1, self.hidden_units])
##            self.out = tf.nn.softplus(tf.matmul(self.out, w));
#            self.out = tf.nn.softplus(tf.matmul(self.out, self.w2) + b2)
#            self.noisy = tf.nn.softplus(tf.matmul(self.noisy, self.w2) + b2);
#            self.reg = tf.nn.l2_loss(self.w2);
#            
#            self.w22 = self.net._cons_variable('lay21', 
#                                        [self.hidden_units, self.hidden_units]);
#            b2 = self.net._cons_variable('lay21b', 
#                                        [1, self.hidden_units])
#            self.out = tf.nn.softplus(tf.matmul(self.out, self.w22) + b2)
#            self.noisy = tf.nn.softplus(tf.matmul(self.noisy, self.w22) + b2);
#            self.reg = tf.nn.l2_loss(self.w22);
#            
#            self.w23 = self.net._cons_variable('lay22', 
#                                        [self.hidden_units, self.hidden_units]);
#            b2 = self.net._cons_variable('lay22b', 
#                                        [1, self.hidden_units])
##            self.out = tf.nn.softplus(tf.matmul(self.out, w));
#            self.out = tf.nn.softplus(tf.matmul(self.out, self.w23) + b2)
#            self.noisy = tf.nn.softplus(tf.matmul(self.noisy, self.w23) + b2);
#            self.reg = tf.nn.l2_loss(self.w23);
            
            self.w3 = self.net._norm_variable('lay10', [self.hidden_units, self.out_units]);
            self.out = tf.nn.softmax(tf.matmul(self.out, self.w3));
            self.noisy = tf.nn.softmax(tf.matmul(self.noisy, self.w3));
#            self.out = (tf.matmul(self.out, self.w3));
#            self.noisy = (tf.matmul(self.noisy, self.w3));
            self.reg += tf.nn.l2_loss(self.w3);

#            self.loss1 = tf.reduce_sum(tf.squared_difference(self.out , tf.transpose([self.x[:, -1]])))\
#            + 0.4 * tf.reduce_mean(tf.squared_difference(self.noisy, 0));
            self.loss1 = 0.7 *tf.reduce_mean(-tf.log(self.out[:, 1]))\
#            + 0.3 * tf.reduce_mean(-tf.log(self.noisy[:, 0]));
            
#            self.learn = 1 -  tf.abs(tf.subtract(self.out, tf.transpose([self.x[:, -1]])));
            self.learn = self.out;
#            print(self.out.shape)
#            print(self.learn.shape)
            self.global_step_identify = tf.Variable(
                        name = 'global_step_identify', initial_value = 1);
        
    def loss(self, reg = True):
        return self.loss1;
             
    def optimizer(self, reg = True):
        self.opt_identify = tf.train.RMSPropOptimizer(
                learning_rate = 1e-2).minimize(
                        self.loss(reg), global_step = self.global_step_identify);
        return self.opt_identify;
    
    def learned(self):
        return self.learn;
    
    def mask(self, mask = None):
        if mask is None:
            return self.masked;
        else:
            self.masked = mask;
            return self.masked;
        
    def threshold(self, v = None):
        if v is None:
            return self.v;
        else:
            self.v = v;