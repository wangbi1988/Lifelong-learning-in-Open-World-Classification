# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 20:50:08 2018

@author: bb
"""

import numpy as np;
import matplotlib.pyplot as plt
import tensorflow as tf

from baseNet import NetWork;
from Identify_Net import Identify_Net;
from Classifier_Net import Classifier_Net;
from tensorflow.examples.tutorials.mnist import input_data
from DataSet import DataSet;
import h5py
from scipy.sparse import csr_matrix as csr

class Brain():
    def __init__(self, input_dim, output_dim):
        self.net = NetWork(None);
        self.DELTA = 1e-3;
        self.INPUT_DIM = input_dim;
        self.OUTPUT_DIM = output_dim;
        self.x = tf.placeholder(tf.float32, shape = [None, self.INPUT_DIM]); # 用autoencoder压缩到20维
        self.noisy = tf.placeholder(tf.float32, shape = [None, self.INPUT_DIM]); # 用autoencoder压缩到20维
        self.y = tf.placeholder(tf.float32, shape = [None, output_dim]);
        self.identify_idx = 0;
        self.__build();
#        self.dict_set = {};

    def __build(self):
        '''
        构建网络
        '''
        with tf.variable_scope('identify_net'):
            '''
            用来验证输入的是不是一个已经学习到的知识
            先将知识用超大的fc进行记忆后，再来训练这个网络
            理想状态是可以输出占用的weight位置，二值{0，1}；是个稀疏矩阵，应该进行压缩
            读取的时候应该是对应一个特定的输入，从而输出。
            '''
            n = 20;
            self.identify_nets = np.array([Identify_Net(self.x, self.noisy,
                                                        index = i, delta = 2e-1) for i in range(n)]);
            
            self.opt_identifys = [self.identify_nets[i].optimizer(False) for i in range(n)];
            self.loss_identifys = [self.identify_nets[i].loss(False) for i in range(n)];
            
        with tf.variable_scope('classifier_net'):
            self.classifier_net = Classifier_Net(self.x, self.y, delta = self.DELTA);
            self.opt_classifier = self.classifier_net.optimizer();
            self.loss_classifier = self.classifier_net.loss();
            self.pred_classifier, _ = self.classifier_net.pred();
                    
        self.sess = tf.InteractiveSession();
        self.sess.run(tf.global_variables_initializer());
    
    '''
    广播给所有的验证网络，用来判断是否属于该网络
    返回结果集，known包括具体可被识别的单元id和数据id
    unknown包括不可识别的数据id
    '''
    def identify(self, data):
        idxs = np.arange(data.shape[0]);
        known_set = {};
        unknown_set = {};
        sets = np.zeros([data.shape[0], 1]) - 1;
        if self.identify_idx <= 0:
            return {'known': [], 'unknown': {'idx':idxs}};
        for i in range(self.identify_idx):
            learned = self.identify_nets[i].learned();
#            learned_identify_set = self.sess.run(learned, {self.x: data})[:, 1]; # 找出预测比，而后取得最大值。
            learned_identify_set = self.sess.run(learned, {self.x: data}); # 找出预测比，而后取得最大值。
            learned_identify_set = np.where(learned_identify_set 
                                            < (1 - self.identify_nets[i].threshold()),
                                            0, learned_identify_set)
            sets = np.column_stack((sets, learned_identify_set));
#            print(learned_identify_set)
        max_sets = np.max(sets, 1);
        argmax_sets = np.argmax(sets, 1) - 1;
        idx = (max_sets == 0);
        unknown_set['idx'] = idxs[idx];
        idxs = idxs[~idx];
        argmax_sets = argmax_sets[idxs];
        max_sets = max_sets[idxs];
        for i in np.unique(argmax_sets):
            known_set[i] = idxs[argmax_sets == i];
        return {'known': known_set, 'unknown': unknown_set};
    
    def identify2(self, data):
        sets = np.zeros([data.shape[0], 1]) - 1;
        for i in range(self.identify_idx):
            learned = self.identify_nets[i].learned();
#            learned_identify_set = self.sess.run(learned, {self.x: data})[:, 1]; # 找出预测比，而后取得最大值。
            learned_identify_set = self.sess.run(learned, {self.x: data}); # 找出预测比，而后取得最大值。
            learned_identify_set = np.where(learned_identify_set 
                                            < (1 - self.identify_nets[i].threshold()),
                                            -learned_identify_set, learned_identify_set)
            sets = np.column_stack((sets, learned_identify_set));
            
        return sets;
    
    def identify_whole2one(self, data):
        idxs = np.arange(data.shape[0]);
        known_set = {};
        unknown_set = {};
        for i in range(self.identify_idx):
            learned = self.identify_nets[i].learned();
            learned_identify_set = self.sess.run(learned, {self.x: data});
            
#            learned_identify_set = np.hstack(learned_identify_set);
#            print(learned_identify_set)
            learned_identify_set = np.sum(np.abs(learned_identify_set - 1) < self.identify_nets[i].delta) / learned_identify_set.shape[0];
#            print(learned_identify_set)
            if learned_identify_set <= (1 - 0.3):
                continue;
            # 将集合里面为真的选出来，之后给指定的分类器进行分类预测
#            print(learned_identify_set.shape)
            if known_set.__contains__(i):
                known_set[i] = np.append(known_set[i], idxs);
            else:
                known_set[i] = idxs;

            data = [];
            idxs = np.array([]);
            break;
        unknown_set['idx'] = idxs;
        return {'known': known_set, 'unknown': unknown_set};
    
    def pred(self, known_set, data):
        dict_set = {};
        for i in range(self.identify_idx):
            if not known_set.__contains__(i):
                continue;
            mask = self.identify_nets[i].mask();# 取出mask
            _, opts = self.classifier_net.pred(sess = self.sess, mask = mask);
            idx = known_set[i];
            sub_data = data[idx];
            # 进行区别
            rls = self.sess.run(self.pred_classifier, {self.x: sub_data}); 
            self.sess.run(opts)
            self.classifier_net.renewmask(self.sess);
            dict_set[i] = {'idx': idx, 'rls': rls};
        return dict_set;
            
    def train(self, dataset, MAX_EPOCH = 40, BATCH_SIZE = 256):
        train = dataset;
        
        N = train.data.shape[0];
        ITER = np.floor(N / BATCH_SIZE).astype(np.int32) + 1;
        MAX_EPOCH = np.ceil(15000/ ITER).astype(np.int32);
        loss_set = np.zeros(ITER * MAX_EPOCH);
        print('begin to trian classifier_net');
        with tf.device("/gpu:0"):
            for epoch in range(MAX_EPOCH):
                for i in range(ITER):
                    batch_x, batch_y = train.next_batch(BATCH_SIZE);
                    _, l = self.sess.run(
                            [self.opt_classifier, self.loss_classifier], 
                            {self.x: batch_x, self.y: batch_y});
                    loss_set[epoch * ITER + i] = l;
    
                acc = np.inf;
                if (epoch + 1) % np.ceil(MAX_EPOCH/ 2).astype(np.int32) == 0:
                    mask = self.classifier_net.mask(self.sess);
    #                print(mask['lay2'])
                    print(np.sum(mask['lay1'] > self.DELTA),
                          (np.sum(mask['lay2'] > self.DELTA)))
                if epoch % 100 == 0:
                    print('epoch is {} with loss {}, and devset acc is {}'
                          .format(epoch, np.mean(loss_set[np.arange(ITER) + epoch * ITER]), acc));
#                  
        mask = self.classifier_net.mask(self.sess, end = True);
        print((np.sum(mask['lay1'] > self.DELTA) / np.sum(mask['lay1'] >=0),
               np.sum(mask['lay2'] > self.DELTA) / np.sum(mask['lay2'] >=0)));

    
        #记忆
        identify_net = self.identify_nets[self.identify_idx];
        identify_net.mask(mask = mask);
        
        MAX_EPOCH = MAX_EPOCH * 2;
        opt_identify = self.opt_identifys[self.identify_idx];
        loss_identify = self.loss_identifys[self.identify_idx];
        loss_set = np.zeros(ITER * MAX_EPOCH);
        with tf.device("/gpu:0"):
            for epoch in range(MAX_EPOCH):
                for i in range(ITER):
                    batch_x, batch_y = train.next_batch(BATCH_SIZE);
                    _, l = self.sess.run([opt_identify,
                                          loss_identify], {self.x: batch_x});
                    loss_set[epoch * ITER + i] = l;
                if epoch % 50 == 0:
                    print('epoch is {} with loss {}'.format(epoch, 
                          np.mean(loss_set[np.arange(ITER) + epoch * ITER])));
        identify_net.threshold(np.mean(loss_set[np.arange(ITER) + epoch * ITER]) * 1.2);
#        identify_net.threshold(np.mean(loss_set[np.arange(ITER) + epoch * ITER]) * 1.2);
        self.identify_idx += 1;
        return;


    
