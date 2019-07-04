# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:07:44 2018

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

def load_mnist(src):
    file = h5py.File('{}_encoder.h5'.format(src), 'r');
    train_x = file['train_x'][:];
    train_y = np.transpose(np.array([file['train_y'][:]]));
    train = DataSet(train_x, train_y, onehot = True);
    
    validation_x = file['validation_x'][:];
    validation_y = np.transpose(np.array([file['validation_y'][:]]));
    validation = DataSet(validation_x, validation_y, onehot = True);
    
    test_x = file['test_x'][:];
    test_y = np.transpose(np.array([file['test_y'][:]]));
    test = DataSet(test_x, test_y, onehot = True);
    
    file.close();
    return train, validation, test;

train, validation, test = load_mnist('mnist');
dict_data = {'train':train, 'validation':validation, 'test':test}
net = NetWork(None);
DELTA = 1e-3;
INPUT_DIM = 20;
x = tf.placeholder(tf.float32, shape = [None, INPUT_DIM]); # 用autoencoder压缩到20维
noisy = tf.placeholder(tf.float32, shape = [None, INPUT_DIM]); # 用autoencoder压缩到20维
y = tf.placeholder(tf.float32, shape = [None, 10]);

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
    identify_net_1 = Identify_Net(x, noisy);
#    identify_net_1.build();
#    loss_identify = identify_net_1.loss(reg = True);
#    pred_identify = identify_net_1.learned();
    
with tf.variable_scope('classifier_net'):
    classifier_net = Classifier_Net(x, y, delta = DELTA);
#    loss_classifier = classifier_net.loss(reg = True);
#    pred_classifier = classifier_net.pred();
    

#global_step_identify = tf.Variable(name = 'global_step_identify', initial_value = 1);
#opt_identify = tf.train.AdamOptimizer(
#        learning_rate = 1e-2).minimize(
#                loss_identify, global_step = global_step_identify);
        
#global_step_classifier = tf.Variable(name = 'global_step_classifier', initial_value = 1);
#opt_classifier = tf.train.RMSPropOptimizer(1e-2).minimize(loss_classifier);
#opt_classifier = tf.train.AdamOptimizer(1e-2).minimize(loss_classifier,
#                                        global_step = global_step_classifier);


sess = tf.Session();
opt_classifier = classifier_net.optimizer();
loss_classifier = classifier_net.loss();
pred_classifier = classifier_net.pred();
opt_identify = identify_net_1.optimizer(False);
loss_identify = identify_net_1.loss(False);
pred_identify = identify_net_1.learned();
sess.run(tf.global_variables_initializer())

MAX_EPOCH = 100;
BATCH_SIZE = 256;
N = train.data.shape[0];
ITER = np.floor_divide(N, BATCH_SIZE);
#loss_set = np.zeros(ITER * MAX_EPOCH);
print('begin to trian classifier_net');
#for epoch in range(MAX_EPOCH):
#    for i in range(ITER):
#        batch_x, batch_y = train.next_batch(BATCH_SIZE);
#        _, l = sess.run([opt_classifier, loss_classifier], {x: batch_x, y: batch_y});
#        loss_set[epoch * ITER + i] = l;
#    op1 = sess.run(pred_classifier, {x: validation.data});
#    arg1 = np.argmax(op1, axis = 1);
#    arg2 = np.argmax(validation.labels, axis = 1);
#    acc = np.sum(np.equal(arg1, arg2)) / arg2.shape[0];
#    if (epoch + 1) % 40 == 0:
#        mask = classifier_net.mask(sess);
#        print(np.sum(mask['lay1'] > DELTA), (np.sum(mask['lay2'] > DELTA)))
#    print('epoch is {} with loss {}, and devset acc is {}'
#          .format(epoch, np.mean(loss_set[np.arange(ITER) + epoch * ITER]), acc));
#          
#pred_y = np.argmax(sess.run(pred_classifier, {x: test.data}), 1);
#print(np.sum(np.equal(pred_y, np.argmax(test.labels, 1)))/ pred_y.shape[0]);
#mask = classifier_net.mask(sess);
#print((np.sum(mask['lay1'] > DELTA) / np.sum(mask['lay1'] >=0),
#       np.sum(mask['lay2'] > DELTA) / np.sum(mask['lay2'] >=0)));
#       
       

def composer(dict_data, sub_index, offset = 0):
    x = dict_data['train'].data[sub_index['train'], :] + offset;
    y = dict_data['train'].labels[sub_index['train'], :];
    train = DataSet(x, y, onehot = False);
    x = dict_data['validation'].data[sub_index['validation'], :] + offset;
    y = dict_data['validation'].labels[sub_index['validation'], :];
    validation = DataSet(x, y, onehot = False);
    x = dict_data['test'].data[sub_index['test'], :] + offset;
    y = dict_data['test'].labels[sub_index['test'], :];
    test = DataSet(x, y, onehot = False);
    return {'train':train, 'validation':validation, 'test':test};

train_labels = np.argmax(train.labels, 1);
validation_labels = np.argmax(validation.labels, 1);
test_labels = np.argmax(test.labels, 1);

sub_train_index = np.less_equal(train_labels, 4); # <= 4
sub_validation_index = np.less_equal(validation_labels, 4); # <= 4
sub_test_index = np.less_equal(test_labels, 4); # <= 4

clip_data = [composer(dict_data, {'train':sub_train_index, 
                                  'validation':sub_validation_index,
                                  'test':sub_test_index})];
for i in range(4):
    sub_train_index = np.equal(train_labels, i + 5);
    sub_validation_index = np.equal(validation_labels, i + 5); 
    sub_test_index = np.equal(test_labels, i + 5);
    clip_data.append(composer(dict_data, {'train':sub_train_index, 
                                          'validation':sub_validation_index,
                                          'test':sub_test_index}));

#记忆
#identify_net_1.mask(mask = {'lay1': csr(mask['lay1']), 'lay2': csr(mask['lay2'])});
    
MAX_EPOCH = 200;
N = clip_data[0]['train'].data.shape[0];
ITER = np.floor_divide(N, BATCH_SIZE);
ls = np.ones(ITER);
for epoch in range(MAX_EPOCH):
    for i in range(ITER):
#        batch_x, batch_y = train.next_batch(BATCH_SIZE);
        batch_x, batch_y = clip_data[0]['train'].next_batch(BATCH_SIZE);
        _, l = sess.run([opt_identify, loss_identify], {x: batch_x, 
                        noisy: (0.95 + 0.1 * (np.random.random(batch_x.shape) - 0.5)) - batch_x});
        ls[i] = l;
#        loss_set[epoch * ITER + i] = l;
#    print('epoch is {} with loss {}'.format(epoch,
#          np.mean(loss_set[np.arange(ITER) + epoch * ITER])));
    print('epoch is {} with loss {}'.format(epoch, np.mean(ls)));
    
batch_x, batch_y = clip_data[0]['test'].next_batch(BATCH_SIZE);
p1 = sess.run(pred_identify, {x: batch_x});
print(np.sum(np.argmax(p1, 1) == 1)/ BATCH_SIZE)
print(np.sum(p1[:, 1] >= 0.9)/ BATCH_SIZE)
#print(np.sum(np.abs(p1 - 1) < 2e-1)/ BATCH_SIZE)

batch_x, batch_y = clip_data[1]['test'].next_batch(BATCH_SIZE);
p2 = sess.run(pred_identify, {x: batch_x});
print(np.sum(np.argmax(p2, 1) == 1)/ BATCH_SIZE)
#print(np.sum(np.abs(p2 - 1) < 2e-1)/ BATCH_SIZE)

batch_x, batch_y = clip_data[2]['test'].next_batch(BATCH_SIZE);
p3 = sess.run(pred_identify, {x: batch_x});
print(np.sum(np.argmax(p3, 1) == 1)/ BATCH_SIZE)
print(np.sum(p3[:, 1] >= 0.9)/ BATCH_SIZE)
#print(np.sum(np.abs(p3 - 1) < 2e-1)/ BATCH_SIZE)

#w = identify_net_1.w.eval(sess)
#
#w2 = identify_net_1.w2.eval(sess)
#
#w3 = identify_net_1.w3.eval(sess)