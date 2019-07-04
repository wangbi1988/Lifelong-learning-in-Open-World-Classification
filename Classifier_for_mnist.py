# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 22:08:38 2018

@author: 王碧
"""

import numpy as np;
import matplotlib.pyplot as plt
import tensorflow as tf
from baseNet import NetWork
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/mnist", one_hot=True);
train_x = mnist.train._images;
train_x = np.reshape(train_x, [-1, 28, 28, 1]);
train_y = mnist.train._labels;

validation_x = mnist.validation._images;
validation_x = np.reshape(validation_x, [-1, 28, 28, 1]);
validation_y = mnist.validation._labels;

test_x = mnist.test._images;
test_x = np.reshape(test_x, [-1, 28, 28, 1]);
test_y = mnist.test._labels;

net = NetWork(None);
x = tf.placeholder(tf.float32, shape = [None, 28, 28, 1]);
y = tf.placeholder(tf.float32, shape = [None, 10]);

#o = tf.reshape(x, [-1, 28, 28, 1]);
o = x;
o = net.resn_block(o, [1, 1], name = 'RESN1');
#o = net.resn_block(o, [4, 4], name = 'RESN2');
#o = net.resn_block(o, [8, 4], name = 'RESN3');
o = tf.reshape(o, [-1, 28 * 28 * 1])
o = net.fc_block(o, [28 * 28 * 1, 1024], name = 'FC1');
o = net.fc_block(o, [1024, 10], name = 'FC2', actfunc = False);
o1 = tf.nn.softmax(o);

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = o1));
opt = tf.train.RMSPropOptimizer(1e-1, decay = 0.8, epsilon = 1e-6, momentum = 0.0).minimize(loss);
#opt = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(loss);

sess = tf.InteractiveSession();
sess.run(tf.global_variables_initializer());

BATCH_SIZE = 100 * 11 * 5 * 1;
MAX_SIZE = train_x.shape[0];
IDX = np.arange(MAX_SIZE);
ITER = 1 * 10;
loss_set = np.zeros([500 * ITER]);
for epoch in range(500):
#    IDX = tf.random_shuffle(IDX);
    for i in range(ITER):
        batch_idx = tf.slice(IDX, [0 + BATCH_SIZE * i], [BATCH_SIZE]).eval();
        batch_x = train_x[batch_idx, :];
        batch_y = train_y[batch_idx, :];
        _, l = sess.run([opt, loss], {x: batch_x, y: batch_y});
        loss_set[epoch * ITER + i] = l;
    op1 = sess.run([o1], {x: validation_x})[0];
    arg1 = np.argmax(op1, axis = 1);
    arg2 = np.argmax(validation_y, axis = 1);
    acc = np.sum(np.equal(arg1, arg2)) / arg2.shape[0];
    print('epoch is {} with loss {}, and devset acc is {}'
          .format(epoch, np.mean(loss_set[np.arange(ITER) + epoch * ITER]), acc));
    

op1 = sess.run([o1], {x: test_x})[0];
arg1 = np.argmax(op1, axis = 1);
arg2 = np.argmax(test_y, axis = 1);
acc = np.sum(np.equal(arg1, arg2)) / arg2.shape[0];
    