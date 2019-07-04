# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 21:06:35 2018

@author: 王碧
"""

import numpy as np;
import matplotlib.pyplot as plt
import tensorflow as tf
from baseNet import NetWork
from tensorflow.examples.tutorials.mnist import input_data
import h5py

src = 'mnist';
#src = 'fasion mnist';
mnist = input_data.read_data_sets("data/{}".format(src), one_hot=True);

train_x = mnist.train._images;
train_y = mnist.train._labels;

validation_x = mnist.validation._images;
validation_y = mnist.validation._labels;

test_x = mnist.test._images;
test_y = mnist.test._labels;

TRAIN_DECODER = True;
net = NetWork(None);
x = tf.placeholder(tf.float32, shape = [None, 28 * 28]);

encoder = tf.reshape(x, [-1, 28, 28, 1]);
encoder = net.resn_block(encoder, [1, 1], name = 'RESN1', actfunc = 'relu');
#encoder = net.resn_block(encoder, [1, 1], name = 'RESN2', actfunc = 'relu');
#encoder = net.resn_block(encoder, [1, 1], name = 'RESN3', actfunc = 'relu');
#encoder = net.conv_block(encoder, [3, 3, 1, 1], [1, 1, 1, 1],
#                         name = 'CONV1', padding = 'VALID', actfunc = 'relu');
#encoder = net.conv_block(encoder, [5, 5, 1, 1], [1, 1, 1, 1],
#                         name = 'CONV2', padding = 'VALID', actfunc = 'relu');
#encoder = net.conv_block(encoder, [3, 3, 1, 1], [1, 1, 1, 1],
#                         name = 'CONV3', padding = 'VALID', actfunc = 'sigmoid');
#encoder = x;
#encoder = net.fc_block(encoder, [28 * 28, 1024], name = 'FC11', actfunc = 'sigmoid');
#encoder = net.fc_block(encoder, [1024, 512], name = 'FC12', actfunc = 'sigmoid');
N = encoder.shape[1].value;
N = N * N;
print(encoder.shape)
encoder = tf.reshape(encoder, [-1, N]);
encoder = net.fc_block(encoder, [N, 50], name = 'FC1', actfunc = 'relu');
#encoder = net.fc_block(encoder, [256, 20], name = 'FC2', actfunc = 'sigmoid');

decoder = encoder;
if TRAIN_DECODER:
    decoder = net.fc_block(decoder, [50, 256], name = 'FC4', actfunc = 'relu');
#    decoder = tf.reshape(decoder, [-1, 16, 16, 1]);
#    decoder = net.conv_block(decoder, [1, 1, 1, 1], [1, 1, 1, 1],
#                             name = 'CONV21', padding = 'VALID', actfunc = 'relu');
#    decoder = tf.reshape(decoder, [-1, 16, 16, 1]);
    decoder = net.fc_block(decoder, [16 * 16, 28 * 28], name = 'FC7', actfunc = 'relu');
else:
    file = h5py.File('decoder_weight.h5', 'r');
    FC4_w = file['FC4_w'][:];
    FC4_b = file['FC4_b'][:];
#    CONV21_w = file['CONV21_w'][:];
#    CONV21_b = file['CONV21_b'][:];
#    CONV22_w = file['CONV22_w'][:];
#    CONV22_b = file['CONV22_b'][:];
    FC7_w = file['FC7_w'][:];
    FC7_b = file['FC7_b'][:];
    file.close();
    decoder = tf.nn.relu(tf.add(tf.matmul(decoder, FC4_w), FC4_b));
#    decoder = tf.reshape(decoder, [-1, 16, 16, 1]);
#    decoder = tf.nn.sigmoid(tf.add(tf.nn.conv2d(decoder, CONV21_w,
#                                                [1, 1, 1, 1], 'VALID'), CONV21_b));
#    decoder = tf.nn.sigmoid(tf.add(tf.nn.conv2d(decoder, CONV22_w,
#                                                [1, 1, 1, 1], 'VALID'), CONV22_b));
#    decoder = tf.reshape(decoder, [-1, 256 * 1]);
#    decoder = tf.nn.relu(tf.add(tf.matmul(decoder, FC7_w), FC7_b));
#decoder = net.conv_block(decoder, [1, 1, 2, 4], [1, 1, 1, 1],
#                         name = 'CONV22', padding = 'VALID', actfunc = 'sigmoid');
decoder = tf.minimum(decoder, 1 - 1e-6);
decoder = tf.maximum(decoder, 1e-6);

loss = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(x * 10, decoder * 10), 1));
#loss = -tf.reduce_mean(tf.add(tf.multiply(x ,tf.log(decoder)),
#                              tf.multiply((1 - x), tf.log(1 - decoder))))
global_step = tf.Variable(initial_value = 0);
#opt = tf.train.RMSPropOptimizer(1e-2, momentum = .2).minimize(loss, global_step = global_step);
opt = tf.train.AdamOptimizer(learning_rate = 4e-3).minimize(loss, global_step = global_step);
#opt = tf.train.GradientDescentOptimizer(learning_rate = 1e-2).minimize(loss);
sess = tf.InteractiveSession();
sess.run(tf.global_variables_initializer());

BATCH_SIZE = 128;
MAX_SIZE = validation_x.shape[0];
IDX = np.arange(MAX_SIZE);
ITER = np.floor_divide(MAX_SIZE, BATCH_SIZE);
N = 400;
loss_set = np.zeros([N * ITER]);
for epoch in range(N):
    for i in range(ITER):
        batch_x, _ = mnist.train.next_batch(BATCH_SIZE);
        _, l, d = sess.run([opt, loss, decoder], {x: batch_x});
        loss_set[epoch * ITER + i] = l;
    print('epoch is {} with loss {}'
          .format(epoch, np.mean(loss_set[np.arange(ITER) + ITER * epoch])));


plt.figure();
plt.plot((loss_set))
plt.figure();
tt = test_x[-5:, :];
tt = np.append(tt, sess.run(decoder, {x: tt}));
plt.imshow(np.reshape(tt, [-1, 28]))

#
#train_x = sess.run(encoder, {x: train_x});
#train_y = np.argmax(train_y, axis = 1);
#
#validation_x = sess.run(encoder, {x: validation_x});
#validation_y = np.argmax(validation_y, axis = 1);
#
#test_x = sess.run(encoder, {x: test_x});
#test_y = np.argmax(test_y, axis = 1);
#
#file = h5py.File('{}_encoder.h5'.format(src), 'w');
#file.create_dataset('train_x', data = train_x, compression = 'gzip', compression_opts = 9);
#file.create_dataset('train_y', data = train_y, compression = 'gzip', compression_opts = 9);
#file.create_dataset('validation_x', data = validation_x, compression = 'gzip', compression_opts = 9);
#file.create_dataset('validation_y', data = validation_y, compression = 'gzip', compression_opts = 9);
#file.create_dataset('test_x', data = test_x, compression = 'gzip', compression_opts = 9);
#file.create_dataset('test_y', data = test_y, compression = 'gzip', compression_opts = 9);
#file.close();
#
if TRAIN_DECODER:
    file = h5py.File('decoder_weight.h5', 'w');
    t = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'FC4');
    file.create_dataset('FC4_w', data = t[0].eval(session = sess), compression = 'gzip', compression_opts = 9);
    file.create_dataset('FC4_b', data = t[1].eval(session = sess), compression = 'gzip', compression_opts = 9);
    
#    t = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'CONV21');
#    file.create_dataset('CONV21_w', data = t[0].eval(session = sess), compression = 'gzip', compression_opts = 9);
#    file.create_dataset('CONV21_b', data = t[1].eval(session = sess), compression = 'gzip', compression_opts = 9);
##    
#    t = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'CONV22');
#    file.create_dataset('CONV22_w', data = t[0].eval(session = sess), compression = 'gzip', compression_opts = 9);
#    file.create_dataset('CONV22_b', data = t[1].eval(session = sess), compression = 'gzip', compression_opts = 9);
#    
    t = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'FC7');
    file.create_dataset('FC7_w', data = t[0].eval(session = sess), compression = 'gzip', compression_opts = 9);
    file.create_dataset('FC7_b', data = t[1].eval(session = sess), compression = 'gzip', compression_opts = 9);
    file.close();
