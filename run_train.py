# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:24:48 2018

@author: bb
"""


import numpy as np;
import matplotlib.pyplot as plt
import tensorflow as tf

from Brain import Brain;
from DataSet import DataSet;
import h5py
from scipy.sparse import csr_matrix as csr
import scipy.io as scio
from tensorflow.examples.tutorials.mnist import input_data


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
    return {'train':train, 'validation':validation, 'test':test};

def load_src_mnist():
    mnist = input_data.read_data_sets("data/mnist", one_hot=True);
    train_x = mnist.train._images;
    train_y = mnist.train._labels;
    train = DataSet(train_x, train_y, onehot = False);
    
    validation_x = mnist.validation._images;
    validation_y = mnist.validation._labels;
    validation = DataSet(validation_x, validation_y, onehot = False);
    
    test_x = mnist.test._images;
    test_y = mnist.test._labels;
    test = DataSet(test_x, test_y, onehot = False);
    return {'train':train, 'validation':validation, 'test':test};
#dict_data = load_mnist('mnist');

dict_data = load_src_mnist();

brain = Brain(28 * 28, 10);

train = dict_data['train'];
validation = dict_data['validation'];
test = dict_data['test'];

'''
先训练0-5 类，然后依次增量训练，
最后一类9留下来测试未知情况
'''

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

oo = 3;
sub_train_index = np.less_equal(train_labels, oo); # <= 4
sub_validation_index = np.less_equal(validation_labels, oo); # <= 4
sub_test_index = np.less_equal(test_labels, oo); # <= 4

clip_data = [composer(dict_data, {'train':sub_train_index, 
                                  'validation':sub_validation_index,
                                  'test':sub_test_index})];
oo = oo + 1;
for i in range(3):
    sub_train_index = np.equal(train_labels, i * 2 + oo) | np.equal(train_labels, i * 2 + oo + 1);
    sub_validation_index = np.equal(validation_labels, i * 2 + oo) | np.equal(validation_labels, i * 2 + oo + 1); 
    sub_test_index = np.equal(test_labels, i * 2 + oo) | np.equal(test_labels, i * 2 + oo + 1);
    clip_data.append(composer(dict_data, {'train':sub_train_index, 
                                          'validation':sub_validation_index,
                                          'test':sub_test_index}));

def out(brain, test):
    # 第一步，对输入的数据进行检测，看看有没有学过。
#    train = dict_data['train'];
#    validation = dict_data['validation'];
    
    identify_rls = brain.identify(test.data);
    know_set = identify_rls['known'];
    unknonw_set = identify_rls['unknown']['idx'];
    
    print('识别数据{}，未知数据{}'.format(
            test.data.shape[0] - unknonw_set.shape[0], unknonw_set.shape[0]));
#    rls = {}
    rls = brain.pred(know_set, test.data);
    
    if (test.data.shape[0] - unknonw_set.shape[0]) == 0:
        return brain, rls, know_set, 0;
    # 对数据进行预测
#    dict_set[i] = {'idx': idx, 'rls': rls};
    y = test.labels;
#    pred_y = np.zeros_like(y) - 1;
    rig = 0;
    try:
        for s in rls:
            t = rls[s];
#            pred_y[t['idx']] = t['rls'];
            rig = rig + np.sum(np.equal(np.argmax(t['rls'], 1), np.argmax(y, 1)[t['idx']]));
        acc = rig / y.shape[0];
        print('acc is {}'.format(acc));
    except TypeError:
        print('catch error')
    return brain, rls, know_set, acc;


def pred(brain, dict_data):
    test = dict_data['test'];
    return out(brain, test);

def validation(brain, dict_data):
    test = dict_data['validation'];
    return out(brain, test);

def train(brain, dict_data):
    # 第一步，对输入的数据进行检测，看看有没有学过。
    train_set = dict_data['train'];
    brain.train(train_set);
    _, _, _, acc = validation(brain, dict_data);
#    validation = dict_data['validation'];
#    test = dict_data['test'];
    identify_rls = brain.identify(train_set.data);
#    know_set = identify_rls['known'];
    unknonw_set = identify_rls['unknown']['idx'];
    
    print('识别数据{}，未知数据{}'.format(
            train_set.data.shape[0] - unknonw_set.shape[0], unknonw_set.shape[0]));
    
    # 未知数据存在，是否需要训练
#    if unknonw_set.shape[0] > 0:
#        brain.train(DataSet(train_set.data[unknonw_set, :], 
#                            train_set.labels[unknonw_set, :], onehot = False));
    return brain;

def identify2(brain, dict_data, name):
    identify_rls = brain.identify2(dict_data['test'].data);
    scio.savemat("{}.mat".format(name), {'data': identify_rls});

#scio.savemat("test_data.mat", {'data': dict_data['test'].data, 'labels': dict_data['test'].labels});
brain = train(brain, clip_data[0]);
_, rls0, know_set, _ = pred(brain, dict_data);
identify2(brain, dict_data, 'c0')
brain = train(brain, clip_data[1]);
_, rls1, know_set, _ = pred(brain, dict_data);
identify2(brain, dict_data, 'c1')
brain = train(brain, clip_data[2]);
_, rls2, know_set, _ = pred(brain, dict_data);
identify2(brain, dict_data, 'c2')
brain = train(brain, clip_data[3]);
_, rls3, know_set, _ = pred(brain, dict_data);
identify2(brain, dict_data, 'c3')



'''
接下来是统计工具。
'''

from Tools import Tools
d0 = Tools.saveResult(rls0, 'rls0.mat');
d1 = Tools.saveResult(rls1, 'rls1.mat');
d2 = Tools.saveResult(rls2, 'rls2.mat');
d3 = Tools.saveResult(rls3, 'rls3.mat');

newrls1 = Tools.incrementalChange(d0, d1, 10000, 'ls1');
newrls2 = Tools.incrementalChange(d1, d2, 10000, 'ls2');
newrls3 = Tools.incrementalChange(d2, d3, 10000, 'ls3');

gray = dict_data['test'].data;
rgb = Tools.gray2rgb(gray, [gray.shape[0], 28, 28]);
scio.savemat("rgbpic.mat", {'rgb':rgb});

idxnet_weight, classifier_weight = Tools.poolingMat([32, 14], brain);
scio.savemat("brain_idxnet_weight.mat", idxnet_weight);
scio.savemat("brain_classify_weight.mat", classifier_weight);

t = 1 - brain.classifier_net.identify_mask_1.eval(brain.sess);
print(np.sum(t) / (t.shape[0] * t.shape[1]))
t = 1 - brain.classifier_net.identify_mask_2.eval(brain.sess);
print(np.sum(t) / (t.shape[0] * t.shape[1]))

