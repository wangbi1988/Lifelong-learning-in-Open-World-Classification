# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:42:09 2018

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

class Tools(object):
    
    def saveResult(rlss, name):
        idx = None;
        rls = None;
        for i in rlss:
            d = rlss[i];
            if idx is None:
                idx = d['idx'];
            else:
                idx = np.hstack((idx, d['idx']));
                
            if rls is None:
                rls = d['rls'];
            else:
                rls = np.row_stack((rls, d['rls']));
        d = {'idx': idx, 'rls': rls};
        scio.savemat(name, d);
        return d;
    
    def incrementalChange(d1, d2, MAXIDX, name = None):
        IDX = np.arange(MAXIDX);
        c1 = np.argmax(d1['rls'], 1);
        c2 = np.argmax(d2['rls'], 1);
        basec1 = np.zeros_like(IDX) - 1;
        basec2 = np.zeros_like(IDX) - 1;
        basec1[d1['idx']] = c1;
        basec2[d2['idx']] = c2;
#        找出不同的部分。
        diff = 1 - np.equal(basec1, basec2);
#        因为d1永远是d2的子集。所以这里可以直接把不同的部分映射到d2中；
        diff = np.equal(diff[d2['idx']], 1);
        d = {'idx': d2['idx'][diff], 'rls': d2['rls'][diff, :]};
        if name is not None:
            scio.savemat(name, d);
        return d
        
    
    def pooling(val, strip):
        k = strip[0];
        T = np.divide(val.shape[1], k).astype(np.int32);
        val_t = np.zeros([val.shape[0], T]);
        for i in range(T):
            val_t[:, i] = np.sum(val[:, np.arange(k) + i * k], 1);
            
        
        k = strip[1];
        T = np.divide(val.shape[0], k).astype(np.int32);
        val_t2 = np.zeros([T, val_t.shape[1]]);
        for i in range(T):
            val_t2[i, :] = np.sum(val_t[np.arange(k) + i * k, :], 0);
        return val_t2;

    def poolingMat(strip, brain):
        idxnet_weight = {};
        t = 1 - brain.classifier_net.identify_mask_1.eval(session = brain.sess);
        for i in range(brain.identify_idx):
            val = {};
            idxnet = brain.identify_nets[i];
            val = Tools.pooling(idxnet.masked['lay1'], strip);
            idxnet_weight['lay{}'.format(i)] = val;
            
        classifier_weight = {'lay1': 
            Tools.pooling(np.multiply(brain.classifier_net.w1.eval(session = brain.sess),
                    t), strip)};
        return idxnet_weight, classifier_weight;
    
    def gray2rgb(gray, s):
#        将nxm转到3xnxm
        gray = np.reshape(gray, s);
        rgb = np.zeros(np.hstack((s, 3)));
        if len(s) == 2:
            rgb[:, :, 0] = gray;
            rgb[:, :, 1] = gray;
            rgb[:, :, 2] = gray;
        elif len(s) == 3:
            rgb[:, :, :, 0] = gray;
            rgb[:, :, :, 1] = gray;
            rgb[:, :, :, 2] = gray;
#        gray = np.column_stack((gray, gray, gray));
#        gray = np.reshape(gray, [3, s[0], s[1]]);
        return rgb;
