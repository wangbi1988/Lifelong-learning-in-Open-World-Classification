# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 09:53:22 2018

@author: 王碧
"""

import tensorflow as tf

class NetWork():
    def __init__(self, params):
#        self.is_train = params['is_train'];
#        self.output_n = params['output_n'];
        self.lr_multipliers = {}
        self.rest_names = {}
    
    '''
    统计
    '''
    def __variable_summaries(var):
        mean = tf.reduce_mean(var);
        tf.summary.scalar('mean', mean);
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)));
        tf.summary.scalar('stddev', stddev);
        tf.summary.histogram('histogram', var);
        return;
        
    '''
    存储与恢复
    '''
    def __add_restore(self, var):
        self.rest_names[var.op.name] = var;
        return;
        
    def save(self, sess, save_path = 'my_model', global_step = 0):
        saver = tf.train.Saver(self.rest_names);
        saver.save(sess, save_path = save_path, global_step = global_step);
        return;
        
    def restore(self, sess, save_path = 'my_model'):
        saver = tf.train.Saver(self.rest_names);
        saver.restore(sess, save_path);
        return;
        
    
    '''
    layer wise training
    '''
    @classmethod
    def __set_lr_mult(self, var, lr_mult):
#        self.lr_multipliers[var.op.name] = lr_mult;
        return;

    '''
    定义变量
    '''    
    def __variable(self, name, shape, 
                   initializer,
                   lr_mult = 1.):
        trainable = lr_mult > 0;
        var = tf.get_variable(name, shape = shape, initializer = initializer, trainable = trainable);
        if trainable:
#            self.__set_lr_mult(var, lr_mult);
            NetWork.__variable_summaries(var);
            
        self.__add_restore(var);
        return var;
    
    def _norm_variable(self, name, shape, mean = .0, stddev = 0.05, lr_mult = 1):
        initializer = tf.truncated_normal_initializer(mean, stddev);
        return self.__variable(name, shape, initializer, lr_mult);
        
    def _cons_variable(self, name, shape, mean = .0, lr_mult = 1):
        initializer = tf.constant_initializer(mean);
        return self.__variable(name, shape, initializer, lr_mult);
    
    '''
    常规层
    '''        
    def pool_layer(self, x, ksize, strides, func='max', name='PL', padding = 'SAME'):
        with tf.variable_scope(name):
            if (func=='max'):
                return tf.nn.max_pool(x, ksize = ksize,
                                        strides = strides,  padding = padding);
            else:
                return tf.nn.avg_pool(x, ksize = ksize,
                                        strides = strides,  padding = padding);

    def __dropout_layer_self(self, x, shape, dr = 0.5, lr_mult =1., name = 'DO'):
        with tf.variable_scope(name):
            do = self._norm_variable('do', shape, lr_mult = lr_mult, stddev = 0.);
            do = tf.nn.sigmoid(do);
            do = tf.where(do >= dr, tf.ones(shape, dtype = tf.float32), tf.zeros(shape, dtype = tf.float32));
        return tf.matmul(x, do);
        
    def __fc_layer(self, x, shape, lr_mult = 1, name = 'FC', mean = .0, stddev = 0.5, dr = 0):
        with tf.variable_scope(name):
            w = self._norm_variable('w', shape, lr_mult = lr_mult, mean = mean, stddev = stddev);
            b = self._norm_variable('b', [shape[1]], lr_mult = lr_mult, mean = mean, stddev = stddev);
            if dr > 0:
                w = self.__dropout_layer_self(w, shape, dr = dr, lr_mult = lr_mult);
        return tf.add(tf.matmul(x, w), b);
#        return tf.matmul(x, w);
    
    def __conv_layer(self, x, shape,  #`[filter_height, filter_width, in_channels, out_channels]`
                     padding, # padding: A `string` from: `"SAME", "VALID"`.
                     strides = [1, 1, 1, 1], 
                     lr_mult = 1, 
                     name = 'CONV', 
                     mean = .0, 
                     stddev = 0.5):
        with tf.variable_scope(name):
            w = self._norm_variable('w', shape, lr_mult = lr_mult, mean = mean, stddev = stddev);
            b = self._norm_variable('b', [shape[-1]], lr_mult = lr_mult, mean = mean, stddev = stddev);
        return tf.add(tf.nn.conv2d(x, w, strides, padding), b);
#        return tf.nn.conv2d(x, w, strides, padding);
        
    def __batch_norm_layer(self, x, axes = 0, name = 'BN', lr_mult = 1):
        with tf.variable_scope(name):
            batch_mean, batch_var = tf.nn.moments(x, axes, name='moments');
            t = x.get_shape().as_list()[-1];
            beta = self._norm_variable('beta', [t], lr_mult = lr_mult, stddev = 0.);
            gamma = self._norm_variable('gamma', [t], lr_mult = lr_mult, stddev = 0.);
        return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-5, name = 'bn');

    def __relu(self, x, actfunc):
        if actfunc == 'relu':
            return tf.nn.relu(x);
        elif actfunc == 'sigmoid':
            return tf.nn.sigmoid(x);
        elif actfunc == 'elu':
            return tf.nn.elu(x);
        elif actfunc == 'tanh':
            return tf.nn.tanh(x);
        elif actfunc == 'none':
            return x;
        elif actfunc == 'softplus':
            return tf.nn.softplus(x);
    
    
    '''
    层结构
    '''
    def fc_block(self, x, shape, lr_mult = 1, name = 'FC_BL',
                 mean = .0, stddev = 0.5, dr = 0, actfunc = 'relu'):
        with tf.variable_scope(name):
            x = self.__fc_layer(x, shape, lr_mult = lr_mult, mean = mean, stddev = stddev, dr = dr);
#            x = self.__dropout_layer_self(x, shape, dr = 0.5, lr_mult = lr_mult)
#            x = self.__batch_norm_layer(x, lr_mult = lr_mult);
        if actfunc:
            return self.__relu(x, actfunc);
        else:
            return x;
    
    def conv_block(self, x, shape, 
                     strides = [1, 1, 1, 1], 
                     lr_mult = 1, 
                     name = 'CONV_BL', 
                     padding = 'SAME', # padding: A `string` from: `"SAME", "VALID"`.
                     mean = .0, 
                     stddev = 0.5, actfunc = 'relu'):
        with tf.variable_scope(name):
            x = self.__conv_layer(x, shape, strides = strides,
                                  padding = padding, mean = mean, stddev = stddev);
            x = self.__relu(x, actfunc);
#            x = self.__pool_layer(x, ksize = [1, 2, 2, 1], strides = [1, 1, 1, 1]);
        return x;
        
    def resn_block(self, x, shape, stride = 1, lr_mult = 1,
                   name = 'RESN_BL', mean = .0, stddev = 0.5, dr = 0, actfunc = 'relu'):
        with tf.variable_scope(name):
            #三层卷积
            input_n = shape[-2];
            output_n = shape[-1];
            residual = self.conv_block(x, [1, 1, input_n, input_n], 
                                       strides = [1, stride, stride, 1],
                                       lr_mult = lr_mult, name = 'first',
                                       actfunc = actfunc);
            residual = self.conv_block(residual, [1, 3, input_n, input_n], 
                                       strides = [1, stride, stride, 1], 
                                       lr_mult = lr_mult, name = 'middle',
                                       actfunc = actfunc);
            residual = self.conv_block(residual, [1, 1, input_n, output_n], 
                                       strides = [1, stride, stride, 1], 
                                       lr_mult = lr_mult, name = 'out',
                                       actfunc = actfunc);
            if input_n != output_n:
                x = self.conv_block(x, [1, 1, input_n, output_n], 
                                    strides = [1, stride, stride, 1], 
                                    lr_mult = lr_mult, name = 'trans',
                                    actfunc = actfunc);
        return tf.add(x, residual);
        
    