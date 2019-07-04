# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:26:58 2018

@author: 王碧
"""

"""
存储网络，要强的过拟合，拒绝泛化能力。
希望网络学会说不
"""

from baseNet import NetWork

class Store():
    """
    paras={n_classes:3
           
    }
    """
    def __init__(self, paras):
        if paras is not None:
            self.n_classes = paras['n_classes'];
        else:
            self.n_classes = None;
        self.net = NetWork();
        return;
        
    def put(self, ngram):
        