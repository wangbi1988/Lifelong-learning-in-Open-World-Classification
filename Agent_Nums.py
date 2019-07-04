# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:50:13 2018

@author: bb
"""

import numpy as np;
import matplotlib.pyplot as plt
import tensorflow as tf
from kbNets import Agent, KBNets


class KBNets_Learning(KBNets):
    def __init__(self, params):
        return;

class Agent_Nums(Agent):
    N_FEATURES = 10;
    N_NUMS = 10;
    ACTIVATE_SIGN = np.zeros([1, N_FEATURES]) - 1; #激活信号， 类似于队列的头指针
    NUMS = np.random.random_sample([N_NUMS, N_FEATURES]);
    __TYPE_ADD = 1;
    __TYPE_SUB = 2;
    __ADD = np.zeros([1, N_FEATURES]); ADD[0, -1] = __TYPE_ADD;
    __SUB = np.zeros([1, N_FEATURES]); SUB[0, -1] = __TYPE_SUB;
    OPT = {0 : __ADD, 1 : __SUB}
    N_OPT = 2;
    
    def __init__(self):
        super.__init__(None);
        self.opts = KBNets_Learning();
        self.thinkings = KBNets_Learning();
        self.gamma = 0.9;


    def learn(self, sess): # 暂不支持增量学习，这是个问题。
        infos = {'x_' : Agent_Nums.NUMS, 'y_' : np.ones(Agent_Nums.NUMS.shape[0])};
        self.info_idx.train(infos, sess); # 学习无符号整数
        
        infos = {'x_' : np.row_stack((Agent_Nums.ACTIVATE_SIGN, Agent_Nums.NUMS)),
                 'y_' : np.row_stack((Agent_Nums.NUMS, Agent_Nums.ACTIVATE_SIGN))};
        self.infos.train(infos, sess);
#        ngram = {'n' : 3, 'x_' : x, 'y_' : x, 'a_' : {'n' : 2, 'o': x}};
        # 生成训练数据
        # 做加减法运算只需要教会agent按照顺序数数即可，然后用rl来训练计算次数。和L语言相似
        
        return;
        
    
        
    # 找出两者之间的差值，并返回reward_discount
    # 这里可以是一个完全随机的过程。
    # 随机生成参数，而后通过DQN来学习。
    # x - a - v = y; eg x:3 a:- v:2 = y:1
    def spillikin(self, sess, x = None, y = None, a = None, v = None):
        if x == None:
            x = Agent_Nums.NUMS[np.random.randint(Agent_Nums.N_NUMS), :];
        if y == None:
            y = Agent_Nums.NUMS[np.random.randint(Agent_Nums.N_NUMS), :];
        if a == None:
            a = np.random.randint(Agent_Nums.N_OPT);
        if v == None:
            v = Agent_Nums.NUMS[np.random.randint(Agent_Nums.N_NUMS), :];
            
        s = np.append(x, y);
        
        begin = Agent_Nums.ACTIVATE_SIGN;
#        begin = y;
        r = 0;
        pointer = None;
        if a == Agent_Nums.__TYPE_ADD:
            pointer = x;
        elif a == Agent_Nums.__TYPE_SUB:
            pointer = y;
            
        while(True):
            if not self.isLearn(pointer, sess):
                r = -10;
                break;
            if self.sameNums(begin, v): # 操作到达指定次数
                break;
            pointer, begin = sess.run([self.infos.pred(np.row_stack((pointer, v)), sess)]);
        
        if a == Agent_Nums.__TYPE_ADD:
            x = pointer;
        elif a == Agent_Nums.__TYPE_SUB:
            y = pointer;
            
        if self.sameNums(y, x) and r != -10:
            r = 10;
            
        a = Agent_Nums.OPT[a];
        a = np.append(a, v);
        s_ = np.append(x, y);
        return s, a, s_, r;
    
    # 形式是 x - y - a - v
    # x - y - a - n 返回 v
    # x - n - a - v 返回 y
    # 换言之，在保证输入输出维度相等的情况下，进行泛化
    # 总共 4 个参数， 可以进行 C(4,3)但需要注意的是应该要插入特殊的间隔符，不然分辨不出来
    # 那是不是可以考虑找到通用程序？详细了解通用程序的定义
    # 可以定义一层，用来规范化传入参数。
    # 只会做出正确的推论
    def thinking(self, sess, xy, av):
        # 此处应该是一个递归式。保证a-v 不变
        # 若av 未知，直接输入x,y可知；
        # 若y 未知，进行猜测后求解delta xy， 变成 delta xy, 0, a,v求输出结果并判别
        # 若3 次递归未找到，则放弃。
        return;
        
        
        
    def sameNums(self, x, y):
        # 误差小于某个值，由于过拟合，所以这个误差会很小
        return np.sum(np.abs(x - y), 1) <= 1e-6;