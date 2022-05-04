#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 02:56:23 2022

@author: kirill
"""
import numpy as np
#make GCNN layer
class gcn_layer:
    def __init__(self,A_hat,D_hat,W):
        super(gcn_layer, self).__init__()

        self.A_hat=A_hat
        self.D_hat=D_hat
        self.W=W
    def relu(self,x):
        for i in range(0, len(x[0,:])):
            for j in range(0,len(x[:,0])):
                x[j,i] = max(0, x[j,i])
        return x    
        
    def forward(self, X):
        return self.relu(np.array(self.D_hat**-1 * self.A_hat * X * self.W))