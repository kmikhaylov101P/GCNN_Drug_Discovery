#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kirill
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
import numpy as np
#make GCNN layer
# class gcn_layer(nn.Module):
#     def __init__(self,A_hat,D_hat, size_in, size_out):
#         super(gcn_layer, self).__init__()

#         self.A_hat=torch.Tensor(A_hat)
#         self.D_hat=torch.Tensor(D_hat)
#         self.D_inv=torch.inverse(self.D_hat)
#         self.weight=torch.nn.Parameter(torch.Tensor(size_in, size_out))
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         print(self.weight)

#     def forward(self, X):
#         D_invA=torch.mm(self.D_inv, self.A_hat)
#         D_invAX=torch.mm(D_invA, torch.Tensor(X))
#         D_invAXW=torch.mm(D_invAX,  self.weight)    
        
#         return D_invAXW# nn.ReLU(D_invAXW)

    
# class GCNN(nn.Module):
#     def __init__(self, input_dimension, output_dimension,A_hat,D_hat):
#         super().__init__()
#         self.gcn_layer1 =  gcn_layer(A_hat,D_hat,input_dimension,10)
#         self.gcn_layer2 =  gcn_layer(A_hat,D_hat,10, output_dimension)
    
#     def forward(self, input):
#         out1 = F.relu(self.gcn_layer1(input))
#         out2 = F.relu(self.gcn_layer2(out1))
        
#         return out2

class gcn_layer(nn.Module):
    def __init__(self, size_in, size_out):
        super(gcn_layer, self).__init__()
        self.weight=torch.nn.Parameter(torch.Tensor(size_in, size_out))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self,X,A):
        A = A.to_dense()
            #A = torch.Tensor(A).float()
        self.A_hat=torch.eye(A.size(dim=0)).float()+A
        self.D_hat=torch.diagflat(torch.sum(A, 0)) #torch.Tensor(D_hat)


        # self.D_inv=torch.inverse(self.D_hat)
        # D_invA=torch.mm(self.D_inv, self.A_hat)
        # D_invAX=torch.matmul(D_invA, X)
        # D_invAXW=torch.matmul(D_invAX,  self.weight)    
        self.D_inv=torch.inverse(self.D_hat)
        D_invA=torch.matmul(self.D_inv, self.A_hat)
        D_invAX=torch.matmul(D_invA, X)
        D_invAXW=torch.matmul(D_invAX,  self.weight)
        
        
        return D_invAXW# nn.ReLU(D_invAXW)

    
class GCNN(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.gcn_layer1 =  gcn_layer(input_dimension,100)
        self.gcn_layer2 =  gcn_layer(100, output_dimension)
    
    def forward(self, X, A):
        out1 = F.relu(self.gcn_layer1(X,A))
        out2 = self.gcn_layer2(out1,A)
        out3=torch.mean(out2, 0)
        
        return out3

