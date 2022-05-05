#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kirill
"""

import torch
import torch.nn as nn
import numpy as np
from GCN_Pytorch import  GCNN
from networkx import karate_club_graph, to_numpy_matrix
import networkx as nx
from tqdm import tqdm

import os
import numpy as np
import scipy.sparse as sp
from  dgl.data import QM9Dataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.dataset import random_split

#Benchmarks from Neural Message Passing for Quantum Chemistry 
#Justin Gilmer 1 Samuel S. Schoenholz 1 Patrick F. Riley 2 Oriol Vinyals 3 George E. Dahl 1
#2017
#Target BAML BOB CM ECFP4 HDAD GC GG-NN DTNN enn-s2s enn-s2s-ens5
# mu 4.34 4.23 4.49 4.82 3.34 0.70 1.22 - 0.30 0.20
# alpha 3.01 2.98 4.33 34.54 1.75 2.27 1.55 - 0.92 0.68
# HOMO 2.20 2.20 3.09 2.89 1.54 1.18 1.17 - 0.99 0.74
# LUMO 2.76 2.74 4.26 3.10 1.96 1.10 1.08 - 0.87 0.65
# gap 3.28 3.41 5.32 3.86 2.49 1.78 1.70 - 1.60 1.23
# R2 3.25 0.80 2.83 90.68 1.35 4.73 3.99 - 0.15 0.14
# ZPVE 3.31 3.40 4.80 241.58 1.91 9.75 2.52 - 1.27 1.10
# U0 1.21 1.43 2.98 85.01 0.58 3.02 0.83 - 0.45 0.33
# U 1.22 1.44 2.99 85.59 0.59 3.16 0.86 - 0.45 0.34
# H 1.22 1.44 2.99 86.21 0.59 3.19 0.81 - 0.39 0.30
# G 1.20 1.42 2.97 78.36 0.59 2.95 0.78 .842 0.44 0.34
# Cv 1.64 1.83 2.36 30.29 0.88 1.45 1.19 - 0.80 0.62
# Omega 0.27 0.35 1.32 1.47 0.34 0.32 0.53 - 0.19 0.15
# Average 2.17 2.08 3.37 53.97 1.35 2.59 1.36 - 0.68 0.52

#Get a loss of 1.17 for a very simple GCNN for mu with 1 or 10 epochs of training

data = QM9Dataset(label_keys=['mu'], cutoff=5.0)

dataset_size = len(data)
test_size = int(0.2 * dataset_size)
train_size = dataset_size - test_size

train_dataset, test_dataset = random_split(data,[train_size, test_size])
dataloader_train = GraphDataLoader(train_dataset, batch_size=100, shuffle=True)
dataloader_test = GraphDataLoader(test_dataset, batch_size=100, shuffle=True)
#loss_function = torch.nn.MSELoss()

loss_function = torch.nn.L1Loss()


model = GCNN(4,2 )
# Optimisers specified in the torch.optim package
optimiser = torch.optim.AdamW(model.parameters(), lr=0.001)
number_of_epochs = 10

def train(model,optimiser, dataloader):
    for epoch in range(number_of_epochs):
        loop = tqdm(dataloader)
        for idx, (batched_graph, labels) in enumerate(loop):
            X_Z =batched_graph.ndata['Z'].float()
            X_R =batched_graph.ndata['R'].float()
    
    
            X =torch.cat((X_Z.unsqueeze(1),X_R),1)
            A = batched_graph.adj().float()
    
            model.train()
            optimiser.zero_grad() #zero gradients of parameters
            outputs = torch.unsqueeze( model.forward(X,A),0)
            loss=loss_function(outputs, labels)
            loss.backward()
            optimiser.step()
            loop.set_description(f"Epoch [{epoch}/{number_of_epochs}]")
            loop.set_postfix(loss=loss.item())
        
def test(model,optimiser, dataloader):
    global X, A, outputs, labels
    test_loss =0
    loop = tqdm(dataloader)
    for idx, (batched_graph, labels) in enumerate(loop):
        X_Z =batched_graph.ndata['Z'].float()
        X_R =batched_graph.ndata['R'].float()


        X =torch.cat((X_Z.unsqueeze(1),X_R),1)
        A = batched_graph.adj().float()

        optimiser.zero_grad() #zero gradients of parameters
        outputs = torch.unsqueeze( model.forward(X,A),0)
        loss=loss_function(outputs, labels)
        test_loss += loss.item()*labels.size(0)
    test_loss = test_loss/len(dataloader.sampler) 
    print('test loss=', test_loss)


train(model, optimiser, dataloader_train)
test(model, optimiser, dataloader_test)