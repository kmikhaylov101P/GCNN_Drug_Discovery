#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 04:15:08 2022

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
data = QM9Dataset(label_keys=['mu', 'gap'], cutoff=5.0)
#subsetA, subsetB = random_split(data, 10000)



dataloader = GraphDataLoader(data, batch_size=100, shuffle=True)


#loss_function = torch.nn.MSELoss()

loss_function = torch.nn.L1Loss()


model = GCNN(4,2 )
# Optimisers specified in the torch.optim package
optimiser = torch.optim.AdamW(model.parameters(), lr=0.001)
number_of_epochs = 2

def train(model,optimiser, dataloader):
    global X, A, outputs, labels
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
            # print('Loss =', loss, 'prediction', outputs, 'label', labels)
            # add stuff to progress bar in the end
            loop.set_description(f"Epoch [{epoch}/{number_of_epochs}]")
            loop.set_postfix(loss=loss.item())
        

train(model, optimiser, dataloader)