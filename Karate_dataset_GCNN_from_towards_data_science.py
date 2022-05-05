#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Source for notes this which this follows: https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780
@author: kirill
"""
import numpy as np
from GCN_layer import gcn_layer

#Now add in the Graph from an actual dataset
from networkx import karate_club_graph, to_numpy_matrix
import networkx as nx
zkc = karate_club_graph()
nx.draw(zkc)
order = sorted(list(zkc.nodes()))
A = to_numpy_matrix(zkc, nodelist=order)
I = np.eye(zkc.number_of_nodes())
A_hat = A + I
D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))

#random weight initialisation
W_1 = np.random.normal(
    loc=0, scale=1, size=(zkc.number_of_nodes(), 4))
W_2 = np.random.normal(
    loc=0, size=(W_1.shape[1], 1))


gcn_layer1=gcn_layer(A_hat, D_hat,W_1)
gcn_layer2=gcn_layer(A_hat, D_hat,W_2)

H_1 = gcn_layer1.forward(I)
H_2 = gcn_layer2.forward(H_1)
output = H_2

feature_representations = {
    node: np.array(output)[node] 
    for node in zkc.nodes()}