# -*- coding: utf-8 -*-
"""
Spyder Editor
Source for notes this which this follows: https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780
This is a temporary script file.
"""
import numpy as np
from GCN_layer import gcn_layer
#define a ReLU function
def relu(x):
    for i in range(0, len(x[0,:])):
        for j in range(0,len(x[:,0])):
            x[j,i] = max(0, x[j,i])
    return x
    



#Adjacency/connectivity matrix
A = np.matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 1], 
    [0, 1, 0, 0],
    [1, 0, 1, 0]],
    dtype=float
)


#Make feature matrix
X = np.matrix([
            [i, -i]
            for i in range(A.shape[0])
        ], dtype=float)

#First consider weight matrix as being all 1s
        
initial_prop_test = A*X
#diagonal of adjacency matrix=0 therfore no carry over of features => modify propogation to include A (i.e. et adjacenecy diagonal to 1)
#Notes call this a "self loop"
I=np.matrix(np.eye(A.shape[0])) #identity matrix

mod_prop_test = (A+I)*X 

#Degree Matrix = Diagonal matrix representing the number of edges attached to each vertex (the degree). Note::::::ly undirected graphs
#For normalisation multiply by D^-1. In other words normalise the features w.r.t number of connections at that node.
#f(X,A)=(D^-1)AX
D =np.matrix(np.diagflat(np.sum(np.array(A),axis=0)))
norm_prop_test = D**-1*(A+I)*X

#Now add weights back in
W = np.matrix([
             [1, -1],
             [-1, 1]
         ])
weighted_prop_test = D**-1*(A+I)*X*W 


#Adding  activation function
activated_prop_test = relu(np.array(D**-1*(A+I)*X*W)) 


