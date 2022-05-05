# GCNN_Drug_Discovery
Program in GCN_train.py (main code) and GCN_Pytorch.py (GCNN definition)
Simple GCNN for the QM9 dataset that uses a graph representation for simple molecules with 4 node features: element, and position (x,y,z). 15 potential outputs but mu (Dipole moment) has been the focus so far and seems to show test error results of around 1.06, in line with older benchmarks.

Play codes based upon https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780 in first_GCNN_from_towards_data_science.py GCN_layer.py Karate_dataset_GCNN_from_towards_data_science.py
