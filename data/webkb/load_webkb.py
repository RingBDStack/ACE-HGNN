import os
import pickle as pkl
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from utils.gat_utils import encode_onehot
def load_webkb_data(dataset="cornell",use_feats="True",data_path="./data/"):
    dataset="cornell"
    print('Loading {} dataset...'.format(dataset))
    idx_features_labels = np.genfromtxt("cornell.content".format(data_path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("cornell.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return features.todense(), adj, labels

if __name__ == '__main__':
    dataset="webkb"
    adj, features, labels, idx_train, idx_val, idx_test = load_webkb_data(dataset, use_feats=True, data_path="")
    print("adj is",adj)
    print("features is ",features)
    print("labels is ",labels)