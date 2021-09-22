"""Data utils functions for pre-processing and data loading."""
import os
import time
import pickle as pkl
import sys
import random
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from .gat_utils import encode_onehot,normalize_features,normalize_adj

G = None
PATHDICT = None
NODELIST = None

def load_data(args, datapath):
    random.seed(args.split_seed)
    load_paths(args.dataset, datapath)
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    else:
        start = time.time()
        data = load_data_lp(args.dataset, args.use_feats, datapath)
        print("Data loading finish...time: {}s.".format(time.time() - start))
        adj = data['adj_train']
        if args.task == 'lp':
            start = time.time()
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed
            )
            print("Edge masking finish...time: {}s.".format(time.time() - start))
            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    start = time.time()
    data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    print("Processing finish...time: {}s.".format(time.time() - start))
    if args.dataset == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])
    return data

def load_paths(dataset, data_path):
    print('start loading paths...')
    start = time.time()
    dict_path = os.path.join(data_path, 'paths.npy')
    if dataset in ['pubmed', 'ppi']:
        global NODELIST
        nodes_path = os.path.join(data_path, 'nodes0.npy')
        NODELIST = np.load(nodes_path).tolist()
    paths_dict = np.load(dict_path).item()
    global PATHDICT
    PATHDICT = paths_dict
    end = time.time()
    print('Loaded in {:.4f} seconds.'.format(end - start))


# ############### FEATURES PROCESSING ####################################


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    random.seed(seed)
    num_class = np.max(labels) + 1
    label_dict = dict()
    for i in range(num_class):
        label_dict[i] = []
    for i, l in enumerate(labels):
        label_dict[l].append(i)
    idx_train, idx_val, idx_test = [], [], []
    for i in range(num_class):
        random.shuffle(label_dict[i])
        num_val = round(val_prop * len(label_dict[i]))
        num_test = round(test_prop * len(label_dict[i]))
        idx_val += label_dict[i][:num_val]
        idx_test += label_dict[i][num_val:num_val + num_test]
        idx_train += label_dict[i][num_val + num_test:]

    return idx_val, idx_test, idx_train


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### LINK PREDICTION DATA LOADERS ####################################


def load_data_lp(dataset, use_feats, data_path):
    if dataset in ['cora', 'pubmed', "citeseer"]:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    # elif dataset == 'citeseer':
    #     adj, features = load_citeseer_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'ppi':
        adj, features = load_ppi_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'webkb':
        adj, features = load_webkb_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features}
    return data


# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'pubmed', "citeseer"]:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    elif dataset == 'ppi':
        adj, features, labels = load_ppi_data(dataset, use_feats, data_path)
        val_prop, test_prop = 0.15, 0.15
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)
    elif dataset == 'webkb':
        adj, features, labels = load_webkb_data(dataset, use_feats, data_path)
        val_prop, test_prop = 0.15, 0.15
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)
    else:
        if dataset == 'disease_nc':
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data


# ############### DATASETS ####################################


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_len = test_idx_range.max() - test_idx_range.min() + 1
        tx_ext = sp.csr_matrix((test_idx_len, tx.shape[1]))
        tx_ext[test_idx_range - test_idx_range.min(), :] = tx
        ty_ext = np.zeros((test_idx_len, ty.shape[1]))
        ty_ext[test_idx_range - test_idx_range.min(), :] = ty
        tx, ty = tx_ext, ty_ext

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    
    if not use_feats:
        features = sp.eye(adj.shape[0])
    graph=nx.from_dict_of_lists(graph)
    global G
    G = graph
    print ('number of edges:', graph.size())
    print ('number of nodes:', graph.number_of_nodes())
    
    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    print ('number of edges:', len(edges))
    node_set=set()
    for line in edges:
        n1, n2 = line[0],line[1]
        node_set.add(n1)
        node_set.add(n2)
    print ('number of nodes:',len(list(node_set)))

    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1. 
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    print ('number of edges:', graph.size())
    print ('number of nodes:', graph.number_of_nodes())
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.node[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features


def load_citeseer_data(dataset, use_feats, data_path):
    features_citeseer,sadj_citeseer,labels_citeseer=torch.load("citeseer.pt")
    features=features_citeseer.numpy()
    adj=sadj_citeseer.to_dense().numpy()
    graph = nx.from_numpy_matrix(adj)
    global G
    G = graph
    print ('number of edges:', graph.size())
    print ('number of nodes:', graph.number_of_nodes())
    labels=labels_citeseer.numpy()
    return sp.csr_matrix(adj), features,labels


def load_ppi_data(dataset, use_feats, data_path):
    features_ppi=np.load(data_path+"/features.npy")
    edges=np.loadtxt(data_path+"/edges.txt")
    labels = np.loadtxt(data_path+"/node2label.txt",delimiter=" ")
    labels = labels[:,1]
    adj = np.zeros((len(features_ppi), len(features_ppi)))
    for item in edges:
        adj[int(list(item)[0]), int(list(item)[1])] = 1.
        adj[int(list(item)[1]), int(list(item)[0])] = 1.
    graph = nx.from_numpy_matrix(adj)
    global G
    G = graph
    print ('number of edges:', graph.size())
    print ('number of nodes:', graph.number_of_nodes())
    return sp.csr_matrix(adj), features_ppi, labels


def load_webkb_data(dataset,use_feats="True",data_path="./data/"):
    dataset="webkb"
    print('Loading {} dataset...'.format(dataset))
    idx_features_labels = np.genfromtxt("{}/{}.content".format(data_path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}/{}.cites".format(data_path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    graph = nx.from_numpy_matrix(np.array(adj.todense()))
    global G
    G = graph
    print ('number of edges:', graph.size())
    print ('number of nodes:', graph.number_of_nodes())
    return adj, features, labels
