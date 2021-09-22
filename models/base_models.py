"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import FermiDiracDecoder
import layers.hyp_layers as hyp_layers
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import acc_f1
from QLearning import Nash
import random
PATHS = None
GRAPH = None
NODES = None

class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        self.manifold = getattr(manifolds, args.manifold)()
        if args.c is not None:
            self.c = [torch.tensor([args.c])]*2
            if not args.cuda == -1:
                for i in range(len(self.c)):    # 'item' use copy not ref
                    self.c[i] = self.c[i].to(args.device)
        else:
            # c is None (should not appear in this branch)
            self.c = [nn.Parameter(torch.Tensor([1.]))]*2
        
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)
        self.mlp = getattr(encoders, 'MLP')(self.c, args, mapping=True)
        self.args = args
        from utils.data_utils import G, PATHDICT, NODELIST
        global GRAPH, PATHS, NODES
        GRAPH, PATHS, NODES = G, PATHDICT, NODELIST

    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h

    def mapping(self, x, last_c, c):
        '''Project embeddings with the curvature c.'''
        for item in c:
            if item < 1e-1:
                item = 1e-1
        emb1 = self.manifold.logmap0(x, last_c[0])
        emb2 = self.manifold.logmap0(x, last_c[1])
        emb1 = self.manifold.expmap0(emb1, c[0])
        emb2 = self.manifold.expmap0(emb2, c[1])
        emb_tmp = torch.add(emb1, emb2)
        h = torch.div(emb_tmp, 2)
        return h

    def find_distance(self, embedding, x, y):
        dis = 0
        if x > y:
            x, y = y, x
        hashkey = x * len(GRAPH.nodes) + y
        route = PATHS[hashkey]
        for i in range(len(route)-1):
            k_dis = self.manifold.sqdist(embedding[route[i]], embedding[route[i+1]], self.c[1]).data.cpu().item()
            dis += k_dis
        return dis
        

    def estimation(self, embedding):
        pp_list = []
        global NODES
        if NODES == None:
            NODES = list(GRAPH.nodes)
        nodes_list = random.sample(NODES, 100)
        nodes = 0
        while nodes < 10:
            node = random.choice(nodes_list)
            if len(list(GRAPH.adj[node])) < 2:
                continue

            iters = 0
            p_list = []
            for i in range(self.args.niter):
                b = random.choice(list(GRAPH.adj[node]))
                c = random.choice(list(GRAPH.adj[node]))
                while b == c:
                    c = random.choice(list(GRAPH.adj[node]))
                a = random.choice(nodes_list)
                while a in [node, b, c]:
                    a = random.choice(nodes_list)
                try:
                    d_am = self.find_distance(embedding, a, node)
                    d_bc = self.find_distance(embedding, b, c)
                    d_ab = self.find_distance(embedding, a, b)
                    d_ac = self.find_distance(embedding, a, c)
                except KeyError:
                    continue
                p = (d_am / 2) + ( (d_bc*d_bc) / (8*d_am) ) - ( (d_ab*d_ab+d_ac*d_ac) / (2*d_am) )
                p_list.append(p)
                iters += 1

            if p_list != []: 
                pp_list.append(np.mean(p_list))
                nodes += 1
        return np.mean(pp_list)

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.key_param = 'acc'       
        self.decoder = model2decoder[args.model](self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            pass
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)

    def compute_metrics(self, embeddings, data, split):
        embeddings[embeddings!=embeddings] = 1e-8
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def train_with_RL(self, env, Agent1, Agent2, data, epoch, ace):
        if epoch >= self.args.start_q:
            observation = env.get_observation() 
            pi1, pi2 = Nash(observation, Agent1, Agent2)    
            action1 = Agent1.choose_action(observation, pi1)
            action2 = Agent2.choose_action(observation, pi2)
            actions = (action1, action2)

            rewards, train_metrics = env.step((action1, action2), self, data, ace)  
            observation_ = env.get_observation()
            pi1, pi2 = Nash(observation_, Agent1, Agent2)
            pis = (pi1, pi2)
            Agent1.learn(observation, actions, rewards[0], observation_, pis)
            Agent2.learn(observation, actions, rewards[1], observation_, pis)
            if self.args.epsilon_decay == 1:
                Agent1.update_epsilon()
                Agent2.update_epsilon()
        else: 
            embeddings = self.encode(data['features'], data['adj_train_norm'])
            env.embedding1 = env.embedding2 = embeddings
            train_metrics = self.compute_metrics(embeddings, data, 'train')
            env.acc1_record.append(train_metrics[self.key_param])
            env.r1_record.append(0.0)
            env.c1_record.append(env.c1[:])
            env.acc2_record.append(train_metrics[self.key_param])
            env.c2_record.append(env.c1[:]) 
        return train_metrics
    
    def change_curv(self, curv):
        '''
        Change curvature everywhere in the model.
        '''
        for item in curv:
            if item < 1e-1:
                item = 1e-1
        self.c = [torch.tensor([c]).float() for c in curv]
        if not self.args.cuda == -1:
            for i in range(len(self.c)):
                self.c[i] = self.c[i].to(self.args.device)
        self.encoder.c = self.c
        self.encoder.change_curv(self.c) 
        self.decoder.c = self.c 

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges
        self.key_param = 'ap'   

    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        # decode curvature using last layer's
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c[1])
        probs = self.dc.forward(sqdist)
        return probs

    def compute_metrics(self, embeddings, data, split):
        if torch.isnan(embeddings).any():
            embeddings[embeddings!=embeddings] = 1e-8   
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics


    def train_with_RL(self, env, Agent1, Agent2, data, epoch, ace):
        if epoch >= self.args.start_q:
            observation = env.get_observation()           
            pi1, pi2 = Nash(observation, Agent1, Agent2)   
            action1 = Agent1.choose_action(observation, pi1)
            action2 = Agent2.choose_action(observation, pi2)
            actions = (action1, action2)

            rewards, train_metrics = env.step((action1, action2), self, data, ace)  
            observation_ = env.get_observation()
            pi1, pi2 = Nash(observation_, Agent1, Agent2)
            pis = (pi1, pi2)
            Agent1.learn(observation, actions, rewards[0], observation_, pis)
            Agent2.learn(observation, actions, rewards[1], observation_, pis)
            if self.args.epsilon_decay == 1:
                Agent1.update_epsilon()
                Agent2.update_epsilon()
        else:      
            embeddings = self.encode(data['features'], data['adj_train_norm'])
            env.embedding1 = env.embedding2 = embeddings
            train_metrics = self.compute_metrics(embeddings, data, 'train')
            env.acc1_record.append(train_metrics[self.key_param])
            env.r1_record.append(0.0)
            env.c1_record.append(env.c1[:])
            env.acc2_record.append(train_metrics[self.key_param])
            env.c2_record.append(env.c1[:])   
        return train_metrics

    def change_curv(self, curv):
        '''
        Change curvature everywhere in the model.
        '''
        for item in curv:
            if item < 1e-1:
                item = 1e-1
        self.c = [torch.tensor([c]).float() for c in curv]
        if not self.args.cuda == -1:
            for i in range(len(self.c)):
                self.c[i] = self.c[i].to(self.args.device)
        self.encoder.c = self.c
        self.encoder.change_curv(self.c)    

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

