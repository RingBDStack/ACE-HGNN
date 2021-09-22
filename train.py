from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time
import warnings

import numpy as np
import optimizers
import torch
import matplotlib.pyplot as plt
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics
from env import Env
from QLearning import *

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))
    warnings.filterwarnings(action='ignore')

    # Load data
    data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    # Initialize RL environment
    lr_q = args.lr_q                        # RL Learning Rate
    action_space1 = ['reject', 'accept']    # HGNN action space
    action_space2 = [                       # ACE  action space
        'r, r', 'r, a',
        'a, r', 'a, a'
    ]
    joint_actions = []
    for i in range(len(action_space1)):     # Joint action space
        for j in range(len(action_space2)):
            joint_actions.append((i, j))
    env = Env(theta=args.theta, initial_c=args.c)
    Agent1 = QLearningTable(actions=list(range(len(action_space1))), joint=joint_actions, start=args.start_q, learning_rate=lr_q)
    Agent2 = QLearningTable(actions=list(range(len(action_space2))), joint=joint_actions, start=args.start_q, learning_rate=lr_q)

    hgnn = Model(args)          # Agent1 HGNN
    ace = Model(args)           # Agent2 ACE
    logging.info(str(hgnn))
    optimizer = getattr(optimizers, args.optimizer)(params=hgnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in hgnn.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0 :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        hgnn = hgnn.to(args.device)
        ace = ace.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)

    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = hgnn.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    val_metric_record = []
    train_metric_record = []
    for epoch in range(args.epochs):
        t = time.time()
        hgnn.train()
        ace.train()
        optimizer.zero_grad()

        # train model with RL and return Agent1's train metrics
        # Terminate mechanism
        if epoch > args.start_q + 30:
            r1 = np.array(env.c1_record)[-30:-1, 0]
            r2 = np.array(env.c1_record)[-30:-1, 1]
            if abs(max(r1) - min(r1)) <= 0.03 and not env.stop[0]:
                env.stop[0] = True
                print("Layer1 RL terminate at {:.3f}.".format(env.c1_record[-1][0]))
                counter = args.patience // 2
            if abs(max(r2) - min(r2)) <= 0.03 and not env.stop[1]:
                env.stop[1] = True
                print("Layer2 RL terminate at {:.3f}.".format(env.c1_record[-1][1]))
                counter = args.patience // 2

        train_metrics = hgnn.train_with_RL(env, Agent1, Agent2, data, epoch, ace)
        train_metric_record.append(train_metrics[hgnn.key_param])
        train_metrics['loss'].backward()
        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(hgnn.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()

        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {}'.format(lr_scheduler.get_lr()[0]),
                                   format_metrics(train_metrics, 'train'),
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))
        if (epoch + 1) % args.eval_freq == 0:
            hgnn.eval()
            embeddings = hgnn.encode(data['features'], data['adj_train_norm'])
            val_metrics = hgnn.compute_metrics(embeddings, data, 'val')
            val_metric_record.append(val_metrics[hgnn.key_param])
            
            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))

            if hgnn.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = hgnn.compute_metrics(embeddings, data, 'test')
                best_emb = embeddings.cpu()
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter >= args.patience and epoch > args.min_epochs and all(env.stop):
                    logging.info("Early stopping")
                    break

    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if not best_test_metrics:
        hgnn.eval()
        best_emb = hgnn.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = hgnn.compute_metrics(best_emb, data, 'test')
    logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))
    if args.save:
        # Save embeddings and attentions
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        if hasattr(hgnn.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(hgnn.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        # Save model
        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(hgnn.state_dict(), os.path.join(save_dir, 'model.pth'))

        # Save curvature record and figures
        np.save(os.path.join(save_dir, 'curv1.npy'), np.array(env.c1_record))
        np.save(os.path.join(save_dir, 'curv2.npy'), np.array(env.c2_record))

        # Save acc record
        np.save(os.path.join(save_dir, 'metric_record.npy'), np.array([train_metric_record, val_metric_record]))

        logging.info("Agent1: {}, Agent2: {}".format(env.c1, env.c2))
        logging.info(f"Saved model in {save_dir}")

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
