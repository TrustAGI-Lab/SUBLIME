import argparse
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import load_data
from model import GCN, GCL
from graph_learners import *
from utils import *
from sklearn.cluster import KMeans
import dgl

import random

EOS = 1e-10

class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()


    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)
        dgl.seed(seed)
        dgl.random.seed(seed)


    def loss_cls(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu


    def loss_gcl(self, model, graph_learner, features, anchor_adj):

        # view 1: anchor graph
        if args.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)

        z1, _ = model(features_v1, anchor_adj, 'anchor')

        # view 2: learned graph
        if args.maskfeat_rate_learner:
            mask, _ = get_feat_mask(features, args.maskfeat_rate_learner)
            features_v2 = features * (1 - mask)
        else:
            features_v2 = copy.deepcopy(features)

        learned_adj = graph_learner(features)
        if not args.sparse:
            learned_adj = symmetrize(learned_adj)
            learned_adj = normalize(learned_adj, 'sym', args.sparse)

        z2, _ = model(features_v2, learned_adj, 'learner')

        # compute loss
        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            # random.shuffle(node_idxs)
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss += model.calc_loss(z1[batch], z2[batch]) * weight
        else:
            loss = model.calc_loss(z1, z2)

        return loss, learned_adj


    def evaluate_adj_by_cls(self, Adj, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, args):

        model = GCN(in_channels=nfeats, hidden_channels=args.hidden_dim_cls, out_channels=nclasses, num_layers=args.nlayers_cls,
                    dropout=args.dropout_cls, dropout_adj=args.dropedge_cls, Adj=Adj, sparse=args.sparse)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)

        bad_counter = 0
        best_val = 0
        best_model = None

        if torch.cuda.is_available():
            model = model.cuda()
            train_mask = train_mask.cuda()
            val_mask = val_mask.cuda()
            test_mask = test_mask.cuda()
            features = features.cuda()
            labels = labels.cuda()

        for epoch in range(1, args.epochs_cls + 1):
            model.train()
            loss, accu = self.loss_cls(model, train_mask, features, labels)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if epoch % 10 == 0:
                model.eval()
                val_loss, accu = self.loss_cls(model, val_mask, features, labels)
                if accu > best_val:
                    bad_counter = 0
                    best_val = accu
                    best_model = copy.deepcopy(model)
                else:
                    bad_counter += 1

                if bad_counter >= args.patience_cls:
                    break
        best_model.eval()
        test_loss, test_accu = self.loss_cls(best_model, test_mask, features, labels)
        return best_val, test_accu, best_model


    def train(self, args):

        torch.cuda.set_device(args.gpu)

        if args.gsl_mode == 'structure_refinement':
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original = load_data(args)
        elif args.gsl_mode == 'structure_inference':
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, _ = load_data(args)

        if args.downstream_task == 'classification':
            test_accuracies = []
            validation_accuracies = []
        elif args.downstream_task == 'clustering':
            n_clu_trials = copy.deepcopy(args.ntrials)
            args.ntrials = 1

        for trial in range(args.ntrials):

            self.setup_seed(trial)

            if args.gsl_mode == 'structure_inference':
                if args.sparse:
                    anchor_adj_raw = torch_sparse_eye(features.shape[0])
                else:
                    anchor_adj_raw = torch.eye(features.shape[0])
            elif args.gsl_mode == 'structure_refinement':
                if args.sparse:
                    anchor_adj_raw = adj_original
                else:
                    anchor_adj_raw = torch.from_numpy(adj_original)


            anchor_adj = normalize(anchor_adj_raw, 'sym', args.sparse)

            if args.sparse:
                anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)
                anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)

            if args.type_learner == 'fgp':
                graph_learner = FGP_learner(features.cpu(), args.k, args.sim_function, 6, args.sparse)
            elif args.type_learner == 'mlp':
                graph_learner = MLP_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                     args.activation_learner)
            elif args.type_learner == 'att':
                graph_learner = ATT_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                          args.activation_learner)
            elif args.type_learner == 'gnn':
                graph_learner = GNN_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                     args.activation_learner, anchor_adj)

            model = GCL(nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
                         emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                         dropout=args.dropout, dropout_adj=args.dropedge_rate, sparse=args.sparse)

            optimizer_cl = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=args.lr, weight_decay=args.w_decay)


            if torch.cuda.is_available():
                model = model.cuda()
                graph_learner = graph_learner.cuda()
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                test_mask = test_mask.cuda()
                features = features.cuda()
                labels = labels.cuda()
                if not args.sparse:
                    anchor_adj = anchor_adj.cuda()

            if args.downstream_task == 'classification':
                best_val = 0
                best_val_test = 0
                best_epoch = 0

            for epoch in range(1, args.epochs + 1):

                model.train()
                graph_learner.train()

                loss, Adj = self.loss_gcl(model, graph_learner, features, anchor_adj)

                optimizer_cl.zero_grad()
                optimizer_learner.zero_grad()
                loss.backward()
                optimizer_cl.step()
                optimizer_learner.step()

                # Structure Bootstrapping
                if (1 - args.tau) and (args.c == 0 or epoch % args.c == 0):
                    if args.sparse:
                        learned_adj_torch_sparse = dgl_graph_to_torch_sparse(Adj)
                        anchor_adj_torch_sparse = anchor_adj_torch_sparse * args.tau \
                                                  + learned_adj_torch_sparse * (1 - args.tau)
                        anchor_adj = torch_sparse_to_dgl_graph(anchor_adj_torch_sparse)
                    else:
                        anchor_adj = anchor_adj * args.tau + Adj.detach() * (1 - args.tau)

                print("Epoch {:05d} | CL Loss {:.4f}".format(epoch, loss.item()))

                if epoch % args.eval_freq == 0:
                    if args.downstream_task == 'classification':
                        model.eval()
                        graph_learner.eval()
                        f_adj = Adj

                        if args.sparse:
                            f_adj.edata['w'] = f_adj.edata['w'].detach()
                        else:
                            f_adj = f_adj.detach()

                        val_accu, test_accu, _ = self.evaluate_adj_by_cls(f_adj, features, nfeats, labels,
                                                                               nclasses, train_mask, val_mask, test_mask, args)

                        if val_accu > best_val:
                            best_val = val_accu
                            best_val_test = test_accu
                            best_epoch = epoch

                    elif args.downstream_task == 'clustering':
                        model.eval()
                        graph_learner.eval()
                        _, embedding = model(features, Adj)
                        embedding = embedding.cpu().detach().numpy()

                        acc_mr, nmi_mr, f1_mr, ari_mr = [], [], [], []
                        for clu_trial in range(n_clu_trials):
                            kmeans = KMeans(n_clusters=nclasses, random_state=clu_trial).fit(embedding)
                            predict_labels = kmeans.predict(embedding)
                            cm_all = clustering_metrics(labels.cpu().numpy(), predict_labels)
                            acc_, nmi_, f1_, ari_ = cm_all.evaluationClusterModelFromLabel(print_results=False)
                            acc_mr.append(acc_)
                            nmi_mr.append(nmi_)
                            f1_mr.append(f1_)
                            ari_mr.append(ari_)

                        acc, nmi, f1, ari = np.mean(acc_mr), np.mean(nmi_mr), np.mean(f1_mr), np.mean(ari_mr)

            if args.downstream_task == 'classification':
                validation_accuracies.append(best_val.item())
                test_accuracies.append(best_val_test.item())
                print("Trial: ", trial + 1)
                print("Best val ACC: ", best_val.item())
                print("Best test ACC: ", best_val_test.item())
            elif args.downstream_task == 'clustering':
                print("Final ACC: ", acc)
                print("Final NMI: ", nmi)
                print("Final F-score: ", f1)
                print("Final ARI: ", ari)

        if args.downstream_task == 'classification' and trial != 0:
            self.print_results(validation_accuracies, test_accuracies)


    def print_results(self, validation_accu, test_accu):
        s_val = "Val accuracy: {:.4f} +/- {:.4f}".format(np.mean(validation_accu), np.std(validation_accu))
        s_test = "Test accuracy: {:.4f} +/- {:.4f}".format(np.mean(test_accu),np.std(test_accu))
        print(s_val)
        print(s_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Experimental setting
    parser.add_argument('-dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed'])
    parser.add_argument('-ntrials', type=int, default=5)
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-gsl_mode', type=str, default="structure_inference",
                        choices=['structure_inference', 'structure_refinement'])
    parser.add_argument('-eval_freq', type=int, default=5)
    parser.add_argument('-downstream_task', type=str, default='classification',
                        choices=['classification', 'clustering'])
    parser.add_argument('-gpu', type=int, default=0)

    # GCL Module - Framework
    parser.add_argument('-epochs', type=int, default=1000)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-hidden_dim', type=int, default=512)
    parser.add_argument('-rep_dim', type=int, default=64)
    parser.add_argument('-proj_dim', type=int, default=64)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=0)
    parser.add_argument('-nlayers', type=int, default=2)

    # GCL Module -Augmentation
    parser.add_argument('-maskfeat_rate_learner', type=float, default=0.2)
    parser.add_argument('-maskfeat_rate_anchor', type=float, default=0.2)
    parser.add_argument('-dropedge_rate', type=float, default=0.5)

    # GSL Module
    parser.add_argument('-type_learner', type=str, default='fgp', choices=["fgp", "att", "mlp", "gnn"])
    parser.add_argument('-k', type=int, default=30)
    parser.add_argument('-sim_function', type=str, default='cosine', choices=['cosine', 'minkowski'])
    parser.add_argument('-gamma', type=float, default=0.9)
    parser.add_argument('-activation_learner', type=str, default='relu', choices=["relu", "tanh"])

    # Evaluation Network (Classification)
    parser.add_argument('-epochs_cls', type=int, default=200)
    parser.add_argument('-lr_cls', type=float, default=0.001)
    parser.add_argument('-w_decay_cls', type=float, default=0.0005)
    parser.add_argument('-hidden_dim_cls', type=int, default=32)
    parser.add_argument('-dropout_cls', type=float, default=0.5)
    parser.add_argument('-dropedge_cls', type=float, default=0.25)
    parser.add_argument('-nlayers_cls', type=int, default=2)
    parser.add_argument('-patience_cls', type=int, default=10)

    # Structure Bootstrapping
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-c', type=int, default=0)

    args = parser.parse_args()

    experiment = Experiment()
    experiment.train(args)
