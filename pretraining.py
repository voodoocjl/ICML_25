import os
import sys

import torch
import argparse
import numpy as np
import torch.nn as nn
from Arguments import Arguments

from torch import optim
from configs import configs
# from nasbench.lib import graph_util
from GVAE_model import  VAEReconstructed_Loss, GVAE
import pickle
import json
import torch.nn.functional as F

myargs = Arguments('MNIST-half')

def get_val_acc_vae(model, cfg, X_adj, X_ops, indices):
    def get_accuracy(inputs, targets):
        N, I, _ = inputs[0].shape
        full_ops_recon, adj_recon = inputs[0], inputs[1]
        ops_recon = full_ops_recon[:, :, 0:-(myargs.n_qubits)]
        full_ops, adj = targets[0], targets[1]
        ops = full_ops[:, :, 0:-(myargs.n_qubits)]
        # post processing, assume non-symmetric
        adj_recon, adj = adj_recon.triu(1), adj.triu(1)
        correct_ops = ops_recon.argmax(dim=-1).eq(ops.argmax(dim=-1)).float().mean().item()
        mean_correct_adj = adj_recon[adj.type(torch.bool)].sum().item() / adj.sum()
        mean_false_positive_adj = adj_recon[(~adj.type(torch.bool)).triu(1)].sum().item() / (
                    N * I * (I - 1) / 2.0 - adj.sum())
        threshold = 0.5  # hard threshold
        adj_recon_thre = adj_recon > threshold
        correct_adj = adj_recon_thre.eq(adj.type(torch.bool)).float().triu(1).sum().item() / (N * I * (I - 1) / 2.0)

        ops_correct = ops_recon.argmax(dim=-1).eq(ops.argmax(dim=-1)).float()
        adj_correct = adj_recon_thre.eq(adj.type(torch.bool)).float()
        return correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj
    model.eval()
    bs = 500
    chunks = len(X_adj) // bs
    if len(X_adj) % bs > 0:
        chunks += 1
    X_adj_split = torch.split(X_adj, bs, dim=0)
    X_ops_split = torch.split(X_ops, bs, dim=0)
    indices_split = torch.split(indices, bs, dim=0)
    correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave, acc_ave = 0, 0, 0, 0, 0
    for i, (adj, ops, ind) in enumerate(zip(X_adj_split, X_ops_split, indices_split)):
        # preprocessing
        adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
        # forward
        ops_recon, adj_recon, mu, logvar = model.forward(ops, adj)
        # reverse preprocessing
        adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
        adj, ops = prep_reverse(adj, ops)
        correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj = get_accuracy((ops_recon, adj_recon), (ops, adj))
        correct_ops_ave += correct_ops * len(ind)/len(indices)
        mean_correct_adj_ave += mean_correct_adj * len(ind)/len(indices)
        mean_false_positive_adj_ave += mean_false_positive_adj * len(ind)/len(indices)
        correct_adj_ave += correct_adj * len(ind)/len(indices)

    return correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave
def is_valid_circuit(adj, ops):
    # allowed_gates = ['PauliX', 'PauliY', 'PauliZ', 'Hadamard', 'RX', 'RY', 'RZ', 'CNOT', 'CZ', 'U3', 'SWAP']
    allowed_gates = ['Identity', 'RX', 'RY', 'RZ', 'C(U3)']    # QWAS with data uploading
    if len(adj) != len(ops) or len(adj[0]) != len(ops):
        return False
    if ops[0] != 'START' or ops[-1] != 'END':
        return False
    for i in range(1, len(ops)-1):
        if ops[i] not in allowed_gates:
            return False
    return True
def load_json(f_name):
    """load circuit dataset."""
    with open(f_name, 'r') as file:
        dataset = json.loads(file.read())
    return dataset

def preprocessing(A, H, method, lbd=None):
    # FixMe: Attention multiplying D or lbd are not friendly with the crossentropy loss in GAE
    def stacked_spmm(A, B):
        assert A.dim() == 3
        assert B.dim() == 3
        return torch.matmul(A, B)
    assert A.dim()==3

    if method == 0:
        return A, H

    if method==1:
        # Adding global node with padding
        A = F.pad(A, (0,1), 'constant', 1.0)
        A = F.pad(A, (0,0,0,1), 'constant', 0.0)
        H = F.pad(H, (0,1,0,1), 'constant', 0.0 )
        H[:, -1, -1] = 1.0

    if method==1:
        # using A^T instead of A
        # and also adding a global node
        A = A.transpose(-1, -2)
        D_in = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=1)))
        D_out = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=2)))
        DA = stacked_spmm(D_in, A) # swap D_in and D_out
        DAD = stacked_spmm(DA, D_out)
        return DAD, H

    elif method == 2:
        assert lbd!=None
        # using lambda*A + (1-lambda)*A^T
        A = lbd * A + (1-lbd)*A.transpose(-1, -2)
        D_in = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=1)))
        D_out = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=2)))
        DA = stacked_spmm(D_in, A)  # swap D_in and D_out
        DAD = stacked_spmm(DA, D_out)
        def prep_reverse(DAD, H):
            AD = stacked_spmm(1.0/D_in, DAD)
            A =  stacked_spmm(AD, 1.0/D_out)
            return A.triu(1), H
        return DAD, H, prep_reverse

    elif method == 3:
        # bidirectional DAG
        assert lbd != None
        # using lambda*A + (1-lambda)*A^T
        A = lbd * A + (1 - lbd) * A.transpose(-1, -2)
        def prep_reverse(A, H):
            return 1.0/lbd*A.triu(1), H
        return A, H, prep_reverse

    elif method == 4:
        A = A + A.triu(1).transpose(-1, -2)
        def prep_reverse(A, H):
            return A.triu(1), H
        return A, H, prep_reverse
def transform_operations(max_idx):
    transform_dict = {0: 'START', 1: 'Identity', 2: 'RX', 3: 'RY', 4: 'RZ', 5: 'C(U3)', 6: 'END'}
    ops = []
    for idx in max_idx:
        ops.append(transform_dict[idx.item()])
    return ops


def build_dataset(dataset, list):
    # indices = np.random.permutation(list)
    X_adj = []
    X_ops = []
    for ind in list:
        X_adj.append(torch.Tensor(dataset[ind]['adj_matrix']))
        X_ops.append(torch.Tensor(dataset[ind]['gate_matrix']))
    X_adj = torch.stack(X_adj)
    X_ops = torch.stack(X_ops)
    return X_adj, X_ops, torch.Tensor(list)




def pretraining_model(dataset, cfg, args):
    train_ind_list, val_ind_list = range(int(len(dataset) * 0.9)), range(int(len(dataset) * 0.9), len(dataset))
    X_adj_train, X_ops_train, indices_train = build_dataset(dataset, train_ind_list)
    X_adj_val, X_ops_val, indices_val = build_dataset(dataset, val_ind_list)

    model = GVAE((args.input_dim, 32, 64, 128, 64, 32, args.dim), normalize=True, dropout=args.dropout,
                 **cfg['GAE'])
    # model.load_state_dict(torch.load('pretrained/best_model.pt'))
    acc_ops_val, mean_corr_adj_val, mean_fal_pos_adj_val, acc_adj_val = get_val_acc_vae(model, cfg, X_adj_val,
                                                                                        X_ops_val, indices_val)
    print(
        'validation set: acc_ops:{0:.4f}, mean_corr_adj:{1:.4f}, mean_fal_pos_adj:{2:.4f}, acc_adj:{3:.4f}'.format(
            acc_ops_val, mean_corr_adj_val, mean_fal_pos_adj_val, acc_adj_val))

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
    epochs = args.epochs
    bs = args.bs
    best_acc=0
    loss_total = []
    for epoch in range(0, epochs):
        chunks = len(train_ind_list) // bs
        if len(train_ind_list) % bs > 0:
            chunks += 1
        X_adj_split = torch.split(X_adj_train, bs, dim=0)
        X_ops_split = torch.split(X_ops_train, bs, dim=0)
        indices_split = torch.split(indices_train, bs, dim=0)
        loss_epoch = []
        Z = []
        for i, (adj, ops, ind) in enumerate(zip(X_adj_split, X_ops_split, indices_split)):
            optimizer.zero_grad()
            # preprocessing
            adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
            # forward
            ops_recon, adj_recon, mu, logvar = model(ops, adj)
            Z.append(mu)
            adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
            adj, ops = prep_reverse(adj, ops)
            loss = VAEReconstructed_Loss(**cfg['loss'])((ops_recon, adj_recon), (ops, adj), mu, logvar)  # With KL
            # loss = Reconstructed_Loss(**cfg['loss'])((ops_recon, adj_recon), (ops, adj)) # Without KL
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_epoch.append(loss.item())
            # if i % 100 == 0:
            #     print('epoch {}: batch {} / {}: loss: {:.5f}'.format(epoch, i, chunks, loss.item()))
        Z = torch.cat(Z, dim=0)
        z_mean, z_std = Z.mean(0), Z.std(0)
        validity_counter = 0
        buckets = {}
        model.eval()
        for _ in range(args.latent_points):
            z = torch.randn(X_adj_train[0].shape[0], args.dim)
            z = z * z_std + z_mean
            # if epoch == args.epochs - 1:
            #     torch.save(z, 'z.pt')
            full_op, full_ad = model.decoder(z.unsqueeze(0))
            full_op = full_op.squeeze(0).cpu()
            ad = full_ad.squeeze(0).cpu()
            op = full_op[:, 0:-(myargs.n_qubits)]
            max_idx = torch.argmax(op, dim=-1)
            one_hot = torch.zeros_like(op)
            for i in range(one_hot.shape[0]):
                one_hot[i][max_idx[i]] = 1
            op_decode = transform_operations(max_idx)
            ad_decode = (ad > 0.5).int().triu(1).numpy()
            ad_decode = np.ndarray.tolist(ad_decode)
            if is_valid_circuit(ad_decode, op_decode):
                validity_counter += 1
                

        validity = validity_counter / args.latent_points
        print('Ratio of valid decodings from the prior: {:.4f}'.format(validity))
        print('Ratio of unique decodings from the prior: {:.4f}'.format(len(buckets) / (validity_counter + 1e-8)))
        acc_ops_val, mean_corr_adj_val, mean_fal_pos_adj_val, acc_adj_val = get_val_acc_vae(model, cfg, X_adj_val,
                                                                                            X_ops_val, indices_val)
        print(
            'validation set: acc_ops:{0:.4f}, mean_corr_adj:{1:.4f}, mean_fal_pos_adj:{2:.4f}, acc_adj:{3:.4f}'.format(
                acc_ops_val, mean_corr_adj_val, mean_fal_pos_adj_val, acc_adj_val))
        print('epoch {}: average loss {:.5f}'.format(epoch, sum(loss_epoch) / len(loss_epoch)))
        loss_total.append(sum(loss_epoch) / len(loss_epoch))

        if acc_adj_val> best_acc:
            best_acc = acc_adj_val
            torch.save(model.state_dict(),f'pretrained/{myargs.n_qubits}_best_model.pt')
            print('save model')



    print('loss for epochs: \n', loss_total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretraining')
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    # parser.add_argument('--data', type=str, default=f'circuit\\data\\data_{vc.num_qubits}_qubits.json',
    #                     help='Data file (default: data.json')
    parser.add_argument('--name', type=str, default=f'circuits_{myargs.n_qubits}_qubits.json',
                        help='circuits with correspoding number of qubits')
    parser.add_argument('--cfg', type=int, default=4,
                        help='configuration (default: 4)')
    parser.add_argument('--bs', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=16,
                        help='training epochs (default: 16)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='decoder implicit regularization (default: 0.3)')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='use input normalization')
    parser.add_argument('--input_dim', type=int, default=2 + len(myargs.allowed_gates) + myargs.n_qubits)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=16,
                        help='feature (latent) dimension (default: 16)')
    parser.add_argument('--hops', type=int, default=5)
    parser.add_argument('--mlps', type=int, default=2)
    parser.add_argument('--latent_points', type=int, default=10000,
                        help='latent points for validaty check (default: 10000)')

    args = parser.parse_args()
    # args.epochs = 100
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = configs[args.cfg]
    dataset = load_json(myargs.data_file)
    print('using {}'.format(myargs.data_file))
    print('feat dim {}'.format(args.dim))
    train_ind_list, val_ind_list = range(int(len(dataset) * 0.9)), range(int(len(dataset) * 0.9), len(dataset))
    X_adj_train, X_ops_train, indices_train = build_dataset(dataset, train_ind_list)
    print(X_adj_train[0])
    print(X_ops_train[0])
    print(indices_train[0])
    pretraining_model(dataset, cfg, args)
