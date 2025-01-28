import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
import numpy as np

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
def transform_operations(max_idx):
    transform_dict = {0: 'START', 1: 'Identity', 2: 'RX', 3: 'RY', 4: 'RZ', 5: 'C(U3)', 6: 'END'}
    ops = []
    for idx in max_idx:
        ops.append(transform_dict[idx.item()])
    return ops

# def generate_single_enta(op_results, n_qubits):
#     n_layers = int(len(op_results) / (2 * n_qubits))
#     single=[[i]+[-1]*2*n_layers for i in range(1,1+n_qubits)]
#     enta=[[i]+[-1]*n_layers for i in range(1,1+n_qubits)]
#     for i in range(0,len(op_results),2*n_qubits):
#         for j in range(0, n_qubits):
#             op = op_results[i + j]
#             if (len(op[-1]) == 1):
#                 col=int(i / n_layers)
#                 row=op[-1][0]
#                 single[row][1+col]= int('data' in op_results[i+j])
#                 single[row][2+col]=int('U3' in op_results[i+j])
#         for j in range(n_qubits, 2*n_qubits):
#             op = op_results[i + j]
#             if len(op[-1])==2 and 'C(U3)' in op:
#                 row,num=op[-1]
#                 col=int(i/(n_layers*2))
#                 enta[row][1+col]=num+1
def generate_single_enta(full_op, n_qubits):
    full_op = full_op.squeeze(0).cpu()    
    op = full_op[:, 0:-(n_qubits)]
    max_idx = torch.argmax(op, dim=-1)    
    op_decode = transform_operations(max_idx)
    op_decode_check=op_decode
    op_results = []
    qubit_choices = full_op[:, -(n_qubits):]
    for i in range(qubit_choices.size(0)):
        if op_decode_check[i] == 'C(U3)':
            # Select the two largest values and sort indices by value
            # Control qubit takes the largest value, target qubit takes the second largest
            values, indices = torch.topk(qubit_choices[i], 2)
            indices = indices[values.argsort(descending=True)]
            op_results.append((op_decode[i], indices.tolist()))
        elif op_decode_check[i] == 'RX':
            values, indices = torch.topk(qubit_choices[i], 1)
            op_results.append(('U3', indices.tolist()))
        elif op_decode_check[i] == 'Identity':
            # If 'Identity' in single-qubit position
            if (i-1)%(n_qubits*2) < n_qubits:
                values, indices = torch.topk(qubit_choices[i], 1)
                op_results.append(('Identity', indices.tolist()))
            # If 'Identity' in double-qubit position
            else:
                values, indices = torch.topk(qubit_choices[i], 2)
                indices = indices[values.argsort(descending=True)]
                op_results.append(('Identity', indices.tolist()))
        elif op_decode_check[i] == 'RY':
            values, indices = torch.topk(qubit_choices[i], 1)
            op_results.append(('data', indices.tolist()))
        elif op_decode_check[i] == 'RZ':
            values, indices = torch.topk(qubit_choices[i], 1)
            op_results.append(('data','U3', indices.tolist()))
        else:
            pass  # Skip 'START', 'END' and 'Identity' as they do not change the state
    n_layers = int(len(op_results) / (2 * n_qubits))
    
    single=np.array([[i]+[-1]*2*n_layers for i in range(1,1+n_qubits)])
    enta=np.array([[i]+[-1]*n_layers for i in range(1,1+n_qubits)])
    
    for i in range(0,len(op_results),2*n_qubits):
        for j in range(0, n_qubits):
            op = op_results[i + j]
            if (len(op[-1]) == 1):
                col=int(i / n_qubits)
                row=op[-1][0]
                single[row][1+col]= int('data' in op_results[i+j])
                single[row][2+col]=int('U3' in op_results[i+j])
        for j in range(n_qubits, 2*n_qubits):
            op = op_results[i + j]
            if len(op[-1])==2:
                if 'C(U3)' in op: 
                    row,num=op[-1]
                if 'Identity' in op:
                    row = op[-1][0]
                    num = row
                col=int(i/(n_qubits*2))
                enta[row][1+col]=num+1
    return single,enta

def is_valid_ops_adj(full_op,full_ad, n_qubits):
    full_op = full_op.squeeze(0).cpu()
    ad = full_ad.squeeze(0).cpu()
    op = full_op[:, 0:-(n_qubits)]
    max_idx = torch.argmax(op, dim=-1)
    one_hot = torch.zeros_like(op)
    for i in range(one_hot.shape[0]):
        one_hot[i][max_idx[i]] = 1
    op_decode = transform_operations(max_idx)
    ad_decode = (ad > 0.5).int().triu(1).numpy()
    ad_decode = np.ndarray.tolist(ad_decode)
    res= is_valid_circuit(ad_decode, op_decode)
            
    return res

def stacked_spmm(A, B):
    assert A.dim()==3
    assert B.dim()==3
    return torch.matmul(A, B)

def preprocessing(A, H, method, lbd=None):
    # FixMe: Attention multiplying D or lbd are not friendly with the crossentropy loss in GAE
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

def normalize_adj(A):
    # Compute the sum of each row and column in A
    sum_A_dim1 = A.sum(dim=1)
    sum_A_dim2 = A.sum(dim=2)

    # Check if sum_A_dim1 and sum_A_dim2 contain any zero values
    contains_zero_dim1 = (sum_A_dim1 == 0).any()
    contains_zero_dim2 = (sum_A_dim2 == 0).any()
    # if contains_zero_dim1:
    #     print("sum_A_dim1 contains zero values.")
    # if contains_zero_dim2:
    #     print("sum_A_dim2 contains zero values.")

    # If zero values are present, replace them with a very small number to avoid division by zero
    sum_A_dim1[sum_A_dim1 == 0] = 1e-10
    sum_A_dim2[sum_A_dim2 == 0] = 1e-10

    D_in = torch.diag_embed(1.0 / torch.sqrt(sum_A_dim1))
    D_out = torch.diag_embed(1.0 / torch.sqrt(sum_A_dim2))
    DA = stacked_spmm(D_in, A)  # swap D_in and D_out
    DAD = stacked_spmm(DA, D_out)
    return DAD


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0., bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.dropout = dropout

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        torch.nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, ops, adj):
        ops = F.dropout(ops, self.dropout, self.training)
        support = F.linear(ops, self.weight)
        output = F.relu(torch.matmul(adj, support))

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_features) + '->' + str(self.out_features) + ')'


class GVAE(nn.Module):
    def __init__(self, dims, normalize, dropout, **kwargs):
        super(GVAE, self).__init__()
        self.encoder = VAEncoder(dims, normalize, dropout)
        self.decoder = Decoder(dims[-1], dims[0], dropout, **kwargs)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, ops, adj):
        mu, logvar = self.encoder(ops, adj)
        z = self.reparameterize(mu, logvar)
        ops_recon, adj_recon = self.decoder(z)
        return ops_recon, adj_recon, mu, logvar

class VAEncoder(nn.Module):
    def __init__(self, dims, normalize, dropout):
        super(VAEncoder, self).__init__()
        self.gcs = nn.ModuleList(self.get_gcs(dims, dropout))
        self.gc_mu = GraphConvolution(dims[-2], dims[-1], dropout)
        self.gc_logvar = GraphConvolution(dims[-2], dims[-1], dropout)
        self.normalize = normalize

    def get_gcs(self,dims,dropout):
        gcs = []
        for k in range(len(dims)-1):
            gcs.append(GraphConvolution(dims[k],dims[k+1], dropout))
        return gcs

    def forward(self, ops, adj):
        if self.normalize:
            adj = normalize_adj(adj)
        x = ops
        for gc in self.gcs[:-1]:
            x = gc(x, adj)
        mu = self.gc_mu(x, adj)
        logvar = self.gc_logvar(x, adj)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, embedding_dim, input_dim, dropout, activation_adj=torch.sigmoid, activation_ops=torch.sigmoid, adj_hidden_dim=None, ops_hidden_dim=None):
        super(Decoder, self).__init__()
        if adj_hidden_dim == None:
            self.adj_hidden_dim = embedding_dim
        if ops_hidden_dim == None:
            self.ops_hidden_dim = embedding_dim
        self.activation_adj = activation_adj
        self.activation_ops = activation_ops
        self.weight = torch.nn.Linear(embedding_dim, input_dim)
        self.dropout = dropout

    def forward(self, embedding):
        embedding = F.dropout(embedding, p=self.dropout, training=self.training)
        ops = self.weight(embedding)
        adj = torch.matmul(embedding, embedding.permute(0, 2, 1))
        return self.activation_adj(ops), self.activation_adj(adj)

class VAEReconstructed_Loss(object):
    def __init__(self, w_ops=1.0, w_adj=1.0, loss_ops=None, loss_adj=None):
        super().__init__()
        self.w_ops = w_ops
        self.w_adj = w_adj
        self.loss_ops = loss_ops
        self.loss_adj = loss_adj

    def __call__(self, inputs, targets, mu, logvar):
        ops_recon, adj_recon = inputs[0], inputs[1]
        ops, adj = targets[0], targets[1]
        loss_ops = self.loss_ops(ops_recon, ops)
        loss_adj = self.loss_adj(adj_recon, adj)
        loss = self.w_ops * loss_ops + self.w_adj * loss_adj
        KLD = -0.5 / (ops.shape[0] * ops.shape[1]) * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 2))
        return loss + KLD