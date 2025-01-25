import json
from math import log2, ceil
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from Network import Attention, FCN, normalize, FC
from FusionModel import cir_to_matrix
import time

from GVAE_translator import generate_circuits, get_gate_and_adj_matrix
from GVAE_model import GVAE, preprocessing
from configs import configs

# torch.cuda.is_available = lambda : False


def get_label(energy, tree_height, mean = None):
    # label = energy.clone()
    # if mean and (mean < float('inf')):
    #     energy_mean = mean
    # else:
    #     energy_mean = energy.mean()
    # for i in range(energy.shape[0]):
    #     label[i] = energy[i] > energy_mean

    x = energy    
    a = [[i for i in range(len(x))]]
    for i in range(1,tree_height):
        t = []
        for j in range(2**(i-1)):        
            index = a[j]
            if len(index):
                mean = x[index].mean()
            else:
                mean = []
            t.append(torch.tensor([item for item in index if x[item] >= mean]))
            t.append(torch.tensor([item for item in index if x[item] < mean]))
        a = t
    label = torch.zeros((len(x), tree_height-1))
    for i in range(len(a)):
        index = a[i]
        if len(index):
            for j in range(len(index)):
                string_num = bin(i)[2:].zfill(tree_height-1)
                label[index[j]] = torch.tensor([int(char) for char in string_num])
    return label

def insert_job(change_code, job):
        if type(job[0]) == type([]):
            qubit = [sub[0] for sub in job]
        else:
            qubit = [job[0]]
            job = [job]
        if change_code != None:            
            for change in change_code:
                if change[0] not in qubit:
                    job.append(change)
        return job

def initialize_model_parameters(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def print_grad_norm(model):
    grads = [param.grad.detach().flatten() for param in model.parameters() if param.grad is not None]
    norm = torch.cat(grads).norm()
    print('Grad Norm: ', norm)

class Classifier:
    def __init__(self, samples, arch_code, node_id, tree_height, fold):
        assert type(samples) == type({})        

        self.samples          = samples
        self.arch_code        = arch_code   #[qubits, layers]        
        self.input_dim_2d     = 21
        self.training_counter = 0
        self.node_layer       = ceil(log2(node_id + 2) - 1)
        self.tree_height      = tree_height
        self.fold             = fold
        # self.model            = Linear(self.input_dim_2d, 2)
        # self.model            = Mlp(self.input_dim_2d, 6, 2)        
        self.model            = FCN(arch_code)
        # self.model            = FC(arch_code)
        
        self.loss_fn          = nn.CrossEntropyLoss() #nn.MSELoss()
        self.l_rate           = 0.001
        self.optimizer        = optim.Adam(self.model.parameters(), lr=self.l_rate, betas=(0.9, 0.999), eps=1e-08)
        self.epochs           = []
        self.training_accuracy = [0]
        self.boundary         = -1
        self.nets             = None
        self.maeinv           = None
        self.labels           = None
        self.mean             = 0        
        self.period           = 10

        checkpoint = torch.load('pretrained/best_model.pt', map_location=torch.device('cpu'))
        # checkpoint = torch.load('pretrained/5_best_model.pt', map_location=torch.device('cpu'))

        input_dim = 2 + 5 + int(self.arch_code[0]/self.fold)
        self.GVAE_model = GVAE((input_dim, 32, 64, 128, 64, 32, 16), normalize=True, dropout=0.3, **configs[4]['GAE'])
        self.GVAE_model.load_state_dict(checkpoint)
        


    def update_samples(self, latest_samples, tree_height, layer=0, latest_labels=None):
        assert type(latest_samples) == type(self.samples)
        self.samples = latest_samples        
        if layer == 0:
            sampled_nets = []
            nets_maeinv  = []
            for k, v in latest_samples.items():
                net = json.loads(k)            
                sampled_nets.append(net)
                nets_maeinv.append(v)
            # self.nets = torch.from_numpy(np.asarray(sampled_nets, dtype=np.float32))
            # self.nets = normalize(self.nets)
            self.nets = self.arch_to_z(sampled_nets)
            self.maeinv = torch.from_numpy(np.asarray(nets_maeinv, dtype=np.float32).reshape(-1, 1))
            self.labels = get_label(self.maeinv, tree_height)
            if torch.cuda.is_available():
                self.nets = self.nets.cuda()
                self.maeinv = self.maeinv.cuda()
                self.labels = self.labels.cuda()
        else:
            self.pred_labels = latest_labels


    def train(self):
        
        self.epochs = 100
        
        # in a rare case, one branch has no networks
        if len(self.nets) == 0:
            return
        # linear, mlp
        nets = self.nets
        # labels = 2 * self.labels - 1
        labels = self.labels
        n_heads = self.tree_height - 1
        train_data = TensorDataset(nets, labels)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        initialize_model_parameters(self.model)
        if torch.cuda.is_available():
            self.model.cuda()
        
        # for param in self.model.parameters():
        #     param.requires_grad = True
        # for param in [self.model.cls1.weight, self.model.cls1.bias,
        #               self.model.cls2.weight, self.model.cls2.bias,
        #               self.model.cls3.weight, self.model.cls3.bias]:
        #     param.requires_grad = True
                    
        for epoch in range(self.epochs):
            for x, y in train_loader:                
                # clear grads
                self.optimizer.zero_grad()
                # forward to get predicted values
                outputs = self.model(x)                
                loss = self.loss_fn(outputs[0][:,:,:n_heads], y.long())
                loss.backward(retain_graph=True)  # back props

                # gradient norm
                # print_grad_norm(self.model)

                self.optimizer.step()  # update the parameters

            # pred = self.model(nets)        
            # pred_label = pred[1][:, :n_heads].float().cpu()
            # true_label = self.labels.cpu()        
            # acc = accuracy_score(true_label.numpy(), pred_label.numpy())
            # print(f'Epoch: {epoch}, Training Accuracy: {acc:.2f}')

        # training accuracy
        pred = self.model(nets)
        
        pred_label = pred[1][:, :n_heads].float().cpu()
        true_label = self.labels.cpu()        
        acc = accuracy_score(true_label.numpy(), pred_label.numpy())
        self.training_accuracy.append(acc)

    def arch_to_z(self, archs):
        adj_list, op_list = [], []
        arch_code = [int(self.arch_code[0] / self.fold), self.arch_code[1]]
        for net in archs:
            circuit_ops = generate_circuits(net, arch_code)
            _, gate_matrix, adj_matrix = get_gate_and_adj_matrix(circuit_ops, arch_code)
            ops = torch.tensor(gate_matrix, dtype=torch.float32).unsqueeze(0)
            adj = torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0)            
            adj_list.append(adj)
            op_list.append(ops)

        adj = torch.cat(adj_list, dim=0)
        ops = torch.cat(op_list, dim=0)
        adj, ops, prep_reverse = preprocessing(adj, ops, **configs[4]['prep'])
        encoder = self.GVAE_model.encoder
        encoder.eval()
        mu, logvar = encoder(ops, adj)
        return mu
    
    def predict(self, remaining, arch):
        assert type(remaining) == type({})
        remaining_archs = []
        adj_list, op_list = [], []
        for k, v in remaining.items():
            net = json.loads(k)
            if len(net) == len(arch['single'][0]):            
                net = insert_job(arch['single'], net) 
                net = cir_to_matrix(net, arch['enta'], self.arch_code, self.fold)
            else:
                net = insert_job(arch['enta'], net)
                net = cir_to_matrix(arch['single'], net, self.arch_code, self.fold)
            remaining_archs.append(net)
            
        remaining_archs = self.arch_to_z(remaining_archs)
        
        # remaining_archs = torch.from_numpy(np.asarray(remaining_archs, dtype=np.float32))
        # remaining_archs = normalize(remaining_archs)        
                
        if torch.cuda.is_available():
            remaining_archs = remaining_archs.cuda()
            self.model.cuda()
        t1 = time.time()
        outputs = self.model(remaining_archs)
        print('Prediction time: ', time.time()-t1)
        if torch.cuda.is_available():
            remaining_archs = remaining_archs.cpu()
            # outputs         = outputs.cpu()
        diff = -(outputs[0][:, 0, :] - outputs[0][:, 1, :]).abs().detach().cpu()
        result = []        
        result.append(list(remaining.keys()))
        result.append(outputs[1].tolist())
        
        assert len(result[0]) == len(remaining)
        return result, diff


    def split_predictions(self, remaining, arch, layer=0, delta = None):
        # assert type(remaining) == type({})
        samples_badness = [[], []]
        samples_goodies = [[], []]
        delta_badness = []
        delta_goodies = []

        if layer == 0:
            predictions, delta = self.predict(remaining, arch)  # arch_str -> pred_test_mae
        else:
           predictions = remaining
           remaining = remaining[0]
           if len(remaining) == 0:
            return samples_goodies, samples_badness, delta_goodies, delta_badness, []
           

        for index, (k, v) in enumerate(zip(predictions[0], predictions[1])):
            if v[layer] == 1 :                
                samples_badness[0].append(k)
                samples_badness[1].append(v)
                delta_badness.append(index)  # bad index
            else:
                samples_goodies[0].append(k)
                samples_goodies[1].append(v)
                delta_goodies.append(index)
        delta_badness = delta[delta_badness]
        delta_goodies = delta[delta_goodies]
        assert len(samples_badness[0]) + len(samples_goodies[0]) == len(remaining)

        delta = torch.exp(delta).mean(dim=0)
        return samples_goodies, samples_badness, delta_goodies, delta_badness, delta

    """
    def predict_mean(self):
        if len(self.nets) == 0:
            return 0
        # can we use the actual maeinv?
        outputs = self.model(self.nets)
        pred_np = None
        if torch.cuda.is_available():
            pred_np = outputs.detach().cpu().numpy()
        else:
            pred_np = outputs.detach().numpy()
        return np.mean(pred_np)
    """

    def sample_mean(self):
        if len(self.nets) == 0:
            return 0
        outputs = self.maeinv
        true_np = None
        if torch.cuda.is_available():
            true_np = outputs.cpu().numpy()
        else:
            true_np = outputs.numpy()
        return np.mean(true_np)


    def split_data(self, layer=0, f1 = None):
        samples_badness = {}
        samples_goodies = {}
        samples_badness_labels = {}
        samples_goodies_labels = {}        
        if layer == 0:
            if len(self.nets) == 0:
                return samples_goodies, samples_badness
            
            self.train()
            outputs = self.model(self.nets)
            # if torch.cuda.is_available():
            #     self.nets = self.nets.cpu()
            #     outputs   = outputs.cpu()
            predictions = {}
            for k in range(0, len(self.nets)):
                arch_str = list(self.samples)[k]
                predictions[arch_str] = outputs[1][k].cpu().detach().numpy().tolist()  # arch_str -> pred_label
            assert len(predictions) == len(self.nets) 
        else:
            predictions = self.pred_labels

        for k, v in predictions.items():
            # if v < self.sample_mean():
            if v[layer] == 1 :
                samples_badness[k] = self.samples[k]  # (val_loss, test_mae)
                samples_badness_labels[k] = predictions[k]
            else:
                samples_goodies[k] = self.samples[k]  # (val_loss, test_mae)
                samples_goodies_labels[k] = predictions[k]
        assert len(samples_badness) + len(samples_goodies) == len(self.samples)
        return samples_goodies, samples_badness, samples_goodies_labels, samples_badness_labels
