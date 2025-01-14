import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from math import pi
import torch.nn.functional as F
from Arguments import Arguments
import numpy as np
from FusionModel import gen_arch

args = Arguments('MOSI')

def prune_single(change_code):
    single_dict = {}
    single_dict['current_qubit'] = []
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]
        length = len(change_code[0])
        change_code = np.array(change_code)
        change_qbit = change_code[:,0] - 1
        change_code = change_code.reshape(-1, length)
        single_dict['current_qubit'] = change_qbit
        j = 0
        for i in change_qbit:
            single_dict['qubit_{}'.format(i)] = change_code[:, 1:][j].reshape(2, -1)
            j += 1
    return single_dict

def translator(single_code, enta_code, trainable, base_code, fold):
    updated_design = {}
    updated_design = prune_single(single_code)
    net = gen_arch(enta_code, base_code)

    if trainable == 'full' or enta_code == None:
        updated_design['change_qubit'] = None
    else:
        if type(enta_code[0]) != type([]): enta_code = [enta_code]
        updated_design['change_qubit'] = enta_code[-1][0]

    # num of layers
    updated_design['n_layers'] = args.n_layers

    for layer in range(updated_design['n_layers']):
        # categories of single-qubit parametric gates
        for i in range(args.n_qubits):
            updated_design['rot' + str(layer) + str(i)] = 'Rx'
        # categories and positions of entangled gates
        for j in range(args.n_qubits):
            if net[j + layer * args.n_qubits] > 0:
                updated_design['enta' + str(layer) + str(j)] = ('IsingZZ', [j, net[j + layer * args.n_qubits]-1])
            else:
                updated_design['enta' + str(layer) + str(j)] = ('IsingZZ', [abs(net[j + layer * args.n_qubits])-1, j])

    # updated_design['total_gates'] = updated_design['n_layers'] * args.n_qubits * 2
    return updated_design

class TQLayer(tq.QuantumModule):
    def __init__(self, arguments, design):
        super().__init__()
        self.args = arguments
        self.design = design
        self.n_wires = self.args.n_qubits
        self.rots, self.entas = tq.QuantumModuleList(), tq.QuantumModuleList()
        for layer in range(self.design['n_layers']):
            for q in range(self.n_wires):
                # 'trainable' option
                if self.design['change_qubit'] is None:
                    rot_trainable = True
                    enta_trainable = True
                elif q == self.design['change_qubit']:
                    rot_trainable = False
                    enta_trainable = True
                else:
                    rot_trainable = False
                    enta_trainable = False
                # single-qubit parametric gates
                if self.design['rot' + str(layer) + str(q)] == 'Rx':
                    self.rots.append(tq.RX(has_params=True, trainable=rot_trainable))
                else:
                    self.rots.append(tq.RY(has_params=True, trainable=rot_trainable))
                # entangled gates
                if self.design['enta' + str(layer) + str(q)][0] == 'IsingXX':
                    self.entas.append(tq.RXX(has_params=True, trainable=enta_trainable))
                else:
                    self.entas.append(tq.RZZ(has_params=True, trainable=enta_trainable))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        bsz = x.shape[0]
        x = x.reshape(bsz, self.n_wires, 3)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        for layer in range(self.design['n_layers']):
            for i in range(self.n_wires):
                if not (i in self.design['current_qubit'] and self.design['qubit_{}'.format(i)][0][layer] == 0):
                    tqf.rot(qdev, wires=i, params=x[:, i])
            for j in range(self.n_wires):
                if not (j in self.design['current_qubit'] and self.design['qubit_{}'.format(j)][1][layer] == 0):
                    self.rots[j + layer * self.n_wires](qdev, wires=j)
                if self.design['enta' + str(layer) + str(j)][1][0] != self.design['enta' + str(layer) + str(j)][1][1]:
                    self.entas[j + layer * self.n_wires](qdev, wires=self.design['enta' + str(layer) + str(j)][1])
        return self.measure(qdev)


class QNet(nn.Module):
    def __init__(self, arguments, design):
        super(QNet, self).__init__()
        self.args = arguments
        self.design = design
        self.ClassicalLayer_a = nn.RNN(self.args.a_insize, self.args.a_hidsize)
        self.ClassicalLayer_v = nn.RNN(self.args.v_insize, self.args.v_hidsize)
        self.ClassicalLayer_t = nn.RNN(self.args.t_insize, self.args.t_hidsize)
        self.ProjLayer_a = nn.Linear(self.args.a_hidsize, self.args.a_hidsize)
        self.ProjLayer_v = nn.Linear(self.args.v_hidsize, self.args.v_hidsize)
        self.ProjLayer_t = nn.Linear(self.args.t_hidsize, self.args.t_hidsize)
        self.QuantumLayer = TQLayer(self.args, self.design)
        self.Regressor = nn.Linear(self.args.n_qubits, 1)
        for name, param in self.named_parameters():
            if "QuantumLayer" not in name:
                param.requires_grad = False

    def forward(self, x_a, x_v, x_t):
        x_a = torch.permute(x_a, (1, 0, 2))
        x_v = torch.permute(x_v, (1, 0, 2))
        x_t = torch.permute(x_t, (1, 0, 2))
        a_h = self.ClassicalLayer_a(x_a)[0][-1]
        v_h = self.ClassicalLayer_v(x_v)[0][-1]
        t_h = self.ClassicalLayer_t(x_t)[0][-1]
        a_o = torch.relu(self.ProjLayer_a(a_h))
        v_o = torch.sigmoid(self.ProjLayer_v(v_h)) * pi
        t_o = torch.sigmoid(self.ProjLayer_t(t_h)) * pi
        x_p = torch.cat((a_o, v_o, t_o), 1)
        exp_val = self.QuantumLayer(x_p)
        output = torch.tanh(self.Regressor(exp_val).squeeze(dim=1)) * 3
        return output
   
