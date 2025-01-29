from GVAE_model import generate_single_enta, is_valid_ops_adj
from Classifier import Classifier
from schemes import *
import torch
from FusionModel import *
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import os
from prepare import *
from Node import Color

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

# arch_code = [10,4]
arch_code = [4, 4]
# n_qubits = 10
n_qubits = 4
# fold = 2
fold = 1

dev = qml.device('default.qubit', wires=4)
def plot_op_results(op_results,img_path):
    @qml.qnode(dev)
    def circuit():
        for item in op_results:
            if len(item) == 2:
                gate, qubits = item
                if gate == 'data':  # Use RX to represent data uploading
                    qml.RX(np.pi / 2, wires=qubits[0])
                elif gate == 'U3':
                    qml.U3(np.pi / 2, np.pi / 2, np.pi / 2, wires=qubits[0])
                elif gate == 'Identity':
                    qml.Identity(wires=qubits[0])
                elif gate == 'C(U3)':
                    qml.ctrl(qml.U3, control=qubits[0])(np.pi / 2, np.pi / 2, np.pi / 2, wires=qubits[1])
                else:
                    print(f"Error: Wrong op type - {item}")
            elif len(item) == 3:
                gate1, gate2, qubits = item
                if gate1 == 'data':  # Use RX to represent data uploading
                    qml.RX(np.pi / 2, wires=qubits[0])
                elif gate1 == 'U3':
                    qml.U3(np.pi / 2, np.pi / 2, np.pi / 2, wires=qubits[0])
                elif gate1 == 'Identity':
                    qml.Identity(wires=qubits[0])
                else:
                    print(f"Error: Wrong op type - {item}")

                if gate2 == 'data':  # Use RX to represent data uploading
                    qml.RX(np.pi / 2, wires=qubits[0])
                elif gate2 == 'U3':
                    qml.U3(np.pi / 2, np.pi / 2, np.pi / 2, wires=qubits[0])
                elif gate2 == 'Identity':
                    qml.Identity(wires=qubits[0])
                else:
                    print(f"Error: Wrong op type - {item}")
            else:
                print(f"Error: Item has more than 3 elements - {item}")
                continue
        return qml.state()

    fig, ax = qml.draw_mpl(circuit)()
    plt.show()

    plot_dir = f"circuit_plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    fig.savefig(img_path)

    return 0
classifier = Classifier({}, arch_code, 0, 3, fold)

def generate_circuits(net):
    def GVAE_translator(data_uploading, rot, enta):

        single_list = []
        enta_list = []
        # n_qubits = 5
        n_layers = 4

        for i in range(0, n_layers):
            single_item = []
            for j in range(0, n_qubits):
                if int(data_uploading[i][j])==1:
                    single_item.append(('data', [j]))
                if int(rot[i][j])==1:
                    single_item.append(('U3', [j]))
                # combination = f'{d}{r}'
                # if combination == '00':
                #     single_item.append(('Identity', [j]))
                # elif combination == '01':
                #     single_item.append(('RX', [j]))
                # elif combination == '10':
                #     single_item.append(('RY', [j]))
                # elif combination == '11':
                #     single_item.append(('RZ', [j]))
            single_list.append(single_item)

            enta_item = []
            for j, et in enumerate(enta[i]):
                if j == int(et) - 1:
                    enta_item.append(('Identity', [j]))
                else:
                    enta_item.append(('C(U3)', [j, int(et) - 1]))
            enta_list.append(enta_item)

        circuit_ops = []
        for layer in range(0, n_layers):
            circuit_ops.extend(single_list[layer])
            circuit_ops.extend(enta_list[layer])

        return circuit_ops
    data_uploading = []
    rot = []
    enta = []

    for i in range(0, len(net), 3):
        data_uploading.append(net[i])
        rot.append(net[i + 1])
        enta.append(net[i + 2])

    circuit_ops = GVAE_translator(data_uploading, rot, enta)

    return circuit_ops

def Langevin_update(x, n_steps=20, step_size=0.01):        
    snr = 1
    x = classifier.arch_to_z([x])
    x_norm = torch.norm(x.reshape(x.shape[0], -1), dim=-1).mean()
    x_valid_list = []
    for i in range(1000):            
        noise = torch.randn_like(x)            
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (x_norm / noise_norm) / snr
        x_new = x + step_size * noise
        x_new = classifier.GVAE_model.decoder(x_new)
        if is_valid_ops_adj(x_new[0],x_new[1], int(arch_code[0]/fold)):
            single,enta,op_result = generate_single_enta(x_new[0], int(n_qubits/fold))
            x_valid_list.append([single, enta, op_result])
    return x_valid_list

def projection(arch_next, single, enta):
    # Define the projection logic here
    single = sorted(single, key=lambda x: x[0])
    enta = sorted(enta, key=lambda x: x[0])
    single = np.array(single)
    enta = np.array(enta)
    projected_archs = []
    for arch in arch_next:
        new_single = single * (arch[0]==-1) + arch[0] * (arch[0]!=-1)
        new_enta = enta * (arch[1]==-1) + arch[1] * (arch[1]!=-1)
        projected_archs.append([new_single.tolist(), new_enta.tolist()])
    return projected_archs

def new_translator(single_code, enta_code, trainable, arch_code, fold=1):
    single_code = qubit_fold(single_code, 0, fold)
    enta_code = qubit_fold(enta_code, 1, fold)
    n_qubits = arch_code[0]
    n_layers = arch_code[1]

    updated_design = {}
    updated_design = prune_single(single_code)
    net = gen_arch(enta_code, arch_code)

    if trainable == 'full' or enta_code == None:
        updated_design['change_qubit'] = None
    else:
        if type(enta_code[0]) != type([]): enta_code = [enta_code]
        updated_design['change_qubit'] = enta_code[-1][0]

    # num of layers
    updated_design['n_layers'] = n_layers

    for layer in range(updated_design['n_layers']):
    # categories of single-qubit parametric gates
        for i in range(n_qubits):
            updated_design['rot' + str(layer) + str(i)] = 'U3'
        # categories and positions of entangled gates
        for j in range(n_qubits):
            if net[j + layer * n_qubits] > 0:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [j, net[j + layer * n_qubits]-1])
            else:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [abs(net[j + layer * n_qubits])-1, j])

    updated_design['total_gates'] = updated_design['n_layers'] * n_qubits * 2
    return updated_design

single = [[1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 1, 1, 1, 1, 1, 1, 1, 1], [3, 1, 1, 1, 1, 1, 1, 1, 1], [4, 1, 1, 1, 1, 1, 1, 1, 1]]
enta = [[3, 3, 2, 2, 1], [1, 2, 2, 2, 2], [2, 3, 3, 3, 3], [4, 1, 1, 1, 1]]

# single = [[i] + [random.randint(0, 1) for _ in range(8)] for i in range(1, 5)]
# enta = [[i] + [random.randint(1, 4) for _ in range(8)] for i in range(1, 5)]


arch = cir_to_matrix(single, enta, arch_code, fold)

arch_next = Langevin_update(arch)
imp_arch_list = projection(arch_next, single, enta)

# plot_op_results(generate_circuits(arch),f"circuit_plots/arch.jpg")
# plot_op_results(arch_next[0][-1],f"circuit_plots/arch_next_0.jpg")

# design = translator(single, enta, 'full', arch_code, fold)
# best_model, report = Scheme(design, 'MNIST', 'init', 30)
# weight = best_model.state_dict()

weight = torch.load('init_weights/init_weight_MNIST_4')

arch_next_acc = []
imp_arch_acc = []

for i in range(1, 11):

    new_single,new_enta=imp_arch_list[i]
    new_arch=cir_to_matrix(new_single,new_enta,arch_code,fold)
    # plot_op_results(generate_circuits(new_arch),f"circuit_plots/imp_arch_list_0.jpg")

    print(Color.CYAN + f'arch_next[{i}]:' + Color.RESET)
    design = new_translator(arch_next[i][0].tolist(), arch_next[i][1].tolist(), 'full', arch_code, fold)       
    best_model, report = Scheme(design, 'MNIST', weight, 20)
    acc = report['mae']
    arch_next_acc.append(acc)
    print(f'arch_next[{i}] Test ACC: {acc}')

    print('---------------------------------')

    print(Color.CYAN + f'implicit_arch[{i}]:' + Color.RESET)
    design = translator(new_single, new_enta, 'full', arch_code, fold)
    best_model, report = Scheme(design, 'MNIST', weight, 20)
    acc = report['mae']
    imp_arch_acc.append(acc)
    print(f'imp_arch_list[{i}] Test ACC: {acc}')

    print('---------------------------------')

print('arch_next_acc:', arch_next_acc)
print('imp_arch_acc:', imp_arch_acc)





