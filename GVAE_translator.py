import torch
import numpy as np
from torch.nn import functional as F

from Arguments import Arguments
args = Arguments()

def GVAE_translator(data_uploading, rot, enta, arch_code):

    single_list = []
    enta_list = []
    n_qubits = arch_code[0]
    n_layers = arch_code[1]

    for i in range(0, n_layers):
        single_item = []
        for j in range(0, n_qubits):
            d = int(data_uploading[i][j])
            r = int(rot[i][j])
            combination = f'{d}{r}'
            if combination == '00':
                single_item.append(('Identity', j))
            elif combination == '01':
                angle = np.random.uniform(0, 2 * np.pi)
                single_item.append(('RX', j, angle))
            elif combination == '10':
                angle = np.random.uniform(0, 2 * np.pi)
                single_item.append(('RY', j, angle))
            elif combination == '11':
                angle = np.random.uniform(0, 2 * np.pi)
                single_item.append(('RZ', j, angle))
        single_list.append(single_item)

        enta_item = []
        for j, et in enumerate(enta[i]):
            if j == int(et) - 1:
                enta_item.append(('Identity', j))
            else:
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, 2 * np.pi)
                delta = np.random.uniform(0, 2 * np.pi)
                enta_item.append(('C(U3)', j, int(et) - 1, theta, phi, delta))
        enta_list.append(enta_item)

    circuit_ops = []
    for layer in range(0, n_layers):
        circuit_ops.extend(single_list[layer])
        circuit_ops.extend(enta_list[layer])

    return circuit_ops

def generate_circuits(net, arch_code):
    data_uploading = []
    rot = []
    enta = []

    for i in range(0, len(net), 3):
        data_uploading.append(net[i])
        rot.append(net[i + 1])
        enta.append(net[i + 2])

    circuit_ops = GVAE_translator(data_uploading, rot, enta, arch_code)

    return circuit_ops

# encode allowed gates in one-hot encoding
def encode_gate_type():
    gate_dict = {}
    ops = args.allowed_gates.copy()
    ops.insert(0, 'START')
    ops.append('END')
    ops_len = len(ops)
    ops_index = torch.tensor(range(ops_len))
    type_onehot = F.one_hot(ops_index, num_classes=ops_len)
    for i in range(ops_len):
        gate_dict[ops[i]] = type_onehot[i]
    return gate_dict

def get_wires(op):
    if op[0] == 'C(U3)':
        return [op[1], op[2]]
    else:
        return [op[1]]
    
def get_gate_and_adj_matrix(circuit_list, arch_code):
    n_qubits = arch_code[0]
    gate_matrix = []
    op_list = []
    cl = list(circuit_list).copy()
    if cl[0] != 'START':
        cl.insert(0, 'START')
    if cl[-1] != 'END':
        cl.append('END')
    # cg = get_circuit_graph(circuit_list)
    gate_dict = encode_gate_type()
    gate_matrix.append(gate_dict['START'].tolist() + [1] * n_qubits)
    op_list.append('START')
    for op in circuit_list:
        op_list.append(op)
        op_qubits = [0] * n_qubits
        op_wires = get_wires(op)
        for i in op_wires:
            op_qubits[i] = 1
        op_vector = gate_dict[op[0]].tolist() + op_qubits
        gate_matrix.append(op_vector)
    gate_matrix.append(gate_dict['END'].tolist() + [1] * n_qubits)
    op_list.append('END')

    op_len = len(op_list)
    adj_matrix = np.zeros((op_len, op_len), dtype=int)
    for index, op in enumerate(circuit_list):
        ancestors = []
        target_wires = get_wires(op)
        if op[0] == 'C(U3)':
            found_wires = {target_wires[0]: False, target_wires[1]: False}
            max_ancestors = 2
        else:
            found_wires = {target_wires[0]: False}
            max_ancestors = 1

        for i in range(index - 1, -1, -1):
            op_wires = get_wires(circuit_list[i])
            if any(not found_wires[w] for w in op_wires if w in found_wires):
                ancestors.append(circuit_list[i])

                for w in op_wires:
                    if w in found_wires:
                        found_wires[w] = True
                if len(ancestors) >= max_ancestors:
                    break
        if len(ancestors) == 0:
            adj_matrix[0][op_list.index(op)] = 1
        else:
            for j in range(len(ancestors)):
                adj_matrix[op_list.index(ancestors[j])][op_list.index(op)] = 1

        descendants = []
        if op[0] == 'C(U3)':
            found_wires = {target_wires[0]: False, target_wires[1]: False}
            max_descendants = 2
        else:
            found_wires = {target_wires[0]: False}
            max_descendants = 1

        for i in range(index + 1, len(circuit_list)):
            op_wires = get_wires(circuit_list[i])
            if any(not found_wires[w] for w in op_wires if w in found_wires):
                descendants.append(circuit_list[i])
                for w in op_wires:
                    if w in found_wires:
                        found_wires[w] = True
                if len(descendants) >= max_descendants:
                    break
        if len(descendants) < max_descendants:
            adj_matrix[op_list.index(op)][op_len - 1] = 1

    return cl, gate_matrix, adj_matrix