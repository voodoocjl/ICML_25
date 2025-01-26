import pickle
import os
import csv

def check_file(task):
    if os.path.exists('results') == False:
        os.makedirs('results')
    result = 'results/{}.csv'.format(task)
    result_fine = 'results/{}_fine.csv'.format(task)
    if os.path.isfile(result) == False:
        with open(result, 'w+', newline='') as res:
            writer = csv.writer(res)
            writer.writerow(['sample_id', 'arch_code', 'sample_node', 'ACC', 'p_ACC'])

    if os.path.isfile(result_fine) == False:
        with open(result_fine, 'w+', newline='') as res:
            writer = csv.writer(res)
            writer.writerow(['iteration', 'arch_code', 'ACC'])

def check_file_with_prefix(path, prefix):
    files = os.listdir(path)
    for file in files:
        if os.path.isfile(os.path.join(path, file)):
            if file.startswith(prefix):
                return file
    return False

state_path = 'states'
if os.path.exists(state_path) == False:
    os.makedirs(state_path)
files = os.listdir(state_path)

init_weight_path = 'init_weights'
if os.path.exists(init_weight_path) == False:
    os.makedirs(init_weight_path)
init_weights = os.listdir(init_weight_path)

weight_path = 'weights'
if os.path.exists(weight_path) == False:
    os.makedirs(weight_path)
weights = os.listdir(weight_path)

def empty_arch(n_layers, n_qubits): 
    single = [[i] + [0]* (2*n_layers) for i in range(1,n_qubits+1)]
    enta = [[i] *(n_layers+1) for i in range(1,n_qubits+1)]
    return [single, enta]

check_file('MNIST')

def get_list_dimensions(lst):
    if isinstance(lst, list):
        if len(lst) == 0:
            return 1
        return 1 + get_list_dimensions(lst[0])
    return 0



