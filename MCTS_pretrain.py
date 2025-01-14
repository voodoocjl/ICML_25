import pickle
import os
import random
import json
import csv
import numpy as np
import torch
import argparse
from Node import Node, Color
# from schemes import Scheme
from FusionModel import cir_to_matrix 
import time
from sampling import sampling_node
import copy
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
from prepare import *
from draw import plot_2d_array
from Arguments import Arguments


class MCTS:
    def __init__(self, search_space, tree_height, fold, arch_code):
        assert type(search_space)    == type([])
        assert len(search_space)     >= 1
        assert type(search_space[0]) == type([])

        self.search_space   = search_space 
        self.ARCH_CODE      = arch_code
        self.ROOT           = None
        self.Cp             = 0.2
        self.nodes          = []
        self.samples        = {}
        self.samples_true   = {}
        self.samples_compact = {}
        self.TASK_QUEUE     = []
        self.DISPATCHED_JOB = {}
        self.mae_list    = []
        self.JOB_COUNTER    = 0
        self.TOTAL_SEND     = 0
        self.TOTAL_RECV     = 0
        self.ITERATION      = 0
        self.MAX_MAEINV     = 0
        self.MAX_SAMPNUM    = 0
        self.sample_nodes   = []
        self.stages         = 0
        self.sampling_num   = 0   
        self.acc_mean       = 0

        self.tree_height    = tree_height

        # initialize a full tree
        total_nodes = 2**tree_height - 1
        for i in range(1, total_nodes + 1):
            is_good_kid = False
            if (i-1) > 0 and (i-1) % 2 == 0:
                is_good_kid = False
            if (i-1) > 0 and (i-1) % 2 == 1:
                is_good_kid = True

            parent_id = i // 2 - 1
            if parent_id == -1:
                self.nodes.append(Node(tree_height, fold, None, is_good_kid, self.ARCH_CODE, True))
            else:
                self.nodes.append(Node(tree_height, fold, self.nodes[parent_id], is_good_kid, self.ARCH_CODE, False))

        self.ROOT = self.nodes[0]
        self.CURT = self.ROOT
        self.weight = 'init'        
        self.explorations = {'phase': 0, 'iteration': 0, 'single':None, 'enta': None, 'rate': [0.002, 0.001, 0.003], 'rate_decay': [0.006, 0.004, 0.002, 0]}
        self.best = {'acc': 0, 'model':[]}
        self.task = ''
        self.history = [[], []]
        self.qubit_used = []
            
    
    def init_train(self, numbers=50):
        # random.seed(40)
        for i in range(0, numbers):
            net = random.choice(self.search_space)
            self.search_space.remove(net)
            self.TASK_QUEUE.append(net)
            self.sample_nodes.append('random')        
        
        print("\ncollect " + str(len(self.TASK_QUEUE)) + " nets for initializing MCTS")

    def re_init_tree(self, mode=None):
        
        self.TASK_QUEUE = []
        self.sample_nodes = []
        
        self.stages += 1
        phase = self.explorations['phase']        
        # strategy = 'base'
        strategy = self.weight
        round = 3


        sorted_changes = [k for k, v in sorted(self.samples_compact.items(), key=lambda x: x[1], reverse=True)]
        epochs = 3
        samples = 50
        file_single = args.file_single
        file_enta = args.file_enta
    
        sorted_changes = [change for change in sorted_changes if len(eval(change)) == self.stages]

        # pick best 2 and randomly choose one
        random.seed(self.ITERATION)
        
        best_changes = [eval(sorted_changes[i]) for i in range(2)]
        best_change = random.choice(best_changes)
        self.ROOT.base_code = best_change
        qubits = [code[0] for code in self.ROOT.base_code]
       
        print('Current Change: ', best_change)
        

        if phase == 0:
            best_change_full = self.insert_job(self.explorations['single'], best_change)
            single = best_change_full
            enta = self.explorations['enta']
        else:
            best_change_full = self.insert_job(self.explorations['enta'], best_change)
            single = self.explorations['single']
            enta = best_change_full
        arch = cir_to_matrix(single, enta, self.ARCH_CODE, args.fold)
        # plot_2d_array(arch)
        design = translator(single, enta, 'full', self.ARCH_CODE, args.fold)        
        best_model, report = Scheme(design, self.task, strategy, epochs)
        self.weight = best_model.state_dict()
        self.samples_true[json.dumps(np.int8(arch).tolist())] = report['mae']
        self.samples_compact = {}
        if report['mae'] > self.best['acc']:
            self.best['acc'] = report['mae']
            self.best['model'] = arch

        with open('results_{}_pretrain.csv'.format(self.task), 'a+', newline='') as res:
            writer = csv.writer(res)
            metrics = report['mae']
            writer.writerow([self.ITERATION, best_change_full, metrics])
        
        if self.stages == 3:
            self.stages = 0
            self.history[phase].append(qubits)
            phase = 1 - phase       # switch phase
            self.ROOT.base_code = None         
            qubits = []
            self.qubit_used = qubits
            self.set_arch(phase, best_change_full)
            self.samples_compact = {}
            self.explorations['iteration'] += 1
            print(Color.BLUE + 'Phase Switch: {}'.format(phase) + Color.RESET)

        
        if phase == 0:
            filename = file_single
        else:
            filename = file_enta
        with open(filename, 'rb') as file:
            search_space = pickle.load(file)
        self.search_space = [x for x in search_space if x[0] not in qubits] 

        random.seed(self.ITERATION)

        for i in range(0, samples):
            net = random.choice(self.search_space)
            while net[0] in qubits:
                net = random.choice(self.search_space)
            self.search_space.remove(net)
            if self.ROOT.base_code != None:
                net_ = self.ROOT.base_code.copy()
                net_.append(net)
            else:
                net_ = [net]
            self.TASK_QUEUE.append(net_)
            self.sample_nodes.append('random')
        print("\ncollect " + str(len(self.TASK_QUEUE)) + " nets for re-initializing MCTS {}".format(self.ROOT.base_code))

        self.qubit_used = qubits

    def dump_all_states(self, num_samples):
        node_path = 'states/mcts_agent'
        self.reset_node_data()
        with open(node_path+'_'+str(num_samples), 'wb') as outfile:
            pickle.dump(self, outfile)


    def reset_node_data(self):
        for i in self.nodes:
            i.clear_data()

    def set_arch(self, phase, best_change):
        if phase == 0:            
            self.explorations['enta'] = best_change
            # self.explorations['single'] = None
        else:
            self.explorations['single'] = best_change
            # self.explorations['enta'] = None

        self.explorations['phase'] = phase        


    def populate_training_data(self):
        self.reset_node_data()
        for k, v in self.samples.items():
            self.ROOT.put_in_bag(json.loads(k), v)


    def populate_prediction_data(self):
        # self.reset_node_data()
        for k in self.search_space:
            self.ROOT.put_in_bag(k, 0.0)


    def train_nodes(self):
        for i in self.nodes:
            i.train(self.tree_height)


    def predict_nodes(self, method = None, dataset =None):
        for i in self.nodes:            
            if dataset:
                i.predict_validation()
            else:
                i.predict(self.explorations, method)


    def check_leaf_bags(self):
        counter = 0
        for i in self.nodes:
            if i.is_leaf is True:
                counter += len(i.bag[0])
        assert counter == len(self.search_space)


    def reset_to_root(self):
        self.CURT = self.ROOT


    def print_tree(self):
        print('\n'+'-'*100)
        for i in self.nodes:
            print(i)
        print('-'*100)


    def select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        self.ROOT.counter += 1
        while curt_node.is_leaf == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append(i.get_uct(self.Cp))
            if torch.rand(1) < curt_node.delta:
                id = torch.randint(0, len(curt_node.kids), (1,))
            else:
                id = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            curt_node = curt_node.kids[id]
            self.nodes[curt_node.id].counter += 1
        return curt_node

    def insert_job(self, change_code, job_input):
        job = copy.deepcopy(job_input)
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


    def evaluate_jobs_before(self):
        jobs = []
        designs =[]        
        archs = []
        nodes = []
        while len(self.TASK_QUEUE) > 0:            
           
            job = self.TASK_QUEUE.pop()
            sample_node = self.sample_nodes.pop()               
            if type(job[0]) != type([]):
                job = [job]                             
            if self.explorations['phase'] == 0:
                single = self.insert_job(self.explorations['single'], job)
                enta = self.explorations['enta']
            else:
                single = self.explorations['single']
                enta = self.insert_job(self.explorations['enta'], job)
            design = translator(single, enta, 'full', self.ARCH_CODE, args.fold)
            arch = cir_to_matrix(single, enta, self.ARCH_CODE, args.fold)
            
            jobs.append(job)
            designs.append(design)
            archs.append(arch)
            nodes.append(sample_node)

        return jobs, designs, archs, nodes

    def evaluate_jobs_after(self, results, jobs, archs, nodes):
        for i in range(len(jobs)):
            acc = results[i]
            job = jobs[i]  
            job_str = json.dumps(job)
            arch = archs[i]
            arch_str = json.dumps(np.int8(arch).tolist())
            
           
            if self.task != 'MOSI':
                exploration, gate_numbers = count_gates(arch, self.explorations['rate'])
            else:
                if self.explorations['phase'] == 0:
                    zero_counts = [job[i].count(0) for i in range(len(job))]
                    gate_reduced = np.sum(zero_counts)
                else:
                    zero_counts = [(job[i].count(job[i][0])-1) for i in range(len(job))]
                    gate_reduced = np.sum(zero_counts)
                exploration = gate_reduced * self.explorations['rate_decay'][self.stages]
            p_acc = acc - exploration
            # p_acc = acc
            self.samples[arch_str] = p_acc
            self.samples_true[arch_str] = acc
            self.samples_compact[job_str] = p_acc
            self.mae_list.append(acc)

    def early_search(self, iter):       
        # save current state
        self.ITERATION = iter  
        if self.ITERATION > 0:
            self.dump_all_states(self.sampling_num + len(self.samples))
        print("\niteration:", self.ITERATION)
        period = 1
        number = 50

        if (self.ITERATION % period == 0): 
            if self.ITERATION == 0:
                self.init_train(number)                    
            else:
                self.re_init_tree()
        
        # evaluate jobs:
        print("\nevaluate jobs...")
        self.mae_list = []
        jobs, designs, archs, nodes = self.evaluate_jobs_before()

        return jobs, designs, archs, nodes
    
    def late_search(self, jobs, results, archs, nodes):
                
        self.evaluate_jobs_after(results, jobs, archs, nodes)    
        print("\nfinished all jobs in task queue")            

        # assemble the training data:
        print("\npopulate training data...")
        self.populate_training_data()
        print("finished")

        # training the tree
        print("\ntrain classifiers in nodes...")
        if torch.cuda.is_available():
            print("using cuda device")
        else:
            print("using cpu device")
        
        start = time.time()
        self.train_nodes()
        print("finished")
        end = time.time()
        print("Running time: %s seconds" % (end - start))
       
        # clear the data in nodes
        self.reset_node_data()                      

        print("\npopulate prediction data...")
        self.populate_prediction_data()
        print("finished")        
        print("\npredict and partition nets in search space...")
        self.predict_nodes()        
        self.check_leaf_bags()
        print("finished")
        self.print_tree()

def Scheme_mp(design, job, task, weight, i, q=None):
    step = len(design)    
    if task != 'MOSI':
        from schemes import Scheme, pretrain
        epoch = 1
    else:
        from schemes_mosi import Scheme
        epoch = 3
    for j in range(step):
        print('Arch:', job[j][-1])
        report = pretrain(design[j], task, weight)
        q.put([i*step+j, report])

def count_gates(arch, coeff=None):
    # x = [item for i in [2,3,4,1] for item in [1,1,i]]
    qubits = args.n_qubits
    layers = args.n_layers
    x = [[0, 0, i]*4 for i in range(1,qubits+1)]    
    x = np.transpose(x, (1,0))
    x = np.sign(abs(x-arch))
    if coeff != None:
        coeff = np.reshape(coeff * 4, (-1,1))
        y = (x * coeff).sum()
    else:
        y = 0
    stat = {}
    stat['uploading'] = x[[3*i for i in range(layers)]].sum()
    stat['single'] = x[[3*i+1 for i in range(layers)]].sum()
    stat['enta'] = x[[3*i+2 for i in range(layers)]].sum()
    return y, stat

def analysis_result(samples, ranks):
    gate_stat = []    
    sorted_changes = [k for k, v in sorted(samples.items(), key=lambda x: x[1], reverse=True)]
    for i in range(ranks):
        _, gates = count_gates(eval(sorted_changes[i]))
        gate_stat.append(list(gates.values()))
    mean = np.mean(gate_stat, axis=0)
    return mean

def sampling_arch(search_space, qubits):
    arch_list = []
    while len(qubits) > 0:    
        arch = random.sample(search_space, 1)
        if arch[0][0] in qubits:
            qubits.remove(arch[0][0])
            arch_list.append(arch[0])
    return arch_list

def create_agent(task, arch_code, node=None):
    if files:
        files.sort(key=lambda x: os.path.getmtime(os.path.join(state_path, x)))
        node_path = os.path.join(state_path, files[-1])
        if node: node_path = node        
        with open(node_path, 'rb') as json_data:
            agent = pickle.load(json_data)
        print("\nresume searching,", agent.ITERATION, "iterations completed before")
        print("=====>loads:", len(agent.samples), "samples")        
        print("=====>loads:", len(agent.TASK_QUEUE), 'tasks')
    else:
        
        path = [args.file_single, args.file_enta]
        n_qubit = args.n_qubits
        n_layer = args.n_layers
        n_single = 2
        n_enta = 2
        
        with open(path[0], 'rb') as file:
            search_space_single = pickle.load(file)        

        with open(path[1], 'rb') as file:
            search_space_enta = pickle.load(file)

        agent = MCTS(search_space_single, 2, args.fold, arch_code)
        agent.task = task       
                  

        # empty = empty_arch(n_layer, n_qubit)   #layers, qubits
        # qubits = random.sample([i for i in range(1, n_qubit+1)],n_single)
        # single = sampling_arch(search_space_single, qubits)

        # qubits = random.sample([i for i in range(1, n_qubit+1)],n_enta)
        # enta = sampling_arch(search_space_enta, qubits)

        # single = agent.insert_job(empty[0], single)
        # enta = agent.insert_job(empty[1], enta)

        # strong entanglement
        n_qubits = int(arch_code[0]/2)
        n_layers = arch_code[1]
        single = [[i]+[1]*2*n_layers for i in range(1,n_qubits+1)]
        enta = [[i]+[i+1]*n_layers for i in range(1,n_qubits)]+[[n_qubits]+[1]*n_layers]
        agent.explorations['single'] = single
        agent.explorations['enta'] = enta
        
        design = translator(single, enta, 'full', arch_code, args.fold)
        weight_file = 'init_weight_half_10'
        if (init_weights and weight_file in init_weights):            
            agent.weight = torch.load(os.path.join(init_weight_path, weight_file))
        else: 
            best_model, report = Scheme(design, task, 'init', 30, None, 'save')           
            agent.weight = best_model.state_dict()
            with open('results_{}_fine.csv'.format(task), 'a+', newline='') as res:
                writer = csv.writer(res)
                writer.writerow([0, [single, enta], report['mae']])            

    return agent


if __name__ == '__main__':
    
     # set random seed
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)

    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('-task', type=str, required=True, help='Task name, e.g., MNIST or MNIST-10')
    parser.add_argument('-output', type=str, required=True, help='filename of output pretraining weights, e.g. pre_weights')

    args_c = parser.parse_args()
    task = args_c.task

    mp.set_start_method('spawn')

    saved = None
    task = args_c.task

    if task in ['MNIST', 'FASHION']:
        arch_code = [4, 4]  #MNIST-4
    else:
        arch_code = [10, 4]  # qubits, layer  

    from schemes import Scheme
    from FusionModel import translator

    num_processes = 1 
    
    check_file(task)  
    
    args = Arguments(task)

    agent = create_agent(task, arch_code, saved)
    ITERATION = agent.ITERATION
    
    for iter in range(ITERATION, 30):
        jobs, designs, archs, nodes = agent.early_search(iter)
        results = {}
        n_jobs = len(jobs)
        step = n_jobs // num_processes
        res = n_jobs % num_processes 
        t = time.time()       
        with Manager() as manager:
            q = manager.Queue()
            with mp.Pool(processes = num_processes) as pool:        
                pool.starmap(Scheme_mp, [(designs[i*step : (i+1)*step], jobs[i*step : (i+1)*step], task, agent.weight, i, q) for i in range(num_processes)])            
                pool.starmap(Scheme_mp, [(designs[n_jobs-i-1 : n_jobs-i], jobs[i*step : (i+1)*step], task, agent.weight, n_jobs-i-1, q) for i in range(res)])
            while not q.empty():
                [i, acc] = q.get()
                results[i] = acc
        print('time: ', time.time() - t)
        agent.late_search(jobs, results, archs, nodes)

    torch.save(agent.nodes[0].classifier.model.state_dict(), os.path.join(init_weight_path, args_c.output))
    print('The pretraining weights have been saved!')
    
    
    
        
