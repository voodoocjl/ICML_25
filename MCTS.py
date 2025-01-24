import pickle
import os
import random
import json
import csv
import numpy as np
import torch
from Node import Node, Color
# from schemes import Scheme
import datetime
from FusionModel import cir_to_matrix 
import time
from sampling import sampling_node
import copy
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
from prepare import *
from draw import plot_2d_array
from Arguments import Arguments
import argparse
import torch.nn as nn

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
        self.explorations = {'phase': 0, 'iteration': 0, 'single':None, 'enta': None, 'rate': [0.001, 0.0005, 0.002], 'rate_decay': [0.006, 0.004, 0.002, 0]}
        self.best = {'acc': 0, 'model':[]}
        self.task = ''
        self.history = [[] for i in range(2)]
        self.qubit_used = []
        self.period = 1

    def init_train(self, numbers=50):
        
        print("\npopulate search space...")
        self.populate_prediction_data()
        print("finished")
        print("\npredict and partition nets in search space...")
        self.predict_nodes()
        self.check_leaf_bags()
        print("finished")
        self.print_tree()

        self.sampling_arch(numbers)
        
        print("\ncollect " + str(len(self.TASK_QUEUE)) + " nets for initializing MCTS")

    def re_init_tree(self, mode=None):
        
        self.TASK_QUEUE = []
        self.sample_nodes = []
        
        self.stages += 1
        phase = self.explorations['phase']        
        # strategy = 'base'
        strategy = self.weight
        round = 3

        file_single = args.file_single
        file_enta = args.file_enta
        if self.task != 'MOSI':
            sorted_changes = [k for k, v in sorted(self.samples_compact.items(), key=lambda x: x[1], reverse=True)]
            epochs = 20
            samples = 20            
        else:
            sorted_changes = [k for k, v in sorted(self.samples_compact.items(), key=lambda x: x[1])]
            epochs = 3
            samples = 30
        sorted_changes = [change for change in sorted_changes if len(eval(change)) == self.stages]

        filename = [file_single, file_enta]

        # pick best 2 and randomly choose one
        random.seed(self.ITERATION)
        
        best_changes = [eval(sorted_changes[i]) for i in range(1)]
        best_change = random.choice(best_changes)
        self.ROOT.base_code = best_change
        qubits = [code[0] for code in self.ROOT.base_code]
        
        print('Current Change: ', best_change)
       
        # if phase == 0:
        if len(best_change[0]) == len(self.explorations['single'][0]):
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
        model_weight = check_file_with_prefix('weights', 'weight_{}_'.format(self.ITERATION))        
        if model_weight:            
            best_model, report = Scheme_eval(design, self.task, model_weight)
            print('Test ACC: ', report['mae'])
        else:
            best_model, report = Scheme(design, self.task, strategy, epochs)
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime('%m-%d-%H')
            torch.save(best_model.state_dict(), 'weights/weight_{}_{}'.format(self.ITERATION, formatted_time))

        self.weight = best_model.state_dict()
        self.samples_true[json.dumps(np.int8(arch).tolist())] = report['mae']
        self.samples_compact = {}
        # arch_next = self.Langevin_update(arch)

        if report['mae'] > self.best['acc']:
            if self.task != 'MOSI':
                self.best['acc'] = report['mae']
                self.best['model'] = arch
        else:
            if self.task == 'MOSI':
                self.best['acc'] = report['mae']
                self.best['model'] = arch

        with open('results/{}_fine.csv'.format(self.task), 'a+', newline='') as res:
            writer = csv.writer(res)
            metrics = report['mae']
            writer.writerow([self.ITERATION, best_change_full, metrics])        
        
        if self.stages == self.period:
            self.stages = 0
            self.history.append(qubits)            
            phase = 1 - phase       # switch phase
            self.ROOT.base_code = None
            # if self.history[phase] != []:
            #     qubits = self.history[phase][-1]
            # else:
            #     qubits = []
            # qubits = []
            self.qubit_used = self.history[-2:]
            self.set_arch(phase, best_change_full)
            self.samples_compact = {}
            self.explorations['iteration'] += 1
            print(Color.BLUE + 'Phase Switch: {}'.format(phase) + Color.RESET)

        arch_last = single + enta
        
        with open('search_space/search_space_mnist_4', 'rb') as file:
            self.search_space = pickle.load(file)
         # remove last configuration
        for i in range(len(arch_last)):
            try:
                self.search_space.remove(arch_last[i])
            except ValueError:
                pass
        self.search_space = [x for x in self.search_space if [x[0]] not in self.qubit_used]       

        random.seed(self.ITERATION)

        self.init_train(samples)
        # for i in range(0, samples):
        #     net = random.choice(self.search_space)
        #     while net[0] in qubits:
        #         net = random.choice(self.search_space)
        #     self.search_space.remove(net)
        #     if self.ROOT.base_code != None:
        #         net_ = self.ROOT.base_code.copy()
        #         net_.append(net)
        #     else:
        #         net_ = [net]
        #     self.TASK_QUEUE.append(net_)
        #     self.sample_nodes.append('random')
        # print("\ncollect " + str(len(self.TASK_QUEUE)) + " nets for re-initializing MCTS {}".format(self.ROOT.base_code))

        self.qubit_used = qubits

    def get_grad(self, x):
        
        model = self.ROOT.classifier.model
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        x = self.nodes[0].classifier.arch_to_z([x]).cuda()
        x.requires_grad_(True)
        x.retain_grad()
        n_heads = 3
        y = torch.tensor([[1, 1, 1]]).cuda()

        # clear grads        
        optimizer.zero_grad()

        # forward to get predicted values
        outputs = model(x)
        loss = loss_fn(outputs[0], y.long())
        loss.backward(retain_graph=True)
        return x, x.grad

    def Langevin_update(self, x, n_steps=20, step_size=0.01):
        
        target_snr = 0.1
        
        # alpha = torch.ones_like(t)

        for i in range(n_steps):

            x, grad = self.get_grad(x)
            noise = torch.randn_like(grad)
            
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()

            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2
            x_mean = x + step_size * grad
            # x = x_mean + torch.sqrt(step_size * 2) * noise
            x = x_mean + noise

        return x, x_mean

    def dump_all_states(self, num_samples):
        node_path = 'states/mcts_agent'
        self.reset_node_data()
        with open(node_path+'_'+str(num_samples), 'wb') as outfile:
            pickle.dump(self, outfile)


    def reset_node_data(self):
        for i in self.nodes:
            i.clear_data()

    def set_arch(self, phase, best_change):
        # if phase == 0:
        if len(best_change[0]) == len(self.explorations['single'][0]):          
            self.explorations['single'] = best_change
            # self.explorations['single'] = None
        else:
            self.explorations['enta'] = best_change
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
                # id = torch.randint(0, len(curt_node.kids), (1,))
                id = np.random.choice(np.argwhere(UCT == np.amin(UCT)).reshape(-1), 1)[0]
            else:
                id = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            curt_node = curt_node.kids[id]
            self.nodes[curt_node.id].counter += 1
        return curt_node
    
    def sampling_arch(self, number=10):
        print('Used Qubits:', self.qubit_used)
        h = 2 ** (self.tree_height-1) - 1
        for i in range(0, number):
            # select
            target_bin   = self.select()           
            qubits = self.qubit_used
            sampled_arch = target_bin.sample_arch(qubits)
            # NOTED: the sampled arch can be None 
            if sampled_arch is not None:                    
                # push the arch into task queue                
                self.TASK_QUEUE.append(sampled_arch)                    
                self.sample_nodes.append(target_bin.id-h)
            else:
                # trail 1: pick a network from the left leaf
                for n in self.nodes:
                    if n.is_leaf == True:
                        sampled_arch = n.sample_arch(qubits)
                        if sampled_arch is not None:
                            print("\nselected node" + str(n.id-7) + " in leaf layer")                            
                            self.TASK_QUEUE.append(sampled_arch)                                
                            self.sample_nodes.append(n.id-h)
                            break
                        else:
                            continue
            if type(sampled_arch[0]) == type([]):
                arch = sampled_arch[-1]
            else:
                arch = sampled_arch
            self.search_space.remove(arch)        

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
            # if self.explorations['phase'] == 0:
            if len(job[0]) == len(self.explorations['single'][0]):
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
            
            # self.DISPATCHED_JOB[job_str] = acc
            # if self.task != 'MOSI':
            #     exploration, gate_numbers = count_gates(arch, self.explorations['rate'])
            # else:
            #     if self.explorations['phase'] == 0:
            #         zero_counts = [job[i].count(0) for i in range(len(job))]
            #         gate_reduced = np.sum(zero_counts)
            #     else:
            #         zero_counts = [(job[i].count(job[i][0])-1) for i in range(len(job))]
            #         gate_reduced = np.sum(zero_counts)
            #     exploration = gate_reduced * self.explorations['rate_decay'][self.stages]
            # p_acc = acc - exploration
            p_acc = acc
            self.samples[arch_str] = p_acc
            self.samples_true[arch_str] = acc
            self.samples_compact[job_str] = p_acc
            sample_node = nodes[i]
            with open('results/{}.csv'.format(self.task), 'a+', newline='') as res:
                writer = csv.writer(res)                                        
                num_id = len(self.samples)
                writer.writerow([num_id, job_str, sample_node, acc, p_acc])
            self.mae_list.append(acc)
                        
            # if job_str in dataset and self.explorations['iteration'] == 0:
            #     report = {'mae': dataset.get(job_str)}
            #     # print(report)
            # self.explorations[job_str]   = ((abs(np.subtract(self.topology[job[0]], job))) % 2.4).round().sum()
            

    def early_search(self, iter):       
        # save current state
        self.ITERATION = iter
        if self.ITERATION > 0:
            self.dump_all_states(self.sampling_num + len(self.samples))
        print("\niteration:", self.ITERATION)
        if self.task == 'MOSI':
            period = 5
            number = 50
        else:
            period = 1
            number = 20

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
        print(self.ROOT.delta_history[-1])
        self.print_tree()
        # # sampling nodes
        # # nodes = [0, 1, 2, 3, 8, 12, 13, 14, 15]
        # nodes = [0, 3, 12, 15]
        # sampling_node(self, nodes, dataset, self.ITERATION)
        
        random.seed(self.ITERATION)
        self.sampling_arch(10)


def Scheme_mp(design, job, task, weight, i, q=None):
    step = len(design)    
    if task != 'MOSI':
        from schemes import Scheme
        epoch = 1
    else:
        from schemes_mosi import Scheme
        epoch = 3
    for j in range(step):
        print('Arch:', job[j][-1])
        _, report = Scheme(design[j], task, weight, epoch, verbs=1)
        q.put([i*step+j, report['mae']])

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

def sampling_qubits(search_space, qubits):
    arch_list = []
    while len(qubits) > 0:    
        arch = random.sample(search_space, 1)
        if arch[0][0] in qubits:
            qubits.remove(arch[0][0])
            arch_list.append(arch[0])
    return arch_list

def create_agent(task, arch_code, pre_file, node=None):
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
        n_single = int(n_qubit/2)
        n_enta = int(n_qubit/2)
        
        with open('search_space/search_space_mnist_4', 'rb') as file:
            search_space = pickle.load(file)

        n_qubits = arch_code[0]
        n_layers = arch_code[1]
        
        if task == 'MNIST-10':
            with open('search_space/search_space_mnist_10', 'rb') as file:
                search_space = pickle.load(file)
            n_qubits = 5
        

        agent = MCTS(search_space, 4, args.fold, arch_code)
        agent.task = task

        if pre_file in init_weights:
            agent.nodes[0].classifier.model.load_state_dict(torch.load(os.path.join(init_weight_path, pre_file)), strict= True)
       
        # strong entanglement
        # n_qubits = arch_code[0]
        n_layers = arch_code[1]
        
        single = [[i]+[1]*2*n_layers for i in range(1,n_qubits+1)]
        enta = [[i]+[i+1]*n_layers for i in range(1,n_qubits)]+[[n_qubits]+[1]*n_layers]
        
        agent.explorations['single'] = single
        agent.explorations['enta'] = enta
        
        design = translator(single, enta, 'full', arch_code, args.fold)
                
        if args.init_weight in init_weights:
            agent.weight = torch.load(os.path.join(init_weight_path, args.init_weight))
        else:
            if task != 'MOSI':
                best_model, report = Scheme(design, task, 'init', 30, None, 'save')
            else:
                best_model, report = Scheme(design, task, 'init', 1)
            agent.weight = best_model.state_dict()
            with open('results/{}_fine.csv'.format(task), 'a+', newline='') as res:
                writer = csv.writer(res)
                writer.writerow([0, [single, enta], report['mae']]) 
        
    return agent


if __name__ == '__main__':
    
     # set random seed
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)

    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('-task', type=str, required=False, default='MNIST', help='Task name, e.g., MNIST or MNIST-10')
    parser.add_argument('-pretrain', type=str, required=False, default='no_pretrain', help='filename of pretraining weights, e.g. pre_weights')

    args_c = parser.parse_args()
    task = args_c.task
    # task = 'MNIST-10'
    task = 'MNIST'

    mp.set_start_method('spawn')

    saved = None
    # saved = 'states/mcts_agent_20'
    
    if task != 'MOSI':
        from schemes import Scheme, Scheme_eval
        from FusionModel import translator
        num_processes = 2
        if task in ['MNIST', 'FASHION']:
            arch_code = [4, 4]  #MNIST-4
            Range = [0.8, 0.82]
        else:
            arch_code = [10, 4]  # qubits, layer 
            Range = [0.5, 0.55]
    else:
        from schemes_mosi import Scheme, Scheme_eval
        from Mosi_Model import translator
        num_processes = 1
        arch_code = [7, 5]
    
    check_file(task)
    
    args = Arguments(task)
    agent = create_agent(task, arch_code, args_c.pretrain, saved)
    ITERATION = agent.ITERATION
     

    for iter in range(ITERATION, 50):
        jobs, designs, archs, nodes = agent.early_search(iter)
        results = {}
        n_jobs = len(jobs)
        step = n_jobs // num_processes
        res = n_jobs % num_processes
        debug = False
        if not debug:
            with Manager() as manager:
                q = manager.Queue()
                with mp.Pool(processes = num_processes) as pool:        
                    pool.starmap(Scheme_mp, [(designs[i*step : (i+1)*step], jobs[i*step : (i+1)*step], task, agent.weight, i, q) for i in range(num_processes)])            
                    pool.starmap(Scheme_mp, [(designs[n_jobs-i-1 : n_jobs-i], jobs[i*step : (i+1)*step], task, agent.weight, n_jobs-i-1, q) for i in range(res)])
                while not q.empty():
                    [i, acc] = q.get()
                    results[i] = acc
        else:
            for i in range(n_jobs):
                results[i] = random.uniform(0.75, 0.8)

        agent.late_search(jobs, results, archs, nodes)

    print('The best model: ', agent.best['acc'])
    agent.dump_all_states(agent.sampling_num + len(agent.samples))
    # plot_2d_array(agent.best['model'])
    
    if task != 'MOSI':
        rank = 20        
        print('<{}:'.format(Range[0]), sum(value < Range[0] for value in list(agent.samples_true.values())))
        print('({}, {}):'.format(Range[0], Range[1]), sum((value in Range)  for value in list(agent.samples_true.values())))
        print('>{}:'.format(Range[1]), sum(value > Range[1]  for value in list(agent.samples_true.values())))
        print('Gate numbers of top {}: {}'.format(rank, analysis_result(agent.samples_true, rank)))
    
        
