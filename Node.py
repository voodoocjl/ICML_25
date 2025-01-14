import json
import math
from math import log2, ceil
import random
import numpy as np
from Classifier import Classifier
import copy


class Node:
    obj_counter   = 0

    def __init__(self, tree_height, fold, parent = None, is_good_kid = False, arch_code = 0, is_root = False):
        # Note: every node is initialized as a leaf,
        # only non-leaf nodes are equipped with classifiers to make decisions
        if not is_root:
            assert type(parent) == type(self)

        self.parent        = parent
        self.is_good_kid   = is_good_kid
        self.ARCH_CODE     = arch_code
        self.is_root       = is_root
        self.x_bar         = float("inf")
        self.n             = 0
        self.uct           = 0
        self.counter       = 1

        self.kids          = []
        self.bag           = {}
        self.validation    = {}
        self.f1            = [0]

        # data for good and bad kids, respectively
        self.good_kid_data = {}
        self.bad_kid_data  = {}
        self.bad_labels    = {}
        self.good_labels   = {}
        self.good_kid_delta = {}
        self.bad_kid_delta = {}
        self.delta_history = []

        self.is_leaf       = True
        self.id            = Node.obj_counter
        self.layer         = ceil(log2(self.id + 2) - 1)
        self.classifier    = Classifier({}, self.ARCH_CODE, self.id, tree_height, fold)

        self.base_code     = None        
        # self.explorations = {'phase': 0, 'iteration': 0, 'single':None, 'enta': None, 'rate': 0.006, 'rate_decay': [0.006, 0.004, 0.002, 0]}

        # insert current node into the kids of parent
        if parent is not None:
            self.parent.kids.append(self)
            if self.parent.is_leaf == True:
                self.parent.is_leaf = False
            assert len(self.parent.kids) <= 2

        Node.obj_counter += 1


    def clear_data(self):
        self.bag.clear()
        self.bad_kid_data.clear()
        self.good_kid_data.clear()
        if not self.is_leaf:
            self.bad_labels.clear()       
            self.good_labels.clear()
            self.good_kid_delta = []
            self.bad_kid_delta = []

    def set_arch(self, phase, code):
        if phase == 0:            
            self.explorations['enta'] = code
            self.explorations['single'] = None
        else:
            self.explorations['single'] = code
            self.explorations['enta'] = None

        self.explorations['phase'] = phase


    def put_in_bag(self, net, maeinv):
        assert type(net) == type([])
        if self.base_code != None and type(net[0]) != type([]):
            net_ = self.base_code.copy()
            net_.append(net)
            net = net_
        net_k = json.dumps(net)
        self.bag[net_k] = (maeinv)


    def get_name(self):
        # state is a list of jsons
        return "node" + str(self.id)


    def pad_str_to_8chars(self, ins):
        if len(ins) <= 14:
            ins += ' ' * (14 - len(ins))
            return ins
        else:
            return ins


    def __str__(self):
        name = self.get_name()
        name = self.pad_str_to_8chars(name)
        name += (self.pad_str_to_8chars('lf:' + str(self.is_leaf)))

        name += (self.pad_str_to_8chars(' val:{0:.4f}   '.format(round(self.get_xbar(), 4))))
        name += (self.pad_str_to_8chars(' uct:{0:.4f}   '.format(round(self.get_uct(Cp=0.2), 4))))
        name += self.pad_str_to_8chars('n:' + str(self.n))
        name += self.pad_str_to_8chars('visit:' + str(self.counter))
        if self.is_leaf == False:
            name += self.pad_str_to_8chars('acc:{0:.4f} '.format(round(self.classifier.training_accuracy[-1], 4)))          

        else:
            name += self.pad_str_to_8chars('acc: ---- ')           

        if self.is_root:
            name += self.pad_str_to_8chars('sp:' + str(len(self.bag)))
        else:
            name += self.pad_str_to_8chars('sp:' + str(len(self.bag[0])))

        parent = '----'
        if self.parent is not None:
            parent = self.parent.get_name()
        parent = self.pad_str_to_8chars(parent)

        name += (' parent:' + parent)        
        if self.is_leaf:
            name = Color.YELLOW + name +Color.RESET
        elif self.layer == 2:
            name = Color.GREEN + name +Color.RESET

        return name


    def get_uct(self, Cp):
        if self.is_root and self.parent == None:
            return float('inf')
        if self.n == 0 and len(self.bag) > 0:
            return float('inf')
        # coeff = 2 ** (5 - ceil(log2(self.id + 2)))
        if len(self.bag) == 0:
            return 0
        return self.x_bar
        # return self.x_bar + 2 * Cp*math.sqrt(2*math.log(self.parent.counter)/self.counter)


    def get_xbar(self):
        return self.x_bar


    def train(self, tree_height):
        if self.parent == None and self.is_root == True:
        # training starts from the bag
            assert len(self.bag) > 0
            self.classifier.update_samples(self.bag, tree_height)
            self.good_kid_data, self.bad_kid_data, self.good_labels, self.bad_labels = self.classifier.split_data(self.layer)
        elif self.is_leaf:
            if self.is_good_kid:
                self.bag = self.parent.good_kid_data
            else:
                self.bag = self.parent.bad_kid_data
        else:
            if self.is_good_kid:
                self.bag = self.parent.good_kid_data
                self.pred_labels = self.parent.good_labels
                self.classifier.update_samples(self.parent.good_kid_data, tree_height, self.layer, self.pred_labels)
                self.good_kid_data, self.bad_kid_data, self.good_labels, self.bad_labels = self.classifier.split_data(self.layer)
            else:
                self.bag = self.parent.bad_kid_data
                self.pred_labels = self.parent.bad_labels
                self.classifier.update_samples(self.parent.bad_kid_data, tree_height, self.layer, self.pred_labels)
                self.good_kid_data, self.bad_kid_data, self.good_labels, self.bad_labels = self.classifier.split_data(self.layer)
        self.x_bar = np.mean(np.array(list(self.bag.values())))
        self.n     = len(self.bag.values())


    def predict(self, arch, method = None):
        if self.parent == None and self.is_root == True and self.is_leaf == False:
            self.good_kid_data, self.bad_kid_data, self.good_kid_delta, self.bad_kid_delta, delta = self.classifier.split_predictions(self.bag, arch)
            self.delta = max(delta[self.layer], 0.1)
            print(delta)
            self.delta_history.append(delta)
        elif self.is_leaf:
            if self.is_good_kid:
                self.bag = self.parent.good_kid_data
            else:
                self.bag = self.parent.bad_kid_data
        else:
            if self.is_good_kid:
                self.bag = self.parent.good_kid_data 
                self.good_kid_data, self.bad_kid_data, self.good_kid_delta, self.bad_kid_delta, delta = self.classifier.split_predictions(self.bag, arch, self.layer, self.parent.good_kid_delta)
                if len(delta) > 0:
                    self.delta = delta[self.layer]
                else:
                    self.delta = 1
            else:
                self.bag = self.parent.bad_kid_data
                self.good_kid_data, self.bad_kid_data, self.good_kid_delta, self.bad_kid_delta, delta = self.classifier.split_predictions(self.bag, arch, self.layer, self.parent.bad_kid_delta)
                if len(delta) > 0:
                    self.delta = delta[self.layer]
                else:
                    self.delta = 1
        


    def predict_validation(self):
        if self.is_leaf == False:
            self.good_kid_data, self.bad_kid_data, _ = self.classifier.split_predictions(self.validation)
        if self.is_good_kid:
            self.bag = self.parent.good_kid_data


    def get_performance(self):
        i = 0
        for k in self.bag.keys():
            if k in self.validation:
                i += 1
        precision = i / (len(self.bag) + 1e-6)
        i = 0
        for k in self.validation.keys():
            if k in self.bag:
                i += 1
        recall = i / len(self.validation)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return f1


    def sample_arch(self, qubits):
        if len(self.bag[0]) == 0:
            return None
        net_str = random.choice(self.bag[0])
        type_ = type(eval(net_str)[-1])
        if type_ == list:
            qubit = eval(net_str)[-1][0]
        else:
            qubit = eval(net_str)[0]
        
        i = 0
        while qubit in qubits:
            net_str = random.choice(list(self.bag.keys()))
            if type_ == list:
                qubit = eval(net_str)[-1][0]
            else:
                qubit = eval(net_str)[0]
            i += 1
            if i > 200:
                return None
            
        self.bag[0].remove(net_str)
       
        return json.loads(net_str)


class Color:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'
