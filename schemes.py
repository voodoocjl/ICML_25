import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, f1_score
from datasets import MNISTDataLoaders, MOSIDataLoaders
from FusionModel import QNet
from FusionModel import translator

from Arguments import Arguments
import random


def get_param_num(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total:', total_num, 'trainable:', trainable_num)


def display(metrics):
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'

    print(YELLOW + "\nTest Accuracy: {}".format(metrics) + RESET)

    
def train(model, data_loader, optimizer, criterion, args):
    model.train()
    for feed_dict in data_loader:
        images = feed_dict['image'].to(args.device)
        targets = feed_dict['digit'].to(args.device)    
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets)        
        loss.backward()
        optimizer.step()

def test(model, data_loader, criterion, args):
    model.eval()
    total_loss = 0
    target_all = torch.Tensor()
    output_all = torch.Tensor()
    with torch.no_grad():
        for feed_dict in data_loader:
            images = feed_dict['image'].to(args.device)
            targets = feed_dict['digit'].to(args.device)        
            output = model(images)
            instant_loss = criterion(output, targets).item()
            total_loss += instant_loss
            target_all = torch.cat((target_all, targets), dim=0)
            output_all = torch.cat((output_all, output), dim=0) 
    total_loss /= len(data_loader)
    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    
    return total_loss, accuracy

def evaluate(model, data_loader, args):
    model.eval()
    metrics = {}
    
    with torch.no_grad():
        for feed_dict in data_loader:
            images = feed_dict['image'].to(args.device)
            targets = feed_dict['digit'].to(args.device)        
            output = model(images)

    _, indices = output.topk(1, dim=1)
    masks = indices.eq(targets.view(-1, 1).expand_as(indices))
    size = targets.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    metrics = accuracy    
    return metrics

def Scheme_eval(design, task, weight):
    result = {}  
    args = Arguments(task) 
    path = 'weights/'  
    if task == 'MOSI':
        dataloader = MOSIDataLoaders(args)
    else:
        dataloader = MNISTDataLoaders(args, task)
   
    train_loader, val_loader, test_loader = dataloader
    model = QNet(args, design).to(args.device)
    model.load_state_dict(torch.load(path+weight), strict= False)
    result['mae'] = evaluate(model, test_loader, args)
    return model, result

def Scheme(design, task, weight='base', epochs=None, verbs=None, save=None):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    args = Arguments(task)
    if epochs == None:
        epochs = args.epochs
    
    if task == 'MOSI':
        dataloader = MOSIDataLoaders(args)
    else:
        dataloader = MNISTDataLoaders(args, task)
   
    train_loader, val_loader, test_loader = dataloader
    model = QNet(args, design).to(args.device)
    if weight != 'init':
        if weight != 'base':
            model.load_state_dict(weight, strict= False)            
        else:            
            model.load_state_dict(torch.load('init_weights/base_fashion'))
    criterion = nn.NLLLoss()   
    optimizer = optim.Adam(model.QuantumLayer.parameters(), lr=args.qlr)
    train_loss_list, val_loss_list = [], []
    best_val_loss = 0
    # print('val_loss: ', evaluate(model, val_loader, args))
    start = time.time()
    for epoch in range(epochs):
        try:
            train(model, train_loader, optimizer, criterion, args)
        except Exception as e:
            print('No parameter gate exists')
        train_loss = test(model, train_loader, criterion, args)
        train_loss_list.append(train_loss)        
        val_loss = evaluate(model, val_loader, args)
        val_loss_list.append(val_loss)
        metrics = evaluate(model, test_loader, args)
        val_loss = 0.5 *(val_loss+train_loss[-1])
        if val_loss > best_val_loss:
            best_val_loss = val_loss
            if not verbs: print(epoch, train_loss, val_loss_list[-1], metrics, 'saving model')
            best_model = copy.deepcopy(model)           
        else:
            if not verbs: print(epoch, train_loss, val_loss_list[-1], metrics)        
    end = time.time()    
    # best_model = model
    metrics = evaluate(best_model, test_loader, args)
    display(metrics)
    print("Running time: %s seconds" % (end - start))
    report = {'train_loss_list': train_loss_list, 'val_loss_list': val_loss_list,
              'best_val_loss': best_val_loss, 'mae': metrics}
    
    if save:
        torch.save(best_model.state_dict(), 'weights/init_weight')
    return best_model, report

def pretrain(design, task, weight):    

    args = Arguments(task)
    
    if task == 'MOSI':
        dataloader = MOSIDataLoaders(args)
    else:
        dataloader = MNISTDataLoaders(args, task)
   
    train_loader, val_loader, test_loader = dataloader
    model = QNet(args, design).to(args.device)
    model.load_state_dict(weight, strict= True)
    
    val_loss = evaluate(model, val_loader, args)
    display(val_loss)
    
    return val_loss


if __name__ == '__main__':
    single = None
    enta = None
    
    arch_code = [10, 4]
    n_layers = 4
    n_qubits = 5
    single = [[i]+[1]*2*n_layers for i in range(1,n_qubits+1)]
    enta = [[i]+[i+1]*n_layers for i in range(1,n_qubits)]+[[n_qubits]+[1]*n_layers]

    task = 'MNIST-10'
    args = Arguments(task)
    
    design = translator(single, enta, 'full', arch_code, args.fold)
    
    # best_model, report = Scheme(design, 'MNIST', 'init', 30)
    weight = torch.load('weights/init_weight_half_10')
    report = pretrain(design, task, weight)  

    # torch.save(best_model.state_dict(), 'weights/base_fashion')