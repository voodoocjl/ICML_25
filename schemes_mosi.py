import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, f1_score
from datasets import MOSIDataLoaders
from Mosi_Model import QNet, translator
from Arguments import Arguments
import random


def get_param_num(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total:', total_num, 'trainable:', trainable_num)


def display(metrics):
    print("\nTest mae: {}".format(metrics['mae']))
    # print("Test correlation: {}".format(metrics['corr']))
    # print("Test multi-class accuracy: {}".format(metrics['multi_acc']))
    # print("Test binary accuracy: {}".format(metrics['bi_acc']))
    # print("Test f1 score: {}".format(metrics['f1']))


def train(model, data_loader, optimizer, criterion, args):
    model.train()
    for data_a, data_v, data_t, target in data_loader:
        data_a, data_v, data_t = data_a.to(args.device), data_v.to(args.device), data_t.to(args.device)
        target = target.to(args.device)
        optimizer.zero_grad()
        output = model(data_a, data_v, data_t)
        loss = criterion(output, target)
        # loss = output[1]
        loss.backward()
        optimizer.step()


def test(model, data_loader, criterion, args):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data_a, data_v, data_t, target in data_loader:
            data_a, data_v, data_t = data_a.to(args.device), data_v.to(args.device), data_t.to(args.device)
            target = target.to(args.device)
            output = model(data_a, data_v, data_t)
            instant_loss = criterion(output, target).item()
            total_loss += instant_loss
    total_loss /= len(data_loader.dataset)
    return total_loss


def evaluate(model, data_loader, args):
    model.eval()
    metrics = {}
    with torch.no_grad():
        data_a, data_v, data_t, target = next(iter(data_loader))
        data_a, data_v, data_t = data_a.to(args.device), data_v.to(args.device), data_t.to(args.device)
        output = model(data_a, data_v, data_t)
    output = output.cpu().numpy()
    target = target.numpy()
    metrics['mae'] = np.mean(np.absolute(output - target)).item()
    metrics['corr'] = np.corrcoef(output, target)[0][1].item()
    metrics['multi_acc'] = round(sum(np.round(output) == np.round(target)) / float(len(target)), 5).item()
    true_label = (target >= 0)
    pred_label = (output >= 0)
    metrics['bi_acc'] = accuracy_score(true_label, pred_label)
    metrics['f1'] = f1_score(true_label, pred_label, average='weighted')
    return metrics


def Scheme(design, task, weight='base', epochs=None, verbs=None):
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)

    args = Arguments(task)        # task='MOSI'
    if epochs == None:
        epochs = args.epochs
    # if torch.cuda.is_available() and args.device == 'cuda':
    #     print("using cuda device")
    # else:
    #     print("using cpu device")
    train_loader, val_loader, test_loader = MOSIDataLoaders(args)
    model = QNet(args, design).to(args.device)    
    if weight != 'init':
        model.load_state_dict(weight, strict=False)
    else:
        model.load_state_dict(torch.load('init_weights/base_weight_MOSI'), strict=False)
    criterion = nn.L1Loss(reduction='sum')    
    optimizer = optim.Adam(model.QuantumLayer.parameters(), lr=args.qlr)
    train_loss_list, val_loss_list = [], []
    best_val_loss = 10000

    start = time.time()
    for epoch in range(epochs):
        try:
            train(model, train_loader, optimizer, criterion, args)
        except Exception as e:
            print('No parameter gate exists')
        train_loss = test(model, train_loader, criterion, args)
        train_loss_list.append(train_loss)
        val_loss = test(model, val_loader, criterion, args)
        val_loss_list.append(val_loss)
        metrics = evaluate(model, test_loader, args)
        val_loss = 0.5 *(val_loss+train_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if not verbs: print(epoch, train_loss, val_loss_list[-1], metrics['mae'], 'saving model')
            best_model = copy.deepcopy(model)           
        else:
            if not verbs: print(epoch, train_loss, val_loss_list[-1], metrics['mae'])
    end = time.time()
    # print("Running time: %s seconds" % (end - start))
    best_model = model
    metrics = evaluate(best_model, test_loader, args)
    display(metrics)
    report = {'train_loss_list': train_loss_list, 'val_loss_list': val_loss_list,
              'best_val_loss': best_val_loss, 'mae': metrics['mae']}

    ## store classical weights
    # del best_model.QuantumLayer
    # torch.save(best_model.state_dict(), 'base_weight_tq_2')
    return best_model, report

def Scheme_eval(design, task, weight):
    result = {}  
    args = Arguments(task) 
    path = 'weights/'  
    dataloader = MOSIDataLoaders(args)

    train_loader, val_loader, test_loader = dataloader
    model = QNet(args, design).to(args.device)
    model.load_state_dict(torch.load(path+weight), strict= False)
    result = evaluate(model, test_loader, args)
    return model, result


if __name__ == '__main__':
    # change_code = None
    # change_code = [6, 1, 1, 2, 1, 2]
    change_code = [5, 3, 6, 4, 5, 4]
    design = translator(change_code)
    # design = translator(change_code)
    best_model, report = Scheme(design)
