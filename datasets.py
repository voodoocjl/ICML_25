import pickle
import torch
from torch.utils.data import Dataset
from Arguments import Arguments


class CustomDataset(Dataset):
    def __init__(self, audio, visual, text, target):
        self.audio = audio
        self.visual = visual
        self.text = text
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        audio_val = self.audio[index]
        visual_val = self.visual[index]
        text_val = self.text[index]
        target = self.target[index]
        return audio_val, visual_val, text_val, target


def MOSIDataLoaders(args):
    with open('data/mosi', 'rb') as file:
        tensors = pickle.load(file)
    
    AUDIO = 'COVAREP'
    VISUAL = 'FACET_4.2'
    TEXT = 'glove_vectors'
    TARGET = 'Opinion Segment Labels'

    train_data = tensors[0]
    train_audio = torch.from_numpy(train_data[AUDIO]).float()
    train_visual = torch.from_numpy(train_data[VISUAL]).float()
    train_text = torch.from_numpy(train_data[TEXT]).float()
    train_target = torch.from_numpy(train_data[TARGET]).squeeze()

    val_data = tensors[1]
    val_audio = torch.from_numpy(val_data[AUDIO]).float()
    val_visual = torch.from_numpy(val_data[VISUAL]).float()
    val_text = torch.from_numpy(val_data[TEXT]).float()
    val_target = torch.from_numpy(val_data[TARGET]).squeeze()

    test_data = tensors[2]
    test_audio = torch.from_numpy(test_data[AUDIO]).float()
    test_visual = torch.from_numpy(test_data[VISUAL]).float()
    test_text = torch.from_numpy(test_data[TEXT]).float()
    test_target = torch.from_numpy(test_data[TARGET]).squeeze()

    train = CustomDataset(train_audio, train_visual, train_text, train_target)
    val = CustomDataset(val_audio, val_visual, val_text, val_target)
    test = CustomDataset(test_audio, test_visual, test_text, test_target)
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset=val, batch_size=len(val), pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=len(test), shuffle = False, pin_memory=True)
    return train_loader, val_loader, test_loader

import torch
from torchquantum.dataset import MNIST


def MNISTDataLoaders(args, task):
    if task in ('MNIST', 'MNIST-10'):
        FAHION = False
    else:
        FAHION = True
    dataset = MNIST(
        root='data',
        train_valid_split_ratio=args.train_valid_split_ratio,
        center_crop=args.center_crop,
        resize=args.resize,
        resize_mode='bilinear',
        binarize=False,
        binarize_threshold=0.1307,
        digits_of_interest=args.digits_of_interest,
        n_test_samples=None,
        n_valid_samples=None,
        fashion=FAHION,
        n_train_samples=None
        )
    dataflow = dict()
    for split in dataset:
        if split == 'train':
            sampler = torch.utils.data.RandomSampler(dataset[split])
            batch_size = args.batch_size
        else:
            # for valid and test, use SequentialSampler to make the train.py
            # and eval.py results consistent
            sampler = torch.utils.data.SequentialSampler(dataset[split])
            batch_size = len(dataset[split])

        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=True)

    return dataflow['train'], dataflow['valid'], dataflow['test']
