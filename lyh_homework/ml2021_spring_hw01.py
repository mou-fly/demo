import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import csv

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

my_seed = 3147
np.random.seed(my_seed)
torch.manual_seed(my_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(my_seed)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class COVID19Dataset(Dataset):
    def __init__(self, path, mode="train", target_only=False):
        self.mode = mode
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)

        feats = []
        if not target_only:
            feats = list(range(93))
        else:
            pass

        if mode == "test":
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            target = data[:, -1]
            data = data[:, feats]
            indices = []
            if mode == "train":
                indices = [i for i in range(data.shape[0]) if i % 10 != 0]
            elif mode == "dev":
                indices = [i for i in range(data.shape[0]) if i % 10 == 0]
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # norm
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        if self.mode == "test":
            return self.data[index]
        elif self.mode in ["train", "dev"]:
            return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)


def prep_dataloader(path, mode, batch_size, target_only=False):
    dataset = COVID19Dataset(path, mode, target_only)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        return self.criterion(pred, target)


def train(tr_set, dv_set, model, config, device):
    n_epochs = config['n_epochs']

    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
            mse_loss.backward()
            optimizer.step()
            loss_record['train'].append(mse_loss.item())

        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                  .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt >= config['early_stop']:
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


def dev(dv_set, model, device):
    model.eval()  # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:  # iterate through the dataloader
        x, y = x.to(device), y.to(device)  # move data to device (cpu/cuda)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)  # compute averaged loss

    return total_loss


def test(tt_set, model, device):
    model.eval()
    pred_ls = []
    for x in tt_set:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            pred_ls.append(pred.detach().cpu())
    pred_ls = torch.cat(pred_ls, dim=0).numpy()
    return pred_ls


device = get_device()  # get the current available device ('cpu' or 'cuda')
os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
target_only = False  # TODO: Using 40 states & 2 tested_positive features

# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 3000,  # maximum number of epochs
    'batch_size': 128,  # mini-batch size for dataloader
    'optimizer': 'SGD',  # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {  # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.0001,  # learning rate of SGD
        'momentum': 0.9  # momentum for SGD
    },
    'early_stop': 200,  # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.pth'  # your model will be saved here
}

tr_set = prep_dataloader("covid.train.csv", 'train', config['batch_size'], target_only=target_only)
dv_set = prep_dataloader("covid.train.csv", 'dev', config['batch_size'], target_only=target_only)
tt_set = prep_dataloader("covid.test.csv", 'test', config['batch_size'], target_only=target_only)

model = NeuralNet(tr_set.dataset.dim).to(device)

model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)
preds = test(tt_set, model, device)
