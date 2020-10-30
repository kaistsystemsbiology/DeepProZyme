import re
import logging
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import basic python packages
import numpy as np

from Bio import SeqIO

# import torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F

# import scikit learn packages
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score


def argument_parser(version=None):
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--seq_file', required=False, 
                        default='./Dataset/input_dataset.fa', help='Sequence data')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory')
    parser.add_argument('-e', '--epoch', required=False, type=int,
                        default=30, help='Total epoch number')
    parser.add_argument('-b', '--batch_size', required=False, type=int,
                        default=32, help='Batch size')
    parser.add_argument('-r', '--learning_rate', required=False, type=float,
                        default=1e-3, help='Learning rate')
    parser.add_argument('-p', '--patience', required=False, type=int,
                        default=5, help='Patience limit for early stopping')
    parser.add_argument('-g', '--gpu', required=False, 
                        default='cuda:0', help='Specify gpu')
    parser.add_argument('-cpu', '--cpu_num', required=False, type=int,
                        default=1, help='Number of cpus to use')  
    parser.add_argument('-ckpt', '--checkpoint', required=False, 
                        default='checkpoint.pt', help='Checkpoint file')
    parser.add_argument('-l', '--log_dir', required=False, 
                        default='CNN_training.log', help='Log file directory')
    parser.add_argument('-third', '--third_level', required=False, type=boolean_string,
                        default=False, help='Predict upto third EC level')      
    return parser



# plot the accuracy and loss value of each model.
###################
def draw(avg_train_losses, avg_valid_losses, output_dir, file_name='CNN_loss_fig.png'):
    fig = plt.figure(figsize=(9,6))

    avg_train_losses = np.array(avg_train_losses)
    avg_train_losses = avg_train_losses[avg_train_losses.nonzero()]
    avg_valid_losses = np.array(avg_valid_losses)
    avg_valid_losses = avg_valid_losses[avg_valid_losses.nonzero()]
    min_position = avg_valid_losses.argmin()+1

    plt.plot(range(1, len(avg_train_losses)+1), avg_train_losses, label='Training loss')
    plt.plot(range(1, len(avg_valid_losses)+1), avg_valid_losses, label='Validation loss')
    plt.axvline(min_position, linestyle='--', color='r', label='Early stopping checkpoint')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim(left=0)
    plt.ylim(bottom=0, )

    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/{file_name}', dpi=600)
    plt.show()
    return


def save_losses(avg_train_losses, avg_valid_losses, output_dir, file_name='losses.txt'):
    with open(f'{output_dir}/{file_name}', 'w') as fp:
        fp.write('Epoch\tAverage_train_loss\tAverage_valid_loss\n')
        cnt = 0
        for train_loss, valid_loss in zip(avg_train_losses, avg_valid_losses):
            cnt += 1
            fp.write(f'{cnt}\t{train_loss:0.12f}\t{valid_loss:0.12f}\n')
    return


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha==None:
            self.alpha=1
        else:
            self.alpha = torch.Tensor(alpha).view(-1, 1)
        
    def forward(self, pred, label):
        BCE_loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()


class DeepECConfig():
    def __init__(self,
                 model = None,
                 optimizer = None,
                 criterion = None,
                 scheduler = None,
                 n_epochs = 50,
                 device = 'cpu',
                 patience = 5,
                 save_name = './deepec.log',
                 train_source = None,
                 val_source = None, 
                 test_source = None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        self.device = device
        self.patience = patience
        self.save_name = save_name
        self.train_source = train_source
        self.val_source = val_source
        self.test_source = test_source

def _getEC32EC4map(explainECs_short, explainECs):
    result = torch.zeros((len(explainECs), len(explainECs_short)))
    for ec4_ind, ec4 in enumerate(explainECs):
        tmp = torch.zeros(len(explainECs_short))
        for i, ec3 in enumerate(explainECs_short):
            if ec4.startswith(ec3):
                tmp[i] = 1
        result[ec4_ind] = tmp
    return result


def _getCommonECs(ec3_pred, ec4_pred, ec2ec_map, device):
    common_pred = torch.zeros(ec4_pred.shape).to(device)
    for i in range(len(ec4_pred)):
        ec4_activemap = torch.matmul(ec2ec_map, ec3_pred[i])
        common_EC = ec4_activemap * ec4_pred[i]
        common_pred[i] = common_EC
    return common_pred

