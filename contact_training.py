import os
import re
import math
import copy
import random
import logging
from typing import Optional, Any
# import basic python packages
import numpy as np
from tqdm.auto import tqdm

# import torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from deepec.process_data import read_EC_Fasta
from deepec.data_loader import DeepContactECDataset
from deepec.utils import argument_parser, FocalLoss, DeepECConfig
from deepec.model import ProtBertStrEC
from transformers import BertConfig

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
    
    
class EarlyStopping:
    def __init__(self, save_name='checkpoint.pt', patience=5, verbose=False, delta=0, explainProts=[]):
        self.save_name = save_name
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.explainProts = explainProts

    def __call__(self, model, optimizer, epoch, val_loss, loss_1, loss_2):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, loss_1, loss_2, model, optimizer, epoch)
        elif score < self.best_score - self.delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, loss_1, loss_2, model, optimizer, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, loss_1, loss_2, model, optimizer, epoch):
        if self.verbose:
            logging.info(f'Epoch {epoch}: Total loss decreased ({self.val_loss_min:.10f} --> {val_loss:.10f}). Loss 1: ({loss_1:.10f}). Loss 2: ({loss_2:.10f}) Saving model ...')
        
        ckpt = {'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'best_acc':self.best_score,
                'epoch':epoch,
                'explainECs':self.explainProts}
        torch.save(ckpt, self.save_name)
        torch.save(model, self.save_name.replace('checkpoint.pt', 'model.pth'))
        self.val_loss_min = val_loss

        
def train_bert_model(config):
    device = config.device 
    train_loader = config.train_source
    model = config.model
    optimizer = config.optimizer
    criterion_1 = config.criterion_1
    criterion_2 = config.criterion_2
    alpha = config.alpha
    train_losses = 0
    losses_1 = 0
    losses_2 = 0
    n = 0

    model.train()
    for data in tqdm(train_loader):
        inputs = {key:val.to(device) for key, val in data.items() if key!='maps'}
        optimizer.zero_grad()
        output, maps = model(**inputs)
        loss_1 = criterion_1(output, inputs['labels'])
        # loss_2 = criterion_2(maps, data['maps'].to(device))
        # loss_2 = loss_2/(inputs['attention_mask'].sum(dim=1)-2).pow(2).sum()
        # loss_2 *= alpha
        # loss = loss_1 + loss_2
        loss = loss_1
        loss.backward()
        optimizer.step()

        train_losses += loss.item()
        losses_1 += loss_1.item()
        # losses_2 += loss_2.item()
        n += inputs['labels'].size(0)
    return train_losses/n, losses_1/n, losses_2/n


def train_bert(config):
    early_stopping = EarlyStopping(save_name=config.save_name, 
                                   patience=config.patience, 
                                   verbose=True,
                                   explainProts=config.explainProts
                                   )

    device = config.device
    n_epochs = config.n_epochs

    avg_train_losses = torch.zeros(n_epochs).to(device)
    logging.info('Training start')
    for epoch in tqdm(range(n_epochs)):
        train_loss, loss_1, loss_2 = train_bert_model(config)

        if config.scheduler != None:
            config.scheduler.step()

        avg_train_losses[epoch] = train_loss
        early_stopping(config.model, config.optimizer, epoch, train_loss, loss_1, loss_2)
        if early_stopping.early_stop:
            logging.info('Early stopping')
            break
            
    logging.info('Training end')
    return avg_train_losses.tolist()


    

if __name__ == '__main__':
    parser = argument_parser()
    options = parser.parse_args()

    output_dir = options.output_dir
    log_dir = options.log_dir

    device = options.gpu
    num_epochs = options.epoch
    batch_size = options.batch_size
    learning_rate = options.learning_rate
    patience = options.patience

    checkpt_file = options.checkpoint
    input_data_file = options.seq_file
    input_contact_map = './Dataset/processed_distance_maps.npz'
    
    num_cpu = options.cpu_num

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(f'{output_dir}/{log_dir}')
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    seed_num = 123 # random seed for reproducibility
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    np.random.seed(seed_num)

    torch.set_num_threads(num_cpu)

    gamma = 0
    alpha = 0.0001

    logging.info(f'\nInitial Setting\
                  \nEpoch: {num_epochs}\
                  \tGamma: {gamma}\
                  \tAlpha: {alpha}\
                  \tBatch size: {batch_size}\
                  \tLearning rate: {learning_rate}\
                  \tGPU: {device}')
    logging.info(f'Input file directory: {input_data_file}')


    input_seqs, input_ecs, input_ids = read_EC_Fasta(input_data_file)
    distance_maps = np.load(input_contact_map, )
    input_maps = [distance_maps[seq_id] for seq_id in input_ids]
    

    explainECs = []
    for ecs in input_ecs:
        explainECs += ecs
    explainECs = list(set(explainECs))
    explainECs.sort()


    trainDataset = DeepContactECDataset(data_X=input_seqs, data_ec=input_ecs, data_map=input_maps, explainECs=explainECs)
    trainDataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

    config = BertConfig.from_pretrained("Rostlab/prot_bert")
    config.update(
        {'hidden_size':128, 
        'intermediate_size':256,
        'max_position_embeddings': 1000,
        'num_attention_heads':8, 
        'num_hidden_layers':2}
    )
    
    const = {
        'explainProts': explainECs,
    }
    
    model = ProtBertStrEC(config, const)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(device)
    logging.info(f'Model Architecture: \n{model}')
    num_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of trainable parameters: {num_train_params}')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, )
    criterion_1 = FocalLoss(gamma=gamma)
    criterion_2 = nn.BCELoss(reduction='sum')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    logging.info(f'Learning rate scheduling: step size: 1\tgamma: 0.95')

    config = DeepECConfig()
    config.model = model 
    config.optimizer = optimizer
    config.criterion_1 = criterion_1
    config.criterion_2 = criterion_2
    config.scheduler = scheduler
    config.n_epochs = num_epochs
    config.alpha = alpha
    config.device = device
    config.save_name = f'{output_dir}/{checkpt_file}'
    config.patience = patience
    config.train_source = trainDataloader
    config.explainProts = explainECs


    train_losses = train_bert(config)
    
    with open(output_dir+'/train_losses.txt', 'w') as fp:
        fp.write('Epoch\tLoss\n')
        for i, loss in enumerate(train_losses):
            fp.write(f'{i}\t{loss}\n')