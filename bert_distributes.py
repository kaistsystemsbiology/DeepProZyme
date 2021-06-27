import os
import re
import math
import copy
import random
import logging
import argparse
import warnings
from typing import Optional, Any
# import basic python packages
import numpy as np
from torch.nn.parallel.distributed import DistributedDataParallel
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

# import torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from deepec.process_data import read_EC_Fasta
from deepec.data_loader import DeepECDataset
from deepec.utils import argument_parser, draw, save_losses, FocalLoss, DeepECConfig
from deepec.train import train_bert, evaluate_bert
from deepec.model import ProtBertEC
from transformers import BertConfig

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    

def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'


parser = argparse.ArgumentParser(description='pytorch distributed')
parser.add_argument('-i', '--seq_file', required=False, 
                        default='./Dataset/uniprot_dataset.fa', help='Sequence data')
parser.add_argument('-o', '--output_dir', required=True, help='Output directory')
parser.add_argument('-e', '--epoch', required=False, type=int,
                    default=30, help='Total epoch number')
parser.add_argument('-b', '--batch_size', required=False, type=int,
                    default=32, help='Batch size')
parser.add_argument('-r', '--learning_rate', required=False, type=float,
                    default=1e-3, help='Learning rate')
parser.add_argument('-gamma', '--gamma', required=False, type=float,
                    default=1.0, help='Focal loss gamma')
parser.add_argument('-p', '--patience', required=False, type=int,
                    default=5, help='Patience limit for early stopping')
parser.add_argument('-g', '--gpu', required=False, type=int,
                    default=None, help='Specify gpu')
parser.add_argument('-cpu', '--cpu_num', required=False, type=int,
                    default=4, help='Number of cpus to use')  
parser.add_argument('-ckpt', '--checkpoint', required=False, 
                    default='checkpoint.pt', help='Checkpoint file')
parser.add_argument('-l', '--log_dir', required=False, 
                    default='bert_dist_training.log', help='Log file directory')
parser.add_argument('--seed', default=123, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-j', '--workers', required=False, default=4, type=int, metavar='N',
                    help='number of data loading workers (defualt: 4')
parser.add_argument('--start-epoch', required=False, default=0, type=int, metavar='N',
                    help='manual epoch number')
parser.add_argument('--world-size', required=False, default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, 
                    help='distributed backend')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch')
parser.add_argument("--local_rank", type=int)


best_acc1 = 0


def main():

    '''
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    '''
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(f'{args.output_dir}/{args.log_dir}')
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


    logging.info(f'\nInitial Setting\
                  \nEpoch: {args.epoch}\
                  \tGamma: {args.gamma}\
                  \tBatch size: {args.batch_size}\
                  \tLearning rate: {args.learning_rate}\
                  \tGPU: {args.gpu}')
    logging.info(f'Input file directory: {args.seq_file}')



    input_seqs, input_ecs, _ = read_EC_Fasta(args.seq_file)

    input_seqs = input_seqs[:10000]
    input_ecs = input_ecs[:10000]

    train_seqs, test_seqs = train_test_split(input_seqs, test_size=0.1, random_state=args.seed)
    train_ecs, test_ecs = train_test_split(input_ecs, test_size=0.1, random_state=args.seed)
    # train_ids, test_ids = train_test_split(input_ids, test_size=0.1, random_state=seed_num)

    train_seqs, val_seqs = train_test_split(train_seqs, test_size=1/9, random_state=args.seed)
    train_ecs, val_ecs = train_test_split(train_ecs, test_size=1/9, random_state=args.seed)
    # train_ids, val_ids = train_test_split(input_ids, test_size=1/9, random_state=seed_num)

    logging.info(f'Number of sequences used- Train: {len(train_seqs)}')
    logging.info(f'Number of sequences used- Validation: {len(val_seqs)}')
    logging.info(f'Number of sequences used- Test: {len(test_seqs)}')


    explainECs = []
    for ecs in input_ecs:
        explainECs += ecs
    explainECs = list(set(explainECs))
    explainECs.sort()

    p = re.compile('(EC:\d+[.].*[.].*[.]).*')
    thirdECs = [p.match(ec).group(1) for ec in explainECs]
    thirdECs = list(set(thirdECs))
    thirdECs.sort()

    const = {
        'explainProts': explainECs,
        'thirdECs': thirdECs,
    }

    datas = {
        'train': [train_seqs, train_ecs],
        'val': [val_seqs, val_ecs],
        'test': [test_seqs, test_ecs]
    }


    train_ec_types = []
    for ecs in train_ecs:
        train_ec_types += ecs
    train_ec_types = set(train_ec_types)

    val_ec_types = []
    for ecs in val_ecs:
        val_ec_types += ecs
    val_ec_types = set(val_ec_types)
    
    test_ec_types = []
    for ecs in test_ecs:
        test_ec_types += ecs
    test_ec_types = set(test_ec_types)

    logging.info(f'Number of ECs in train data: {len(train_ec_types)}')
    logging.info(f'Number of ECs in validation data: {len(val_ec_types)}')
    logging.info(f'Number of ECs in test data: {len(test_ec_types)}')


    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, const, datas, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, const, datas, args)



def main_worker(gpu, ngpus_per_node, const, datas, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)


    config = BertConfig.from_pretrained("Rostlab/prot_bert")
    config.update(
        {'hidden_size':128, 
        'intermediate_size':256,
        'max_position_embeddings': 1000,
        'num_attention_heads':2, 
        'num_hidden_layers':1}
    )

    model = ProtBertEC(config, const)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        model = DDP(model, device_ids=[args.gpu])
    else:
        model.cuda()
        model = DDP(model)
    
    criterion = FocalLoss(gamma=args.gamma).cuda(args.gpu)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    cudnn.benchmark = True

    explainECs = const['explainProts']
    trainDataset = DeepECDataset(data_X=datas['train'][0], data_Y=datas['train'][1], explainECs=explainECs)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainDataset)
    else:
        train_sampler = None

    trainDataloader = DataLoader(
        trainDataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    validDataloader = DataLoader(
        DeepECDataset(data_X=datas['val'][0], data_Y=datas['val'][1], explainECs=explainECs),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    early_stopping = EarlyStopping(
        output_dir=args.output_dir, 
        patience=args.patience, 
        verbose=True,
        explainProts=explainECs
    )

    logging.info('Training start')
    for epoch in range(args.start_epoch, args.epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(trainDataloader, model, criterion, optimizer, args)
        valid_loss = validate(validDataloader, model, criterion, args)
        early_stopping(model, optimizer, epoch, valid_loss)
        scheduler.step()

        if early_stopping.early_stop:
            logging.info('Early stopping')
            break
    logging.info('Training end')


    model = torch.load(f'{args.output_dir}/model.pth')
    model.cuda()
    # model = DDP(model)

    test_source = (datas['test'][0], datas['test'][1])
    y_true, y_score, y_pred = evaluate(test_source, model, explainECs, args)
    precision = precision_score(y_true, y_pred, average='macro')
    print('Precision', precision)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    logging.info(f'(Macro) Precision: {precision}\tRecall: {recall}\tF1: {f1}')
    
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    logging.info(f'(Micro) Precision: {precision}\tRecall: {recall}\tF1: {f1}')





def train(train_loader, model, criterion, optimizer, args):
    losses = AverageMeter('Loss', ':.4e')
    model.train()
    for i, data in enumerate(train_loader):
        data = {key:val.cuda(args.gpu, non_blocking=True) for key, val in data.items()}
        output = model(**data)
        loss = criterion(output, data['labels'])
        losses.update(loss.item(), data['labels'].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4e')
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            data = {key:val.cuda(args.gpu, non_blocking=True) for key, val in data.items()}
            output = model(**data)
            loss = criterion(output, data['labels'])
            losses.update(loss.item(), data['labels'].size(0))

        return losses.avg


def evaluate(test_source, model, explainECs, args):
    num_data = test_source.dataset.__len__()
    len_ECs = len(explainECs)

    y_pred = torch.zeros([num_data, len_ECs])
    y_score = torch.zeros([num_data, len_ECs])
    y_true = torch.zeros([num_data, len_ECs])

    test_loader = DataLoader(
        DeepECDataset(data_X=test_source[0], data_Y=test_source[1], explainECs=explainECs),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )


    model.eval()
    with torch.no_grad():
        cnt = 0
        for i, data in enumerate(test_loader):
            data = {key:val.cuda(args.gpu, non_blocking=True) for key, val in data.items()}
            output = model(**data)
            output = torch.sigmoid(output)
            pred = output > 0.5
            output = output.cpu()
            pred = pred.float().cpu()

            step = data['labels'].shape[0]
            y_pred[cnt:cnt+step] = pred
            y_score[cnt:cnt+step] = output
            y_true[cnt:cnt+step] = data['labels'].cpu()
            
            cnt += step
        
        del data
        del output
    
    return y_true.numpy(), y_score.numpy(), y_pred.numpy()



class EarlyStopping:
    def __init__(self, output_dir='.', patience=5, verbose=False, delta=0, explainProts=[]):
        self.output_dir = output_dir
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.explainProts = explainProts

    def __call__(self, model, optimizer, epoch, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
        elif score < self.best_score - self.delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        if self.verbose:
            logging.info(f'Epoch {epoch}: Validation loss decreased ({self.val_loss_min:.12f} --> {val_loss:.12f}).  Saving model ...')
        
        ckpt = {'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'best_acc':self.best_score,
                'epoch':epoch,
                'explainECs':self.explainProts}

        torch.save(ckpt, self.output_dir + '/checkpt.pt')
        torch.save(model, self.output_dir + '/model.pth')
        self.val_loss_min = val_loss



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)




if __name__ == '__main__':
    main()