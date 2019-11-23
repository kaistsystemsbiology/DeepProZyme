import os
import random
import logging
# import basic python packages
import numpy as np

from Bio import SeqIO

# import torch packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from process_data import readFasta, read_SP_Fasta, deleteLowConf, \
                         getExplainedEC, getExplainedEC_short, \
                         getExplainableData, convertECtoLevel3

from data_loader import ECDataset, EnzymeDataset

from utils import EarlyStopping, draw, \
                  train_model, evalulate_deepEC
    
from model import DeepEC

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler('Sequence_process.log')
file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

seed_num = 123 # random seed for reproducibility
torch.manual_seed(seed_num)
random.seed(seed_num)
torch.cuda.manual_seed_all(seed_num)
np.random.seed(seed_num)

num_cpu = 4
torch.set_num_threads(num_cpu)



train_data_file = './Dataset/7.train_gold_standard_seq.txt'
val_data_file = './Dataset/7.val_gold_standard_seq.txt'
test_data_file = './Dataset/7.test_gold_standard_seq.txt'
lf_data_file = './Dataset/low_conf_protein_seq.txt'

id2seq_train, id2ec_train = readFasta(train_data_file)
id2seq_val, id2ec_val = readFasta(val_data_file)
id2seq_test, id2ec_test = readFasta(test_data_file)
id2seq_lf, id2ec_lf = readFasta(lf_data_file)

deleteLowConf(id2seq_train, id2ec_train, id2ec_lf)
deleteLowConf(id2seq_val, id2ec_val, id2ec_lf)
deleteLowConf(id2seq_test, id2ec_test, id2ec_lf)

explainECs = getExplainedEC(id2ec_train, id2ec_val, id2ec_test)

train_seqs, train_ecs = getExplainableData(id2seq_train, id2ec_train, explainECs)
val_seqs, val_ecs = getExplainableData(id2seq_val, id2ec_val, explainECs)
test_seqs, test_ecs = getExplainableData(id2seq_test, id2ec_test, explainECs)


with open('./Dataset/ec_train_seq.fasta', 'w') as fp:
    cnt = 0
    for seq, ec_list in zip(train_seqs, train_ecs):
        cnt += 1
        ecs = ';'.join(ec_list)
        fp.write(f'>seq{cnt}\t{ecs}\n{seq}\n')

with open('./Dataset/ec_valid_seq.fasta', 'w') as fp:
    cnt = 0
    for seq, ec_list in zip(val_seqs, val_ecs):
        cnt += 1
        ecs = ';'.join(ec_list)
        fp.write(f'>seq{cnt}\t{ecs}\n{seq}\n')

with open('./Dataset/ec_test_seq.fasta', 'w') as fp:
    cnt = 0
    for seq, ec_list in zip(test_seqs, test_ecs):
        cnt += 1
        ecs = ';'.join(ec_list)
        fp.write(f'>seq{cnt}\t{ecs}\n{seq}\n')

# with open('./Dataset/ec_train_seq.fasta', 'w') as fp:
#     for seqID in id2seq_train:
#         ecs = ';'.join(id2ec_train[seqID])
#         sequence = id2seq_train[seqID]
#         fp.write(f'>{seqID}\t{ecs}\n{sequence}\n')

# with open('./Dataset/ec_valid_seq.fasta', 'w') as fp:
#     for seqID in id2seq_val:
#         ecs = ';'.join(id2ec_val[seqID])
#         sequence = id2seq_val[seqID]
#         fp.write(f'>{seqID}\t{ecs}\n{sequence}\n')

# with open('./Dataset/ec_test_seq.fasta', 'w') as fp:
#     for seqID in id2seq_test:
#         ecs = ';'.join(id2ec_test[seqID])
#         sequence = id2seq_test[seqID]
#         fp.write(f'>{seqID}\t{ecs}\n{sequence}\n')