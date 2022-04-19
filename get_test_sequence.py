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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# import torch packages
import torch

from deepec.process_data import read_EC_Fasta

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
    


if __name__ == '__main__':
    input_data_file = './Dataset/uniprot_dataset.fa'
    # output_dir = './Dataset/uniprot_dataset_test.fa'
#     output_dir = './Dataset/uniprot_dataset_val.fa'
    output_dir = './Dataset/uniprot_dataset_train.fa'


    seed_num = 123 # random seed for reproducibility
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    np.random.seed(seed_num)


    input_seqs, input_ecs, input_ids = read_EC_Fasta(input_data_file)

    train_seqs, test_seqs = train_test_split(input_seqs, test_size=0.1, random_state=seed_num)
    train_ecs, test_ecs = train_test_split(input_ecs, test_size=0.1, random_state=seed_num)
    train_ids, test_ids = train_test_split(input_ids, test_size=0.1, random_state=seed_num)

    train_seqs, val_seqs = train_test_split(train_seqs, test_size=1/9, random_state=seed_num)
    train_ecs, val_ecs = train_test_split(train_ecs, test_size=1/9, random_state=seed_num)
    train_ids, val_ids = train_test_split(train_ids, test_size=1/9, random_state=seed_num)


    # target_ids = test_ids
    # target_seqs = test_seqs
    # target_ecs = test_ecs
    # target_ids = val_ids
    # target_seqs = val_seqs
    # target_ecs = val_ecs
    target_ids = train_ids
    target_seqs = train_seqs
    target_ecs = train_ecs

    f = open(output_dir, 'w')
    for i, seq_id in enumerate(target_ids):
        seq = target_seqs[i]
        ecs = target_ecs[i]
        ecs = ';'.join(ecs)
        f.write(f'>{seq_id}\t{ecs}\n{seq}\n')
    f.close()



