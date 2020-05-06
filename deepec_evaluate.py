import os
import random
import logging
# import basic python packages
import numpy as np
from sklearn.model_selection import train_test_split
# import torch packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from deepec.process_data import read_EC_Fasta
from deepec.data_loader import ECDataset
from deepec.utils import argument_parser, evalulate_deepEC
from deepec.old_models import DeepEC

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')


if __name__ == '__main__':
    parser = argument_parser()
    options = parser.parse_args()

    output_dir = options.output_dir
    log_dir = options.log_dir

    device = options.gpu
    num_cpu = options.cpu_num
    batch_size = options.batch_size

    checkpt_file_cnn1 = options.checkpoint_CNN1 # CNN1
    checkpt_file_cnn2 = options.checkpoint_CNN2 # CNN2
    checkpt_file_cnn3 = options.checkpoint_CNN3 # CNN3

    input_data_file = options.seq_file

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

    logging.info(f'\nInitial Setting\
                  \tBatch size: {batch_size}\
                  \nCNN1 checkpoint: {checkpt_file_cnn1}\
                  \tCNN2 checkpoint: {checkpt_file_cnn2}\
                  \tCNN3 checkpoint: {checkpt_file_cnn3}')

    input_seqs, input_ecs, input_ids = read_EC_Fasta(input_data_file)
    _, test_seqs = train_test_split(input_seqs, test_size=0.1, random_state=seed_num)
    _, test_ecs = train_test_split(input_ecs, test_size=0.1, random_state=seed_num)
    # train_ids, test_ids = train_test_split(input_ids, test_size=0.1, random_state=seed_num)

    logging.info(f'Number of sequences used- Test: {len(test_seqs)}')

    ckpt1 = torch.load(checkpt_file_cnn1, map_location=device)
    ckpt2 = torch.load(checkpt_file_cnn2, map_location=device)
    ckpt3 = torch.load(checkpt_file_cnn3, map_location=device)

    cnn1 = DeepEC(out_features=ckpt1['explainECs'])
    cnn1 = cnn1.to(device)
    cnn2 = DeepEC(out_features=ckpt2['explainECs'])
    cnn2 = cnn2.to(device)
    cnn3 = DeepEC(out_features=ckpt3['explainECs'])
    cnn3 = cnn3.to(device)

    cnn1.load_state_dict(ckpt1['model'])
    cnn2.load_state_dict(ckpt2['model'])
    cnn3.load_state_dict(ckpt3['model'])
    
    testDataset = ECDataset(test_seqs, test_ecs, cnn3.explainECs)
    testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)

    evalulate_deepEC(cnn1, cnn2, cnn3, testDataloader, len(testDataset), device)