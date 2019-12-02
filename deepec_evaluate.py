import os
import random
import logging
# import basic python packages
import numpy as np

# import torch packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from process_data import read_EC_Fasta, getExplainedEC_short
from data_loader import ECDataset, EnzymeDataset
from utils import argument_parser, evalulate_deepEC
from model import DeepEC

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')


if __name__ == '__main__':
    parser = argument_parser()
    options = parser.parse_args()

    output_dir = options.output_dir
    log_dir = options.log_dir

    device = options.gpu
    batch_size = options.batch_size

    checkpt_file_cnn1 = options.checkpoint_CNN1 # CNN1
    checkpt_file_cnn2 = options.checkpoint_CNN2 # CNN2
    checkpt_file_cnn3 = options.checkpoint_CNN3 # CNN3

    train_data_file = options.training_data
    val_data_file = options.validation_data
    test_data_file = options.test_data


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

    num_cpu = 4
    torch.set_num_threads(num_cpu)

    logging.info(f'\nInitial Setting\
                  \tBatch size: {batch_size}\
                  \nCNN1 checkpoint: {checkpt_file_cnn1}\
                  \tCNN2 checkpoint: {checkpt_file_cnn2}\
                  \tCNN3 checkpoint: {checkpt_file_cnn3}')

    _, train_ecs = read_EC_Fasta(train_data_file)
    _, val_ecs = read_EC_Fasta(val_data_file)
    test_seqs, test_ecs = read_EC_Fasta(test_data_file)

    len_test_seq = len(test_seqs)
    logging.info(f'Number of sequences used- Test: {len_test_seq}')

    explainECs = []
    for ec_data in [train_ecs, val_ecs, test_ecs]:
        for ecs in ec_data:
            for each_ec in ecs:
                if each_ec not in explainECs:
                    explainECs.append(each_ec)
    explainECs.sort()

    explainECs_short = getExplainedEC_short(explainECs)

    testDataset = ECDataset(test_seqs, test_ecs, explainECs)
    testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)

    cnn1 = DeepEC(out_features=1)
    cnn1 = cnn1.to(device)
    cnn2 = DeepEC(out_features=len(explainECs_short))
    cnn2 = cnn2.to(device)
    cnn3 = DeepEC(out_features=len(explainECs))
    cnn3 = cnn3.to(device)

    cnn1.load_state_dict(\
            torch.load(checkpt_file_cnn1, map_location=device)['model'])
    cnn2.load_state_dict(\
            torch.load(checkpt_file_cnn2, map_location=device)['model'])
    cnn3.load_state_dict(\
            torch.load(checkpt_file_cnn3, map_location=device)['model'])


    evalulate_deepEC(cnn1, cnn2, cnn3, testDataloader, len(testDataset), explainECs, explainECs_short, device)