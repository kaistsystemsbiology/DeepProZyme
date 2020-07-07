import os
import random
import logging
# import basic python packages
import numpy as np
import matplotlib.pyplot as plt
# import torch packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from deepec.process_data import read_SP_Fasta, split_EnzNonenz, read_actual_Fasta

from deepec.data_loader import EnzymeDataset

from deepec.utils import argument_parser, EarlyStopping, \
                  draw, save_losses, train_model, calculateTestAccuracy, evalulate_model
    
from deepec.tf_models import DeepTFactor

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')


if __name__ == '__main__':
    parser = argument_parser()
    options = parser.parse_args()

    output_dir = options.output_dir
    log_dir = options.log_dir

    device = options.gpu
    checkpt_file = options.checkpoint

    enzyme_data_file = options.enzyme_data
    nonenzyme_data_file = options.nonenzyme_data

    num_cpu = options.cpu_num
    batch_size = options.batch_size

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

    logging.info('Test sequence analysis at cutoff 0.5')
    logging.info(f'Checkpoint dir: {checkpt_file}')

    enzyme_seqs = read_SP_Fasta(enzyme_data_file)
    nonenzyme_seqs = read_SP_Fasta(nonenzyme_data_file)
    nonenzyme_seqs = random.sample(nonenzyme_seqs, len(enzyme_seqs) * 3)

    train_data, valid_data, test_data = \
        split_EnzNonenz(enzyme_seqs, nonenzyme_seqs, seed_num)


    len_test_seq = len(test_data[0])

    logging.info(f'TF sequence dir: {enzyme_data_file}')
    logging.info(f'Non-TF sequence dir: {nonenzyme_data_file}')

    logging.info(f'Number of sequences used- Test: {len_test_seq}')


    testDataset = EnzymeDataset(test_data[0], test_data[1])
    testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)


    model = DeepTFactor(out_features=[1])
    model = model.to(device)

    ckpt = torch.load(f'{checkpt_file}', map_location=device)
    model.load_state_dict(ckpt['model'])

    accuracy = calculateTestAccuracy(model, testDataloader, device, cutoff=0.5)