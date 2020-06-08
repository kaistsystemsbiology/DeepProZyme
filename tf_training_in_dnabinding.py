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

from deepec.process_data import read_SP_Fasta, split_EnzNonenz

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
    num_epochs = options.epoch
    batch_size = options.batch_size
    learning_rate = options.learning_rate
    patience = options.patience

    checkpt_file = options.checkpoint

    enzyme_data_file = options.enzyme_data
    nonenzyme_data_file = options.nonenzyme_data

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

    logging.info(f'\nInitial Setting\
                  \nEpoch: {num_epochs}\
                  \tBatch size: {batch_size}\
                  \tLearning rate: {learning_rate}')


    enzyme_seqs = read_SP_Fasta(enzyme_data_file)
    nonenzyme_seqs = read_SP_Fasta(nonenzyme_data_file)

    train_data, valid_data, test_data = \
        split_EnzNonenz(enzyme_seqs, nonenzyme_seqs, seed_num)

    len_train_seq = len(train_data[0])
    len_valid_seq = len(valid_data[0])
    len_test_seq = len(test_data[0])

    logging.info(f'TF sequence dir: {enzyme_data_file}')
    logging.info(f'Non-TF sequence dir: {nonenzyme_data_file}')

    logging.info(f'Number of sequences used- Train: {len_train_seq}')
    logging.info(f'Number of sequences used- Validation: {len_valid_seq}')
    logging.info(f'Number of sequences used- Test: {len_test_seq}')


    trainDataset = EnzymeDataset(train_data[0], train_data[1])
    valDataset = EnzymeDataset(valid_data[0], valid_data[1])
    testDataset = EnzymeDataset(test_data[0], test_data[1])

    trainDataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    validDataloader = DataLoader(valDataset, batch_size=batch_size, shuffle=True)
    testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)


    model = DeepTFactor(out_features=[1])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    model, avg_train_losses, avg_valid_losses = train_model(
                model, optimizer, criterion, device,
                batch_size, patience, num_epochs, 
                trainDataloader, validDataloader, 
                save_name=f'{output_dir}/{checkpt_file}')

    save_losses(avg_train_losses, avg_valid_losses, output_dir=output_dir)
    draw(avg_train_losses, avg_valid_losses, output_dir=output_dir)

    ckpt = torch.load(f'{output_dir}/{checkpt_file}', map_location=device)
    model.load_state_dict(ckpt['model'])

    fpr, tpr, thrd = evalulate_model(model, testDataloader, len(testDataset), [1], device)
    sensitivity = tpr
    specificity = 1-fpr
    j = sensitivity + specificity - 1
    ind = np.argmax(j)
    cutoff = thrd[ind]
    ckpt['cutoff'] = cutoff
    logging.info(f'Cutoff of the prediction score: {cutoff}')
    torch.save(ckpt, f'{output_dir}/{checkpt_file}',)
    accuracy = calculateTestAccuracy(model, testDataloader, device, cutoff=cutoff)

    
    fig = plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, 'b',linewidth=2)
    plt.xlim(left=0, right=1)
    plt.ylim(bottom=0, top=1)
    plt.savefig(output_dir + '/AUC_curve.png')
