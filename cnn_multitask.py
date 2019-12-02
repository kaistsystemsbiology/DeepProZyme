# import
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

from process_data import read_EC_Fasta, \
                         getExplainedEC, getExplainedEC_short, \
                         getExplainableData, convertECtoLevel3


from data_loader import ECDataset_multitask

from utils import argument_parser, EarlyStopping, draw, save_losses,\
                  train_model_multitask, evalulate_model_multitask
    
from model import DeepEC_multitask

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
    train_data_file = options.training_data
    val_data_file = options.validation_data
    test_data_file = options.test_data

    third_level = options.third_level
    alpha = options.alpha

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
                  \nEpoch: {num_epochs}\
                  \tBatch size: {batch_size}\
                  \tLearning rate: {learning_rate}\
                  \tAlpha: {alpha}')


    train_seqs, train_ecs = read_EC_Fasta(train_data_file)
    val_seqs, val_ecs = read_EC_Fasta(val_data_file)
    test_seqs, test_ecs = read_EC_Fasta(test_data_file)


    len_train_seq = len(train_seqs)
    len_valid_seq = len(val_seqs)
    len_test_seq = len(test_seqs)

    logging.info(f'Number of sequences used- Train: {len_train_seq}')
    logging.info(f'Number of sequences used- Validation: {len_valid_seq}')
    logging.info(f'Number of sequences used- Test: {len_test_seq}')


    explainECs = []
    for ec_data in [train_ecs, val_ecs, test_ecs]:
        for ecs in ec_data:
            for each_ec in ecs:
                if each_ec not in explainECs:
                    explainECs.append(each_ec)
    explainECs.sort()


    explainECs_short = getExplainedEC_short(explainECs)
    train_ecs_short = convertECtoLevel3(train_ecs)
    val_ecs_short = convertECtoLevel3(val_ecs)
    test_ecs_short = convertECtoLevel3(test_ecs)


    trainDataset = ECDataset_multitask(train_seqs, \
                                    train_ecs, train_ecs_short, 
                                    explainECs, explainECs_short)
    valDataset = ECDataset_multitask(val_seqs, \
                                    val_ecs, val_ecs_short, 
                                    explainECs, explainECs_short)
    testDataset = ECDataset_multitask(test_seqs, \
                                    test_ecs, test_ecs_short,
                                    explainECs, explainECs_short)

    trainDataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    validDataloader = DataLoader(valDataset, batch_size=batch_size, shuffle=True)
    testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)


    num_class4 = len(explainECs)
    num_class3 = len(explainECs_short)
    model = DeepEC_multitask(out_features1=num_class3, out_features2=num_class4)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    model, avg_train_losses, avg_valid_losses, \
           avg_train_losses_1, avg_valid_losses_1, \
           avg_train_losses_2, avg_valid_losses_2 \
           = train_model_multitask(model, optimizer, criterion, device,
                                   batch_size, patience, num_epochs, 
                                   trainDataloader, validDataloader,
                                   f'{output_dir}/{checkpt_file}', alpha
                                   )


    loss_graph_file_EC3 = 'loss_for_EC3_lr4.png'
    loss_graph_file_EC4 = 'loss_for_EC4_lr4.png'
    loss_text_file_EC3 = 'loss_EC3.txt'
    loss_text_file_EC4 = 'loss_EC4.txt'

    save_losses(avg_train_losses, avg_valid_losses, \
                output_dir=output_dir)
    save_losses(avg_train_losses_1, avg_valid_losses_1, \
                output_dir=output_dir, file_name=loss_text_file_EC3)
    save_losses(avg_train_losses_2, avg_valid_losses_2, \
                output_dir=output_dir, file_name=loss_text_file_EC4)

    draw(avg_train_losses, avg_valid_losses, \
        output_dir=output_dir)
    draw(avg_train_losses_1, avg_valid_losses_1, \
        output_dir=output_dir, file_name=loss_graph_file_EC3)
    draw(avg_train_losses_2, avg_valid_losses_2, \
        output_dir=output_dir, file_name=loss_graph_file_EC4)

    ckpt = torch.load(f'{output_dir}/{checkpt_file}')
    model.load_state_dict(ckpt['model'])

    precision, recall, f1 = evalulate_model_multitask(model, testDataloader, \
                                                     explainECs, explainECs_short, \
                                                     device, alpha=alpha)