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

from deepec.process_data import read_EC_Fasta, \
                                getExplainedEC_short, \
                                convertECtoLevel3

from deepec.data_loader import ECDataset

from deepec.utils import argument_parser, EarlyStopping, \
                         draw, save_losses, train_model_CAM, evalulate_model_CAM
    
from deepec.model import DeepEC_CAM

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
    basal_net = options.basal_net
    patience = options.patience

    checkpt_file = options.checkpoint
    seq_file = options.seq_file

    third_level = options.third_level
    num_cpu = options.cpu_num
    resume = options.training_resume

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
                  \tLearning rate: {learning_rate}\
                  \tPredict upto 3 level: {third_level}')


    scheduling = True
    if scheduling:
        logging.info('Learnig rate scheduling')

    sequences, ecs, _ = read_EC_Fasta(seq_file)
    train_x, test_x, train_y, test_y = train_test_split(sequences, 
                                                        ecs, 
                                                        test_size=0.1, 
                                                        random_state=seed_num)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, 
                                                        train_y, 
                                                        test_size=1/9, 
                                                        random_state=seed_num)


    len_train_seq = len(train_x)
    len_valid_seq = len(valid_x)
    len_test_seq = len(test_x)

    logging.info(f'Number of sequences used- Train: {len_train_seq}')
    logging.info(f'Number of sequences used- Validation: {len_valid_seq}')
    logging.info(f'Number of sequences used- Test: {len_test_seq}')

    if resume:
        ckpt = torch.load(f'{output_dir}/{checkpt_file}', map_location=device)
        explainECs = ckpt['explainECs']
    else:
        explainECs = []
        for ec_data in [train_y, valid_y, test_y]:
            for ecs in ec_data:
                for each_ec in ecs:
                    if each_ec not in explainECs:
                        explainECs.append(each_ec)
        explainECs.sort()


    if third_level:
        logging.info('Predict EC number upto third level')
        explainECs = getExplainedEC_short(explainECs)
        train_y = convertECtoLevel3(train_y)
        valid_y = convertECtoLevel3(valid_y)
        test_y = convertECtoLevel3(test_y)
    else:
        logging.info('Predict EC number upto fourth level')

    trainDataset = ECDataset(train_x, train_y, explainECs)
    valDataset = ECDataset(valid_x, valid_y, explainECs)
    testDataset = ECDataset(test_x, test_y, explainECs)

    trainDataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    validDataloader = DataLoader(valDataset, batch_size=batch_size, shuffle=True)
    testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)

    if resume:
        model = DeepEC_CAM(out_features=len(explainECs), basal_net=basal_net)
        model.load_state_dict(ckpt['model'])
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(ckpt['optimizer'])
        logging.info('Training resumed')
    else:
        model = DeepEC_CAM(out_features=len(explainECs), basal_net=basal_net)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if scheduling:
        # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=learning_rate*3/30)
        # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//6, gamma=0.1)
        # exp_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10], gamma=0.1) # CNN17_4, CNN18_3
        exp_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 9, 14], gamma=0.3) # CNN17_5
    else:
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=1)

    logging.info(f'Model Architecture: \n{model}')
    num_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of trainable parameters: {num_train_params}')

    model.explainECs = explainECs
    criterion = nn.BCELoss()

    model, avg_train_losses, avg_valid_losses = train_model_CAM(
        model, optimizer, exp_lr_scheduler, 
        criterion, device,
        batch_size, patience, num_epochs, 
        trainDataloader, validDataloader,
        f'{output_dir}/{checkpt_file}'
        )

    save_losses(avg_train_losses, avg_valid_losses, output_dir=output_dir)
    draw(avg_train_losses, avg_valid_losses, output_dir=output_dir)

    ckpt = torch.load(f'{output_dir}/{checkpt_file}')
    model.load_state_dict(ckpt['model'])

    y_true, y_pred = evalulate_model_CAM(model, testDataloader, len(testDataset), explainECs, device)