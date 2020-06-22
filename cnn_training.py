import os
import random
import logging
# import basic python packages
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score

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
                         draw, save_losses, train_model, evalulate_model
    
from deepec.model import DeepECv2

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
    input_data_file = options.seq_file

    third_level = options.third_level
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
                  \tLearning rate: {learning_rate}\
                  \tPredict upto 3 level: {third_level}')


    input_seqs, input_ecs, input_ids = read_EC_Fasta(input_data_file)

    train_seqs, test_seqs = train_test_split(input_seqs, test_size=0.1, random_state=seed_num)
    train_ecs, test_ecs = train_test_split(input_ecs, test_size=0.1, random_state=seed_num)
    # train_ids, test_ids = train_test_split(input_ids, test_size=0.1, random_state=seed_num)

    train_seqs, val_seqs = train_test_split(train_seqs, test_size=1/9, random_state=seed_num)
    train_ecs, val_ecs = train_test_split(train_ecs, test_size=1/9, random_state=seed_num)
    # train_ids, val_ids = train_test_split(input_ids, test_size=1/9, random_state=seed_num)

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

    if third_level:
        logging.info('Predict EC number upto third level')
        explainECs = getExplainedEC_short(explainECs)
        train_ecs = convertECtoLevel3(train_ecs)
        val_ecs = convertECtoLevel3(val_ecs)
        test_ecs = convertECtoLevel3(test_ecs)
    else:
        logging.info('Predict EC number upto fourth level')

    train_ec_types = []
    for ecs in train_ecs:
        for each_ec in ecs:
            if each_ec not in train_ec_types:
                train_ec_types.append(each_ec)
    val_ec_types = []
    for ecs in val_ecs:
        for each_ec in ecs:
            if each_ec not in val_ec_types:
                val_ec_types.append(each_ec)
    test_ec_types = []
    for ecs in test_ecs:
        for each_ec in ecs:
            if each_ec not in test_ec_types:
                test_ec_types.append(each_ec)

    logging.info(f'Number of ECs in train data: {len(train_ec_types)}')
    logging.info(f'Number of ECs in validation data: {len(val_ec_types)}')
    logging.info(f'Number of ECs in test data: {len(test_ec_types)}')

    trainDataset = ECDataset(train_seqs, train_ecs, explainECs)
    valDataset = ECDataset(val_seqs, val_ecs, explainECs)
    testDataset = ECDataset(test_seqs, test_ecs, explainECs)

    trainDataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    validDataloader = DataLoader(valDataset, batch_size=batch_size, shuffle=True)
    testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)


    model = DeepECv2(out_features=explainECs)
    logging.info(f'Model Architecture: \n{model}')
    model = model.to(device)
    num_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of trainable parameters: {num_train_params}')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    model, avg_train_losses, avg_valid_losses = train_model(
        model, optimizer, criterion, device,
        batch_size, patience, num_epochs, 
        trainDataloader, validDataloader,
        f'{output_dir}/{checkpt_file}'
        )

    save_losses(avg_train_losses, avg_valid_losses, output_dir=output_dir)
    draw(avg_train_losses, avg_valid_losses, output_dir=output_dir)

    ckpt = torch.load(f'{output_dir}/{checkpt_file}')
    model.load_state_dict(ckpt['model'])

    y_true, y_score, y_pred = evalulate_model(
        model, testDataloader, len(testDataset), explainECs, device
        )
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    logging.info(f'Precision: {precision}\tRecall: {recall}\tF1: {f1}')
    