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
    
from deepec.model import DeepECv2_3 as DeepECv2

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
    learning_rate = options.learning_rate

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

    logging.info(f'ckpt dir: {checkpt_file}')


    input_seqs, input_ecs, input_ids = read_EC_Fasta(input_data_file)
    _, test_seqs = train_test_split(input_seqs, test_size=0.1, random_state=seed_num)
    _, test_ecs = train_test_split(input_ecs, test_size=0.1, random_state=seed_num)
    len_test_seq = len(test_seqs)
    logging.info(f'Number of sequences used- Test: {len_test_seq}')

    ckpt = torch.load(f'{checkpt_file}', map_location=device)
    explainECs = ckpt['explainECs']

    testDataset = ECDataset(test_seqs, test_ecs, explainECs)
    testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)

    model = DeepECv2(out_features=explainECs)
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    num_train_params_1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_train_params_2 = sum(p.numel() for p in model.cnn0.parameters() if p.requires_grad)
    logging.info(f'Number of trainable parameters: {num_train_params_1 + num_train_params_2}')

    y_true, y_score, y_pred = evalulate_model(
        model, testDataloader, len(testDataset), explainECs, device
        )

    del model
    del ckpt
    torch.cuda.empty_cache()

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    logging.info(f'Precision: {precision}\tRecall: {recall}\tF1: {f1}')
    