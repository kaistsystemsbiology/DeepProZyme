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

from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import f1_score, precision_score, recall_score

# import torch packages
import torch
from torch.utils.data import DataLoader

from deepec.process_data import read_EC_Fasta
from deepec.data_loader import ECEmbedDataset, DeepECDataset
from deepec.utils import argument_parser, draw, save_losses, FocalLoss, DeepECConfig
from deepec.train import train_bert, evaluate_bert
from deepec.model import ProtBertConvEC
from transformers import BertConfig
from transformers import Trainer, TrainingArguments

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')


'''python bert_get_thrd.py -i ./Dataset/uniprot_dataset_val.fa -o ./output/bert_05/validation_set_thrd_tuning -ckpt ./output/bert_05/model_single.pth -g cuda:3 -b 128'''
if __name__ == '__main__':
    parser = argument_parser()
    options = parser.parse_args()

    output_dir = options.output_dir

    device = options.gpu
    batch_size = options.batch_size

    checkpt_file = options.checkpoint
    input_data_file = options.seq_file
    num_cpu = options.cpu_num

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    seed_num = 123 # random seed for reproducibility
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    np.random.seed(seed_num)

    torch.set_num_threads(num_cpu)

    logging.info(f'Input file directory: {input_data_file}')


    model = torch.load(checkpt_file)
    model = model.to(device)
    model.eval()
    explainECs = model.explainECs


    input_seqs, input_ecs, input_ids = read_EC_Fasta(input_data_file)

    proteinDataset = DeepECDataset(data_X=input_seqs, data_Y=input_ecs, explainECs=explainECs)
    proteinDataloader = DataLoader(proteinDataset, batch_size=batch_size, shuffle=False)


    with torch.no_grad():
        y_score = torch.zeros([len(input_seqs), len(explainECs)])
        y_true = torch.zeros([len(input_seqs), len(explainECs)])
        logging.info('Prediction starts on the dataset')
        cnt = 0
        for batch, data in enumerate(tqdm(proteinDataloader)):
            inputs = {key:val.to(device) for key, val in data.items()}
            output = model(**inputs)
            step = data['input_ids'].shape[0]
            y_score[cnt:cnt+step] = torch.sigmoid(output).cpu()
            y_true[cnt:cnt+step] = inputs['labels'].cpu()
            cnt += step
        logging.info('Prediction Ended on test dataset')

    
    len_ECs = len(explainECs)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    prec = dict()
    rec = dict()
    f1s = dict()

    f = open(f'{output_dir}/F_score_thrds.txt', 'w')
    f.write('EC\tPrecision\tRecall\tF1\tThrd\n')
    for i in tqdm(range(len_ECs)):
        ec_number = explainECs[i]
        precision, recall, thrds = precision_recall_curve(y_true[:, i], y_score[:, i])
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)
        thrd = thrds[ix]

        f.write(f'{ec_number}\t{precision[ix]:0.4f}\t{recall[ix]:0.4f}\t{fscore[ix]:0.4f}\t{thrd:0.4f}\n')

    f.close()
