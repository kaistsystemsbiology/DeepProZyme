import os
import random
import logging
# import basic python packages
import numpy as np

# import torch packages
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deepec.process_data import read_EC_actual_Fasta
from deepec.data_loader import DeepECDataset
from deepec.utils import argument_parser

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')



if __name__ == '__main__':
    parser = argument_parser()
    options = parser.parse_args()

    output_dir = options.output_dir
    checkpt_file = options.checkpoint
    input_data_file = options.seq_file

    device = options.gpu
    num_epochs = options.epoch
    batch_size = options.batch_size
    num_cpu = options.cpu_num

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    seed_num = 123 # random seed for reproducibility
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    np.random.seed(seed_num)
    torch.set_num_threads(num_cpu)

    model = torch.load(checkpt_file)
    model = model.to(device)
    explainECs = model.explainECs
    
    input_seqs, input_ids = read_EC_actual_Fasta(input_data_file)
    pseudo_labels = np.zeros((len(input_seqs)))

    proteinDataset = DeepECDataset(data_X=input_seqs, data_Y=pseudo_labels, explainECs=explainECs, pred=True)
    proteinDataloader = DataLoader(proteinDataset, batch_size=batch_size, shuffle=False)
    
    model.eval() # training session with train dataset
    with torch.no_grad():
        y_pred = torch.zeros([len(input_seqs), len(explainECs)])
        y_score = torch.zeros([len(input_seqs), len(explainECs)])
        logging.info('Prediction starts on the dataset')
        cnt = 0
        for batch, data in enumerate(proteinDataloader):
            inputs = {key:val.to(device) for key, val in data.items()}
            output = model(**inputs)
            prediction = torch.sigmoid(output) > 0.5
            prediction = prediction.float()
            step = data['input_ids'].shape[0]
            y_pred[cnt:cnt+step] = prediction.cpu()
            y_score[cnt:cnt+step] = torch.sigmoid(output).cpu()
            cnt += step
        logging.info('Prediction Ended on test dataset')

    with open(f'{output_dir}/prediction_result.txt', 'w') as fp:
        fp.write('sequence_ID\tprediction\tscore\n')
        for i, ith_pred in enumerate(y_pred):
            nonzero_preds = torch.nonzero(ith_pred, as_tuple=False)
            if len(nonzero_preds) == 0:
                fp.write(f'{input_ids[i]}\tNone\t0.0\n')
                continue
            for j in nonzero_preds:
                pred_ec = explainECs[j]
                pred_score = y_score[i][j].item()
                fp.write(f'{input_ids[i]}\t{pred_ec}\t{pred_score:0.4f}\n')