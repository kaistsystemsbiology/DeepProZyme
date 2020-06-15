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
from deepec.utils import argument_parser, evalulate_model
from deepec.tf_models import DeepTFactor

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')


if __name__ == '__main__':
    parser = argument_parser()
    options = parser.parse_args()

    output_dir = options.output_dir
    device = options.gpu
    batch_size = options.batch_size

    checkpt_file = options.checkpoint

    enzyme_data_file = options.enzyme_data
    nonenzyme_data_file = options.nonenzyme_data

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


    tf_seqs, tf_ids = read_actual_Fasta(enzyme_data_file)
    nontf_seqs, nontf_ids = read_actual_Fasta(nonenzyme_data_file)

    logging.info(f'TF sequence dir: {enzyme_data_file}')
    logging.info(f'Non-TF sequence dir: {nonenzyme_data_file}')

    tfDataset = EnzymeDataset(tf_seqs, np.ones((len(tf_seqs))))
    nontfDataset = EnzymeDataset(nontf_seqs, np.zeros((len(nontf_seqs))))

    tfDataloader = DataLoader(tfDataset, batch_size=batch_size, shuffle=False)
    nontfDataloader = DataLoader(nontfDataset, batch_size=batch_size, shuffle=False)

    model = DeepTFactor(out_features=[1])
    model = model.to(device)

    ckpt = torch.load(f'{checkpt_file}', map_location=device)
    model.load_state_dict(ckpt['model'])
    cutoff = ckpt['cutoff']
    # cutoff = 0.5

    fp = open(f'{output_dir}/prediction_result.txt', 'w')
    fp.write('sequence_ID\ttype\tprediction\tscore\n')

    y_pred = torch.zeros([len(tf_ids), 1])
    with torch.no_grad():
        model.eval()
        cnt = 0
        for x, _ in tfDataloader:
            x = x.type(torch.FloatTensor)
            x_length = x.shape[0]
            output = model(x.to(device))
            prediction = output.cpu()
            y_pred[cnt:cnt+x_length] = prediction
            cnt += x_length
    scores = y_pred[:,0]
    for seq_id, score in zip(tf_ids, scores):
        if score > cutoff:
            tf = True
        else:
            tf = False
        fp.write(f'{seq_id}\t{tf}\tTF\t{score:0.4f}\n')

    y_pred = torch.zeros([len(nontf_ids), 1])
    with torch.no_grad():
        model.eval()
        cnt = 0
        for x, _ in nontfDataloader:
            x = x.type(torch.FloatTensor)
            x_length = x.shape[0]
            output = model(x.to(device))
            prediction = output.cpu()
            y_pred[cnt:cnt+x_length] = prediction
            cnt += x_length
    scores = y_pred[:,0]
    for seq_id, score in zip(nontf_ids, scores):
        if score > cutoff:
            tf = True
        else:
            tf = False
        fp.write(f'{seq_id}\t{tf}\tNonTF\t{score:0.4f}\n')