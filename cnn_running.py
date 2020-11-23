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

from deepec.process_data import read_EC_actual_Fasta, \
                                getExplainedEC_short, \
                                convertECtoLevel3
from deepec.data_loader import ECDataset, ECEmbedDataset
from deepec.utils import argument_parser
from deepec.model import DeepECv2_3, DeepEC

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

if __name__ == '__main__':
    parser = argument_parser()
    options = parser.parse_args()

    output_dir = options.output_dir
    checkpt_file = options.checkpoint
    input_data_file = options.seq_file

    device = options.gpu
    num_cpu = options.cpu_num
    batch_size = options.batch_size

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    seed_num = 123 # random seed for reproducibility
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    np.random.seed(seed_num)
    torch.set_num_threads(num_cpu)

    ckpt = torch.load(f'{checkpt_file}', map_location=device)
    explainECs = ckpt['explainECs']

    ntokens = 21
    emsize = 64 # embedding dimension
    nhid = 128 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    # model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout, explainECs).to(device)
    model = DeepEC(out_features=explainECs).to(device)
    model.load_state_dict(ckpt['model'])

    input_seqs, input_ids = read_EC_actual_Fasta(input_data_file)
    pseudo_labels = np.zeros((len(input_seqs)))

    # proteinDataset = ECEmbedDataset(input_seqs, pseudo_labels, explainECs, pred=True)
    proteinDataset = ECDataset(input_seqs, pseudo_labels, explainECs, pred=True)
    proteinDataloader = DataLoader(proteinDataset, batch_size=batch_size, shuffle=False)

    model.eval() # training session with train dataset
    with torch.no_grad():
        y_pred = torch.zeros([len(input_seqs), len(explainECs)])
        logging.info('Prediction starts on the dataset')
        cnt = 0
        for batch, data in enumerate(proteinDataloader):
            data = data.type(torch.FloatTensor)
            data = data.to(device)
            output = model(data)
            prediction = torch.sigmoid(output) > 0.5
            prediction = prediction.float()
            y_pred[cnt:cnt+data.shape[0]] = prediction.cpu()
            cnt += data.shape[0]
        logging.info('Prediction Ended on test dataset')

    with open(f'{output_dir}/prediction_result.txt', 'w') as fp:
        fp.write('sequence_ID\tprediction\n')
        for i, ith_pred in enumerate(y_pred):
            if len(ith_pred.nonzero()) == 0:
                fp.write(f'{input_ids[i]}\tNone\n')
                continue
            pred_ecs = [explainECs[j] for j in ith_pred.nonzero()]
            pred_ecs = ';'.join(pred_ecs)
            fp.write(f'{input_ids[i]}\t{pred_ecs}\n')
            # for j in ith_pred.nonzero():
            #     fp.write(f'{input_ids[i]}\t{explainECs[j]}\n')