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


from deepec.data_loader import ECDataset

from deepec.utils import argument_parser, EarlyStopping, \
                         draw, save_losses, train_model, evalulate_model
    
from deepec.old_models import DeepEC

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
    input_data_file = options.enzyme_data

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

    model = DeepEC(out_features=explainECs, basal_net='CNN0')
    model = model.to(device)
    model.load_state_dict(ckpt['model'])

    input_seqs, input_ids = read_EC_actual_Fasta(input_data_file)
    pseudo_labels = np.zeros((len(input_seqs)))

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
            prediction = output > 0.5
            prediction = prediction.float()
            y_pred[cnt:cnt+data.shape[0]] = prediction.cpu()
            cnt += data.shape[0]
        logging.info('Prediction Ended on test dataset')

    with open(f'{output_dir}/prediction_result.txt', 'w') as fp:
        fp.write('sequence_ID\tprediction\n')
        for i, ith_pred in enumerate(y_pred):
            for j in ith_pred.nonzero():
                fp.write(f'{input_ids[i]}\t{explainECs[j]}\n')