# import
import random
import logging
# import basic python packages
import numpy as np

# import torch packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from process_data import readFasta, read_SP_Fasta, deleteLowConf, \
                         getExplainedEC, getExplainedEC_short, \
                         getExplainableData, convertECtoLevel3

from data_loader import ECDataset_multitask, EnzymeDataset

from utils import EarlyStopping, draw, \
                  train_model, evalulate_model, 
                  calculateTestAccuracy
    
from model import DeepEC

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler('CNN2_training.log', 'w')
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


# parameters
device = 'cuda:0' # The specific gpu I used for training. Convert the device to your own gpu
num_epochs = 30 # Number of maximum epochs
batch_size = 32 # Batch size for the training/validation data


train_data_file = './Dataset/7.train_gold_standard_seq.txt'
val_data_file = './Dataset/7.val_gold_standard_seq.txt'
test_data_file = './Dataset/7.test_gold_standard_seq.txt'
lf_data_file = './Dataset/low_conf_protein_seq.txt'

enzyme_data_file = './Dataset/processedEnzSeq.fasta'
nonnzyme_data_file = './Dataset/processedNonenzSeq.fasta'


id2seq_train, id2ec_train = readFasta(train_data_file)
id2seq_val, id2ec_val = readFasta(val_data_file)
id2seq_test, id2ec_test = readFasta(test_data_file)
id2seq_lf, id2ec_lf = readFasta(lf_data_file)

deleteLowConf(id2seq_train, id2ec_train, id2ec_lf)
deleteLowConf(id2seq_val, id2ec_val, id2ec_lf)
deleteLowConf(id2seq_test, id2ec_test, id2ec_lf)

explainECs = getExplainedEC(id2ec_train, id2ec_val, id2ec_test)


test_seqs, test_ecs = getExplainableData(id2seq_test, id2ec_test, explainECs)

len_test_seq = len(test_seqs)
logging.info(f'Test: {len_test_seq}')


explainECs_short = getExplainedEC_short(explainECs)
test_ecs_short = convertECtoLevel3(test_ecs)



testDataset = ECDataset(test_seqs, test_ecs, explainECs)


testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)
