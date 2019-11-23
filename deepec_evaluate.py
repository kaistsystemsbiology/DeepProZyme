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

from process_data import read_EC_Fasta, getExplainedEC_short
from data_loader import ECDataset, EnzymeDataset
from utils import evalulate_deepEC
from model import DeepEC

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler('DeepEC_evaluation.log', 'w')
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
device = 'cuda:1'
batch_size = 32 # Batch size for the training/validation data

checkpt_file_cnn2 = 'checkpoint_CNN2.pt'
checkpt_file_cnn3 = 'checkpoint_CNN3.pt'

train_data_file = './Dataset/ec_train_seq.fasta'
val_data_file = './Dataset/ec_valid_seq.fasta'
test_data_file = './Dataset/ec_test_seq.fasta'


_, train_ecs = read_EC_Fasta(train_data_file)
_, val_ecs = read_EC_Fasta(val_data_file)
test_seqs, test_ecs = read_EC_Fasta(test_data_file)


len_test_seq = len(test_seqs)
logging.info(f'Number of sequences used- Test: {len_test_seq}')

explainECs = []
for ec_data in [train_ecs, val_ecs, test_ecs]:
    for ecs in ec_data:
        for each_ec in ecs:
            if each_ec not in explainECs:
                explainECs.append(each_ec)
explainECs.sort()

explainECs_short = getExplainedEC_short(explainECs)


testDataset = ECDataset(test_seqs, test_ecs, explainECs)
testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)


cnn2 = DeepEC(out_features=len(explainECs_short))
cnn2 = cnn2.to(device)
cnn3 = DeepEC(out_features=len(explainECs))
cnn3 = cnn3.to(device)

cnn2.load_state_dict(\
        torch.load(checkpt_file_cnn2, map_location=device)['model'])
cnn3.load_state_dict(\
        torch.load(checkpt_file_cnn3, map_location=device)['model'])


evalulate_deepEC(cnn2, cnn3, testDataloader, test_seqs, explainECs, explainECs_short, device)