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

from sklearn.model_selection import train_test_split

from process_data import read_SP_Fasta, deleteLowConf, \
                         getExplainedEC, getExplainedEC_short, \
                         getExplainableData, convertECtoLevel3

from data_loader import EnzymeDataset

from utils import EarlyStopping, draw, \
                  train_model, calculateTestAccuracy
    
from model import DeepEC

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler('CNN1_training.log')
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
device = 'cuda:1' # The specific gpu I used for training. Convert the device to your own gpu
num_epochs = 30 # Number of maximum epochs
batch_size = 32 # Batch size for the training/validation data
learning_rate = 1e-3
patience = 5
checkpt_file = 'checkpoint_CNN1.pt'

enzyme_data_file = './Dataset/processedEnzSeq.fasta'
nonnzyme_data_file = './Dataset/processedNonenzSeq.fasta'

loss_graph_file = 'CNN1_loss_fig.png'

enzyme_seqs = read_SP_Fasta(enzyme_data_file)
nonenzyme_seqs = read_SP_Fasta(nonnzyme_data_file)

enzyme_train, enzyme_test = train_test_split(
    enzyme_seqs, test_size=0.1, shuffle=True, random_state=seed_num
    )

nonenzyme_train, nonenzyme_test = train_test_split(
    nonenzyme_seqs, test_size=0.1, shuffle=True, random_state=seed_num
    )

enzyme_train, enzyme_valid = train_test_split(
    enzyme_train, test_size=1/9, shuffle=True, random_state=seed_num
    )

nonenzyme_train, nonenzyme_valid = train_test_split(
    nonenzyme_train, test_size=1/9, shuffle=True, random_state=seed_num
    )

train_inputs = np.array(
    enzyme_train + nonenzyme_train
    )
train_labels = np.hstack(
    (np.ones(len(enzyme_train)), np.zeros(len(nonenzyme_train)))
    )

valid_inputs = np.array(
    enzyme_valid + nonenzyme_valid
    )
valid_labels = np.hstack(
    (np.ones(len(enzyme_valid)), np.zeros(len(nonenzyme_valid)))
    )

test_inputs = np.array(
    enzyme_test + nonenzyme_test
    )
test_labels = np.hstack(
    (np.ones(len(enzyme_test)), np.zeros(len(nonenzyme_test)))
    )



len_train_seq = len(train_inputs)
len_valid_seq = len(valid_inputs)
len_test_seq = len(test_inputs)

logging.info(f'Number of sequences used- Train: {len_train_seq}')
logging.info(f'Number of sequences used- Validation: {len_valid_seq}')
logging.info(f'Number of sequences used- Test: {len_test_seq}')



trainDataset = EnzymeDataset(train_inputs, train_labels)
valDataset = EnzymeDataset(valid_inputs, valid_labels)
testDataset = EnzymeDataset(test_inputs, test_labels)

trainDataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
validDataloader = DataLoader(valDataset, batch_size=batch_size, shuffle=True)
testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)



model = DeepEC(out_features=1)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

# model, avg_train_losses, avg_valid_losses = train_model(
#     model, optimizer, criterion, device,
#     batch_size, patience, num_epochs, 
#     trainDataloader, validDataloader,
#     checkpt_file
#     )


# draw(avg_train_losses, avg_valid_losses, file_name=loss_graph_file)

ckpt = torch.load(checkpt_file)
model.load_state_dict(ckpt['model'])

accuracy = calculateTestAccuracy(model, testDataloader, device)