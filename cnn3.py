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

from process_data import read_EC_Fasta, \
                         getExplainedEC, getExplainedEC_short, \
                         getExplainableData, convertECtoLevel3


from data_loader import ECDataset

from utils import EarlyStopping, draw,\
                  train_model, evalulate_model
    
from model import DeepEC

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler('CNN3_training.log')
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
device = 'cuda:3' # The specific gpu I used for training. Convert the device to your own gpu
num_epochs = 30 # Number of maximum epochs
batch_size = 32 # Batch size for the training/validation data
learning_rate = 1e-3
patience = 5
checkpt_file = 'checkpoint_CNN3.pt'


train_data_file = './Dataset/ec_train_seq.fasta'
val_data_file = './Dataset/ec_valid_seq.fasta'
test_data_file = './Dataset/ec_test_seq.fasta'

loss_graph_file = 'CNN2_loss_fig.png'


train_seqs, train_ecs = read_EC_Fasta(train_data_file)
val_seqs, val_ecs = read_EC_Fasta(val_data_file)
test_seqs, test_ecs = read_EC_Fasta(test_data_file)

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

third_level = False
if third_level==True:
    logging.info('Predict EC number upto third level')
    explainECs = getExplainedEC_short(explainECs)
    train_ecs = convertECtoLevel3(train_ecs)
    val_ecs = convertECtoLevel3(val_ecs)
    test_ecs = convertECtoLevel3(test_ecs)
else:
    logging.info('Predict EC number upto fourth level')

trainDataset = ECDataset(train_seqs, train_ecs, explainECs)
valDataset = ECDataset(val_seqs, val_ecs, explainECs)
testDataset = ECDataset(test_seqs, test_ecs, explainECs)

trainDataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
validDataloader = DataLoader(valDataset, batch_size=batch_size, shuffle=True)
testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)


model = DeepEC(out_features=len(explainECs))
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

model, avg_train_losses, avg_valid_losses = train_model(
    model, optimizer, criterion, device,
    batch_size, patience, num_epochs, 
    trainDataloader, validDataloader,
    checkpt_file
    )
draw(avg_train_losses, avg_valid_losses, file_name=loss_graph_file)

ckpt = torch.load(checkpt_file)
model.load_state_dict(ckpt['model'])

precision, recall, f1 = evalulate_model(model, testDataloader, device)