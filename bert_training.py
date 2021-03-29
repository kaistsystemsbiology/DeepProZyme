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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score

# import torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from deepec.process_data import read_EC_Fasta, \
                                getExplainedEC_short, \
                                convertECtoLevel3
from deepec.data_loader import ECEmbedDataset
from deepec.utils import argument_parser, draw, save_losses, FocalLoss, DeepECConfig
# from deepec.train import train_mask, evalulate_mask
# from transformers import AdamW
from transformers import AutoTokenizer
from transformers import BertConfig
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, BertModel, BertPreTrainedModel, BertForSequenceClassification

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')

class DeepECDataset(Dataset):
    def __init__(self, data_X, data_Y, explainECs, tokenizer_name='Rostlab/prot_bert_bfd', max_length=1000, pred=False):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.max_length = max_length
        self.data_X = data_X
        self.data_Y = data_Y
        self.pred = pred
        self.map_EC = self.getECmap(explainECs)
        
        
    def __len__(self):
        return len(self.data_X)
    
    
    def getECmap(self, explainECs):
        ec_vocab = list(set(explainECs))
        ec_vocab.sort()
        map = {}
        for i, ec in enumerate(ec_vocab):
            baseArray = np.zeros(len(ec_vocab))
            baseArray[i] = 1
            map[ec] = baseArray
        return map
    

    def convert2onehot_EC(self, EC):
        map_EC = self.map_EC
        single_onehot = np.zeros(len(map_EC))
        for each_EC in EC:
            single_onehot += map_EC[each_EC]
        return single_onehot
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        seq = " ".join(str(self.data_X[idx]))
        seq = re.sub(r"[UZOB]", "X", seq)
        
        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)
        
        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        labels = self.data_Y[idx]
        labels = self.convert2onehot_EC(labels)
        labels = labels.reshape(-1)
        sample['labels'] = torch.tensor(labels)
        return sample
    
    
class ProtBertMultiLabelClassification(BertForSequenceClassification):
    def __init__(self, config, out_features=[], fc_gamma=0):
        super(ProtBertMultiLabelClassification, self).__init__(config)
        self.explainECs = out_features
        self.fc_alpha = 1
        self.fc_gamma = fc_gamma
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, len(out_features))
        nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
            
            
class EarlyStopping:
    def __init__(self, save_name='checkpoint.pt', patience=5, verbose=False, delta=0, explainProts=[]):
        self.save_name = save_name
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.explainProts = explainProts

    def __call__(self, model, optimizer, epoch, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
        elif score < self.best_score - self.delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        if self.verbose:
            logging.info(f'Epoch {epoch}: Validation loss decreased ({self.val_loss_min:.12f} --> {val_loss:.12f}).  Saving model ...')
        
        ckpt = {'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'best_acc':self.best_score,
                'epoch':epoch,
                'explainECs':self.explainProts}
        torch.save(ckpt, self.save_name)
        self.val_loss_min = val_loss
        

class WarmupOpt:
    def __init__(self, optimizer, model_size, warmup_step=10000, unit_step=20):
        self.optimizer = optimizer
        self.model_size = model_size
        self.warmup_step = warmup_step
        self.unit_step = unit_step
        self._step = 0
        self._rate = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.model_size**(-0.5)*min((step/self.unit_step)**(-0.5), (step/self.unit_step)*self.warmup_step**(-1.5))
    
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()



def train_bert_model(config):
    device = config.device 
    train_loader = config.train_source
    model = config.model
    optimizer = config.optimizer
    criterion = config.criterion
    train_losses = 0
    n = 0

    model.train()
    for batch, data in enumerate(train_loader):
        inputs = {key:val.to(device) for key, val in data.items()}
        optimizer.zero_grad()
        output = model(**inputs)
        loss = criterion(output, inputs['labels'])
        loss.backward()
        optimizer.step()

        train_losses += loss.item()
        n += inputs['labels'].size(0)
    return train_losses/n


def eval_bert_model(config):
    device = config.device
    val_loader = config.val_source
    model = config.model
    criterion = config.criterion
    valid_losses = 0
    n = 0

    model.eval()
    with torch.no_grad():
        for batch, data in enumerate(val_loader):
            inputs = {key:val.to(device) for key, val in data.items()}
            output = model(**inputs)
            loss = criterion(output, inputs['labels'])
            valid_losses += loss.item()
            n += inputs['labels'].size(0)
    return valid_losses/n


def train_bert(config):
    early_stopping = EarlyStopping(save_name=config.save_name, 
                                   patience=config.patience, 
                                   verbose=True,
                                   explainProts=config.explainProts
                                   )

    device = config.device
    n_epochs = config.n_epochs

    avg_train_losses = torch.zeros(n_epochs).to(device)
    avg_valid_losses = torch.zeros(n_epochs).to(device)
    
    logging.info('Training start')
    for epoch in range(n_epochs):
        train_loss = train_bert_model(config)
        valid_loss = eval_bert_model(config)
        if config.scheduler != None:
            config.scheduler.step()

        avg_train_losses[epoch] = train_loss
        avg_valid_losses[epoch] = valid_loss
        early_stopping(config.model, config.optimizer, epoch, valid_loss)
        if early_stopping.early_stop:
            logging.info('Early stopping')
            break
            
    logging.info('Training end')
    return avg_train_losses.tolist(), avg_valid_losses.tolist()


def evaluate_bert(config):
    model = config.model
    model.eval() # training session with train dataset
    num_data = config.test_source.dataset.__len__()
    len_ECs = len(config.explainProts)
    device = config.device

    with torch.no_grad():
        y_pred = torch.zeros([num_data, len_ECs])
        y_score = torch.zeros([num_data, len_ECs])
        y_true = torch.zeros([num_data, len_ECs])
        logging.info('Prediction starts on test dataset')
        cnt = 0
        for batch, data in enumerate(config.test_source):
            inputs = {key:val.to(device) for key, val in data.items()}
            output = model(**inputs)
            output = torch.sigmoid(output)
            prediction = output > 0.5
            prediction = prediction.float().cpu()

            y_pred[cnt:cnt+inputs['labels'].shape[0]] = prediction
            y_score[cnt:cnt+inputs['labels'].shape[0]] = output.cpu()
            y_true[cnt:cnt+inputs['labels'].shape[0]] = inputs['labels'].cpu()
            cnt += inputs['labels'].shape[0]
        logging.info('Prediction Ended on test dataset')

        del inputs
        del output

        y_true = y_true.numpy()
        y_score = y_score.numpy()
        y_pred = y_pred.numpy()

    return y_true, y_score, y_pred



if __name__ == '__main__':
    parser = argument_parser()
    options = parser.parse_args()

    output_dir = options.output_dir
    log_dir = options.log_dir

    device = options.gpu
    num_epochs = options.epoch
    batch_size = options.batch_size
    learning_rate = options.learning_rate
    patience = options.patience

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

    gamma = 1

    logging.info(f'\nInitial Setting\
                  \nEpoch: {num_epochs}\
                  \tGamma: {gamma}\
                  \tBatch size: {batch_size}\
                  \tLearning rate: {learning_rate}\
                  \tGPU: {device}\
                  \tPredict upto 3 level: {third_level}')
    logging.info(f'Input file directory: {input_data_file}')


    input_seqs, input_ecs, input_ids = read_EC_Fasta(input_data_file)
    # input_seqs, input_ecs, input_ids = input_seqs[:3000], input_ecs[:3000], input_ids[:3000]

    train_seqs, test_seqs = train_test_split(input_seqs, test_size=0.1, random_state=seed_num)
    train_ecs, test_ecs = train_test_split(input_ecs, test_size=0.1, random_state=seed_num)
    # train_ids, test_ids = train_test_split(input_ids, test_size=0.1, random_state=seed_num)

    train_seqs, val_seqs = train_test_split(train_seqs, test_size=1/9, random_state=seed_num)
    train_ecs, val_ecs = train_test_split(train_ecs, test_size=1/9, random_state=seed_num)
    # train_ids, val_ids = train_test_split(input_ids, test_size=1/9, random_state=seed_num)


    logging.info(f'Number of sequences used- Train: {len(train_seqs)}')
    logging.info(f'Number of sequences used- Validation: {len(val_seqs)}')
    logging.info(f'Number of sequences used- Test: {len(test_seqs)}')

    explainECs = []
    for ecs in input_ecs:
        explainECs += ecs
    explainECs = list(set(explainECs))
    explainECs.sort()

    if third_level:
        logging.info('Predict EC number upto third level')
        explainECs = getExplainedEC_short(explainECs)
        train_ecs = convertECtoLevel3(train_ecs)
        val_ecs = convertECtoLevel3(val_ecs)
        test_ecs = convertECtoLevel3(test_ecs)
    else:
        logging.info('Predict EC number upto fourth level')

    train_ec_types = []
    for ecs in train_ecs:
        train_ec_types += ecs
    len_train_ecs = len(set(train_ec_types))

    val_ec_types = []
    for ecs in val_ecs:
        val_ec_types += ecs
    len_val_ecs = len(set(val_ec_types))
    
    test_ec_types = []
    for ecs in test_ecs:
        test_ec_types += ecs
    len_test_ecs = len(set(test_ec_types))

    logging.info(f'Number of ECs in train data: {len_train_ecs}')
    logging.info(f'Number of ECs in validation data: {len_val_ecs}')
    logging.info(f'Number of ECs in test data: {len_test_ecs}')


    trainDataset = DeepECDataset(data_X=train_seqs, data_Y=train_ecs, explainECs=explainECs)
    valDataset = DeepECDataset(data_X=val_seqs, data_Y=val_ecs, explainECs=explainECs)
    testDataset = DeepECDataset(data_X=test_seqs, data_Y=test_ecs, explainECs=explainECs)

    trainDataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    validDataloader = DataLoader(valDataset, batch_size=batch_size, shuffle=True)
    testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)

    config = BertConfig.from_pretrained("Rostlab/prot_bert")
    config.update(
        {'hidden_size':128, 
        'intermediate_size':256,
        'max_position_embeddings': 1000,
        'num_attention_heads':8, 
        'num_hidden_layers':2}
    )
    model = ProtBertMultiLabelClassification(config, out_features=explainECs)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(device)
    logging.info(f'Model Architecture: \n{model}')
    num_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of trainable parameters: {num_train_params}')

    # optimizer_adam = optim.Adam(model.parameters(), lr=learning_rate, )
    # emsize = 128
    # warmup_step = 10000
    # unit_step = 20
    # optimizer = WarmupOpt(optimizer_adam, emsize, warmup_step=warmup_step, unit_step=unit_step)
    # scheduler = None
    # logging.info(f'Learning rate scheduling: Warmup step: {warmup_step}\tUnit step: {unit_step}')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, )
    criterion = FocalLoss(gamma=gamma)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    logging.info(f'Learning rate scheduling: step size: 1\tgamma: 0.95')

    criterion = FocalLoss(gamma=gamma)

    config = DeepECConfig()
    config.model = model 
    config.optimizer = optimizer
    config.criterion = criterion
    config.scheduler = scheduler
    config.n_epochs = num_epochs
    config.device = device
    config.save_name = f'{output_dir}/{checkpt_file}'
    config.patience = patience
    config.train_source = trainDataloader
    config.val_source = validDataloader
    config.test_source = testDataloader
    config.explainProts = explainECs


    avg_train_losses, avg_val_losses = train_bert(config)
    save_losses(avg_train_losses, avg_val_losses, output_dir=output_dir)
    draw(avg_train_losses, avg_val_losses, output_dir=output_dir)

    ckpt = torch.load(f'{output_dir}/{checkpt_file}')
    model.load_state_dict(ckpt['model'])

    y_true, y_score, y_pred = evaluate_bert(config)
    precision = precision_score(y_true, y_pred, average='macro')
    print('Precision', precision)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    logging.info(f'(Macro) Precision: {precision}\tRecall: {recall}\tF1: {f1}')
    
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    logging.info(f'(Micro) Precision: {precision}\tRecall: {recall}\tF1: {f1}')
    
    # len_ECs = len(explainECs)

    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # prec = dict()
    # rec = dict()
    # f1s = dict()

    # for i in range(len_ECs):
    #     fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #     prec[i] = precision_score(y_true[:, i], y_pred[:, i], )
    #     rec[i] = recall_score(y_true[:, i], y_pred[:, i])
    #     f1s[i] = f1_score(y_true[:, i], y_pred[:, i])

    # fp = open(f'{output_dir}/performance_indices.txt', 'w')
    # fp.write('EC\tAUC\tPrecision\tRecall\tF1\n')
    # for ind in roc_auc:
    #     ec = explainECs[ind]
    #     fp.write(f'{ec}\t{roc_auc[ind]}\t{prec[ind]}\t{rec[ind]}\t{f1s[ind]}\n')
    # fp.close()