import re
import logging
import argparse

# import basic python packages
import numpy as np

# import torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F

# import scikit learn packages
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score



# early stopping with validation dataset 
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


def train_model(config):
    device = config.device 
    train_loader = config.train_source
    model = config.model
    optimizer = config.optimizer
    criterion = config.criterion
    train_losses = 0
    n = 0

    model.train()
    for batch, (data, label) in enumerate(train_loader):
        data = data.float().to(device)
        label = label.float().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_losses += loss.item()
        n += data.size(0)
    return train_losses/n


def eval_model(config):
    device = config.device
    val_loader = config.val_source
    model = config.model
    criterion = config.criterion
    valid_losses = 0
    n = 0

    model.eval()
    with torch.no_grad():
        for batch, (data, label) in enumerate(val_loader):
            data = data.float().to(device)
            label = label.float().to(device)
            output = model(data)
            loss = criterion(output, label)
            valid_losses += loss.item()
            n += data.size(0)
    return valid_losses/n


def train(config):
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
        train_loss = train_model(config)
        valid_loss = eval_model(config)
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


def evalulate(config):
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
        for batch, (data, label) in enumerate(config.test_source):
            data = data.float().to(device)
            label = label.float()
            output = model(data)
            output = torch.sigmoid(output)
            prediction = output > 0.5
            prediction = prediction.float().cpu()

            y_pred[cnt:cnt+data.shape[0]] = prediction
            y_score[cnt:cnt+data.shape[0]] = output.cpu()
            y_true[cnt:cnt+data.shape[0]] = label
            cnt += data.shape[0]
        logging.info('Prediction Ended on test dataset')

        del data
        del output

        y_true = y_true.numpy()
        y_score = y_score.numpy()
        y_pred = y_pred.numpy()

    return y_true, y_score, y_pred


def evalulate_mcdropout(config):
    model = config.model
    model.eval() # training session with train dataset
    num_data = config.test_source.dataset.__len__()
    len_ECs = len(config.explainProts)
    device = config.device

    def enable_dropout(model):
        """ Function to enable the dropout layers during test-time """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                print(m.__class__.__name__)
                m.train()

    enable_dropout((model))

    with torch.no_grad():
        y_pred = torch.zeros([num_data, len_ECs])
        y_pred_2 = torch.zeros([num_data, len_ECs])
        y_score = torch.zeros([num_data, len_ECs])
        y_true = torch.zeros([num_data, len_ECs])
        logging.info('Prediction starts on test dataset')
        cnt = 0
        for batch, (data, label, mask) in enumerate(config.test_source):
            data = data.type(torch.FloatTensor).to(device)
            label = label.type(torch.FloatTensor)
            mask = mask.to(device)
            outputs = []
            for i in range(100):
                output = model(data, mask)
                output = torch.sigmoid(output)
                outputs.append(output)
            outputs = torch.stack(outputs)
            output = outputs.mean(dim=0)
            output_std = outputs.std(dim=0)

            prediction = output > 0.5
            prediction = prediction.float().cpu()
            y_pred[cnt:cnt+data.shape[0]] = prediction

            prediction = output - output_std > 0.5
            prediction = prediction.float().cpu()
            y_pred_2[cnt:cnt+data.shape[0]] = prediction

            y_score[cnt:cnt+data.shape[0]] = output.cpu()
            y_true[cnt:cnt+data.shape[0]] = label
            cnt += data.shape[0]
        logging.info('Prediction Ended on test dataset')

        del data
        del output
        del outputs
        del output_std

        y_true = y_true.numpy()
        y_score = y_score.numpy()
        y_pred = y_pred.numpy()
        y_pred_2 = y_pred_2.numpy()

    return y_true, y_score, y_pred, y_pred_2



######################


def train_model_emb(config):
    device = config.device 
    train_loader = config.train_source
    train_loader_emb = config.train_source_emb
    model = config.model
    optimizer = config.optimizer
    criterion = config.criterion
    train_losses = 0
    n = 0

    model.train()
    for (data, label), (data_emb, _) in zip(train_loader, train_loader_emb):
        data = data.type(torch.float).to(device)
        data_emb = data_emb.type(torch.long).to(device)
        label = label.type(torch.float).to(device)
        optimizer.zero_grad()
        output = model(data, data_emb)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_losses += loss.item()
        n += data.size(0)
    return train_losses/n


def eval_model_emb(config):
    device = config.device
    val_loader = config.val_source
    val_loader_emb = config.val_source_emb
    model = config.model
    criterion = config.criterion
    valid_losses = 0
    n = 0

    model.eval()
    with torch.no_grad():
        for (data, label), (data_emb, _) in zip(val_loader, val_loader_emb):
            data = data.type(torch.float).to(device)
            data_emb = data_emb.type(torch.long).to(device)
            label = label.type(torch.float).to(device)
            output = model(data, data_emb)
            loss = criterion(output, label)
            valid_losses += loss.item()
            n += data.size(0)
    return valid_losses/n


def train_emb(config):
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
        train_loss = train_model_emb(config)
        valid_loss = eval_model_emb(config)
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


def evalulate_emb(config):
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
        for (data, label), (data_emb, _), in zip(config.test_source, config.test_source_emb):
            data = data.type(torch.float).to(device)
            data_emb = data_emb.type(torch.long).to(device)
            label = label.type(torch.float)
            output = model(data, data_emb)
            output = torch.sigmoid(output)
            prediction = output > 0.5
            prediction = prediction.float().cpu()

            y_pred[cnt:cnt+data.shape[0]] = prediction
            y_score[cnt:cnt+data.shape[0]] = output.cpu()
            y_true[cnt:cnt+data.shape[0]] = label
            cnt += data.shape[0]
        logging.info('Prediction Ended on test dataset')

        del data
        del output

        y_true = y_true.numpy()
        y_score = y_score.numpy()
        y_pred = y_pred.numpy()

    return y_true, y_score, y_pred




def train_mask_model(config):
    device = config.device 
    train_loader = config.train_source
    model = config.model
    optimizer = config.optimizer
    criterion = config.criterion
    train_losses = 0
    n = 0

    model.train()
    for batch, (data, mask, label) in enumerate(train_loader):
        data = data.float().to(device)
        label = label.float().to(device)
        mask = mask.bool().to(device)
        optimizer.zero_grad()
        output = model(data, mask)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_losses += loss.item()
        n += data.size(0)
    return train_losses/n


def eval_mask_model(config):
    device = config.device
    val_loader = config.val_source
    model = config.model
    criterion = config.criterion
    valid_losses = 0
    n = 0

    model.eval()
    with torch.no_grad():
        for batch, (data, mask, label) in enumerate(val_loader):
            data = data.float().to(device)
            label = label.float().to(device)
            mask = mask.bool().to(device)
            output = model(data, mask)
            loss = criterion(output, label)
            valid_losses += loss.item()
            n += data.size(0)
    return valid_losses/n


def train_mask(config):
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
        train_loss = train_mask_model(config)
        valid_loss = eval_mask_model(config)
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


def evalulate_mask(config):
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
        for batch, (data, mask, label) in enumerate(config.test_source):
            data = data.float().to(device)
            label = label.float()
            mask = mask.bool().to(device)
            output = model(data, mask)
            output = torch.sigmoid(output)
            prediction = output > 0.5
            prediction = prediction.float().cpu()

            y_pred[cnt:cnt+data.shape[0]] = prediction
            y_score[cnt:cnt+data.shape[0]] = output.cpu()
            y_true[cnt:cnt+data.shape[0]] = label
            cnt += data.shape[0]
        logging.info('Prediction Ended on test dataset')

        del data
        del output

        y_true = y_true.numpy()
        y_score = y_score.numpy()
        y_pred = y_pred.numpy()

    return y_true, y_score, y_pred



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