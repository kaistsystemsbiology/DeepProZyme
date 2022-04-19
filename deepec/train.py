import re
import logging

# import basic python packages
import numpy as np

# import torch packages
import torch

from tqdm.auto import tqdm


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
            self.save_checkpoint(val_loss, model, optimizer, epoch, best=True)
        elif score < self.best_score - self.delta:
            self.counter += 1
            self.save_checkpoint(val_loss, model, optimizer, epoch, best=False)
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, best=True)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch, best=True):
        if self.verbose:
            logging.info(f'Epoch {epoch}: Validation loss decreased ({self.val_loss_min:.12f} --> {val_loss:.12f}).  Saving model ...')
        
        ckpt = {'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'best_acc':self.best_score,
                'epoch':epoch,
                'explainECs':self.explainProts}
        torch.save(ckpt, self.save_name)
        if best:
            torch.save(model, self.save_name.replace('checkpoint.pt', 'model.pth'))
        torch.save(model, self.save_name.replace('checkpoint.pt', f'model_{epoch}.pth'))
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
    for batch, (data, label) in enumerate(tqdm(train_loader)):
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
        for batch, (data, label) in enumerate(tqdm(val_loader)):
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
    for epoch in tqdm(range(n_epochs)):
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
        for batch, (data, label) in enumerate(tqdm(config.test_source)):
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




def train_bert_model(config):
    device = config.device 
    train_loader = config.train_source
    model = config.model
    optimizer = config.optimizer
    criterion = config.criterion
    train_losses = 0
    n = 0

    model.train()
    for batch, data in enumerate(tqdm(train_loader)):
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
        for batch, data in enumerate(tqdm(val_loader)):
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