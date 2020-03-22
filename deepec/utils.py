import re
import logging
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import basic python packages
import numpy as np

from Bio import SeqIO

# import torch packages
import torch
import torch.nn as nn

# import scikit learn packages
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score


def argument_parser(version=None):
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory')
    parser.add_argument('-l', '--log_dir', required=False, 
                        default='CNN_training.log', help='Log file directory')
    parser.add_argument('-g', '--gpu', required=False, 
                        default='cuda:0', help='Specify gpu')
    parser.add_argument('-e', '--epoch', required=False, type=int,
                        default=30, help='Total epoch number')
    parser.add_argument('-b', '--batch_size', required=False, type=int,
                        default=32, help='Batch size')
    parser.add_argument('-r', '--learning_rate', required=False, type=float,
                        default=1e-3, help='Learning rate')
    parser.add_argument('-p', '--patience', required=False, type=int,
                        default=5, help='Patience limit for early stopping')
    parser.add_argument('-c', '--checkpoint', required=False, 
                        default='checkpoint.pt', help='Checkpoint file')
    parser.add_argument('-t', '--seq_file', required=False, 
                        default='./Dataset/ec_seq.fa', help='Sequence data')
    parser.add_argument('-enz', '--enzyme_data', required=False, 
                        default='./Dataset/processedEnzSeq.fasta', help='Enzyme data')
    parser.add_argument('-nonenz', '--nonenzyme_data', required=False, 
                        default='./Dataset/processedNonenzSeq.fasta', help='Nonenzyme data')
    parser.add_argument('-basal', '--basal_net', required=False, 
                        default='CNN0_3', help='Select basal net')
    parser.add_argument('-third', '--third_level', required=False, type=boolean_string,
                        default=False, help='Predict upto third EC level')
    parser.add_argument('-resume', '--training_resume', required=False, type=boolean_string,
                        default=False, help='Resume training based on the previous checkpoint')
    parser.add_argument('-cpu', '--cpu_num', required=False, type=int,
                        default=1, help='Number of cpus to use')
    
    return parser


# early stopping with validation dataset 
class EarlyStopping:
    def __init__(self, save_name='checkpoint.pt', patience=7, verbose=False, delta=0):
        self.save_name = save_name
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer, epoch):
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
            logging.info(f'Validation loss decreased ({self.val_loss_min:.9f} --> {val_loss:.9f}).  Saving model ...')
        
        ckpt = {'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'best_acc':self.best_score,
                'epoch':epoch,
                'explainECs':model.explainECs}
        torch.save(ckpt, self.save_name)
        self.val_loss_min = val_loss


# plot the accuracy and loss value of each model.
###################
def draw(avg_train_losses, avg_valid_losses, output_dir, file_name='CNN_loss_fig.png'):
    fig = plt.figure(figsize=(9,6))

    min_position = avg_valid_losses.index(min(avg_valid_losses)) + 1

    plt.plot(range(1, len(avg_train_losses)+1), avg_train_losses, label='Training loss')
    plt.plot(range(1, len(avg_valid_losses)+1), avg_valid_losses, label='Validation loss')
    plt.axvline(min_position, linestyle='--', color='r', label='Early stopping checkpoint')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim(left=0)
    plt.ylim(bottom=0, )

    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/{file_name}', dpi=600)
    plt.show()
    return


def save_losses(avg_train_losses, avg_valid_losses, output_dir, file_name='losses.txt'):
    with open(f'{output_dir}/{file_name}', 'w') as fp:
        fp.write('Epoch\tAverage_train_loss\tAverage_valid_loss\n')
        cnt = 0
        for train_loss, valid_loss in zip(avg_train_losses, avg_valid_losses):
            cnt += 1
            fp.write(f'{cnt}\t{train_loss:9f}\t{valid_loss:9f}\n')
    return


def train_model(model, optimizer, criterion, device,
               batch_size, patience, n_epochs, 
               train_loader, valid_loader, save_name='checkpoint.pt'):
    early_stopping = EarlyStopping(
                                save_name=save_name, 
                                patience=patience, 
                                verbose=True)

    avg_train_losses = torch.zeros(n_epochs).to(device)
    avg_valid_losses = torch.zeros(n_epochs).to(device)
    
    logging.info('Training start')
    for epoch in range(n_epochs):
        train_losses = torch.zeros(len(train_loader)).to(device)
        valid_losses = torch.zeros(len(train_loader)).to(device)
        model.train() # training session with train dataset
        for batch, (data, label) in enumerate(train_loader):
            data = data.type(torch.FloatTensor)
            label = label.type(torch.FloatTensor)
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_losses[batch] = loss.item()
        avg_train_losses[epoch] = torch.mean(train_losses)

        model.eval() # validation session with validation dataset
        with torch.no_grad():
            for batch, (data, label) in enumerate(valid_loader):
                data = data.type(torch.FloatTensor)
                label = label.type(torch.FloatTensor)
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                loss = criterion(output, label)

                valid_losses[batch] = loss.item()
            
            valid_loss = torch.mean(valid_losses)
            avg_valid_losses[epoch] = valid_loss
        
        # decide whether to stop or not based on validation loss
        early_stopping(valid_loss, model, optimizer, epoch) 
        if early_stopping.early_stop:
            logging.info('Early stopping')
            break
            
    logging.info('Training end')
    return model, avg_train_losses.tolist(), avg_valid_losses.tolist()



# def evalulate_model(model, test_loader, num_data, explainECs, device):
#     model.eval() # training session with train dataset
#     with torch.no_grad():
#         y_pred = torch.zeros([num_data, len(explainECs)]).to(device)
#         y_true = torch.zeros([num_data, len(explainECs)]).to(device)
#         logging.info('Prediction starts on test dataset')
#         cnt = 0
#         for batch, (data, label) in enumerate(test_loader):
#             data = data.type(torch.FloatTensor)
#             label = label.type(torch.FloatTensor)
#             data = data.to(device)
#             label = label.to(device)
#             output = model(data)
#             prediction = output > 0.5
#             prediction = prediction.float()

#             y_pred[cnt:cnt+data.shape[0]] = prediction
#             y_true[cnt:cnt+data.shape[0]] = label
#             cnt += data.shape[0]
#         logging.info('Prediction Ended on test dataset')

#         y_true = y_true.cpu().numpy()
#         y_pred = y_pred.cpu().numpy()
#         precision = precision_score(y_true, y_pred, average='macro')
#         recall = recall_score(y_true, y_pred, average='macro')
#         f1 = f1_score(y_true, y_pred, average='macro')

#     logging.info(f'Precision: {precision}\tRecall: {recall}\tF1: {f1}')
#     return precision, recall, f1


def evalulate_model(model, test_loader, num_data, explainECs, device):
    model.eval() # training session with train dataset
    with torch.no_grad():
        y_pred = torch.zeros([num_data, len(explainECs)])
        y_true = torch.zeros([num_data, len(explainECs)])
        logging.info('Prediction starts on test dataset')
        cnt = 0
        for batch, (data, label) in enumerate(test_loader):
            data = data.type(torch.FloatTensor)
            label = label.type(torch.FloatTensor)
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            prediction = output > 0.5
            prediction = prediction.float()

            y_pred[cnt:cnt+data.shape[0]] = prediction.cpu()
            y_true[cnt:cnt+data.shape[0]] = label.cpu()
            cnt += data.shape[0]
        logging.info('Prediction Ended on test dataset')

        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

    logging.info(f'Precision: {precision}\tRecall: {recall}\tF1: {f1}')
    return precision, recall, f1



def calculateTestAccuracy(model, testDataloader, device):
    with torch.no_grad():
        model.eval()
        n=0
        test_loss=0
        test_acc=0
        for x, y in testDataloader:
            x = x.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)
            test_seq = x.to(device)
            test_label = y.to(device)
            output = model(test_seq)
            prediction = output > 0.5
            prediction = prediction.float()
            prediction = prediction.cpu()

            test_label = test_label.float()
            test_label = test_label.cpu()

            test_acc += (prediction==test_label).float().sum()
            n += test_label.size(0)
            
        test_acc /= n
        logging.info('Test accuracy: %0.6f'%(test_acc.item()))
    return test_acc.item()



def train_model_CAM(model, optimizer, criterion, 
               device, batch_size, patience, n_epochs, 
               train_loader, valid_loader, save_name='checkpoint.pt'):
    early_stopping = EarlyStopping(
                                save_name=save_name, 
                                patience=patience, 
                                verbose=True)

    avg_train_losses = torch.zeros(n_epochs).to(device)
    avg_valid_losses = torch.zeros(n_epochs).to(device)
    
    logging.info('Training start')
    model.train() # training session with train dataset
    for epoch in range(n_epochs):
        train_losses = torch.zeros(len(train_loader)).to(device)
        valid_losses = torch.zeros(len(train_loader)).to(device)
        for batch, (data, label) in enumerate(train_loader):
            data = data.type(torch.FloatTensor)
            label = label.type(torch.FloatTensor)
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_losses[batch] = loss.item()
        avg_train_losses[epoch] = torch.mean(train_losses)

        model.eval() # validation session with validation dataset
        with torch.no_grad():
            for batch, (data, label) in enumerate(valid_loader):
                data = data.type(torch.FloatTensor)
                label = label.type(torch.FloatTensor)
                data = data.to(device)
                label = label.to(device)
                output, _ = model(data)
                loss = criterion(output, label)

                valid_losses[batch] = loss.item()
            
            valid_loss = torch.mean(valid_losses)
            avg_valid_losses[epoch] = valid_loss
        
        # decide whether to stop or not based on validation loss
        early_stopping(valid_loss, model, optimizer, epoch) 
        if early_stopping.early_stop:
            logging.info('Early stopping')
            break
            
    logging.info('Training end')
    return model, avg_train_losses.tolist(), avg_valid_losses.tolist()


def evalulate_model_CAM(model, test_loader, num_data, explainECs, device):
    model.eval() # training session with train dataset
    with torch.no_grad():
        y_pred = torch.zeros([num_data, len(explainECs)]).to(device)
        y_true = torch.zeros([num_data, len(explainECs)]).to(device)
        logging.info('Prediction starts on test dataset')
        cnt = 0
        for batch, (data, label) in enumerate(test_loader):
            data = data.type(torch.FloatTensor).to(device)
            label = label.type(torch.FloatTensor).to(device)
            output, _ = model(data)
            prediction = output > 0.5
            prediction = prediction.float()

            y_pred[cnt:cnt+data.shape[0]] = prediction
            y_true[cnt:cnt+data.shape[0]] = label
            cnt += data.shape[0]
        logging.info('Prediction Ended on test dataset')

        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

    logging.info(f'Precision: {precision}\tRecall: {recall}\tF1: {f1}')
    return y_true, y_pred