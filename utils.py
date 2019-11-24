import logging
import re
# import
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
            logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        ckpt = {'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'best_acc':self.best_score,
                'epoch':epoch}
        torch.save(ckpt, self.save_name)
        self.val_loss_min = val_loss


# plot the accuracy and loss value of each model.
###################
def draw(avg_train_losses, avg_valid_losses, file_name='loss_fig.png'):
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
    plt.savefig(f'./{file_name}', dpi=600)
    plt.show()
    return



# training the model using training and validation set
def train_model(model, optimizer, criterion, device,
               batch_size, patience, n_epochs, 
               train_loader, valid_loader, save_name='checkpoint.pt'):
    early_stopping = EarlyStopping(
                                save_name=save_name, 
                                patience=patience, 
                                verbose=True)
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    
    for epoch in range(n_epochs):
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
            train_losses.append(loss.item())    
        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)

        model.eval() # validation session with validation dataset
        with torch.no_grad():
            for data, label in valid_loader:
                data = data.type(torch.FloatTensor)
                label = label.type(torch.FloatTensor)
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                loss = criterion(output, label)
                valid_losses.append(loss.item())
            valid_loss = np.average(valid_losses)
            avg_valid_losses.append(valid_loss)
        
        train_losses = []
        valid_losses = []
        
        # decide whether to stop or not based on validation loss
        early_stopping(valid_loss, model, optimizer, epoch) 
        if early_stopping.early_stop:
            logging.info('Early stopping')
            break
            
    return model, avg_train_losses, avg_valid_losses


def train_model_multitask(model, optimizer, criterion, device,
                        batch_size, patience, n_epochs, 
                        train_loader, valid_loader, 
                        save_name='checkpoint.pt', alpha=10):

    early_stopping = EarlyStopping(
                                save_name=save_name, 
                                patience=patience, 
                                verbose=True)
    train_losses = []
    train_losses_1 = []
    train_losses_2 = []

    valid_losses = []
    valid_losses_1 = []
    valid_losses_2 = []

    avg_train_losses = []
    avg_train_losses_1 = []
    avg_train_losses_2 = []

    avg_valid_losses = []
    avg_valid_losses_1 = []
    avg_valid_losses_2 = []
    
    logging.info('Training start')
    for epoch in range(n_epochs):
        model.train() # training session with train dataset
        for batch, (data, label1, label2) in enumerate(train_loader):
            data = data.type(torch.FloatTensor)
            label1 = label1.type(torch.FloatTensor)
            label2 = label2.type(torch.FloatTensor)
            data = data.to(device)
            label1 = label1.to(device)
            label2 = label2.to(device)
            optimizer.zero_grad()
            output1, output2 = model(data)
            loss1 = criterion(output1, label1)
            loss2 = criterion(output2, label2)
            loss = alpha*loss1 + loss2
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_losses_1.append(loss1.item())
            train_losses_2.append(loss2.item())
        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)
        train_loss_1 = np.average(train_losses_1)
        avg_train_losses_1.append(train_loss)
        train_loss_2 = np.average(train_losses_2)
        avg_train_losses_1.append(train_loss)

        model.eval() # validation session with validation dataset
        with torch.no_grad():
            for data, label1, label2 in valid_loader:
                data = data.type(torch.FloatTensor)
                label1 = label1.type(torch.FloatTensor)
                label2 = label2.type(torch.FloatTensor)
                data = data.to(device)
                label1 = label1.to(device)
                label2 = label2.to(device)
                optimizer.zero_grad()
                output1, output2 = model(data)
                loss1 = criterion(output1, label1)
                loss2 = criterion(output2, label2)
                loss = alpha*loss1 + loss2

                valid_losses.append(loss.item())
                valid_losses_1.append(loss1.item())
                valid_losses_2.append(loss2.item())

            valid_loss = np.average(valid_losses)
            avg_valid_losses.append(valid_loss)
            valid_loss_1 = np.average(valid_losses_1)
            avg_valid_losses_1.append(valid_loss_1)
            valid_loss_2 = np.average(valid_losses_2)
            avg_valid_losses_2.append(valid_loss_2)
        
        train_losses = []
        train_losses_1 = []
        train_losses_2 = []
        valid_losses = []
        valid_losses_1 = []
        valid_losses_2 = []
        
        # decide whether to stop or not based on validation loss
        early_stopping(valid_loss, model, optimizer, epoch) 
        if early_stopping.early_stop:
            logging.info('Early stopping')
            break
            
    logging.info('Training end')
    return model, avg_train_losses, avg_valid_losses, avg_train_losses_1, avg_valid_losses_1, avg_train_losses_2, avg_valid_losses_2


def evalulate_model(model, test_loader, device):
    model.eval() # training session with train dataset
    with torch.no_grad():
        y_true = torch.Tensor().to(device)
        y_pred = torch.Tensor().to(device)
        for batch, (data, label) in enumerate(test_loader):
            data = data.type(torch.FloatTensor)
            label = label.type(torch.FloatTensor)
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            prediction = output > 0.5
            prediction = prediction.float()

            y_true = torch.cat((y_true, label))
            y_pred = torch.cat((y_pred, prediction))

        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

    logging.info(f'Precision: {precision}\tRecall: {recall}\tF1: {f1}')
    return precision, recall, f1


def evalulate_model_multitask(model, test_loader, device, alpha=10):
    model.eval()
    with torch.no_grad():
        y_pred1 = torch.Tensor().to(device)
        y_pred2 = torch.Tensor().to(device)
        for batch, (data, label1, label2) in enumerate(test_loader):
            data = data.type(torch.FloatTensor)
            label1 = label1.type(torch.FloatTensor)
            label2 = label2.type(torch.FloatTensor)
            data = data.to(device)
            label1 = label1.to(device)
            label2 = label2.to(device)

            output1, output2 = model(data)
            prediction1 = output1 > 0.5
            prediction1 = prediction1.float()
            prediction2 = output2 > 0.5
            prediction2 = prediction2.float()
            
            y_pred1 = torch.cat((y_pred1, prediction1))
            y_pred2 = torch.cat((y_pred1, prediction2))

        logging.info('Prediction Ended on test dataset')
        ec2ec_map = getEC32EC4map(explainECs_short, explainECs).to(device)
        prediction = getCommonECs(y_pred1, y_pred2, ec2ec_map, device)
        prediction = prediction.cpu().numpy()
        logging.info('Got common ECs from the prediction')

        y_pred1 = 0
        y_pred2 = 0

        y_true = torch.Tensor().to(device)
        for _, label in test_loader:
            label = label.type(torch.FloatTensor).to(device)
            y_true = torch.cat((y_true, label))
        y_true = y_true.cpu()
        logging.info('Label were collected in a single tensor')

        precision = precision_score(y_true, prediction, average='macro')
        recall = recall_score(y_true, prediction, average='macro')
        f1 = f1_score(y_true, prediction, average='macro')

        logging.info(f'Precision: {precision}\tRecall: {recall}\tF1: {f1}')
    return precision, recall, f1


def evalulate_deepEC(model1, model2, test_loader, explainECs, explainECs_short, device):
    with torch.no_grad():
        model1.eval() # CNN2
        model2.eval() # CNN3
        y_pred1 = torch.Tensor().to(device)
        y_pred2 = torch.Tensor().to(device)
        logging.info('Prediction starts on test dataset')
        for batch, (data, _) in enumerate(test_loader):
            data = data.type(torch.FloatTensor)
            data = data.to(device)
            output1 = model1(data)
            output2 = model2(data)

            prediction1 = output1 > 0.5
            prediction1 = prediction1.float()
            prediction2 = output2 > 0.5
            prediction2 = prediction2.float()

            y_pred1 = torch.cat((y_pred1, prediction1))
            y_pred2 = torch.cat((y_pred2, prediction2))


        logging.info('Prediction Ended on test dataset')
        ec2ec_map = getEC32EC4map(explainECs_short, explainECs).to(device)
        prediction = getCommonECs(y_pred1, y_pred2, ec2ec_map, device)
        prediction = prediction.cpu().numpy()
        logging.info('Got common ECs from the prediction')

        y_pred1 = None
        y_pred2 = None
        
        y_true = torch.Tensor().to(device)
        for _, label in test_loader:
            label = label.type(torch.FloatTensor).to(device)
            y_true = torch.cat((y_true, label))
        y_true = y_true.cpu()
        logging.info('Label were collected in a single tensor')

        precision = precision_score(y_true, prediction, average='macro')
        recall = recall_score(y_true, prediction, average='macro')
        f1 = f1_score(y_true, prediction, average='macro')

        logging.info(f'Precision: {precision}\tRecall: {recall}\tF1: {f1}')
    return precision, recall, f1


def getEC32EC4map(explainECs_short, explainECs):
    result = torch.zeros((len(explainECs), len(explainECs_short)))
    for ec4_ind, ec4 in enumerate(explainECs):
        tmp = torch.zeros(len(explainECs_short))
        for i, ec3 in enumerate(explainECs_short):
            if ec4.startswith(ec3):
                tmp[i] = 1
        result[ec4_ind] = tmp
    return result


def getCommonECs(ec3_pred, ec4_pred, ec2ec_map, device):
    common_pred = torch.zeros(ec4_pred.shape).to(device)
    for i in range(len(ec4_pred)):
        ec4_activemap = torch.matmul(ec2ec_map, ec3_pred[i])
        common_EC = ec4_activemap * ec4_pred[i]
        common_pred[i] = common_EC
    return common_pred


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