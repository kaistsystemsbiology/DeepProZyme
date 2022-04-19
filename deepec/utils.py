import logging
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# import torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F


def argument_parser(version=None):
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--seq_file', required=False, 
                        default='./Dataset/uniprot_dataset.fa', help='Sequence data')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory')
    parser.add_argument('-e', '--epoch', required=False, type=int,
                        default=30, help='Total epoch number')
    parser.add_argument('-b', '--batch_size', required=False, type=int,
                        default=32, help='Batch size')
    parser.add_argument('-r', '--learning_rate', required=False, type=float,
                        default=1e-3, help='Learning rate')
    parser.add_argument('-gamma', '--gamma', required=False, type=float,
                        default=1.0, help='Focal loss gamma')
    parser.add_argument('-p', '--patience', required=False, type=int,
                        default=5, help='Patience limit for early stopping')
    parser.add_argument('-g', '--gpu', required=False, 
                        default='cuda:0', help='Specify gpu')
    parser.add_argument('-cpu', '--cpu_num', required=False, type=int,
                        default=4, help='Number of cpus to use')  
    parser.add_argument('-ckpt', '--checkpoint', required=False, 
                        default='checkpoint.pt', help='Checkpoint file')
    parser.add_argument('-l', '--log_dir', required=False, 
                        default='CNN_training.log', help='Log file directory')
    parser.add_argument('-third', '--third_level', required=False, type=boolean_string,
                        default=False, help='Predict upto third EC level')      
    return parser



# plot the accuracy and loss value of each model.
###################
def draw(avg_train_losses, avg_valid_losses, output_dir, file_name='CNN_loss_fig.png'):
    fig = plt.figure(figsize=(9,6))

    avg_train_losses = np.array(avg_train_losses)
    avg_train_losses = avg_train_losses[avg_train_losses.nonzero()]
    avg_valid_losses = np.array(avg_valid_losses)
    avg_valid_losses = avg_valid_losses[avg_valid_losses.nonzero()]
    min_position = avg_valid_losses.argmin()+1

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
            fp.write(f'{cnt}\t{train_loss:0.12f}\t{valid_loss:0.12f}\n')
    return


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha==None:
            self.alpha=1
        else:
            self.alpha = torch.Tensor(alpha).view(-1, 1)
        
    def forward(self, pred, label):
        BCE_loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()


class DeepECConfig():
    def __init__(self,
                 model = None,
                 optimizer = None,
                 criterion = None,
                 scheduler = None,
                 n_epochs = 50,
                 device = 'cpu',
                 patience = 5,
                 save_name = './deepec.log',
                 train_source = None,
                 val_source = None, 
                 test_source = None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        self.device = device
        self.patience = patience
        self.save_name = save_name
        self.train_source = train_source
        self.val_source = val_source
        self.test_source = test_source




def run_neural_net(model, proteinDataloader, pred_thrd, device):
        num_data = len(proteinDataloader.dataset)
        num_ecs = len(proteinDataloader.dataset.map_EC)
        pred_thrd = pred_thrd.to(device)
        model.eval() # training session with train dataset
        with torch.no_grad():
            y_pred = torch.zeros([num_data, num_ecs])
            y_score = torch.zeros([num_data, num_ecs])
            logging.info('Deep leanrning prediction starts on the dataset')
            cnt = 0
            for batch, data in enumerate(tqdm(proteinDataloader)):
                inputs = {key:val.to(device) for key, val in data.items()}
                output = model(**inputs)
                output = torch.sigmoid(output)
                prediction = output > pred_thrd
                prediction = prediction.float()
                step = data['input_ids'].shape[0]
                y_pred[cnt:cnt+step] = prediction.cpu()
                y_score[cnt:cnt+step] = output.cpu()
                cnt += step
            logging.info('Deep learning prediction ended on test dataset')
        return y_pred, y_score


def save_dl_result(y_pred, y_score, input_ids, explainECs, output_dir):
    failed_cases = []
    with open(f'{output_dir}/DL_prediction_result.txt', 'w') as fp:
        fp.write('sequence_ID\tprediction\tscore\n')
        for i, ith_pred in enumerate(y_pred):
            nonzero_preds = torch.nonzero(ith_pred, as_tuple=False)
            if len(nonzero_preds) == 0:
                fp.write(f'{input_ids[i]}\tNone\t0.0\n')
                failed_cases.append(input_ids[i])
                continue
            for j in nonzero_preds:
                pred_ec = explainECs[j]
                pred_score = y_score[i][j].item()
                fp.write(f'{input_ids[i]}\t{pred_ec}\t{pred_score:0.4f}\n')
    return failed_cases