# import torch packages
import torch
from torch.utils.data import Dataset

import numpy as np


# create a data loading class for the use of pytorch
# I put the one hot encoding in here
class ECDataset(Dataset):
    def __init__(self, data_X, data_Y, explainECs, pred=False):
        self.pred = pred
        self.data_X = data_X
        self.map_AA = self.getAAmap()

        if not pred:
            self.data_Y = data_Y
            self.map_EC = self.getECmap(explainECs)
        
    
    def __len__(self):
        return len(self.data_X)
    
    
    def getAAmap(self):
        aa_vocab = ['A', 'C', 'D', 'E', 
                    'F', 'G', 'H', 'I', 
                    'K', 'L', 'M', 'N', 
                    'P', 'Q', 'R', 'S',
                    'T', 'V', 'W', 'X', 
                    'Y']
        map = {}
        for i, char in enumerate(aa_vocab):
            baseArray = np.zeros(len(aa_vocab))
            baseArray[i] = 1
            map[char] = baseArray
        return map
    
    
    def getECmap(self, explainECs):
        ec_vocab = list(set(explainECs))
        ec_vocab.sort()
        map = {}
        for i, ec in enumerate(ec_vocab):
            baseArray = np.zeros(len(ec_vocab))
            baseArray[i] = 1
            map[ec] = baseArray
        return map


    def convert2onehot_seq(self, single_seq, max_seq=1000):
        single_onehot = np.zeros((max_seq, len(self.map_AA)))
        for i, x in enumerate(single_seq):
            single_onehot[i] = np.asarray(self.map_AA[x])
        return single_onehot
    
    
    def convert2onehot_EC(self, EC):
        map_EC = self.map_EC
        single_onehot = np.zeros(len(map_EC))
        for each_EC in EC:
            single_onehot += map_EC[each_EC]
        return single_onehot
    
    
    def __getitem__(self, idx):
        x = self.data_X[idx]
        x = self.convert2onehot_seq(x)
        x = x.reshape((1,) + x.shape)

        if self.pred:
            return x

        y = self.data_Y[idx]
        y = self.convert2onehot_EC(y)
        y = y.reshape(-1)
        return x, y


# create a data loading class for the use of pytorch
# I put the one hot encoding in here
class EnzymeDataset(Dataset):
    def __init__(self, data_X, data_Y):
        self.getAAmap()
        self.data_X = data_X
        self.data_Y = data_Y
        
    
    def __len__(self):
        return len(self.data_X)
    
    
    def getAAmap(self):
        aa_vocab = ['A', 'C', 'D', 'E', 
                    'F', 'G', 'H', 'I', 
                    'K', 'L', 'M', 'N', 
                    'P', 'Q', 'R', 'S',
                    'T', 'V', 'W', 'X', 
                    'Y', '_']
        map = {}
        for i, char in enumerate(aa_vocab):
            baseArray = np.zeros(len(aa_vocab)-1)
            if char != '_':
                baseArray[i] = 1
            map[char] = baseArray
        self.map = map
        return

        
    def convert2onehot(self, single_seq):
        single_onehot = []
        for x in single_seq:
            single_onehot.append(self.map[x])
        return np.asarray(single_onehot)
    
    
    def __getitem__(self, idx):
        x = self.data_X[idx]
        x = self.convert2onehot(x)
        y = self.data_Y[idx]
        
        x = x.reshape((1,) + x.shape)
        y = y.reshape(-1)
        return x, y


class ECDataset_multitask(Dataset):
    def __init__(self, data_X, \
                 data_Y_4, data_Y_3, \
                 explainECs, explainECs_short):
        self.data_X = data_X
        self.data_Y_4 = data_Y_4
        self.data_Y_3 = data_Y_3
        self.map_AA = self.getAAmap()
        self.map_EC_4 = self.getECmap(explainECs)
        self.map_EC_3 = self.getECmap(explainECs_short)
        
    
    def __len__(self):
        return len(self.data_X)
    
    
    def getAAmap(self):
        aa_vocab = ['A', 'C', 'D', 'E', 
                    'F', 'G', 'H', 'I', 
                    'K', 'L', 'M', 'N', 
                    'P', 'Q', 'R', 'S',
                    'T', 'V', 'W', 'X', 
                    'Y', '_']
        map = {}
        for i, char in enumerate(aa_vocab):
            baseArray = np.zeros(len(aa_vocab)-1)
            if char != '_':
                baseArray[i] = 1
            map[char] = baseArray
        return map
    
    
    def getECmap(self, explainECs):
        ec_vocab = list(set(explainECs))
        ec_vocab.sort()
        map = {}
        for i, ec in enumerate(ec_vocab):
            baseArray = np.zeros(len(ec_vocab))
            baseArray[i] = 1
            map[ec] = baseArray
        return map


    def convert2onehot_seq(self, single_seq):
        single_onehot = []
        for x in single_seq:
            single_onehot.append(self.map_AA[x])
        return np.asarray(single_onehot)
    
    
    def convert2onehot_EC(self, EC, map_EC):
        single_onehot = np.zeros(len(map_EC))
        for each_EC in EC:
            single_onehot += map_EC[each_EC]
        return single_onehot
    
    
    def __getitem__(self, idx):
        x = self.data_X[idx]
        x = self.convert2onehot_seq(x)

        y1 = self.data_Y_3[idx]
        y1 = self.convert2onehot_EC(y1, self.map_EC_3)

        y2 = self.data_Y_4[idx]
        y2 = self.convert2onehot_EC(y2, self.map_EC_4)
        
        x = x.reshape((1,) + x.shape)
        y1 = y1.reshape(-1)
        y2 = y2.reshape(-1)
        return x, y1, y2