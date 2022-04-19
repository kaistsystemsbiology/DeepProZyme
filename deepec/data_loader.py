import re
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer


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

        if self.pred:
            return sample
            
        labels = self.data_Y[idx]
        labels = self.convert2onehot_EC(labels)
        labels = labels.reshape(-1)
        sample['labels'] = torch.tensor(labels)
        return sample