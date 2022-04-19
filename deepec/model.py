import torch.nn as nn
from transformers import BertModel, BertForSequenceClassification


class ProtBertConvEC(BertForSequenceClassification):
    def __init__(self, config, out_features=[], fc_gamma=0):
        super(ProtBertConvEC, self).__init__(config)
        self.explainECs = out_features
        self.fc_alpha = 1
        self.fc_gamma = fc_gamma
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, len(out_features))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4, config.hidden_size))
        self.bn1 = nn.BatchNorm2d(num_features=128)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1))
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.pool1 = nn.MaxPool2d(kernel_size=(1000-4+1-4+1, 1), stride=1)
        self.relu = nn.ReLU()
        
        nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0)
        for conv_layer in [self.conv1, self.conv2]:
            nn.init.xavier_uniform_(conv_layer.weight)
            conv_layer.bias.data.fill_(0)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        x, _ = self.bert(input_ids, token_type_ids, attention_mask)
        bs, seq_len, hidden = x.size()
        x = x.view(bs, 1, seq_len, hidden)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.pool1(x)
        x = x.view(-1, hidden)
        logits = self.classifier(x)
        return logits
