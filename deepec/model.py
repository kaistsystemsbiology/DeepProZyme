import math
import copy
from typing import Optional, Any
# import torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import BertModel, BertForSequenceClassification


class CNN0(nn.Module):
    '''=
    Use second level convolution.
    channel size: 4 -> 16 
                  8 -> 12
                  16 -> 4
    '''
    def __init__(self):
        super(CNN0, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
           
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,20))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,20))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,20))

        self.batchnorm1 = nn.BatchNorm2d(num_features=128)
        self.batchnorm2 = nn.BatchNorm2d(num_features=128)
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)

        self.conv1_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(16,1))
        self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(12,1))
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,1))

        self.batchnorm1_1 = nn.BatchNorm2d(num_features=128)
        self.batchnorm2_1 = nn.BatchNorm2d(num_features=128)
        self.batchnorm3_1 = nn.BatchNorm2d(num_features=128)

        self.conv = nn.Conv2d(in_channels=128*3, out_channels=128*3, kernel_size=(1,1))
        self.batchnorm = nn.BatchNorm2d(num_features=128*3)
        self.pool = nn.MaxPool2d(kernel_size=(982,1), stride=1)

        
    def forward(self, x):
        x1 = self.dropout(self.relu(self.batchnorm1(self.conv1(x))))
        x2 = self.dropout(self.relu(self.batchnorm2(self.conv2(x))))
        x3 = self.dropout(self.relu(self.batchnorm3(self.conv3(x))))
        x1 = self.dropout(self.relu(self.batchnorm1_1(self.conv1_1(x1))))
        x2 = self.dropout(self.relu(self.batchnorm2_1(self.conv2_1(x2))))
        x3 = self.dropout(self.relu(self.batchnorm3_1(self.conv3_1(x3))))

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.relu(self.batchnorm(self.conv(x)))
        x = self.pool(x)
        return x


class DeepECv2(nn.Module):
    def __init__(self, out_features):
        super(DeepECv2, self).__init__()
        self.explainECs = out_features
        self.cnn0 = CNN0()
        self.fc1 = nn.Linear(in_features=128*3, out_features=512)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=len(out_features))
        self.bn2 = nn.BatchNorm1d(num_features=len(out_features))
        self.relu = nn.ReLU()
        self.out_act = nn.Sigmoid()
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        
    def forward(self, x):
        x = self.cnn0(x)
        x = x.view(-1, 128*3)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return x


        
class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 1),
                               stride=stride, padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

        
class DeepEC_emb(nn.Module):
    def __init__(self, explainECs, num_blocks, block=Bottleneck):
        super(DeepEC_emb, self).__init__()
        self.explainECs = explainECs
        self.in_planes = 64 

        self.embedding = nn.Embedding(20, 10) # 20 AA + X + blank
        self.conv = nn.Conv2d(1, 64, kernel_size=(3, 10),
                              stride=1, padding=(1, 0), bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)
        self.linear = nn.Linear(512*block.expansion, len(explainECs))
        self.pool = nn.MaxPool2d(kernel_size=(1000-3, 1))
        self.relu = nn.ReLU()

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

        
    def forward(self, x):
        x = self.embedding(x.type(torch.long))
        x = self.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class DeepEC(nn.Module):
    def __init__(self, out_features):
        super(DeepEC, self).__init__()
        self.explainECs = out_features
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,20))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,20))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,20))
        self.batchnorm1 = nn.BatchNorm2d(num_features=128)
        self.batchnorm2 = nn.BatchNorm2d(num_features=128)
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)
        self.pool1 = nn.MaxPool2d(kernel_size=(1000-4+1,1), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(1000-8+1,1), stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(1000-16+1,1), stride=1)

        self.fc1 = nn.Linear(in_features=128*3, out_features=512)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=len(out_features))
        self.bn2 = nn.BatchNorm1d(num_features=len(out_features))
        self.relu = nn.ReLU()
        self.out_act = nn.Sigmoid()
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        
    def forward(self, x):
        x1 = self.relu(self.batchnorm1(self.conv1(x)))
        x2 = self.relu(self.batchnorm2(self.conv2(x)))
        x3 = self.relu(self.batchnorm3(self.conv3(x)))
        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        x = x.view(-1, 128*3)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class Embeddings(nn.Module):
    def __init__(self, ntoken, ninp):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp 
    
    def forward(self, src):
        return self.emb(src) * math.sqrt(self.ninp)
    
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)
    
    
    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

        
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src2, weight = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weight
    

class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        
    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = src
        weights = []

        for mod in self.layers:
            output, weight = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            weights.append(weight)
        if self.norm is not None:
            output = self.norm(output)

        return output, weights

class DeepTransformer(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, explainProts=[]):
        super(DeepTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.seq_len = 1000
        self.explainProts = explainProts
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = Embeddings(ntoken, ninp)
        self.ninp = ninp

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128*1, kernel_size=(4, ninp), stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=128*1)
        self.conv2 = nn.Conv2d(in_channels=128*1, out_channels=128*3, kernel_size=(4, 1), stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=128*3)

        self.pool = nn.MaxPool2d(kernel_size=(self.seq_len-4+1-4+1, 1))
        self.fc2 = nn.Linear(128*3, len(explainProts))

        self.relu = nn.ReLU()

        self.init_weights()


    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask):
        src = src.type(torch.long).T
        src = self.encoder(src)
        src = self.pos_encoder(src)
        src, _ = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        src.transpose_(0, 1)
        src = src.view(-1, 1, self.seq_len, self.ninp)
        src = self.relu(self.bn1(self.conv1(src)))
        src = self.relu(self.bn2(self.conv2(src)))
        src = self.pool(src).view(-1, 128*3)
        return self.fc2(src)



class DeepEC2(nn.Module):
    def __init__(self, out_features):
        super(DeepEC2, self).__init__()
        self.explainECs = out_features
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5,20))
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5,1))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5,1))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5,1))
        self.bn1 = nn.BatchNorm2d(num_features=128)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.pool1 = nn.MaxPool2d(kernel_size=(4,1), stride=3)
        self.pool2 = nn.MaxPool2d(kernel_size=(4,1), stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=(3,1), stride=3)
        self.pool4 = nn.MaxPool2d(kernel_size=(4,1), stride=3)

        self.fc1 = nn.Linear(in_features=128*9, out_features=512)
        self.bn5 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=len(out_features))
        self.bn6 = nn.BatchNorm1d(num_features=len(out_features))
        self.relu = nn.ReLU()
        self.out_act = nn.Sigmoid()
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        
    def forward(self, x):
        x = self.relu(self.pool1(self.bn1(self.conv1(x))))        
        x = self.relu(self.pool2(self.bn2(self.conv2(x))))
        x = self.relu(self.pool3(self.bn3(self.conv3(x))))
        x = self.relu(self.pool4(self.bn4(self.conv4(x))))
        x = x.view(-1, 128*9)
        x = self.relu(self.bn5(self.fc1(x)))
        x = self.bn6(self.fc2(x))
        return x


class DeepEC3(nn.Module):
    def __init__(self, out_features, dropout=0.1):
        super(DeepEC3, self).__init__()
        self.explainECs = out_features
        self.embed = nn.Embedding(num_embeddings=21, embedding_dim=32)
        self.pos_encoder = PositionalEncoding(d_model=32, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5,32))
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5,1))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5,1))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5,1))
        self.bn1 = nn.BatchNorm2d(num_features=128)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.pool1 = nn.MaxPool2d(kernel_size=(4,1), stride=3)
        self.pool2 = nn.MaxPool2d(kernel_size=(4,1), stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=(3,1), stride=3)
        self.pool4 = nn.MaxPool2d(kernel_size=(4,1), stride=3)

        self.fc1 = nn.Linear(in_features=128*9, out_features=512)
        self.bn5 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=len(out_features))
        self.bn6 = nn.BatchNorm1d(num_features=len(out_features))
        self.relu = nn.ReLU()
        self.out_act = nn.Sigmoid()
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        
    def forward(self, x):
        x = self.embed(x.type(torch.long))
        x = self.pos_encoder(x)
        bs, length, emb_dim = x.size()
        x = x.view(-1, 1, length, emb_dim)
        x = self.pool1(self.dropout(self.relu(self.bn1(self.conv1(x)))))
        x = self.pool2(self.dropout(self.relu(self.bn2(self.conv2(x)))))
        x = self.pool3(self.dropout(self.relu(self.bn3(self.conv3(x)))))
        x = self.pool4(self.dropout(self.relu(self.bn4(self.conv4(x)))))
        x = x.view(-1, 128*9)
        x = self.relu(self.bn5(self.fc1(x)))
        x = self.bn6(self.fc2(x))
        return x


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class InceptionEC(nn.Module):
    def __init__(self, out_features, *out_channels):
        super(InceptionEC, self).__init__()
        self.explainECs = out_features
        in_channels = 1
        out_channels = out_channels[0]
        self.branch1x1 = BasicConv(in_channels, out_channels[0], kernel_size=(1,20))
        self.branch3x1_1 = BasicConv(in_channels, out_channels[1], kernel_size=(1,20))
        self.branch3x1_2 = BasicConv(out_channels[1], out_channels[2], kernel_size=(5,1), padding=(2,0))
        self.branch5x1_1 = BasicConv(in_channels, out_channels[3], kernel_size=(1,20))
        self.branch5x1_2 = BasicConv(out_channels[3], out_channels[4], kernel_size=(15,1), padding=(7,0))
        self.branch_pool_1 = nn.AvgPool2d(kernel_size=(3,20), padding=(1,0), stride=1)
        self.branch_pool_2 = BasicConv(in_channels, out_channels[5], kernel_size=(1,1))
        
        num_channels = out_channels[0]+out_channels[2]+out_channels[4]+out_channels[5]
        self.pool = nn.MaxPool2d(kernel_size=(1000, 1))
        self.fc = nn.Linear(num_channels, len(out_features))


    def forward(self, x):
        x1 = self.branch1x1(x)
        x2 = self.branch3x1_1(x)
        x2 = self.branch3x1_2(x2)
        x3 = self.branch5x1_1(x)
        x3 = self.branch5x1_2(x3)
        x4 = self.branch_pool_1(x)
        x4 = self.branch_pool_2(x4)
        x = torch.cat([x1, x2, x3, x4], 1)
        x = self.pool(x)
        bs, num_channels, _, _ = x.size()
        x = x.view(-1, num_channels)
        return self.fc(x)



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