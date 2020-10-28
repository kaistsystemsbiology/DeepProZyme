import math
# import torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN0(nn.Module):
    '''
    Use second level convolution.
    channel size: 4 -> 16 
                  8 -> 12
                  16 -> 4
    '''
    def __init__(self):
        super(CNN0, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
           
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,21))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,21))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,21))

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


class CNN1(nn.Module):
    '''
    Use second level convolution.
    channel size: 4 -> 16 
                  8 -> 12
                  16 -> 4
    '''
    def __init__(self):
        super(CNN1, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
           
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,10))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,10))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,10))

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


class CNN2(nn.Module):
    '''
    Use second level convolution.
    channel size: 4 -> 16 
                  8 -> 12
                  16 -> 4
    '''
    def __init__(self):
        super(CNN2, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
           
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,21))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,21))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,21))

        self.conv1_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(16,1))
        self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(12,1))
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,1))

        self.conv = nn.Conv2d(in_channels=128*3, out_channels=128*3, kernel_size=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(982,1), stride=1)

        
    def forward(self, x):
        x1 = self.dropout(self.relu(self.conv1(x)))
        x2 = self.dropout(self.relu(self.conv2(x)))
        x3 = self.dropout(self.relu(self.conv3(x)))
        x1 = self.dropout(self.relu(self.conv1_1(x1)))
        x2 = self.dropout(self.relu(self.conv2_1(x2)))
        x3 = self.dropout(self.relu(self.conv3_1(x3)))

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.relu(self.conv(x))
        x = self.pool(x)
        return x


class DeepECv2_2(nn.Module):
    def __init__(self, out_features):
        super(DeepECv2_2, self).__init__()
        self.explainECs = out_features
        self.embedding = nn.Embedding(21, 10) # 20 AA + X + blank
        self.cnn0 = CNN1()
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
        x = self.embedding(x)
        x = self.cnn0(x)
        x = x.view(-1, 128*3)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.out_act(self.bn2(self.fc2(x)))
        return x



class DeepECv2_3(nn.Module):
    def __init__(self, out_features):
        super(DeepECv2_3, self).__init__()
        self.explainECs = out_features
        # self.embedding = nn.Embedding(1000, 21) # 20 AA + X + blank
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
        # x = self.out_act(x)
        return x


class DeepECv2_4(nn.Module):
    def __init__(self, out_features):
        super(DeepECv2_4, self).__init__()
        self.explainECs = out_features
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
           
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,21))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,21))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,21))

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

        self.fc1 = nn.Linear(in_features=128*3, out_features=len(out_features))

        self.out_act = nn.Sigmoid()
        self.init_weights()

        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


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

        x = x.view(-1, 128*3)
        return self.out_act(self.fc1(x))


class DeepECv2_5(nn.Module):
    def __init__(self, out_features):
        super(DeepECv2_5, self).__init__()
        self.explainECs = out_features
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,21))
        self.batchnorm1 = nn.BatchNorm2d(num_features=128)
           
        self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,1))
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(8,1))
        self.conv2_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(16,1))

        self.batchnorm2_1 = nn.BatchNorm2d(num_features=128)
        self.batchnorm2_2 = nn.BatchNorm2d(num_features=128)
        self.batchnorm2_3 = nn.BatchNorm2d(num_features=128)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(16,1))
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(12,1))
        self.conv3_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,1))

        self.batchnorm3_1 = nn.BatchNorm2d(num_features=128)
        self.batchnorm3_2 = nn.BatchNorm2d(num_features=128)
        self.batchnorm3_3 = nn.BatchNorm2d(num_features=128)

        self.conv = nn.Conv2d(in_channels=128*3, out_channels=128*3, kernel_size=(1,1))
        self.batchnorm = nn.BatchNorm2d(num_features=128*3)

        self.pool = nn.MaxPool2d(kernel_size=(979,1), stride=1)

        self.fc1 = nn.Linear(in_features=128*3, out_features=512)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=len(out_features))
        self.bn2 = nn.BatchNorm1d(num_features=len(out_features))
        

        self.out_act = nn.Sigmoid()
        self.init_weights()

        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


    def forward(self, x):
        x = self.dropout(self.relu(self.batchnorm1(self.conv1(x))))
        x1 = self.dropout(self.relu(self.batchnorm2_1(self.conv2_1(x))))
        x2 = self.dropout(self.relu(self.batchnorm2_2(self.conv2_2(x))))
        x3 = self.dropout(self.relu(self.batchnorm2_3(self.conv2_3(x))))
        x1 = self.dropout(self.relu(self.batchnorm3_1(self.conv3_1(x1))))
        x2 = self.dropout(self.relu(self.batchnorm3_2(self.conv3_2(x2))))
        x3 = self.dropout(self.relu(self.batchnorm3_3(self.conv3_3(x3))))

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.relu(self.batchnorm(self.conv(x)))
        x = self.pool(x)

        x = x.view(-1, 128*3)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return self.out_act(x)


        
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

        self.embedding = nn.Embedding(21, 10) # 20 AA + X + blank
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


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
           
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,21))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,21))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,21))

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

        self.init_weights()

        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        
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


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, explainProts=[]):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.explainProts = explainProts
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.cnn = CNN()
        self.fc1 = nn.Linear(128*3, 512)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=len(explainProts))
        self.relu = nn.ReLU()

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        

    def forward(self, src):
        src = src.type(torch.long)
        if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, self.src_mask)
        x = self.decoder(x)
        bs, length, ntkn = x.size()
        x = x.view(-1, 1, length, ntkn)
        x = self.cnn(x)
        x = x.view(-1, 128*3)
        x = self.relu(self.bn1(self.fc1(x)))
        return self.fc2(x)


class TransformerModule(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModule, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder1 = nn.Linear(ninp, ntoken)
        self.decoder2 = nn.Linear(ntoken, 1)
        self.relu = nn.ReLU()
        
        self.init_weights()

        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data.uniform_(-initrange, initrange)
        self.decoder2.bias.data.zero_()
        self.decoder2.weight.data.uniform_(-initrange, initrange)

            
    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, self.src_mask)
        x = self.relu(self.decoder1(x))
        x = self.decoder2(x)
        bs, length, ntkn = x.size()
        x = x.view(-1, length*ntkn)
        return x


class HybridModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, explainProts=[]):
        super(HybridModel, self).__init__()
        self.explainProts = explainProts
        self.transformer = TransformerModule(ntoken, ninp, nhead, nhid, nlayers, dropout=0.5)
        self.convolution = CNN()
        self.linear = nn.Linear(in_features=128*3, out_features=1000)
        self.bn = nn.BatchNorm1d(num_features=1000)
        self.relu = nn.ReLU()
        self.outlayer = nn.Linear(1000, len(explainProts))

        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x_onehot, x_emb):
        x  = self.convolution(x_onehot)
        x = x.view(-1, 128*3)
        x = self.relu(self.bn(self.linear(x)))
        x += self.transformer(x_emb)
        return self.outlayer(x)