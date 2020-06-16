# import torch packages
import torch
import torch.nn as nn


class DeepECv2(nn.Module):
    def __init__(self, out_features):
        super(DeepECv2, self).__init__()
        self.explainECs = out_features
        # self.embedding = nn.Embedding(1000, 21) # 20 AA + X + blank
        self.cnn0 = CNN0()
        self.fc = nn.Linear(in_features=128*3, out_features=len(out_features))
        self.bn1 = nn.BatchNorm1d(num_features=len(out_features))
        self.out_act = nn.Sigmoid()
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        
    def forward(self, x):
        # x = self.embedding(x)
        x = self.cnn0(x)
        x = x.view(-1, 128*3)
        x = self.out_act(self.bn1(self.fc(x)))
        return x

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
        self.dropout = nn.Dropout(p=0.1)
           
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


class BasicBlock(nn.Module):
    def __init__(self, in_channels):
        super(BasicBlock, self).__init__()
        
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3,1), padding=(1,0))
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(3,1), padding=(1,0))
        self.bn2 = nn.BatchNorm2d(num_features=in_channels)
    
    def forward(self, x):
        skip_x = self.relu(self.bn1(self.conv1(x)))
        skip_x = self.bn2(self.conv2(skip_x))
        return self.relu(x+skip_x)


class ResTF(nn.Module):
    def __init__(self, out_features, blocks):
        super(ResTF, self).__init__()
        self.explainECs = out_features
        
        input_channels = 128
        self.input_channels = input_channels
    
        self.relu = nn.ReLU()
        self.cnn0 = nn.Conv2d(1, input_channels, kernel_size=(4,21))
        self.batchnorm0 = nn.BatchNorm2d(num_features=input_channels)
        
        layers = []
        for i in range(blocks):
            layers.append(BasicBlock(input_channels))
            
        self.layer = nn.Sequential(*layers)
        self.max_pool = nn.MaxPool2d(kernel_size=(997, 1), stride=1)
        self.fc = nn.Linear(in_features=input_channels, out_features=len(out_features))
        self.bn1 = nn.BatchNorm1d(num_features=len(out_features))
        self.out_act = nn.Sigmoid()
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        
    def forward(self, x):
        x = self.relu(self.batchnorm0(self.cnn0(x)))
        x = self.layer(x)
        x = self.max_pool(x)
        x = x.view(-1, self.input_channels)
        x = self.out_act(self.bn1(self.fc(x)))
        return x


class ResEC(nn.Module):
    def __init__(self, out_features, blocks):
        super(ResEC, self).__init__()

        
        input_channels = 128
        self.input_channels = input_channels
    
        self.relu = nn.ReLU()
        self.cnn0 = nn.Conv2d(1, input_channels, kernel_size=(4,21))
        self.batchnorm0 = nn.BatchNorm2d(num_features=input_channels)
        
        layers = []
        for i in range(blocks):
            layers.append(BasicBlock(input_channels))
            
        self.layer = nn.Sequential(*layers)

        deconv_components = [nn.ConvTranspose2d(in_channels=128, out_channels=512, kernel_size=(4,1)),
                             nn.BatchNorm2d(num_features=512),
                             nn.LeakyReLU()]
        self.deconv = nn.Sequential(*deconv_components)
        self.conv = nn.Conv2d(in_channels=512, out_channels=out_features, kernel_size=(1,1))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        
    def forward(self, x):
        x = self.relu(self.batchnorm0(self.cnn0(x)))
        x = self.layer(x)
        x = self.deconv(x)
        x = self.conv(x)
        return x