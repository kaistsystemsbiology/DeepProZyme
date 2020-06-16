# import torch packages
import torch
import torch.nn as nn


class DeepTFactor(nn.Module):
    def __init__(self, out_features):
        super(DeepTFactor, self).__init__()
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


class DeepEC(nn.Module):
    def __init__(self, out_features, basal_net='CNN0'):
        super(DeepEC, self).__init__()
        self.explainECs = out_features
        if basal_net == 'CNN0':
            self.cnn0 = CNN0()
        elif basal_net == 'CNN0_0':
            self.cnn0 = CNN0_0()
        elif basal_net == 'CNN0_01':
            self.cnn0 = CNN0_01()
        elif basal_net == 'CNN0_02':
            self.cnn0 = CNN0_02()
        elif basal_net == 'CNN0_03':
            self.cnn0 = CNN0_03()
        elif basal_net == 'CNN0_3':
            self.cnn0 = CNN0_3()
        elif basal_net == 'CNN0_4':
            self.cnn0 = CNN0_4()
        else:
            raise ValueError
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
        x = self.cnn0(x)
        x = x.view(-1, 128*3)
        x = self.out_act(self.bn1(self.fc(x)))
        return x



class InceptionModule(nn.Module):
    def __init__(self,in_channels, *out_channels):
        super(InceptionModule, self).__init__()
        out_channels=out_channels[0]
        self.relu = nn.ReLU()

        self.branch1x1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=(1,1))
        self.branch3x1_1 = nn.Conv2d(in_channels, out_channels[1], kernel_size=(1,1))
        self.branch3x1_2 = nn.Conv2d(out_channels[1], out_channels[2], kernel_size=(3,1), padding=(1,0))
        self.branch5x1_1 = nn.Conv2d(in_channels, out_channels[3], kernel_size=(1,1))
        self.branch5x1_2 = nn.Conv2d(out_channels[3], out_channels[4], kernel_size=(5,1), padding=(2,0))
        self.branch_pool_1 = nn.AvgPool2d(kernel_size=(3,1), padding=(1,0), stride=1)
        self.branch_pool_2 = nn.Conv2d(in_channels, out_channels[5], kernel_size=(1,1))

        
    def forward(self,x): 
        branch1x1 = self.branch1x1(x)
        branch3x1 = self.branch3x1_1(x)
        branch3x1 = self.branch3x1_2(branch3x1)
        branch5x1 = self.branch5x1_1(x)
        branch5x1 = self.branch5x1_2(branch5x1)
        branch_pool = self.branch_pool_1(x)
        branch_pool = self.branch_pool_2(branch_pool)
        return torch.cat([branch1x1, branch3x1, branch5x1, branch_pool], axis=1)
    
    
class InceptionModule_A(nn.Module):
    def __init__(self,in_channels, *out_channels):
        super(InceptionModule_A, self).__init__()
        out_channels=out_channels[0]
        self.relu = nn.ReLU()

        self.branch1x1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=(1,1))
        self.branch3x1_1 = nn.Conv2d(in_channels, out_channels[1], kernel_size=(1,1))
        self.branch3x1_2 = nn.Conv2d(out_channels[1], out_channels[2], kernel_size=(3,1), padding=(1,0))
        self.branch5x1_1 = nn.Conv2d(in_channels, out_channels[3], kernel_size=(1,1))
        self.branch5x1_2 = nn.Conv2d(out_channels[3], out_channels[3], kernel_size=(3,1), padding=(1,0))
        self.branch5x1_3 = nn.Conv2d(out_channels[3], out_channels[4], kernel_size=(3,1), padding=(1,0))
        self.branch_pool_1 = nn.AvgPool2d(kernel_size=(3,1), padding=(1,0), stride=1)
        self.branch_pool_2 = nn.Conv2d(in_channels, out_channels[5], kernel_size=(1,1))
        
        self.branch1x1_bn = nn.BatchNorm2d(num_features=out_channels[0])
        self.branch3x1_1_bn = nn.BatchNorm2d(num_features=out_channels[1])
        self.branch3x1_2_bn = nn.BatchNorm2d(num_features=out_channels[2])
        self.branch5x1_1_bn = nn.BatchNorm2d(num_features=out_channels[3])
        self.branch5x1_2_bn = nn.BatchNorm2d(num_features=out_channels[3])
        self.branch5x1_3_bn = nn.BatchNorm2d(num_features=out_channels[4])

        self.branch_pool_2_bn = nn.BatchNorm2d(num_features=out_channels[5])

        
    def forward(self,x): 
        branch1x1 = self.branch1x1_bn(self.relu(self.branch1x1(x)))
        branch3x1 = self.branch3x1_1_bn(self.relu(self.branch3x1_1(x)))
        branch3x1 = self.branch3x1_2_bn(self.relu(self.branch3x1_2(branch3x1)))
        branch5x1 = self.branch5x1_1_bn(self.relu(self.branch5x1_1(x)))
        branch5x1 = self.branch5x1_2_bn(self.relu(self.branch5x1_2(branch5x1)))
        branch5x1 = self.branch5x1_3_bn(self.relu(self.branch5x1_3(branch5x1)))
        branch_pool = self.branch_pool_1(x)
        branch_pool = self.branch_pool_2_bn(self.relu(self.branch_pool_2(branch_pool)))
        return torch.cat([branch1x1, branch3x1, branch5x1, branch_pool], axis=1)


class InceptionEC(nn.Module):
    def __init__(self, out_features):
        super(InceptionEC, self).__init__()
        self.explainECs = out_features
    
        input_channels = 64
        self.cnn0 = nn.Conv2d(1, input_channels, kernel_size=(4,21))
        self.batchnorm0 = nn.BatchNorm2d(num_features=input_channels)
        self.relu = nn.ReLU()
        self.inception = nn.Sequential(
                            InceptionModule_A(input_channels, [64, 32, 128, 16, 32, 32]),
                            InceptionModule_A(256, [128, 128, 192, 32, 96, 64]),
                            InceptionModule_A(480, [192, 96, 208, 16, 48, 64]),
                            InceptionModule_A(512, [160, 112, 224, 24, 64, 64]),
                            InceptionModule_A(512, [128, 128, 256, 24, 64, 64]),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=(997, 1), stride=1)
        self.fc = nn.Linear(in_features=512, out_features=len(out_features))
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
        x = self.inception(x)
        x = self.max_pool(x)
        x = x.view(-1, 512)
        x = self.out_act(self.bn1(self.fc(x)))
        return x