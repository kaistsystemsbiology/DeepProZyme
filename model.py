# import torch packages
import torch
import torch.nn as nn


class CNN0(nn.Module):
    def __init__(self):
        super(CNN0, self).__init__()
        self.relu = nn.ReLU()
           
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,21))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,21))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,21))

        self.batchnorm1 = nn.BatchNorm2d(num_features=128)
        self.batchnorm2 = nn.BatchNorm2d(num_features=128)
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)

        self.pool1 = nn.MaxPool2d(kernel_size=(1000-4+1,1), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(1000-8+1,1), stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(1000-16+1,1), stride=1)
        
    def forward(self, x):
        x1 = self.relu(self.batchnorm1(self.conv1(x)))
        x2 = self.relu(self.batchnorm2(self.conv2(x)))
        x3 = self.relu(self.batchnorm3(self.conv3(x)))
        
        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)

        x = torch.cat((x1, x2, x3), dim=1)
        return x 



class DeepEC(nn.Module):
    def __init__(self, explainEcs, basal_net='CNN0'):
        super(DeepEC, self).__init__()
        self.explainEcs = explainEcs
        if basal_net == 'CNN0':
            self.cnn0 = CNN0()
        elif basal_net == 'CNN0_1':
            self.cnn0 = CNN0_1()
        elif basal_net == 'CNN0_2':
            self.cnn0 = CNN0_2()
        elif basal_net == 'CNN0_3':
            self.cnn0 = CNN0_3()
        elif basal_net == 'CNN0_4':
            self.cnn0 = CNN0_4()
        else:
            raise ValueError
        self.fc = nn.Linear(in_features=128*3, out_features=len(explainEcs))
        self.bn1 = nn.BatchNorm1d(num_features=len(explainEcs))
        self.out_act = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
        
        
    def forward(self, x):
        x = self.cnn0(x)
        x = x.view(-1, 128*3)
        x = self.out_act(self.bn1(self.fc(x)))
        return x



class CNN0_1(nn.Module):
    '''
    CAM calculation model. Replace maxpooling into GAP
    Use second level convolutional layers
    CNN2 (upto third EC level)
    '''
    def __init__(self):
        super(CNN0_1, self).__init__()
        self.relu = nn.ReLU()
           
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

        self.conv4 = nn.Conv2d(in_channels=128*3, out_channels=128*3, kernel_size=(8,1))
        self.batchnorm4 = nn.BatchNorm2d(num_features=128*3)

        self.conv5 = nn.Conv2d(in_channels=128*3, out_channels=128*3, kernel_size=(8,1))
        self.batchnorm5 = nn.BatchNorm2d(num_features=128*3)

        self.conv6 = nn.Conv2d(in_channels=128*3, out_channels=216, kernel_size=(8,1))

        
    def forward(self, x):
        x1 = self.relu(self.batchnorm1(self.conv1(x)))
        x2 = self.relu(self.batchnorm2(self.conv2(x)))
        x3 = self.relu(self.batchnorm3(self.conv3(x)))
        x1 = self.relu(self.batchnorm1_1(self.conv1_1(x1)))
        x2 = self.relu(self.batchnorm2_1(self.conv2_1(x2)))
        x3 = self.relu(self.batchnorm3_1(self.conv3_1(x3)))

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.relu(self.batchnorm4(self.conv4(x)))
        x = self.relu(self.batchnorm5(self.conv5(x)))
        x = self.conv6(x)
        return x


class CNN0_2(nn.Module):
    '''
    CAM calculation model. Replace maxpooling into GAP
    Use second level convolutional layers
    CNN3 (upto fourth EC level)
    '''
    def __init__(self):
        super(CNN0_2, self).__init__()
        self.relu = nn.ReLU()
           
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

        self.conv4 = nn.Conv2d(in_channels=128*3, out_channels=512, kernel_size=(8,1))
        self.batchnorm4 = nn.BatchNorm2d(num_features=512)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(8,1))
        self.batchnorm5 = nn.BatchNorm2d(num_features=1024)

        self.conv6 = nn.Conv2d(in_channels=1024, out_channels=2531, kernel_size=(8,1))

        
    def forward(self, x):
        x1 = self.relu(self.batchnorm1(self.conv1(x)))
        x2 = self.relu(self.batchnorm2(self.conv2(x)))
        x3 = self.relu(self.batchnorm3(self.conv3(x)))
        x1 = self.relu(self.batchnorm1_1(self.conv1_1(x1)))
        x2 = self.relu(self.batchnorm2_1(self.conv2_1(x2)))
        x3 = self.relu(self.batchnorm3_1(self.conv3_1(x3)))

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.relu(self.batchnorm4(self.conv4(x)))
        x = self.relu(self.batchnorm5(self.conv5(x)))
        x = self.conv6(x)
        return x


class DeepEC_CAM(nn.Module):
    def __init__(self, explainEcs, basal_net='CNN0_1'):
        super(DeepEC_CAM, self).__init__()
        self.explainEcs = explainEcs
        if basal_net == 'CNN0':
            self.cnn0 = CNN0()
        if basal_net == 'CNN0_1':
            self.cnn0 = CNN0_1()
        elif basal_net == 'CNN0_2':
            self.cnn0 = CNN0_2()
        elif basal_net == 'CNN0_3':
            self.cnn0 = CNN0_3()
        elif basal_net == 'CNN0_4':
            self.cnn0 = CNN0_4()
        else:
            raise ValueError
        self.fc = nn.Linear(in_features=len(explainEcs), out_features=len(explainEcs))
        self.bn1 = nn.BatchNorm1d(num_features=len(explainEcs))
        self.out_act = nn.Sigmoid()
      
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
        
        
    def forward(self, x):
        x = self.cnn0(x)
        out = torch.sum(x, axis=[2,3])
        out = out.view(-1, len(self.explainEcs))
        out = self.out_act(self.bn1(self.fc(out)))
        return out, x
    