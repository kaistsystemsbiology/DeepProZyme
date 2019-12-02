# import torch packages
import torch
import torch.nn as nn


class CNN0(nn.Module):
    def __init__(self):
        super(CNN0, self).__init__()
        self.relu = nn.ReLU()
        self.out_act = nn.Sigmoid()
           
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,21))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,21))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,21))

        self.batchnorm1 = nn.BatchNorm2d(num_features=128)
        self.batchnorm2 = nn.BatchNorm2d(num_features=128)
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)

        self.pool1 = nn.MaxPool2d(kernel_size=(997,1), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(993,1), stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(985,1), stride=1)

        self.init_weights()


    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

        
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
    def __init__(self, out_features):
        super(DeepEC, self).__init__()
        self.cnn0 = CNN0()
        self.fc = nn.Linear(in_features=128*3, out_features=out_features)
        # nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='relu')
        nn.init.normal_(self.fc.weight)
        self.bn1 = nn.BatchNorm1d(num_features=out_features)
        self.out_act = nn.Sigmoid()
        
        
    def forward(self, x):
        x = self.cnn0(x)
        x = x.view(-1, 128*3)
        x = self.out_act(self.bn1(self.fc(x)))
        return x


class DeepEC_multitask(nn.Module):
    def __init__(self, out_features1, out_features2):
        super(DeepEC_multitask, self).__init__()
        self.cnn0 = CNN0()
        self.fc1 = nn.Linear(in_features=128*3, out_features=out_features1)
        self.fc2 = nn.Linear(in_features=128*3, out_features=out_features2)
        # nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.normal_(self.fc1.weight)
        nn.init.normal_(self.fc2.weight)
        self.bn1 = nn.BatchNorm1d(num_features=out_features1)
        self.bn2 = nn.BatchNorm1d(num_features=out_features2)
        self.out_act = nn.Sigmoid()
        
        
    def forward(self, x):
        x = self.cnn0(x).view(-1, 128*3)
        x1 = self.out_act(self.bn1(self.fc1(x)))
        x2 = self.out_act(self.bn2(self.fc2(x)))
        return x1, x2