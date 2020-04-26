# import torch packages
import torch
import torch.nn as nn



class DeepEC_CAM(nn.Module):
    def __init__(self, out_features, basal_net='CNN0_3'):
        super(DeepEC_CAM, self).__init__()
        self.explainEcs = None
        self.num_ECs = out_features
        if basal_net == 'CNN0':
            self.cnn0 = CNN0()
        if basal_net == 'CNN0_1':
            self.cnn0 = CNN0_1()
        elif basal_net == 'CNN0_2':
            self.cnn0 = CNN0_2()
        elif basal_net == 'CNN0_3':
            self.cnn0 = CNN0_3(out_features)
        elif basal_net == 'CNN0_4':
            self.cnn0 = CNN0_4(out_features)
        elif basal_net == 'CNN0_5':
            self.cnn0 = CNN0_5(out_features)
        elif basal_net == 'CNN0_6':
            self.cnn0 = CNN0_6(out_features)
        elif basal_net == 'CNN0_7':
            self.cnn0 = CNN0_7(out_features)
        elif basal_net == 'CNN0_8':
            self.cnn0 = CNN0_8(out_features)
        elif basal_net == 'CNN0_9':
            self.cnn0 = CNN0_9(out_features)
        elif basal_net == 'CNN0_10':
            self.cnn0 = CNN0_10(out_features)
        else:
            raise ValueError
        self.fc = nn.Linear(in_features=out_features, out_features=out_features)
        self.bn1 = nn.BatchNorm1d(num_features=out_features)
        self.out_act = nn.Sigmoid()
      
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
        
        
    def forward(self, x):
        x = self.cnn0(x)
        out = torch.sum(x, axis=[2,3])
        out = out.view(-1, self.num_ECs)
        out = self.out_act(self.bn1(self.fc(out)))
        return out, x




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
    def __init__(self, out_features, basal_net='CNN0'):
        super(DeepEC, self).__init__()
        self.explainECs = out_features
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
        self.fc = nn.Linear(in_features=128*3, out_features=len(out_features))
        self.bn1 = nn.BatchNorm1d(num_features=len(out_features))
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


class CNN0_3(nn.Module):
    '''
    CAM calculation model. Replace maxpooling into GAP
    Use second level convolutional layers
    CNN3 (upto fourth EC level)
    '''
    def __init__(self, out_feature):
        super(CNN0_3, self).__init__()
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
           
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,21))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,21))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,21))

        self.batchnorm1 = nn.BatchNorm2d(num_features=128)
        self.batchnorm2 = nn.BatchNorm2d(num_features=128)
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)

        self.conv1_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(16,1))
        self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(12,1))
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4,1))

        self.batchnorm1_1 = nn.BatchNorm2d(num_features=256)
        self.batchnorm2_1 = nn.BatchNorm2d(num_features=256)
        self.batchnorm3_1 = nn.BatchNorm2d(num_features=256)

        self.conv4 = nn.Conv2d(in_channels=256*3, out_channels=1024, kernel_size=(8,1))
        self.batchnorm4 = nn.BatchNorm2d(num_features=1024)
        
        self.deconv = nn.ConvTranspose2d(in_channels=1024, out_channels=out_feature, kernel_size=(26,1))
        
    def forward(self, x):
        x1 = self.lrelu(self.batchnorm1(self.conv1(x)))
        x2 = self.lrelu(self.batchnorm2(self.conv2(x)))
        x3 = self.lrelu(self.batchnorm3(self.conv3(x)))
        x1 = self.lrelu(self.batchnorm1_1(self.conv1_1(x1)))
        x2 = self.lrelu(self.batchnorm2_1(self.conv2_1(x2)))
        x3 = self.lrelu(self.batchnorm3_1(self.conv3_1(x3)))

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.lrelu(self.batchnorm4(self.conv4(x)))
        x = self.deconv(x)
        return x


class CNN0_4(nn.Module):
    '''
    CAM calculation model. Replace maxpooling into GAP
    Use second level convolutional layers
    CNN3 (upto fourth EC level)
    Use residual connection
    '''
    def __init__(self, out_feature):
        super(CNN0_4, self).__init__()
        self.lrelu = nn.LeakyReLU()
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

        self.conv5 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(26,21))
        self.batchnorm5 = nn.BatchNorm2d(num_features=512)
        
        self.deconv = nn.ConvTranspose2d(in_channels=512, out_channels=out_feature, kernel_size=(26,1))
        
    def forward(self, input):
        x1 = self.lrelu(self.batchnorm1(self.conv1(input)))
        x2 = self.lrelu(self.batchnorm2(self.conv2(input)))
        x3 = self.lrelu(self.batchnorm3(self.conv3(input)))
        x1 = self.lrelu(self.batchnorm1_1(self.conv1_1(x1)))
        x2 = self.lrelu(self.batchnorm2_1(self.conv2_1(x2)))
        x3 = self.lrelu(self.batchnorm3_1(self.conv3_1(x3)))

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.lrelu(self.batchnorm4(self.conv4(x)))
        x += self.relu(self.batchnorm5(self.conv5(input)))
        x = self.deconv(x)
        return x


class CNN0_5(nn.Module):
    '''
    CAM calculation model. Replace maxpooling into GAP
    Use second level convolutional layers
    CNN3 (upto fourth EC level)
    '''
    def __init__(self, out_feature):
        super(CNN0_5, self).__init__()
        self.relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
           
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128*1, kernel_size=(5,21), padding=(2,1))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=128*1, kernel_size=(9,21), padding=(4,1))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=128*1, kernel_size=(13,21), padding=(6,1))
        self.conv4 = nn.Conv2d(in_channels=128*3, out_channels=out_feature, kernel_size=(9,1), padding=(4,1))

        self.batchnorm1 = nn.BatchNorm2d(num_features=128*1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=128*1)
        self.batchnorm3 = nn.BatchNorm2d(num_features=128*1)
        
        
    def forward(self, x):
        x1 = self.relu(self.batchnorm1(self.conv1(x)))
        x2 = self.relu(self.batchnorm2(self.conv2(x)))
        x3 = self.relu(self.batchnorm3(self.conv3(x)))
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv4(x)
        return x


class CNN0_6(nn.Module):
    '''
    CAM calculation model. Replace maxpooling into GAP
    Use second level convolutional layers
    CNN3 (upto fourth EC level)
    Use Deconv layers
    '''
    def __init__(self, out_feature):
        super(CNN0_6, self).__init__()
        self.relu = nn.LeakyReLU()
        # self.relu = nn.ReLU()
           
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,21))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,21))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,21))
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(16,1))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(12,1))
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,1))
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=128*3, out_channels=512, kernel_size=(9,1))
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=out_feature, kernel_size=(11,1))

        self.batchnorm1 = nn.BatchNorm2d(num_features=128)
        self.batchnorm2 = nn.BatchNorm2d(num_features=128)
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)
        self.batchnorm4 = nn.BatchNorm2d(num_features=128)
        self.batchnorm5 = nn.BatchNorm2d(num_features=128)
        self.batchnorm6 = nn.BatchNorm2d(num_features=128)
        
        self.batchnorm7 = nn.BatchNorm2d(num_features=512)
        
    def forward(self, x):
        x1 = self.relu(self.batchnorm1(self.conv1(x)))
        x2 = self.relu(self.batchnorm2(self.conv2(x)))
        x3 = self.relu(self.batchnorm3(self.conv3(x)))
        x1 = self.relu(self.batchnorm4(self.conv4(x1)))
        x2 = self.relu(self.batchnorm5(self.conv5(x2)))
        x3 = self.relu(self.batchnorm6(self.conv6(x3)))
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.relu(self.batchnorm7(self.deconv1(x)))
        x = self.deconv2(x)
        return x


    
class CNN0_7(nn.Module):
    '''
    CAM calculation model. Replace maxpooling into GAP
    Use second level convolutional layers
    CNN3 (upto fourth EC level)
    Use Deconv layers
    '''
    def __init__(self, out_feature):
        super(CNN0_7, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,21))
        self.batchnorm1 = nn.BatchNorm2d(num_features=128)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,21))
        self.batchnorm2 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,21))
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(16,1))
        self.batchnorm4 = nn.BatchNorm2d(num_features=128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(12,1))
        self.batchnorm5 = nn.BatchNorm2d(num_features=128)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,1))
        self.batchnorm6 = nn.BatchNorm2d(num_features=128)
        self.deconv1 = nn.ConvTranspose2d(in_channels=128*3, out_channels=512, kernel_size=(9,1))
        self.batchnorm7 = nn.BatchNorm2d(num_features=512)
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=out_feature, kernel_size=(11,1))
        
    def forward(self, x):
        x1 = self.relu(self.batchnorm1(self.conv1(x)))
        x2 = self.relu(self.batchnorm1(self.conv2(x)))
        x3 = self.relu(self.batchnorm3(self.conv3(x)))
        x1 = self.relu(self.batchnorm4(self.conv4(x1)))
        x2 = self.relu(self.batchnorm5(self.conv5(x2)))
        x3 = self.relu(self.batchnorm6(self.conv6(x3)))
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.relu(self.batchnorm7(self.deconv1(x)))
        x = self.deconv2(x)
        return x


class CNN0_8(nn.Module):
    '''
    CAM calculation model. Replace maxpooling into GAP
    Use second level convolutional layers
    CNN3 (upto fourth EC level)
    Use Deconv layers
    '''
    def __init__(self, out_feature):
        super(CNN0_8, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,21)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(16,1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,21)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(12,1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,21)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128*3, out_channels=128*3, kernel_size=(6,1)),
            nn.BatchNorm2d(num_features=128*3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128*3, out_channels=128*3, kernel_size=(6,1)),
            nn.BatchNorm2d(num_features=128*3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128*3, out_channels=512, kernel_size=(5,1)),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=out_feature, kernel_size=(5,1))
        )
        
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.deconv(x)
        return x


class CNN0_9(nn.Module):
    '''
    CAM calculation model. Replace maxpooling into GAP
    Use second level convolutional layers
    CNN3 (upto fourth EC level)
    '''
    def __init__(self, out_feature):
        super(CNN0_9, self).__init__()
        self.relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128*1, kernel_size=(5,21), padding=(2,1)),
            nn.BatchNorm2d(num_features=128*1),
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128*1, kernel_size=(9,21), padding=(4,1)),
            nn.BatchNorm2d(num_features=128*1),
            nn.LeakyReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128*1, kernel_size=(13,21), padding=(6,1)),
            nn.BatchNorm2d(num_features=128*1),
            nn.LeakyReLU()
        )
           
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=128*3, out_channels=512, kernel_size=(5,1), padding=(2,1)),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=out_feature, kernel_size=(5,1), padding=(2,1))
        )
        

        
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv(x)
        return x



class CNN0_10(nn.Module):
    def __init__(self, out_feature):
        super(CNN0_10, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3,21)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1,1)),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU()
        )
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(16,1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(8,1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(12,1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(16,1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )
        

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128*3, out_channels=128*3, kernel_size=(7,1)),
            nn.BatchNorm2d(num_features=128*3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128*3, out_channels=128*3, kernel_size=(7,1)),
            nn.BatchNorm2d(num_features=128*3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128*3, out_channels=512, kernel_size=(7,1)),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(),
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=out_feature, kernel_size=(3,1))
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x_res = self.conv2(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.deconv1(x)
        x = x + x_res
        x = self.deconv2(x)
        return x



class CNN11(nn.Module):
    '''
    CAM calculation model. Replace maxpooling into GAP
    Upto fourth EC level
    Use Deconv layers
    '''
    def __init__(self, out_feature):
        super(CNN11, self).__init__()
        layer1_components = [nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,21)),
                             nn.BatchNorm2d(num_features=128),
                             nn.LeakyReLU()]

        for i in range(4):
            layer1_components.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,1)))
            layer1_components.append(nn.BatchNorm2d(num_features=128))
            layer1_components.append(nn.LeakyReLU())

        self.layer1 = nn.Sequential(*layer1_components)

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,21)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(9,1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,21)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )

        deconv1_components = []
        for i in range(4):
            deconv1_components += [nn.ConvTranspose2d(in_channels=128*3, out_channels=128*3, kernel_size=(4,1)),
                                   nn.BatchNorm2d(num_features=128*3),
                                   nn.LeakyReLU()]
        self.deconv1 = nn.Sequential(*deconv1_components)
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128*3, out_channels=out_feature, kernel_size=(4,1))
        )
        
        
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x



class CNN12(nn.Module):
    '''
    CAM calculation model. Replace maxpooling into GAP
    Upto fourth EC level
    Use Deconv layers
    '''
    def __init__(self, out_feature):
        super(CNN12, self).__init__()
        layer1_components = [nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,21)),
                             nn.BatchNorm2d(num_features=128),
                             nn.LeakyReLU()]

        for i in range(4):
            layer1_components += [nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,1)),
                                  nn.BatchNorm2d(num_features=128),
                                  nn.LeakyReLU()]

        self.layer1_1 = nn.Sequential(*layer1_components[0:3*2])
        self.layer1_2 = nn.Sequential(*layer1_components[3*2:3*4])
        self.layer1_3 = nn.Sequential(*layer1_components[3*4:])

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,21)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(9,1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,21)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )


        self.deconv1_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128*3, out_channels=128*3, kernel_size=(4,1)),
            nn.BatchNorm2d(num_features=128*3),
            nn.LeakyReLU()
        )

        self.deconv1_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128*4, out_channels=128*4, kernel_size=(4,1)),
            nn.BatchNorm2d(num_features=128*4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128*4, out_channels=128*4, kernel_size=(4,1)),
            nn.BatchNorm2d(num_features=128*4),
            nn.LeakyReLU()
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128*5, out_channels=128*5, kernel_size=(4,1)),
            nn.BatchNorm2d(num_features=128*5),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128*5, out_channels=out_feature, kernel_size=(4,1))
        )

        
    def forward(self, x):
        x1 = self.layer1_1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)

        x1_2 = self.layer1_2(x1)
        x1_3 = self.layer1_3(x1_2)

        x = torch.cat((x1_3, x2, x3), dim=1)
        x = self.deconv1_1(x)
        x = torch.cat((x, x1_2), dim=1)
        x = self.deconv1_2(x)
        x = torch.cat((x, x1), dim=1)
        x = self.deconv2(x)
        return x

        

class CNN13(nn.Module):
    '''
    CAM calculation model. Replace maxpooling into GAP
    Upto fourth EC level
    Use Deconv layers
    '''
    def __init__(self, out_feature):
        super(CNN13, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,21)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=844, kernel_size=(4,1))
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,21)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=844, kernel_size=(8,1))
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,21)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=843, kernel_size=(16,1))
        )
        
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        return x


class CNN14(nn.Module):
    '''
    CAM calculation model. Replace maxpooling into GAP
    Upto fourth EC level
    Use Deconv layers
    '''
    def __init__(self, out_feature):
        super(CNN14, self).__init__()
        layer1_components = [nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,21)),
                             nn.BatchNorm2d(num_features=128),
                             nn.LeakyReLU()]

        for i in range(4):
            layer1_components += [nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,1)),
                                  nn.BatchNorm2d(num_features=128),
                                  nn.LeakyReLU()]

        self.layer1_1 = nn.Sequential(*layer1_components[0:3*2])
        self.layer1_2 = nn.Sequential(*layer1_components[3*2:3*4])
        self.layer1_3 = nn.Sequential(*layer1_components[3*4:])

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,21)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(9,1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,21)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )


        self.deconv1_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128*3, out_channels=128, kernel_size=(4,1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )

        self.deconv1_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4,1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4,1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128*5, kernel_size=(4,1)),
            nn.BatchNorm2d(num_features=128*5),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128*5, out_channels=out_feature, kernel_size=(4,1))
        )

        
    def forward(self, x):
        x1_1 = self.layer1_1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)

        x1_2 = self.layer1_2(x1_1)
        x1_3 = self.layer1_3(x1_2)

        x = torch.cat((x1_3, x2, x3), dim=1)
        x = self.deconv1_1(x)
        x += x1_2
        x = self.deconv1_2(x)
        x += x1_1
        x = self.deconv2(x)
        return x