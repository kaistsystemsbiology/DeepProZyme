# import torch packages
import torch
import torch.nn as nn




class DeepEC_CAM(nn.Module):
    def __init__(self, out_features, basal_net='CNN15'):
        super(DeepEC_CAM, self).__init__()
        self.explainECs = None
        self.num_ECs = out_features
        if basal_net == 'CNN15':
            self.cnn0 = CNN15(out_features)
        elif basal_net == 'CNN16':
            self.cnn0 = CNN16(out_features, blocks=25)
        elif basal_net == 'ResEC':
            self.cnn0 = ResEC(out_features)
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


class CNN16(nn.Module):
    def __init__(self, out_feature):
        super(CNN16, self).__init__()
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

        deconv_components = []
        for i in range(5):
            deconv_components += [nn.ConvTranspose2d(in_channels=128*3, out_channels=128*3, kernel_size=(4,1)),
                                   nn.BatchNorm2d(num_features=128*3),
                                   nn.LeakyReLU()]
        self.deconv = nn.Sequential(*deconv_components)
        
        self.conv = nn.Conv2d(in_channels=128*3, out_channels=out_feature, kernel_size=(1,1))
        
        
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.deconv(x)
        x = self.conv(x)
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

        deconv_components = [nn.ConvTranspose2d(in_channels=128*3, out_channels=512, kernel_size=(4,1)),
                             nn.BatchNorm2d(num_features=512),
                             nn.LeakyReLU()]
        self.deconv = nn.Sequential(*deconv_components)
        self.conv = nn.Conv2d(in_channels=512, out_channels=out_feature, kernel_size=(1,1))

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