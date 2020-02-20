# import torch packages
import torch
import torch.nn as nn




class CNN15(nn.Module):
    def __init__(self, out_feature):
        super(CNN15, self).__init__()
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
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(3,1)),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU()
        )

        self.conv = nn.Conv2d(in_channels=512, out_channels=out_feature, kernel_size=(1,1))
        

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
        x = self.conv(x)
        return x


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


class CNN17(nn.Module):
    def __init__(self, out_feature):
        super(CNN17, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,21)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4,1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=844, kernel_size=(1,1))
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,21)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(8,1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=844, kernel_size=(1,1))
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,21)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(16,1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=843, kernel_size=(1,1))
        )
        

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        return x


class CNN18(nn.Module):
    def __init__(self, out_feature):
        super(CNN18, self).__init__()
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
            nn.ConvTranspose2d(in_channels=128*5, out_channels=128*5, kernel_size=(4,1)),
            nn.BatchNorm2d(num_features=128*5),
            nn.LeakyReLU()
        )

        self.conv = nn.Conv2d(in_channels=128*5, out_channels=out_feature, kernel_size=(1,1))

        
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
        x = self.conv(x)
        return x



class DeepEC_CAM(nn.Module):
    def __init__(self, out_features, basal_net='CNN15'):
        super(DeepEC_CAM, self).__init__()
        self.explainEcs = None
        self.num_ECs = out_features
        if basal_net == 'CNN15':
            self.cnn0 = CNN15(out_features)
        elif basal_net == 'CNN16':
            self.cnn0 = CNN16(out_features)
        elif basal_net == 'CNN17':
            self.cnn0 = CNN17(out_features)
        elif basal_net == 'CNN18':
            self.cnn0 = CNN18(out_features)
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