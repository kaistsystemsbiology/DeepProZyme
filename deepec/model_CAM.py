# import torch packages
import torch
import torch.nn as nn
import logging




class DeepEC_CAM(nn.Module):
    def __init__(self, out_features, basal_net='CNN16'):
        super(DeepEC_CAM, self).__init__()
        self.explainECs = None
        self.num_ECs = out_features
        if basal_net == 'CNN16':
            self.cnn0 = CNN16(out_features)
        elif basal_net == 'ResEC':
            self.cnn0 = ResEC(out_features , blocks=15)
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
        
        self.conv = nn.Conv2d(in_channels=128*3, out_channels=1024, kernel_size=(1,1))
        
        
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


class DeepEC_GradCAM(nn.Module):
    def __init__(self, out_features):
        super(DeepEC_GradCAM, self).__init__()
        self.explainECs = None
        self.num_ECs = out_features

        self.cnn0 = CNN_Grad(out_features)
        self.fc = nn.Linear(in_features=1024, out_features=out_features)
        self.bn1 = nn.BatchNorm1d(num_features=out_features)
        self.out_act = nn.Sigmoid()
        # self.pool = nn.MaxPool2d(kernel_size=(1000,1), stride=1)
        self.pool = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(1000, 1))

        self.cnn0.CAM_relu.register_forward_hook(self.forward_hook)
        self.cnn0.CAM_relu.register_backward_hook(self.backward_hook)
        self.pool.CAM_relu.register_forward_hook(self.forward_hook)
        self.pool.CAM_relu.register_backward_hook(self.backward_hook)
      
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)

        
    def forward_hook(self, _, input, output):
        self.forward_result = torch.squeeze(output)

    def backward_hook(self, _, grad_input, grad_output):
        self.backward_result = torch.squeeze(grad_output[0])
        
        
    def forward(self, x):
        # logging.info(x.shape)
        x = self.cnn0(x)
        # logging.info(x.shape)
        x = self.pool(x)
        # logging.info(x.shape)
        x = x.view(-1, 1024)
        # logging.info(x.shape)
        x = self.out_act(self.bn1(self.fc(x)))
        # logging.info(x.shape)
        return x


class CNN_Grad(nn.Module):
    def __init__(self, out_feature):
        super(CNN_Grad, self).__init__()
        layer1_components = [nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,21)),
                             nn.BatchNorm2d(num_features=128),
                             nn.ReLU()]

        for i in range(4):
            layer1_components.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,1)))
            layer1_components.append(nn.BatchNorm2d(num_features=128))
            layer1_components.append(nn.ReLU())

        self.layer1 = nn.Sequential(*layer1_components)

        layer2_components = [
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,21)),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        ]
        for i in range(2):
            layer2_components.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,1)))
            layer2_components.append(nn.BatchNorm2d(num_features=128))
            layer2_components.append(nn.ReLU())
        layer2_components += [
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,1)),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        ]
        self.layer2 = nn.Sequential(*layer2_components)

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,21)),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )

        deconv_components = []
        for i in range(5):
            deconv_components += [nn.ConvTranspose2d(in_channels=128*3, out_channels=128*3, kernel_size=(4,1)),
                                   nn.BatchNorm2d(num_features=128*3),
                                   nn.ReLU()]
        self.deconv = nn.Sequential(*deconv_components)
        
        self.conv = nn.Conv2d(in_channels=128*3, out_channels=1024, kernel_size=(1,1))
        self.CAM_relu = nn.ReLU()
        
        
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.deconv(x)
        x = self.conv(x)
        return x


class GReLUInner(torch.autograd.Function):
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input * (input > 0).float()
    
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        return grad_output * (grad_output > 0).float() * (input[0] > 0).float()
