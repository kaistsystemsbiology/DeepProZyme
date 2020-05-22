# import torch packages
import torch
import torch.nn as nn
import logging


class DeepEC_CAM(nn.Module):
    def __init__(self, out_features, basal_net='CNN_grad'):
        super(DeepEC_CAM, self).__init__()
        self.explainECs = None
        self.num_ECs = out_features
        if basal_net == 'CNN_grad':
            self.cnn0 = CNN_grad(out_features)
        elif basal_net == 'ResEC':
            self.cnn0 = ResEC(out_features , blocks=15)
        elif basal_net == 'ResEC_2':
            self.cnn0 = ResEC_2(out_features , blocks=4)
        else:
            raise ValueError

        self.pool = nn.MaxPool2d(kernel_size=(1000,1), stride=1)
        self.fc = nn.Linear(in_features=1024, out_features=out_features)
        self.out_act = nn.Sigmoid()
      
        self.cnn0.CAM_relu.register_forward_hook(self.forward_hook)
        self.cnn0.CAM_relu.register_backward_hook(self.backward_hook)
        self.cnn0.conv.register_forward_hook(self.forward_hook_grad)
        self.cnn0.conv.register_backward_hook(self.backward_hook_grad)
        self.init_weights()


    def forward(self, x):
        x = self.cnn0(x)
        x = self.pool(x)
        x = x.view(-1, 1024)
        x = self.out_act(self.fc(x))
        return x


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

    def forward_hook_grad(self, _, input, output):
        self.forward_result_grad = torch.squeeze(output)

    def backward_hook_grad(self, _, grad_input, grad_output):
       self.backward_result_grad = torch.squeeze(grad_output[0])


    def get_cam(self, label_ind):
        cam = torch.tensordot(
            self.fc.weight[label_ind, :], 
            self.forward_result.squeeze(), dims=1
            ).squeeze()
        return cam

        
    


class CNN_grad(nn.Module):
    def __init__(self, out_feature):
        super(CNN_grad, self).__init__()
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
        self.CAM_relu = nn.ReLU()
        
        
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.deconv(x)
        x = self.CAM_relu(self.conv(x))
        return x