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
        elif basal_net == 'ResEC_2':
            self.cnn0 = ResEC_2(out_features , blocks=4)
        else:
            raise ValueError
        # self.fc = nn.Linear(in_features=1024, out_features=out_features)
        self.fc = nn.Linear(in_features=out_features, out_features=out_features)
        self.bn1 = nn.BatchNorm1d(num_features=out_features)
        self.out_act = nn.Sigmoid()
      
        self.cnn0.CAM_relu.register_forward_hook(self.forward_hook)
        self.cnn0.conv.register_forward_hook(self.forward_hook_grad)
        self.cnn0.conv.register_backward_hook(self.backward_hook_grad)
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

        
    def forward(self, x):
        x = self.cnn0(x)
        x = torch.sum(x, axis=[2,3])
        # x = x.view(-1, 1024) # check the number
        x = x.view(-1, self.num_ECs)
        # x = self.out_act(self.bn1(self.fc(x)))
        x = self.out_act(self.fc(x))
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
        self.pool = nn.MaxPool2d(kernel_size=(1000,1), stride=1)

        self.cnn0.CAM_relu.register_forward_hook(self.forward_hook)
        self.cnn0.CAM_relu.register_backward_hook(self.backward_hook)
      
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
        x = self.cnn0(x)
        x = self.pool(x)
        x = x.view(-1, 1024)
        x = self.out_act(self.bn1(self.fc(x)))
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
        x = self.CAM_relu(self.conv(x))
        return x


class GReLUInner(torch.autograd.Function):
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input * (input > 0).float()
    
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        return grad_output * (grad_output > 0).float() * (input[0] > 0).float()


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
        self.conv = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1,1))
        self.CAM_relu = nn.ReLU()

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
        x = self.CAM_relu(x)
        return x


class DeepEC_CAM_2(nn.Module):
    def __init__(self, out_features, basal_net='CNN16'):
        super(DeepEC_CAM_2, self).__init__()
        self.explainECs = None
        self.num_ECs = out_features
        if basal_net == 'CNN16':
            self.cnn0 = CNN16(out_features)
        elif basal_net == 'CNN_04':
            self.cnn0 = CNN_04(out_features)
        elif basal_net == 'ResEC':
            self.cnn0 = ResEC(out_features , blocks=15)
        elif basal_net == 'ResEC_2':
            self.cnn0 = ResEC_2(out_features , blocks=4)
        elif basal_net == 'ResEC_3':
            self.cnn0 = ResEC_3(out_features , blocks=10)
        else:
            raise ValueError
        # self.fc = nn.Linear(in_features=out_features, out_features=out_features)
        self.fc = nn.Linear(in_features=512, out_features=out_features)
        self.bn1 = nn.BatchNorm1d(num_features=out_features)
        self.out_act = nn.Sigmoid()
      
        self.cnn0.CAM_relu.register_forward_hook(self.forward_hook)
        self.cnn0.conv.register_forward_hook(self.forward_hook_grad)
        self.cnn0.conv.register_backward_hook(self.backward_hook_grad)
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

        
    def forward(self, x):
        x = self.cnn0(x)
        x = torch.sum(x, axis=[2,3])
        x = x.view(-1, 512) # check the number
        # x = x.view(-1, self.num_ECs)
        # x = self.out_act(self.bn1(self.fc(x)))
        x = self.out_act(self.fc(x))
        return x


class ResEC_2(nn.Module):
    def __init__(self, out_features, blocks):
        super(ResEC_2, self).__init__()

        
        input_channels = 128
        self.input_channels = input_channels
    
        self.relu = nn.ReLU()
        self.cnn0 = nn.Conv2d(1, input_channels, kernel_size=(4,21))
        self.batchnorm0 = nn.BatchNorm2d(num_features=input_channels)
        
        layers = []
        for i in range(blocks):
            layers.append(BasicBlock(input_channels))
            
        self.layer = nn.Sequential(*layers)

        deconv_components = [nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=(4,1)),
                             nn.BatchNorm2d(num_features=256),
                             nn.LeakyReLU()]
        self.deconv = nn.Sequential(*deconv_components)

        self.conv = nn.Conv2d(in_channels=256, out_channels=out_features, kernel_size=(1,1))
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_features)
        self.CAM_relu = nn.ReLU()

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
        x = self.batchnorm1(self.conv(x))
        x = self.CAM_relu(x)
        return x


class ResEC_3(nn.Module):
    def __init__(self, out_features, blocks):
        super(ResEC_3, self).__init__()

        
        input_channels = 128
        self.input_channels = input_channels
    
        self.relu = nn.ReLU()
        self.cnn0 = nn.Conv2d(1, input_channels, kernel_size=(4,21))
        self.batchnorm0 = nn.BatchNorm2d(num_features=input_channels)
        
        layers = []
        for i in range(blocks):
            layers.append(BasicBlock(input_channels))
            
        self.layer = nn.Sequential(*layers)

        deconv_components = [nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4,1)),
                             nn.BatchNorm2d(num_features=128),
                             nn.LeakyReLU()]
        self.deconv = nn.Sequential(*deconv_components)

        self.conv = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1,1))
        self.CAM_relu = nn.ReLU()

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
        x = self.CAM_relu(x)
        return x


class CNN_04(nn.Module):
    def __init__(self, out_feature):
        super(CNN_04, self).__init__()
        layer1_components = [nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,21)),
                             nn.BatchNorm2d(num_features=128),
                             nn.LeakyReLU()]

        for i in range(1):
            layer1_components.append(nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4,1)))
            layer1_components.append(nn.BatchNorm2d(num_features=128))
            layer1_components.append(nn.LeakyReLU())

        layer2_components = [nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(8,21)),
                             nn.BatchNorm2d(num_features=128),
                             nn.LeakyReLU()]

        for i in range(1):
            layer2_components.append(nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4,1)))
            layer2_components.append(nn.BatchNorm2d(num_features=128))
            layer2_components.append(nn.LeakyReLU())
        for i in range(2):
            layer2_components.append(nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3,1)))
            layer2_components.append(nn.BatchNorm2d(num_features=128))
            layer2_components.append(nn.LeakyReLU())


        layer3_components = [nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(16,21)),
                             nn.BatchNorm2d(num_features=128),
                             nn.LeakyReLU()]

        for i in range(5):
            layer3_components.append(nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4,1)))
            layer3_components.append(nn.BatchNorm2d(num_features=128))
            layer3_components.append(nn.LeakyReLU())
            
        self.layer1 = nn.Sequential(*layer1_components)
        self.layer2 = nn.Sequential(*layer2_components)
        self.layer3 = nn.Sequential(*layer3_components)

        
        self.conv = nn.Conv2d(in_channels=128*3, out_channels=out_feature, kernel_size=(1,1))
        self.CAM_relu = nn.ReLU()
        
        
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.CAM_relu(self.conv(x))
        return x