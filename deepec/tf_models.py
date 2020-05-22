# import torch packages
import torch
import torch.nn as nn


class DeepTFactor(nn.Module):
    def __init__(self, out_features):
        super(DeepTFactor, self).__init__()
        self.explainECs = out_features

        self.cnn0 = CNN0_0()
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


class CNN0_0(nn.Module):
    '''
    Use second level convolution.
    channel size: 4 -> 16 
                  8 -> 12
                  16 -> 4
    '''
    def __init__(self):
        super(CNN0_0, self).__init__()
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

        self.pool1 = nn.MaxPool2d(kernel_size=(1000-4+1-16+1,1), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(1000-8+1-12+1,1), stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(1000-16+1-4+1,1), stride=1)
        
    def forward(self, x):
        x1 = self.dropout(self.relu(self.batchnorm1(self.conv1(x))))
        x2 = self.dropout(self.relu(self.batchnorm2(self.conv2(x))))
        x3 = self.dropout(self.relu(self.batchnorm3(self.conv3(x))))
        x1 = self.dropout(self.relu(self.batchnorm1_1(self.conv1_1(x1))))
        x2 = self.dropout(self.relu(self.batchnorm2_1(self.conv2_1(x2))))
        x3 = self.dropout(self.relu(self.batchnorm3_1(self.conv3_1(x3))))
        
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


class CNN0_01(nn.Module):
    '''
    Use third level convolution.
    channel size: 4 -> 16 
                  8 -> 12
                  16 -> 4
    concat
    channel size: 8
    '''
    def __init__(self):
        super(CNN0_01, self).__init__()
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

        self.pool1 = nn.MaxPool2d(kernel_size=(1000-4+1-16+1-8+1,1), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(1000-8+1-12+1-8+1,1), stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(1000-16+1-4+1-8+1,1), stride=1)
        
    def forward(self, x):
        x1 = self.relu(self.batchnorm1(self.conv1(x)))
        x2 = self.relu(self.batchnorm2(self.conv2(x)))
        x3 = self.relu(self.batchnorm3(self.conv3(x)))
        x1 = self.relu(self.batchnorm1_1(self.conv1_1(x1)))
        x2 = self.relu(self.batchnorm2_1(self.conv2_1(x2)))
        x3 = self.relu(self.batchnorm3_1(self.conv3_1(x3)))
        
        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.relu(self.batchnorm4(self.conv4(x)))
        return x 


class CNN0_02(nn.Module):
    '''
    Use second level convolution.
    channel size: 4 -> 16 
                  8 -> 12
                  16 -> 4
    '''
    def __init__(self):
        super(CNN0_02, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
           
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(4,21))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(8,21))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(16,21))

        self.batchnorm1 = nn.BatchNorm2d(num_features=256)
        self.batchnorm2 = nn.BatchNorm2d(num_features=256)
        self.batchnorm3 = nn.BatchNorm2d(num_features=256)

        self.conv1_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(16,1))
        self.conv2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(12,1))
        self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4,1))

        self.batchnorm1_1 = nn.BatchNorm2d(num_features=256)
        self.batchnorm2_1 = nn.BatchNorm2d(num_features=256)
        self.batchnorm3_1 = nn.BatchNorm2d(num_features=256)

        self.pool1 = nn.MaxPool2d(kernel_size=(1000-4+1-16+1,1), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(1000-8+1-12+1,1), stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(1000-16+1-4+1,1), stride=1)
        
    def forward(self, x):
        x1 = self.dropout(self.relu(self.batchnorm1(self.conv1(x))))
        x2 = self.dropout(self.relu(self.batchnorm2(self.conv2(x))))
        x3 = self.dropout(self.relu(self.batchnorm3(self.conv3(x))))
        x1 = self.dropout(self.relu(self.batchnorm1_1(self.conv1_1(x1))))
        x2 = self.dropout(self.relu(self.batchnorm2_1(self.conv2_1(x2))))
        x3 = self.dropout(self.relu(self.batchnorm3_1(self.conv3_1(x3))))
        
        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)

        x = torch.cat((x1, x2, x3), dim=1)
        return x 


class CNN0_03(nn.Module):
    '''
    Use third level convolution.
    channel size: 4 -> 16 
                  8 -> 12
                  16 -> 4
    concat
    channel size: 8
    '''
    def __init__(self):
        super(CNN0_03, self).__init__()
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

        self.conv6 = nn.Conv2d(in_channels=128*3, out_channels=128*3, kernel_size=(8,1))
        self.batchnorm6 = nn.BatchNorm2d(num_features=128*3)

        self.conv7 = nn.Conv2d(in_channels=128*3, out_channels=128*3, kernel_size=(8,1))
        self.batchnorm7 = nn.BatchNorm2d(num_features=128*3)

        self.pool1 = nn.MaxPool2d(kernel_size=(1000-4+1-16+1-8+1,1), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(1000-8+1-12+1-8+1,1), stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(1000-16+1-4+1-8+1,1), stride=1)
        
        self.pool = nn.MaxPool2d(kernel_size=(1000-16+1-4+1-8+1-8+1-8+1-8+1,1), stride=1)
        
    def forward(self, x):
        x1 = self.relu(self.batchnorm1(self.conv1(x)))
        x2 = self.relu(self.batchnorm2(self.conv2(x)))
        x3 = self.relu(self.batchnorm3(self.conv3(x)))
        x1 = self.relu(self.batchnorm1_1(self.conv1_1(x1)))
        x2 = self.relu(self.batchnorm2_1(self.conv2_1(x2)))
        x3 = self.relu(self.batchnorm3_1(self.conv3_1(x3)))
        
        # x1 = self.pool1(x1)
        # x2 = self.pool2(x2)
        # x3 = self.pool3(x3)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.relu(self.batchnorm4(self.conv4(x)))
        x = self.relu(self.batchnorm5(self.conv5(x)))
        x = self.relu(self.batchnorm6(self.conv6(x)))
        x = self.relu(self.batchnorm7(self.conv7(x)))
        x = self.pool(x)
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


class DeepTFactor_gradCAM(nn.Module):
    def __init__(self, out_features):
        super(DeepTFactor_gradCAM, self).__init__()
        self.explainECs = out_features

        self.cnn0 = CNN0_1
        self.fc = nn.Linear(in_features=128, out_features=len(out_features))
        self.bn1 = nn.BatchNorm1d(num_features=len(out_features))
        self.out_act = nn.Sigmoid()

        self.cnn0.CAM_relu.register_forward_hook(self.forward_hook)
        self.cnn0.CAM_relu.register_backward_hook(self.backward_hook)
        self.cnn0.conv.register_forward_hook(self.forward_hook_grad)
        self.cnn0.conv.register_backward_hook(self.backward_hook_grad)
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


class CNN0_1(nn.Module):
    '''
    Use second level convolution.
    channel size: 4 -> 16 
                  8 -> 12
                  16 -> 4
    '''
    def __init__(self):
        super(CNN0_1, self).__init__()
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

        self.pool1 = nn.MaxPool2d(kernel_size=(1000-4+1-16+1,1), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(1000-8+1-12+1,1), stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(1000-16+1-4+1,1), stride=1)

        self.conv = nn.Conv2d(in_channels=128*3, out_channels=128, kernel_size=(1,1))
        self.CAM_relu = nn.ReLU()
        
    def forward(self, x):
        x1 = self.dropout(self.relu(self.batchnorm1(self.conv1(x))))
        x2 = self.dropout(self.relu(self.batchnorm2(self.conv2(x))))
        x3 = self.dropout(self.relu(self.batchnorm3(self.conv3(x))))
        x1 = self.dropout(self.relu(self.batchnorm1_1(self.conv1_1(x1))))
        x2 = self.dropout(self.relu(self.batchnorm2_1(self.conv2_1(x2))))
        x3 = self.dropout(self.relu(self.batchnorm3_1(self.conv3_1(x3))))
        
        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.CAM_relu(self.conv(x))
        return x