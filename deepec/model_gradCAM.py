# import torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


## https://github.com/kazuto1011/grad-cam-pytorch/blob/master/LICENSE
class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient


class GuidedBackPropagation(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_in[0]),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam