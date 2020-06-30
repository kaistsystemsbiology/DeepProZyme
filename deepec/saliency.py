# import torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F

# import basic python packages
import numpy as np
import matplotlib.pyplot as plt


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
#         self.probs = F.softmax(self.logits, dim=1)
        self.probs = F.relu(self.logits,)
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
#             if isinstance(module, nn.ReLU):
#                 tmp = F.relu(grad_in[0])
#                 pos_grad = torch.clamp(input=tmp, min=0.0)
#                 return (pos_grad,)

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
        for name, module in self.model.cnn0.named_modules():
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


def save_gradient(filename, gradient, seq_len):
    gradient = gradient[:,:seq_len, :]
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient /= np.abs(gradient).max()
    gradient = gradient[:,:,0]
    plt.figure(figsize=(2, 20))
    plt.imshow(gradient, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.tight_layout()
    num_ticks = seq_len//10
    plt.yticks([i*10 for i in range(num_ticks+1)])
    plt.savefig(filename)
    plt.close()


def save_gradcam(filename, gcam, seq_len):
    gcam = gcam[:seq_len, :]
    gcam = gcam.cpu().numpy()
    gcam -= gcam.min()
    gcam /= gcam.max()
    plt.figure(figsize=(2, 20))
    plt.tight_layout()
    num_ticks = seq_len//10
    plt.yticks([i*10 for i in range(num_ticks+1)])
    plt.imshow(gcam, cmap='Greys')
    
    plt.savefig(filename)
    plt.close()


def analyzeRow(j, i, regions, gradients):
    aa_vocab = ['A', 'C', 'D', 'E', 
                'F', 'G', 'H', 'I', 
                'K', 'L', 'M', 'N', 
                'P', 'Q', 'R', 'S',
                'T', 'V', 'W', 'X', 
                'Y']
    seq_len = len(protein_seqs[j].split('_')[0])
    gradient = torch.mul(regions, gradients)[j]
    gradient = gradient[:,:seq_len, :]
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient /= np.abs(gradient).max()
    gradient = gradient[:,:,0]
    for aa, each_grad in zip(aa_vocab, gradient[i]):
        print(aa, each_grad)
        

def analyzeGradient(j, i, gradients):
    aa_vocab = ['A', 'C', 'D', 'E', 
                'F', 'G', 'H', 'I', 
                'K', 'L', 'M', 'N', 
                'P', 'Q', 'R', 'S',
                'T', 'V', 'W', 'X', 
                'Y']
    seq_len = len(protein_seqs[j].split('_')[0])
    gradient = gradients[j][:,:seq_len, :]
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient /= np.abs(gradient).max()
    gradient = gradient[:,:,0]
    for aa, each_grad in zip(aa_vocab, gradient[i]):
        print(aa, each_grad)
    print('\nhighlighted\n')
    for aa, each_grad in zip(aa_vocab, gradient[i]):
        if each_grad > 0:
            print(aa, each_grad)