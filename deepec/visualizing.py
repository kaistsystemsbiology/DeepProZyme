import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from skimage.transform import resize

from Bio import SeqIO

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from deepec.data_loader import ECDataset


def getCAMMap(weight_sigmoid, features, ec_ind, seq_num):
    weight_c = torch.Tensor(weight_sigmoid[ec_ind])
    F_ks = features[seq_num].cpu()
    M_c = torch.zeros(F_ks.shape[1:])
    for i in range(len(weight_c)):
        M_c += weight_c[i] * F_ks[i]
    cam = M_c - torch.min(M_c)
    cam_img = cam / torch.max(cam)
    return resize(cam_img.numpy(), (1000, 10))


def drawCAMMap(cam_img, seq_len, output_dir, seq_name='seq'):
    plt.imshow(np.transpose(cam_img[:seq_len]))
    plt.savefig(f'{output_dir}/{seq_name}.png', dpi=600)
    plt.close()
    return


def getHighlights(cam_img, criteria=0.5):
    seq_info = np.asarray([item[0] for item in cam_img])
    locs = []
    for i, val in enumerate(seq_info):
        if val > criteria:
            locs.append(i)
    return np.asarray(locs)


def extractCorrectPreds(y_true, y_pred, test_seqs, test_ecs):
    compared = y_true == y_pred
    correct_preds = []
    for i in range(len(compared)):
        if compared[i].all():
            correct_preds.append(i)
    correct_seqs = [test_seqs[i] for i in correct_preds]
    correct_ecs = [test_ecs[i] for i in correct_preds]
    return correct_preds, correct_seqs, correct_ecs


def seeCAMimg(model, weight_sigmoid, seq_num, correct_preds, correct_seqs, correct_ecs, device):
    if seq_num not in correct_preds:
        print('Did not correctly predicted seq')
        return
    tmp_ind = correct_preds.index(seq_num)
    seq = [correct_seqs[tmp_ind]]
    ecs = [correct_ecs[tmp_ind]]
    print('\t'.join(ecs[0]))
    testDataset = ECDataset(seq, ecs, model.explainECs, pred=True)
    testDataloader = DataLoader(testDataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for x in testDataloader:
            _, feature = model(x.type(torch.FloatTensor).to(device))
    return getCAMMap(weight_sigmoid, feature, model.explainECs.index(ecs[0][0]), 0)


def summaryCAM(model, weight_sigmoid, seq_num, correct_preds, correct_seqs, correct_ecs, test_seqs, test_ids, device):
    cam_img = seeCAMimg(model, weight_sigmoid, seq_num, correct_preds, correct_seqs, correct_ecs, device)
    attentions = getHighlights(cam_img[:len(test_seqs[seq_num])])
    print(test_ids[seq_num])
    print(attentions)
    print([str(test_seqs[seq_num])[item] for item in attentions])
    drawCAMMap(cam_img, len(test_seqs[seq_num]), './attentions', f'{test_ids[seq_num]}')
    return