import os
import random
import logging
# import basic python packages
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score

# import torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from deepec.process_data import read_EC_Fasta, \
                                getExplainedEC_short, \
                                convertECtoLevel3


from deepec.data_loader import ECEmbedDataset as ECDataset

from deepec.utils import argument_parser, EarlyStopping, \
                         draw, save_losses, train_model_emb_sch, evalulate_model_emb
    
# from deepec.model import DeepECv2_2 as DeepECv2

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResEC(nn.Module):
    def __init__(self, block, num_blocks, explainECs=[]):
        super(ResEC, self).__init__()
        self.explainECs = explainECs
        num_classes = len(explainECs)
        self.in_planes = 64
        self.embedding = nn.Embedding(21, 10) # 20 AA + X + blank
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=(125,2), stride=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.embedding(x.type(torch.long))
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out




class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha==None:
            self.alpha=1
        else:
            self.alpha = torch.Tensor(alpha).view(-1, 1)
        
    def forward(self, pred, label):
        # pt = label*pred + (1-label)*(1-pred)
        # loss = -(1-pt).pow(self.gamma) * (self.alpha*torch.log(pt))
        # return loss.mean()
        BCE_loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()



if __name__ == '__main__':
    parser = argument_parser()
    options = parser.parse_args()

    output_dir = options.output_dir
    log_dir = options.log_dir

    device = options.gpu
    num_epochs = options.epoch
    batch_size = options.batch_size
    learning_rate = options.learning_rate
    patience = options.patience

    checkpt_file = options.checkpoint
    input_data_file = options.seq_file

    third_level = options.third_level
    num_cpu = options.cpu_num

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(f'{output_dir}/{log_dir}')
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    seed_num = 123 # random seed for reproducibility
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    np.random.seed(seed_num)

    torch.set_num_threads(num_cpu)

    gamma = 3

    logging.info(f'\nInitial Setting\
                  \nEpoch: {num_epochs}\
                  \tGamma: {gamma}\
                  \tBatch size: {batch_size}\
                  \tLearning rate: {learning_rate}\
                  \tPredict upto 3 level: {third_level}')


    input_seqs, input_ecs, input_ids = read_EC_Fasta(input_data_file)

    train_seqs, test_seqs = train_test_split(input_seqs, test_size=0.1, random_state=seed_num)
    train_ecs, test_ecs = train_test_split(input_ecs, test_size=0.1, random_state=seed_num)
    # train_ids, test_ids = train_test_split(input_ids, test_size=0.1, random_state=seed_num)

    train_seqs, val_seqs = train_test_split(train_seqs, test_size=1/9, random_state=seed_num)
    train_ecs, val_ecs = train_test_split(train_ecs, test_size=1/9, random_state=seed_num)
    # train_ids, val_ids = train_test_split(input_ids, test_size=1/9, random_state=seed_num)

    len_train_seq = len(train_seqs)
    len_valid_seq = len(val_seqs)
    len_test_seq = len(test_seqs)

    logging.info(f'Number of sequences used- Train: {len_train_seq}')
    logging.info(f'Number of sequences used- Validation: {len_valid_seq}')
    logging.info(f'Number of sequences used- Test: {len_test_seq}')


    explainECs = []
    for ec_data in [train_ecs, val_ecs, test_ecs]:
        for ecs in ec_data:
            for each_ec in ecs:
                if each_ec not in explainECs:
                    explainECs.append(each_ec)
    explainECs.sort()

    if third_level:
        logging.info('Predict EC number upto third level')
        explainECs = getExplainedEC_short(explainECs)
        train_ecs = convertECtoLevel3(train_ecs)
        val_ecs = convertECtoLevel3(val_ecs)
        test_ecs = convertECtoLevel3(test_ecs)
    else:
        logging.info('Predict EC number upto fourth level')

    train_ec_types = []
    for ecs in train_ecs:
        for each_ec in ecs:
            if each_ec not in train_ec_types:
                train_ec_types.append(each_ec)
    val_ec_types = []
    for ecs in val_ecs:
        for each_ec in ecs:
            if each_ec not in val_ec_types:
                val_ec_types.append(each_ec)
    test_ec_types = []
    for ecs in test_ecs:
        for each_ec in ecs:
            if each_ec not in test_ec_types:
                test_ec_types.append(each_ec)

    logging.info(f'Number of ECs in train data: {len(train_ec_types)}')
    logging.info(f'Number of ECs in validation data: {len(val_ec_types)}')
    logging.info(f'Number of ECs in test data: {len(test_ec_types)}')

    trainDataset = ECDataset(train_seqs, train_ecs, explainECs)
    valDataset = ECDataset(val_seqs, val_ecs, explainECs)
    testDataset = ECDataset(test_seqs, test_ecs, explainECs)

    trainDataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    validDataloader = DataLoader(valDataset, batch_size=batch_size, shuffle=True)
    testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)


    model = ResEC(Bottleneck, [1, 1, 1, 1], explainECs=explainECs)
    logging.info(f'Model Architecture: \n{model}')
    model = model.to(device)
    num_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of trainable parameters: {num_train_params}')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logging.info('Adam Optimizer')
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    criterion = FocalLoss(gamma=gamma)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
    logging.info(f'Learning rate scheduling: step size: 3\tgamma: 0.9')

    model, avg_train_losses, avg_valid_losses = train_model_emb_sch(
        model, optimizer, criterion, device,
        scheduler, batch_size, patience, num_epochs, 
        trainDataloader, validDataloader,
        f'{output_dir}/{checkpt_file}'
        )

    save_losses(avg_train_losses, avg_valid_losses, output_dir=output_dir)
    draw(avg_train_losses, avg_valid_losses, output_dir=output_dir)

    ckpt = torch.load(f'{output_dir}/{checkpt_file}')
    model.load_state_dict(ckpt['model'])

    y_true, y_score, y_pred = evalulate_model_emb(
        model, testDataloader, len(testDataset), explainECs, device
        )
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    logging.info(f'Precision: {precision}\tRecall: {recall}\tF1: {f1}')
    