import os
import random
# import basic python packages
import numpy as np

# import torch packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from deepec.process_data import read_actual_Fasta
from deepec.data_loader import EnzymeDataset
from deepec.utils import argument_parser
from deepec.tf_models import DeepEC



if __name__ == '__main__':
    parser = argument_parser()
    options = parser.parse_args()

    device = options.gpu
    num_cpu = options.cpu_num
    batch_size = options.batch_size

    output_dir = options.output_dir
    checkpt_file = options.checkpoint
    protein_data_file = options.enzyme_data

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    seed_num = 123 # random seed for reproducibility
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    np.random.seed(seed_num)

    torch.set_num_threads(num_cpu)

    protein_seqs, seq_ids = read_actual_Fasta(protein_data_file)
    pseudo_labels = np.zeros((len(protein_seqs)))
    proteinDataset = EnzymeDataset(protein_seqs, pseudo_labels)
    proteinDataloader = DataLoader(proteinDataset, batch_size=batch_size, shuffle=False)


    model = DeepEC(out_features=[1], basal_net='CNN0_0')
    model = model.to(device)

    ckpt = torch.load(f'{checkpt_file}', map_location=device)
    model.load_state_dict(ckpt['model'])

    y_pred = torch.zeros([len(seq_ids), 1])
    with torch.no_grad():
        model.eval()
        cnt = 0
        for x, _ in proteinDataloader:
            x = x.type(torch.FloatTensor)
            x_length = x.shape[0]
            output = model(x.to(device))
            prediction = output.cpu()
            y_pred[cnt:cnt+x_length] = prediction
            cnt += x_length

    scores = y_pred[:,0]
    with open(f'{output_dir}/prediction_result.txt', 'w') as fp:
        fp.write('sequence_ID\tprediction\tscore\n')
        for seq_id, score in zip(seq_ids, scores):
            if score > 0.5:
                tf = True
            else:
                tf = False
            fp.write(f'{seq_id}\t{tf}\t{score:0.4f}\n')


