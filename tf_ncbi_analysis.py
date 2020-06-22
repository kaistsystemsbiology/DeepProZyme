import os
import gzip
import random
import logging

from Bio import SeqIO
# import basic python packages
import numpy as np

# import torch packages
import torch
from torch.utils.data import DataLoader

from deepec.data_loader import EnzymeDataset
from deepec.utils import argument_parser
from deepec.tf_models import DeepTFactor_2 as DeepTFactor


def read_ncbi_data(fasta_dir):
    sequences = []
    ids = []
    acceptable_aas = set([
        'A', 'C', 'D', 'E', 'F', 
        'G', 'H', 'I', 'K', 'L',
        'M', 'N', 'P', 'Q', 'R',
        'S', 'T', 'V', 'W', 'X', 
        'Y'
        ])
    len_criteria = 1000

    with gzip.open(fasta_dir, 'rt') as handle:
        for seq_record in SeqIO.parse(handle, 'fasta'):
            seq = seq_record.seq
            if len(seq) > len_criteria:
                continue
            exclusive_aas = set(list(seq)) - acceptable_aas
            if len(seq) <= len_criteria:
                seq += '_' * (len_criteria-len(seq))
            if len(exclusive_aas) > 0:
                continue
            desc = seq_record.description
            sequences.append(seq)
            ids.append(desc)
    return sequences, ids


if __name__ == '__main__':
    parser = argument_parser()
    options = parser.parse_args()

    device = options.gpu
    num_cpu = options.cpu_num
    batch_size = options.batch_size

    output_dir = options.output_dir
    checkpt_file = options.checkpoint


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.set_num_threads(num_cpu)


    model = DeepTFactor(out_features=[1])
    model = model.to(device)

    ckpt = torch.load(f'{checkpt_file}', map_location=device)
    model.load_state_dict(ckpt['model'])
    # cutoff = ckpt['cutoff']
    cutoff = 0.5

    ncbi_data_dir = '../../../SeqData/NCBI_complete_genome/20200211_genomes/'
    ncbi_data_list = [item for item in os.listdir(ncbi_data_dir) if '.faa.gz' in item]
    logging.info(f'{len(ncbi_data_dir)} complete genomes are in ready')

    for ncbi_data in ncbi_data_list:
        protein_seqs, seq_ids = read_ncbi_data(ncbi_data_dir + ncbi_data)
        pseudo_labels = np.zeros((len(protein_seqs)))
        proteinDataset = EnzymeDataset(protein_seqs, pseudo_labels)
        proteinDataloader = DataLoader(proteinDataset, batch_size=batch_size, shuffle=False)

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
        with open(f'{output_dir}/{ncbi_data}.txt', 'w') as fp:
            fp.write('sequence_ID\tprediction\tscore\n')
            for seq_id, score in zip(seq_ids, scores):
                if score > cutoff:
                    tf = True
                else:
                    tf = False
                fp.write(f'{seq_id}\t{tf}\t{score:0.4f}\n')