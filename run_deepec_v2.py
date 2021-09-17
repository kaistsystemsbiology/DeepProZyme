import os
import shutil
import random
import logging
# import basic python packages
import numpy as np

# import torch packages
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deepec.process_data import read_EC_actual_Fasta
from deepec.data_loader import DeepECDataset
from deepec.utils import argument_parser, run_neural_net, save_dl_result
from deepec.homology import run_blastp, read_best_blast_result, merge_predictions



logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')


'''
python run_deepec_v2.py -i ./Dataset/analysis_seqs/swissprot_20180412_20210531.fa -o ./output/bert_05/deepec_v2_new_seq -ckpt ./output/bert_05/model_single.pth -g cuda:3 -b 128 -cpu 2
'''
if __name__ == '__main__':
    parser = argument_parser()
    options = parser.parse_args()

    output_dir = options.output_dir
    checkpt_file = options.checkpoint
    input_data_file = options.seq_file

    device = options.gpu
    num_epochs = options.epoch
    batch_size = options.batch_size
    num_cpu = options.cpu_num

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(output_dir + '/tmp'):
        os.makedirs((output_dir+'/tmp'))


    seed_num = 123 # random seed for reproducibility
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    np.random.seed(seed_num)
    torch.set_num_threads(num_cpu)

    model = torch.load(checkpt_file)
    model = model.to(device)
    explainECs = model.explainECs
    
    input_seqs, input_ids = read_EC_actual_Fasta(input_data_file)
    pseudo_labels = np.zeros((len(input_seqs)))

    proteinDataset = DeepECDataset(data_X=input_seqs, data_Y=pseudo_labels, explainECs=explainECs, pred=True)
    proteinDataloader = DataLoader(proteinDataset, batch_size=batch_size, shuffle=False)


    id2ind = {seq_id: i for i, seq_id in enumerate(input_ids)}
    ec2ind = {ec: i for i, ec in enumerate(explainECs)}

    pred_thrd = torch.zeros(len(explainECs))
    with open('./output/bert_05/validation_set_thrd_tuning/F_score_thrds.txt', 'r') as fp:
        fp.readline()
        while True:
            line = fp.readline()
            if not line:
                break
            data = line.strip().split('\t')
            ec = data[0]
            thrd = float(data[-1])
            pred_thrd[ec2ind[ec]] = thrd

    y_pred, y_score = run_neural_net(model, proteinDataloader, pred_thrd, device=device)
    failed_cases = save_dl_result(y_pred, y_score, input_ids, explainECs, output_dir+'/tmp')

    if len(failed_cases) > 0:
        blastp_input = f'{output_dir}/tmp/temp_seq.fa'
        blastp_tmp_output = f'{output_dir}/tmp/blast_tmp_result.txt'
        blastp_output = f'{output_dir}/tmp/blast_result.txt'

        with open(blastp_input, 'w') as fp:
            for seq_id in failed_cases:
                idx = id2ind[seq_id]
                seq = input_seqs[idx]
                fp.write(f'>{seq_id}\n{seq}\n')

        run_blastp(blastp_input, blastp_tmp_output, './Dataset/swissprot_enzyme_diamond', threads=num_cpu)
        # run_blastp(blastp_input, blastp_tmp_output, './Dataset/swissprot_enzyme', threads=num_cpu)
        blastp_pred = read_best_blast_result(blastp_tmp_output)
        
        with open(blastp_output, 'w') as fp:
            fp.write('sequence_ID\tprediction\n')
            for seq_id in blastp_pred:
                ec = blastp_pred[seq_id][0]
                fp.write(f'{seq_id}\t{ec}\n')

        merge_predictions(f'{output_dir}/tmp/DL_prediction_result.txt', blastp_output, output_dir)
    
    else:
        shutil.copy(output_dir+'/tmp/DL_prediction_result.txt', output_dir)
        os.rename(output_dir+'/DL_prediction_result.txt', output_dir+'/DeepECv2_result.txt')