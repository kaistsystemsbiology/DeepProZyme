import re
import os
import random
import logging
from copy import deepcopy
# import basic python packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from Bio import SeqIO
from tqdm.auto import tqdm
# import torch packages
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deepec.data_loader import DeepECDataset


def read_EC_Fasta(fasta_file):
    columns_aas = [
        'A', 'C', 'D', 'E', 
        'F', 'G', 'H', 'I', 
        'K', 'L', 'M', 'N', 
        'P', 'Q', 'R', 'S',
        'T', 'V', 'W', 'Y'
    ]
    sequences = []
    ecs = []
    ids = []
    fp = open(fasta_file, 'r')
    for seq_record in SeqIO.parse(fp, 'fasta'):
        seq = seq_record.seq
        if len(seq) >= 1000:
            continue
        if len(set(seq) - set(columns_aas)) > 0:
            continue
        seq_ecs = seq_record.description.split('\t')[1]
        seq_id = seq_record.description.split('\t')[0]
        seq_ecs = seq_ecs.split(';')
        sequences.append(seq)
        ecs.append(seq_ecs)
        ids.append(seq_id)
    fp.close()
    return sequences, ecs, ids


def read_Enzyme_Fasta(fasta_file):
    sequences = []
    enzymes = []
    ids = []
    fp = open(fasta_file, 'r')
    for seq_record in SeqIO.parse(fp, 'fasta'):
        if len(seq) >= 1000:
            continue
        seq = seq_record.seq
        seq_id = seq_record.id
        enzyme = float(seq_record.description.split('\t')[1])
        sequences.append(seq)
        enzymes.append(enzyme)
        ids.append(seq_id)
    fp.close()
    return sequences, enzymes, ids


def read_EC_actual_Fasta(fasta_file):
    sequences = []
    ids = []
    fp = open(fasta_file, 'r')
    for seq_record in SeqIO.parse(fp, 'fasta'):
        seq = seq_record.seq
        seq_id = seq_record.id

        sequences.append(seq)
        ids.append(seq_id)
    fp.close()
    return sequences, ids



def get_attention_map(model, input_ids, token_type_ids, attention_mask, output_attentions=False):
    _, _, attentions = model.bert(input_ids, token_type_ids, attention_mask, )
    return attentions


def _get_highlighted(df_array, input_seq, max_num):
    highlighted_residues = pd.DataFrame(df_array).sum(axis=1).sort_values(ascending=False)
    highlighted_residues = highlighted_residues[:max_num]
    residues = {}
    for ix in highlighted_residues.index:
        residues[ix] = input_seq[ix]
    return residues
    

def get_highlighted_residues(input_seq, attention, head_num):
    columns_aas = [
        'A', 'C', 'D', 'E', 
        'F', 'G', 'H', 'I', 
        'K', 'L', 'M', 'N', 
        'P', 'Q', 'R', 'S',
        'T', 'V', 'W', 'Y'
    ]
    highlighted_residues = {}
    
    for head in range(head_num):
        attn = attention[head][1:len(input_seq)+1, 1:len(input_seq)+1].detach()
        
        df_array = np.zeros((len(input_seq), 20))
        for j, aa in enumerate(input_seq):
            aa_ind = columns_aas.index(aa)
            df_array[j][aa_ind] = attn.mean(dim=0)[j]
        highlighted_residues[head] = _get_highlighted(df_array, input_seq, max_num=2)
    return highlighted_residues
        

def mutated_pts(seq, highlighted_dir):
    p_num = re.compile('(\d+)(.*)')
    head2muts = {}
    with open(highlighted_dir, 'r') as fp:
        fp.readline()
        while True:
            line = fp.readline()
            if not line:
                break
            head, muts = line.strip().split('\t')
            mutations = {}
            for mut in muts.split(';'):
                m_num = p_num.match(mut)
                aa = m_num.group(2)
                if aa == 'A':
                    continue
                pos = int(m_num.group(1))
                mutations[pos] = aa
            head2muts[head] = mutations
    return head2muts

def mut_seq(seq, mutation={}):
    seq2 = list(seq)
    for ix, prev_aa in mutation.items():
        if seq2[ix] == prev_aa:
            seq2[ix] = 'A'
        else:
            print(f'Wrong!!\t{ix}\t{seq2[ix]}\t{prev_aa}')
    seq2 = ''.join(seq2)
    seq3 = ''
    for i in range(len(seq2)//60+1):
        seq3 += seq2[i*60:(i+1)*60]+'\n'
    return seq3



if __name__ == '__main__':
    # device = 'cpu'
    device = 'cuda:2'
    batch_size = 128
    checkpt_file = './Dataset/model.pth'

    # seq_dir = './analyze/Mannheimia/uniprot_mannheimia_enzymes.fa'
    # mut_dir = './analyze/Mannheimia/mutated_sequences'
    # seq_dir = './analyze/Escherichia_coli/uniprot_ecoli_k12_enzymes.fa'
    # mut_dir = './analyze/Escherichia_coli/mutated_sequences'
    seq_dir = './Dataset/swissprot_enzymes.fa'
    mut_dir = './analyze/SwissProt/mutated_sequences'

    if not os.path.exists(mut_dir):
        os.makedirs(mut_dir)

    model = torch.load(checkpt_file)
    model = model.to(device)
    model.bert.config.output_attentions = False
    model.eval()
    explainECs = model.explainECs
    pred_thrd = pred_thrd = model.thresholds.to(device)
    ec2ind = {ec: i for i, ec in enumerate(explainECs)}

    # input_seqs, input_ids = read_EC_actual_Fasta(seq_dir)
    input_seqs, input_ecs, input_ids = read_EC_Fasta(seq_dir)
    pseudo_labels = np.zeros((len(input_seqs)))

    proteinDataset = DeepECDataset(data_X=input_seqs, data_Y=pseudo_labels, explainECs=explainECs, pred=True)
    proteinDataloader = DataLoader(proteinDataset, batch_size=batch_size, shuffle=False)


    print('Step 1')
    ## Predict EC numbers of naive sequences
    with torch.no_grad():
        y_pred = torch.zeros([len(input_seqs), len(explainECs)])
        y_score = torch.zeros([len(input_seqs), len(explainECs)])
        cnt = 0
        for batch, data in enumerate(tqdm(proteinDataloader)):
            inputs = {key:val.to(device) for key, val in data.items()}
            output = model(**inputs)
            output = torch.sigmoid(output)
            prediction = output > pred_thrd
            prediction = prediction.float()
            step = data['input_ids'].shape[0]
            y_pred[cnt:cnt+step] = prediction.cpu()
            y_score[cnt:cnt+step] = output.cpu()
            cnt += step

    original_prediction = {}
    for i, ith_pred in enumerate(y_pred):
        nonzero_preds = torch.nonzero(ith_pred, as_tuple=False)
        if len(nonzero_preds) == 0:
            original_prediction[input_ids[i]] = 'None'
        else:
            pred_ecs = []
            for j in nonzero_preds:
                pred_ecs.append(explainECs[j])
            pred_ecs.sort()
            pred_ecs = ';'.join(pred_ecs)
            original_prediction[input_ids[i]] = pred_ecs


    print('Step 2')
    ## Get highlighted residues and alanine mutated sequences
    if not os.path.exists(mut_dir+'/indiv_seqs'):
        os.makedirs(mut_dir+'/indiv_seqs')
    model.bert.config.output_attentions = True
    with torch.no_grad():
        cnt = 0
        for batch, item in enumerate(tqdm(proteinDataloader)):
            step = item['input_ids'].shape[0]
            attentions = get_attention_map(model, **{key:val.to(device) for key, val in item.items()})
            layer_num = len(attentions)
            head_num = attentions[0].size()[1]

            for i in range(step):
                seq_id = input_ids[cnt + i]
                seq = input_seqs[cnt + i]
                seq_len = len(seq)
                f = open(mut_dir+'/indiv_seqs/'+seq_id+'_mutants.fa', 'w')
                for layer in range(layer_num):
                    attention = attentions[layer][i]
                    highlighted_residues = get_highlighted_residues(seq, attention, head_num) # {head: {ix: AA}}
                    
                    for head, mutation_info in highlighted_residues.items():
                        mutated_seq = mut_seq(seq, mutation_info)
                        mutated_info = []
                        for ix, aa in mutation_info.items():
                            mutated_info.append(f'{aa}{ix+1}A')
                        mutated_info = '_'.join(mutated_info)
                        f.write(f'>{seq_id}_{layer}_{head}_{mutated_info}\n{mutated_seq}')
                f.close()
            cnt += step

    print('Step 3')
    ## Predict EC numbers of the mutated sequences
    model.bert.config.output_attentions = False
    with torch.no_grad():
        for i, seq_id in enumerate(tqdm(input_ids)):
            naive_ec = original_prediction[seq_id]
            mut_seqs, mut_ids = read_EC_actual_Fasta(mut_dir+'/indiv_seqs/'+seq_id+'_mutants.fa')
            pseudo_labels = np.zeros((len(mut_seqs)))
            mutDataset = DeepECDataset(data_X=mut_seqs, data_Y=pseudo_labels, explainECs=explainECs, pred=True)
            mutDataloader = DataLoader(mutDataset, batch_size=len(mut_seqs), shuffle=False)

            for data in mutDataloader:
                inputs = {key:val.to(device) for key, val in data.items()}
                output = model(**inputs)
                output = torch.sigmoid(output)
                prediction = output > pred_thrd
                prediction = prediction.float().cpu()

            diff_prediction = {}
            for j, jth_pred in enumerate(prediction):
                nonzero_preds = torch.nonzero(jth_pred, as_tuple=False)
                if len(nonzero_preds) == 0:
                    mut_ec = 'None'
                else:
                    pred_ecs = []
                    for k in nonzero_preds:
                        pred_ecs.append(explainECs[k])
                    pred_ecs.sort()
                    pred_ecs = ';'.join(pred_ecs)
                    mut_ec = pred_ecs
                if mut_ec != naive_ec:
                    diff_prediction[mut_ids[j]] = (naive_ec, mut_ec, mut_seqs[j])
            

    print('Step 4')
    ## Save the result
    with open(mut_dir + '/' + 'changed_EC_seqs.fa', 'w') as fp:
        for seq_id, (prev_ec, new_ec, seq) in diff_prediction.items():
            fp.write(f'>{seq_id}\t{prev_ec}\t{new_ec}\n{seq}\n')