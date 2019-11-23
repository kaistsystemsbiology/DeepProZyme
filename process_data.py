# import

import re
import logging

# import basic python packages
import numpy as np
import pandas as pd

from Bio import SeqIO

# import torch packages
import torch
from torch.utils.data import DataLoader, Dataset

# import scikit learn packages
from sklearn.model_selection import train_test_split



def read_SP_Fasta(fasta_file, len_criteria=1000):
    result = []
    fp = open(fasta_file, 'r')
    for seq_record in SeqIO.parse(fp, 'fasta'):
        seq = seq_record.seq
        if len(seq) <= len_criteria:
            seq += '_' * (len_criteria-len(seq))
            result.append(str(seq))
    fp.close()
    return result


def read_EC_Fasta(fasta_file):
    sequences = []
    ecs = []
    fp = open(fasta_file, 'r')
    for seq_record in SeqIO.parse(fp, 'fasta'):
        seq = seq_record.seq
        seq_ecs = seq_record.description.split('\t')[1]
        seq_ecs = seq_ecs.split(';')
        sequences.append(seq)
        ecs.append(seq_ecs)
    fp.close()
    return sequences, ecs


def readFasta(fasta_file, len_criteria=1000):
    id2seq = {}
    id2ec = {}
    fp = open(fasta_file, 'r')
    cnt = 0
    for seq_record in SeqIO.parse(fp, 'fasta'):
        seq_info = seq_record.id.split('|')
        seq_id = seq_info[0]
        seq_ec = seq_info[1]
        seq = seq_record.seq
        if 'B' in seq or 'O' in seq or 'U' in seq or 'Z' in seq:
            continue
        if len(seq) <= len_criteria:
            seq += '_' * (len_criteria-len(seq))
            id2seq[seq_id] = str(seq)
            id2ec[seq_id] = []
    fp.close()
    
    fp = open(fasta_file, 'r')
    for seq_record in SeqIO.parse(fp, 'fasta'):
        seq_info = seq_record.id.split('|')
        seq_id = seq_info[0]
        seq_ec = seq_info[1]
        seq = seq_record.seq
        if 'B' in seq or 'O' in seq or 'U' in seq or 'Z' in seq:
            continue
        if len(seq) <= len_criteria:
            if seq_ec not in id2ec[seq_id]:
                id2ec[seq_id].append(seq_ec)
            cnt += 1
    fp.close()
    logging.info(f'{cnt} number of sequences are processed')
    return id2seq, id2ec


def deleteLowConf(id2seq, id2ec, id2ec_low_confi):
    cnt = 0
    for seqID in id2ec_low_confi:
        if seqID in id2seq:
            del id2seq[seqID]
            del id2ec[seqID]
            cnt += 1
    logging.info(f'{cnt} low confidence sequences are removed')
    return


def getExplainedEC(id2ec_train, id2ec_val, id2ec_test):
    ec2cnt = {}
    for id2ec in [id2ec_train, id2ec_val, id2ec_test]:
        for seqid in id2ec:
            for each_ec in id2ec[seqid]:
                ec2cnt[each_ec] = 0
    logging.info('Number of EC types: %d'%len(ec2cnt))

    for id2ec in [id2ec_train, id2ec_val, id2ec_test]:
        for seqid in id2ec:
            for each_ec in id2ec[seqid]:
                ec2cnt[each_ec] += 1
    explainECs = []
    for ec in ec2cnt:
        if ec2cnt[ec] < 10:
            continue
        explainECs.append(ec)
    explainECs.sort()

    logging.info('Number of Explained EC: %d'%len(explainECs))
    return explainECs


def getExplainedEC_short(explainECs):
    p = re.compile('EC:\S[.]\S+[.]\S+[.]')
    tmp = []
    for item in explainECs:
        matched = p.match(item)
        tmp.append(matched.group())
    explainECs_short = list(set(tmp))
    explainECs_short.sort()
    num_ec = len(explainECs_short)
    logging.info(f'Number of Explained EC in 3 level: {num_ec}')
    return explainECs_short


def getExplainableData(id2seq, id2ec, explainECs):
    Xs, Ys = [], []
    for seq_id in id2ec:
        tmp_y = []
        for each_ec in id2ec[seq_id]:
            if each_ec in explainECs:
                tmp_y.append(each_ec)
        if len(tmp_y):
            Xs.append(id2seq[seq_id])
            Ys.append(tmp_y)
    return Xs, Ys


def convertECtoLevel3(ecs):
    ecs_short = []
    p = re.compile('EC:\S[.]\S+[.]\S+[.]')
    for i, item in enumerate(ecs):
        tmp = []
        for each_ec in item:
            tmp_ec = p.match(each_ec).group()
            if tmp_ec not in tmp:
                tmp.append(tmp_ec)
        ecs_short.append(tmp)
    return ecs_short


