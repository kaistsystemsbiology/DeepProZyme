# import
import re
import logging

# import basic python packages
from Bio import SeqIO



def read_EC_Fasta(fasta_file):
    sequences = []
    ecs = []
    ids = []
    fp = open(fasta_file, 'r')
    for seq_record in SeqIO.parse(fp, 'fasta'):
        seq = seq_record.seq
        seq_ecs = seq_record.description.split('\t')[1]
        seq_id = seq_record.description.split('\t')[0]
        seq_ecs = seq_ecs.split(';')
        sequences.append(seq)
        ecs.append(seq_ecs)
        ids.append(seq_id)
    fp.close()
    return sequences, ecs, ids


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