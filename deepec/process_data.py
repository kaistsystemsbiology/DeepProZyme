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