import logging
import subprocess


def run_blastp(target_fasta, blastp_result, db_dir, threads=1):
    logging.info('BLASTp prediction starts on the dataset')
    subprocess.call(
        "diamond blastp -d %s -q %s -o %s --threads %s --id 50 --outfmt 6 qseqid sseqid evalue score qlen slen length pident"%(db_dir, target_fasta, blastp_result, threads), 
        shell=True, 
        stderr=subprocess.STDOUT
    )
    logging.info('BLASTp prediction ended on the dataset')


def read_best_blast_result(blastp_result):
    query_db_set_info = {}
    with open(blastp_result, 'r') as fp:
        for line in fp:
            sptlist = line.strip().split('\t')            
            query_id = sptlist[0].strip()
            db_id = sptlist[1].strip()  
            
            ec_number = db_id.split('|')[1].strip()
            score = float(sptlist[3].strip())
            qlen = sptlist[4].strip()
            length = sptlist[6].strip()
            length = float(length)
            pident = float(sptlist[-1].strip())

            ec_number = ec_number.split(';')
            ec_numbers = []
            for item in ec_number:
                if 'EC:' in item:
                    ec_numbers.append(item)
                else:
                    ec_numbers.append(f'EC:{item}')
            ec_numbers.sort()
            ec_numbers = ';'.join(ec_numbers)

            if pident < 50:
                continue
            coverage = length/float(qlen)*100
            if coverage >= 75:
                if query_id not in query_db_set_info:
                    query_db_set_info[query_id] = [ec_numbers, score]
                else:
                    p_score = query_db_set_info[query_id][1]
                    if score > p_score:
                        query_db_set_info[query_id] = [ec_numbers, score]
    return query_db_set_info


def merge_predictions(dl_pred_result, blastp_pred_result, output_dir):
    dl_pred = {}
    with open(dl_pred_result, 'r') as f1:
        f1.readline()
        while True:
            line = f1.readline()
            if not line:
                break
            seq_id, ec, _  = line.strip().split('\t')
            if seq_id not in dl_pred:
                dl_pred[seq_id] = ec
            else:
                dl_pred[seq_id] += f';{ec}'
    
    blastp_pred = {}
    with open(blastp_pred_result, 'r') as f2:
        f2.readline()
        while True:
            line = f2.readline()
            if not line:
                break
            seq_id, ec = line.strip().split('\t')
            blastp_pred[seq_id] = ec

    merged = {}
    for seq_id, pred in dl_pred.items():
        if pred == 'None':
            if seq_id in blastp_pred:
                merged[seq_id] = blastp_pred[seq_id]
            else:
                merged[seq_id] = 'None'
        else:
            merged[seq_id] = dl_pred[seq_id]
    
    with open(f'{output_dir}/DeepECv2_result.txt', 'w') as fp:
        fp.write('sequence_ID\tprediction\n')
        for seq_id, pred in merged.items():
            fp.write(f'{seq_id}\t{pred}\n')