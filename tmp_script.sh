python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200615_01 -g cuda:0 -e 50 -b 32 -r 0.0001
python tf_running.py -c ./output/tf_deeptfactor_20200615_01/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200615_01/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_01/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200615_01/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_01/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_01/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_01/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_01/new_bacteria_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200615_02 -g cuda:0  -e 50 -b 16 -r 0.0003
python tf_running.py -c ./output/tf_deeptfactor_20200615_02/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200615_02/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_02/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200615_02/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_02/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_02/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_02/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_02/new_bacteria_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200615_03 -g cuda:0  -e 50 -b 16 -r 0.001
python tf_running.py -c ./output/tf_deeptfactor_20200615_03/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200615_03/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_03/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200615_03/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_03/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_03/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_03/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_03/new_bacteria_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200615_04 -g cuda:0  -e 50 -b 16 -r 0.003
python tf_running.py -c ./output/tf_deeptfactor_20200615_04/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200615_04/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_04/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200615_04/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_04/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_04/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_04/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_04/new_bacteria_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200615_05 -g cuda:0  -e 50 -b 16 -r 0.0001
python tf_running.py -c ./output/tf_deeptfactor_20200615_05/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200615_05/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_05/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200615_05/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_05/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_05/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_05/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_05/new_bacteria_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200615_06 -g cuda:0  -e 50 -b 32 -r 0.0003
python tf_running.py -c ./output/tf_deeptfactor_20200615_06/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200615_06/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_06/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200615_06/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_06/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_06/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_06/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_06/new_bacteria_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200615_07 -g cuda:0  -e 50 -b 32 -r 0.001
python tf_running.py -c ./output/tf_deeptfactor_20200615_07/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200615_07/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_07/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200615_07/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_07/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_07/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_07/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_07/new_bacteria_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200615_08 -g cuda:0  -e 50 -b 32 -r 0.003
python tf_running.py -c ./output/tf_deeptfactor_20200615_08/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200615_08/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_08/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200615_08/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_08/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_08/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_08/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_08/new_bacteria_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200615_09 -g cuda:0  -e 50 -b 64 -r 0.0001
python tf_running.py -c ./output/tf_deeptfactor_20200615_09/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200615_09/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_09/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200615_09/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_09/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_09/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_09/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_09/new_bacteria_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200615_10 -g cuda:0  -e 50 -b 64 -r 0.0003
python tf_running.py -c ./output/tf_deeptfactor_20200615_10/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200615_10/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_10/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200615_10/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_10/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_10/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_10/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_10/new_bacteria_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200615_11 -g cuda:0  -e 50 -b 64 -r 0.001
python tf_running.py -c ./output/tf_deeptfactor_20200615_11/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200615_11/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_11/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200615_11/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_11/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_11/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_11/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_11/new_bacteria_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200615_12 -g cuda:0  -e 50 -b 64 -r 0.003
python tf_running.py -c ./output/tf_deeptfactor_20200615_12/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200615_12/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_12/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200615_12/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_12/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_12/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_12/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_12/new_bacteria_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200615_13 -g cuda:0  -e 50 -b 128 -r 0.0001
python tf_running.py -c ./output/tf_deeptfactor_20200615_13/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200615_13/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_13/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200615_13/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_13/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_13/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_13/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_13/new_bacteria_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200615_14 -g cuda:0  -e 50 -b 128 -r 0.0003
python tf_running.py -c ./output/tf_deeptfactor_20200615_14/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200615_14/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_14/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200615_14/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_14/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_14/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_14/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_14/new_bacteria_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200615_15 -g cuda:0  -e 50 -b 128 -r 0.001
python tf_running.py -c ./output/tf_deeptfactor_20200615_15/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200615_15/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_15/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200615_15/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_15/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_15/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_15/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_15/new_bacteria_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200615_16 -g cuda:0  -e 50 -b 128 -r 0.003
python tf_running.py -c ./output/tf_deeptfactor_20200615_16/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200615_16/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_16/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200615_16/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_16/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_16/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200615_16/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200615_16/new_bacteria_nontf -g cuda:0