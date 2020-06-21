python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200619_01 -g cuda:0 -e 50 -b 16 -r 0.0001
python tf_running.py -c ./output/tf_deeptfactor_20200619_01/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200619_01/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_01/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200619_01/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_01/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_01/new_prokaryote_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_01/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_01/new_prokaryote_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200619_02 -g cuda:0 -e 50 -b 16 -r 0.0003
python tf_running.py -c ./output/tf_deeptfactor_20200619_02/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200619_02/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_02/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200619_02/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_02/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_02/new_prokaryote_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_02/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_02/new_prokaryote_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200619_03 -g cuda:0 -e 50 -b 16 -r 0.001
python tf_running.py -c ./output/tf_deeptfactor_20200619_03/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200619_03/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_03/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200619_03/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_03/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_03/new_prokaryote_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_03/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_03/new_prokaryote_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200619_04 -g cuda:0 -e 50 -b 16 -r 0.003
python tf_running.py -c ./output/tf_deeptfactor_20200619_04/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200619_04/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_04/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200619_04/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_04/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_04/new_prokaryote_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_04/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_04/new_prokaryote_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200619_05 -g cuda:0 -e 50 -b 32 -r 0.0001
python tf_running.py -c ./output/tf_deeptfactor_20200619_05/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200619_05/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_05/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200619_05/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_05/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_05/new_prokaryote_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_05/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_05/new_prokaryote_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200619_06 -g cuda:0 -e 50 -b 32 -r 0.0003
python tf_running.py -c ./output/tf_deeptfactor_20200619_06/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200619_06/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_06/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200619_06/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_06/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_06/new_prokaryote_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_06/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_06/new_prokaryote_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200619_07 -g cuda:0 -e 50 -b 32 -r 0.001
python tf_running.py -c ./output/tf_deeptfactor_20200619_07/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200619_07/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_07/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200619_07/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_07/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_07/new_prokaryote_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_07/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_07/new_prokaryote_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200619_08 -g cuda:0 -e 50 -b 32 -r 0.003
python tf_running.py -c ./output/tf_deeptfactor_20200619_08/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200619_08/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_08/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200619_08/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_08/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_08/new_prokaryote_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_08/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_08/new_prokaryote_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200619_09 -g cuda:0 -e 50 -b 64 -r 0.0001
python tf_running.py -c ./output/tf_deeptfactor_20200619_09/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200619_09/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_09/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200619_09/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_09/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_09/new_prokaryote_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_09/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_09/new_prokaryote_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200619_10 -g cuda:0 -e 50 -b 64 -r 0.0003
python tf_running.py -c ./output/tf_deeptfactor_20200619_10/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200619_10/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_10/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200619_10/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_10/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_10/new_prokaryote_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_10/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_10/new_prokaryote_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200619_11 -g cuda:0 -e 50 -b 64 -r 0.001
python tf_running.py -c ./output/tf_deeptfactor_20200619_11/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200619_11/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_11/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200619_11/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_11/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_11/new_prokaryote_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_11/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_11/new_prokaryote_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200619_12 -g cuda:0 -e 50 -b 64 -r 0.003
python tf_running.py -c ./output/tf_deeptfactor_20200619_12/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200619_12/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_12/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200619_12/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_12/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_12/new_prokaryote_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_12/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_12/new_prokaryote_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200619_13 -g cuda:0 -e 50 -b 128 -r 0.0001
python tf_running.py -c ./output/tf_deeptfactor_20200619_13/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200619_13/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_13/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200619_13/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_13/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_13/new_prokaryote_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_13/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_13/new_prokaryote_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200619_14 -g cuda:0 -e 50 -b 128 -r 0.0003
python tf_running.py -c ./output/tf_deeptfactor_20200619_14/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200619_14/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_14/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200619_14/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_14/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_14/new_prokaryote_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_14/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_14/new_prokaryote_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200619_15 -g cuda:0 -e 50 -b 128 -r 0.001
python tf_running.py -c ./output/tf_deeptfactor_20200619_15/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200619_15/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_15/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200619_15/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_15/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_15/new_prokaryote_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_15/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_15/new_prokaryote_nontf -g cuda:0

python tf_training.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTFSeq_20200601.fasta -o ./output/tf_deeptfactor_20200619_16 -g cuda:0 -e 50 -b 128 -r 0.003
python tf_running.py -c ./output/tf_deeptfactor_20200619_16/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200619_16/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_16/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_20200619_16/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_16/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_16/new_prokaryote_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_20200619_16/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_prokaryote_20200530.fasta -o ./output/tf_deeptfactor_20200619_16/new_prokaryote_nontf -g cuda:0