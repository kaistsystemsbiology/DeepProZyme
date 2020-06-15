python tf_training.py -enz ./Dataset/processedDNABinding_20200601.fasta -nonenz ./Dataset/processedNonTF_NonDNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_01 -g cuda:3  -e 50 -b 16 -r 0.0001
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_01/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200601_01/prediction_result -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_01/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_01/nontf_dna_binding_seq_results -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_01/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_01/new_bacteria_tf -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_01/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_01/new_bacteria_nontf -g cuda:3

python tf_training.py -enz ./Dataset/processedDNABinding_20200601.fasta -nonenz ./Dataset/processedNonTF_NonDNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_02 -g cuda:3  -e 50 -b 16 -r 0.0003
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_02/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200601_02/prediction_result -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_02/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_02/nontf_dna_binding_seq_results -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_02/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_02/new_bacteria_tf -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_02/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_02/new_bacteria_nontf -g cuda:3

python tf_training.py -enz ./Dataset/processedDNABinding_20200601.fasta -nonenz ./Dataset/processedNonTF_NonDNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_03 -g cuda:3  -e 50 -b 16 -r 0.001
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_03/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200601_03/prediction_result -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_03/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_03/nontf_dna_binding_seq_results -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_03/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_03/new_bacteria_tf -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_03/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_03/new_bacteria_nontf -g cuda:3

python tf_training.py -enz ./Dataset/processedDNABinding_20200601.fasta -nonenz ./Dataset/processedNonTF_NonDNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_04 -g cuda:3  -e 50 -b 16 -r 0.003
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_04/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200601_04/prediction_result -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_04/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_04/nontf_dna_binding_seq_results -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_04/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_04/new_bacteria_tf -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_04/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_04/new_bacteria_nontf -g cuda:3

python tf_training.py -enz ./Dataset/processedDNABinding_20200601.fasta -nonenz ./Dataset/processedNonTF_NonDNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_05 -g cuda:3  -e 50 -b 32 -r 0.0001
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_05/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200601_05/prediction_result -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_05/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_05/nontf_dna_binding_seq_results -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_05/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_05/new_bacteria_tf -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_05/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_05/new_bacteria_nontf -g cuda:3

python tf_training.py -enz ./Dataset/processedDNABinding_20200601.fasta -nonenz ./Dataset/processedNonTF_NonDNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_06 -g cuda:3  -e 50 -b 32 -r 0.0003
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_06/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200601_06/prediction_result -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_06/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_06/nontf_dna_binding_seq_results -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_06/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_06/new_bacteria_tf -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_06/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_06/new_bacteria_nontf -g cuda:3

python tf_training.py -enz ./Dataset/processedDNABinding_20200601.fasta -nonenz ./Dataset/processedNonTF_NonDNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_07 -g cuda:3  -e 50 -b 32 -r 0.001
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_07/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200601_07/prediction_result -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_07/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_07/nontf_dna_binding_seq_results -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_07/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_07/new_bacteria_tf -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_07/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_07/new_bacteria_nontf -g cuda:3

python tf_training.py -enz ./Dataset/processedDNABinding_20200601.fasta -nonenz ./Dataset/processedNonTF_NonDNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_08 -g cuda:3  -e 50 -b 32 -r 0.003
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_08/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200601_08/prediction_result -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_08/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_08/nontf_dna_binding_seq_results -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_08/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_08/new_bacteria_tf -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_08/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_08/new_bacteria_nontf -g cuda:3

python tf_training.py -enz ./Dataset/processedDNABinding_20200601.fasta -nonenz ./Dataset/processedNonTF_NonDNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_09 -g cuda:3  -e 50 -b 64 -r 0.0001
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_09/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200601_09/prediction_result -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_09/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_09/nontf_dna_binding_seq_results -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_09/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_09/new_bacteria_tf -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_09/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_09/new_bacteria_nontf -g cuda:3

python tf_training.py -enz ./Dataset/processedDNABinding_20200601.fasta -nonenz ./Dataset/processedNonTF_NonDNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_10 -g cuda:3  -e 50 -b 64 -r 0.0003
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_10/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200601_10/prediction_result -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_10/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_10/nontf_dna_binding_seq_results -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_10/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_10/new_bacteria_tf -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_10/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_10/new_bacteria_nontf -g cuda:3

python tf_training.py -enz ./Dataset/processedDNABinding_20200601.fasta -nonenz ./Dataset/processedNonTF_NonDNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_11 -g cuda:3  -e 50 -b 64 -r 0.001
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_11/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_vs_dna_11/prediction_result -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_11/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_11/nontf_dna_binding_seq_results -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_11/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_11/new_bacteria_tf -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_11/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_11/new_bacteria_nontf -g cuda:3

python tf_training.py -enz ./Dataset/processedDNABinding_20200601.fasta -nonenz ./Dataset/processedNonTF_NonDNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_12 -g cuda:3  -e 50 -b 64 -r 0.003
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_12/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_vs_dna_12/prediction_result -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_12/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_12/nontf_dna_binding_seq_results -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_12/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_12/new_bacteria_tf -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_12/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_12/new_bacteria_nontf -g cuda:3

python tf_training.py -enz ./Dataset/processedDNABinding_20200601.fasta -nonenz ./Dataset/processedNonTF_NonDNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_13 -g cuda:3  -e 50 -b 128 -r 0.0001
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_13/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_vs_dna_13/prediction_result -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_13/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_13/nontf_dna_binding_seq_results -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_13/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_13/new_bacteria_tf -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_13/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_13/new_bacteria_nontf -g cuda:3

python tf_training.py -enz ./Dataset/processedDNABinding_20200601.fasta -nonenz ./Dataset/processedNonTF_NonDNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_14 -g cuda:3  -e 50 -b 128 -r 0.0003
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_14/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_vs_dna_14/prediction_result -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_14/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_14/nontf_dna_binding_seq_results -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_14/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_14/new_bacteria_tf -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_14/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_14/new_bacteria_nontf -g cuda:3

python tf_training.py -enz ./Dataset/processedDNABinding_20200601.fasta -nonenz ./Dataset/processedNonTF_NonDNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_15 -g cuda:3  -e 50 -b 128 -r 0.001
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_15/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_vs_dna_15/prediction_result -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_15/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_15/nontf_dna_binding_seq_results -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_15/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_15/new_bacteria_tf -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_15/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_15/new_bacteria_nontf -g cuda:3

python tf_training.py -enz ./Dataset/processedDNABinding_20200601.fasta -nonenz ./Dataset/processedNonTF_NonDNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_16 -g cuda:3  -e 50 -b 128 -r 0.003
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_16/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_vs_dna_16/prediction_result -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_16/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_vs_dna_16/nontf_dna_binding_seq_results -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_16/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_16/new_bacteria_tf -g cuda:3
python tf_running.py -c ./output/tf_deeptfactor_vs_dna_16/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_vs_dna_16/new_bacteria_nontf -g cuda:3