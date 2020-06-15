python tf_training_in_dnabinding.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_01 -g cuda:0  -e 50 -b 16 -r 0.0001
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_01/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_in_dna_20200608_01/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_01/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_01/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_01/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_01/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_01/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_01/new_bacteria_nontf -g cuda:0

python tf_training_in_dnabinding.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_02 -g cuda:0  -e 50 -b 16 -r 0.0003
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_02/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_in_dna_20200608_02/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_02/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_02/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_02/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_02/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_02/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_02/new_bacteria_nontf -g cuda:0

python tf_training_in_dnabinding.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_03 -g cuda:0  -e 50 -b 16 -r 0.001
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_03/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_in_dna_20200608_03/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_03/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_03/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_03/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_03/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_03/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_03/new_bacteria_nontf -g cuda:0

python tf_training_in_dnabinding.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_04 -g cuda:0  -e 50 -b 16 -r 0.003
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_04/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_in_dna_20200608_04/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_04/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_04/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_04/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_04/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_04/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_04/new_bacteria_nontf -g cuda:0

python tf_training_in_dnabinding.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_05 -g cuda:0  -e 50 -b 32 -r 0.0001
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_05/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_in_dna_20200608_05/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_05/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_05/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_05/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_05/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_05/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_05/new_bacteria_nontf -g cuda:0

python tf_training_in_dnabinding.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_06 -g cuda:0  -e 50 -b 32 -r 0.0003
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_06/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_in_dna_20200608_06/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_06/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_06/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_06/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_06/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_06/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_06/new_bacteria_nontf -g cuda:0

python tf_training_in_dnabinding.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_07 -g cuda:0  -e 50 -b 32 -r 0.001
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_07/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_in_dna_20200608_07/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_07/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_07/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_07/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_07/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_07/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_07/new_bacteria_nontf -g cuda:0

python tf_training_in_dnabinding.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_08 -g cuda:0  -e 50 -b 32 -r 0.003
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_08/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_in_dna_20200608_08/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_08/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_08/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_08/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_08/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_08/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_08/new_bacteria_nontf -g cuda:0

python tf_training_in_dnabinding.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_09 -g cuda:0  -e 50 -b 64 -r 0.0001
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_09/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_in_dna_20200608_09/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_09/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_09/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_09/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_09/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_09/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_09/new_bacteria_nontf -g cuda:0

python tf_training_in_dnabinding.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_10 -g cuda:0  -e 50 -b 64 -r 0.0003
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_10/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_in_dna_20200608_10/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_10/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_10/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_10/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_10/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_10/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_10/new_bacteria_nontf -g cuda:0

python tf_training_in_dnabinding.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_11 -g cuda:0  -e 50 -b 64 -r 0.001
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_11/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_in_dna_20200608_11/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_11/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_11/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_11/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_11/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_11/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_11/new_bacteria_nontf -g cuda:0

python tf_training_in_dnabinding.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_12 -g cuda:0  -e 50 -b 64 -r 0.003
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_12/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_in_dna_20200608_12/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_12/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_12/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_12/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_12/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_12/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_12/new_bacteria_nontf -g cuda:0

python tf_training_in_dnabinding.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_13 -g cuda:0  -e 50 -b 128 -r 0.0001
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_13/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_in_dna_20200608_13/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_13/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_13/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_13/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_13/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_13/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_13/new_bacteria_nontf -g cuda:0

python tf_training_in_dnabinding.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_14 -g cuda:0  -e 50 -b 128 -r 0.0003
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_14/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_in_dna_20200608_14/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_14/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_14/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_14/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_14/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_14/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_14/new_bacteria_nontf -g cuda:0

python tf_training_in_dnabinding.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_15 -g cuda:0  -e 50 -b 128 -r 0.001
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_15/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_in_dna_20200608_15/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_15/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_15/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_15/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_15/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_15/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_15/new_bacteria_nontf -g cuda:0

python tf_training_in_dnabinding.py -enz ./Dataset/processedTFSeq_20200601.fasta -nonenz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_16 -g cuda:0  -e 50 -b 128 -r 0.003
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_16/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_in_dna_20200608_16/prediction_result -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_16/checkpoint.pt -enz ./Dataset/processedNonTF_DNAbinding_20200601.fasta -o ./output/tf_deeptfactor_in_dna_20200608_16/nontf_dna_binding_seq_results -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_16/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_TF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_16/new_bacteria_tf -g cuda:0
python tf_running.py -c ./output/tf_deeptfactor_in_dna_20200608_16/checkpoint.pt -enz ../../../SeqData/DeepTFactor_additional/new_seq_NonTF_bacteria_20200530.fasta -o ./output/tf_deeptfactor_in_dna_20200608_16/new_bacteria_nontf -g cuda:0