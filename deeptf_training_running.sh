python tf_training.py -enz ../../../SeqData/y-ome/DeepTFactor_Dataset_preparation/processedTFSeq_extended_20200527_1.fasta -nonenz ../../../SeqData/y-ome/DeepTFactor_Dataset_preparation/processedNonTFSeq_extended_20200527_1.fasta -o ./output/tf_deeptfactor_20200527_01 -g cuda:1  -e 50 -b 32 -r 3e-4
python tf_running.py -c ./output/tf_deeptfactor_20200527_01/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200527_01/prediction_result -g cuda:1
python tf_running.py -c ./output/tf_deeptfactor_20200527_01/checkpoint.pt -enz ../../../SeqData/y-ome/processedNonTF_DNA_Seq_extended_20200523.fasta -o ./output/tf_deeptfactor_20200527_01/nontf_dna_binding_seq_results -g cuda:1

python tf_training.py -enz ../../../SeqData/y-ome/DeepTFactor_Dataset_preparation/processedTFSeq_extended_20200527_1.fasta -nonenz ../../../SeqData/y-ome/DeepTFactor_Dataset_preparation/processedNonTFSeq_extended_20200527_1.fasta -o ./output/tf_deeptfactor_20200527_02 -g cuda:1  -e 50 -b 32 -r 1e-3
python tf_running.py -c ./output/tf_deeptfactor_20200527_02/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200527_02/prediction_result -g cuda:1
python tf_running.py -c ./output/tf_deeptfactor_20200527_02/checkpoint.pt -enz ../../../SeqData/y-ome/processedNonTF_DNA_Seq_extended_20200523.fasta -o ./output/tf_deeptfactor_20200527_02/nontf_dna_binding_seq_results -g cuda:1

python tf_training.py -enz ../../../SeqData/y-ome/DeepTFactor_Dataset_preparation/processedTFSeq_extended_20200527_1.fasta -nonenz ../../../SeqData/y-ome/DeepTFactor_Dataset_preparation/processedNonTFSeq_extended_20200527_1.fasta -o ./output/tf_deeptfactor_20200527_03 -g cuda:1  -e 50 -b 32 -r 3e-3
python tf_running.py -c ./output/tf_deeptfactor_20200527_03/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200527_03/prediction_result -g cuda:1
python tf_running.py -c ./output/tf_deeptfactor_20200527_03/checkpoint.pt -enz ../../../SeqData/y-ome/processedNonTF_DNA_Seq_extended_20200523.fasta -o ./output/tf_deeptfactor_20200527_03/nontf_dna_binding_seq_results -g cuda:1

python tf_training.py -enz ../../../SeqData/y-ome/DeepTFactor_Dataset_preparation/processedTFSeq_extended_20200527_1.fasta -nonenz ../../../SeqData/y-ome/DeepTFactor_Dataset_preparation/processedNonTFSeq_extended_20200527_1.fasta -o ./output/tf_deeptfactor_20200527_04 -g cuda:1  -e 50 -b 64 -r 3e-4
python tf_running.py -c ./output/tf_deeptfactor_20200527_04/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200527_04/prediction_result -g cuda:1
python tf_running.py -c ./output/tf_deeptfactor_20200527_04/checkpoint.pt -enz ../../../SeqData/y-ome/processedNonTF_DNA_Seq_extended_20200523.fasta -o ./output/tf_deeptfactor_20200527_04/nontf_dna_binding_seq_results -g cuda:1

python tf_training.py -enz ../../../SeqData/y-ome/DeepTFactor_Dataset_preparation/processedTFSeq_extended_20200527_1.fasta -nonenz ../../../SeqData/y-ome/DeepTFactor_Dataset_preparation/processedNonTFSeq_extended_20200527_1.fasta -o ./output/tf_deeptfactor_20200527_05 -g cuda:1  -e 50 -b 64 -r 1e-3
python tf_running.py -c ./output/tf_deeptfactor_20200527_05/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200527_05/prediction_result -g cuda:1
python tf_running.py -c ./output/tf_deeptfactor_20200527_05/checkpoint.pt -enz ../../../SeqData/y-ome/processedNonTF_DNA_Seq_extended_20200523.fasta -o ./output/tf_deeptfactor_20200527_05/nontf_dna_binding_seq_results -g cuda:1

python tf_training.py -enz ../../../SeqData/y-ome/DeepTFactor_Dataset_preparation/processedTFSeq_extended_20200527_1.fasta -nonenz ../../../SeqData/y-ome/DeepTFactor_Dataset_preparation/processedNonTFSeq_extended_20200527_1.fasta -o ./output/tf_deeptfactor_20200527_06 -g cuda:1  -e 50 -b 64 -r 3e-3
python tf_running.py -c ./output/tf_deeptfactor_20200527_06/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200527_06/prediction_result -g cuda:1
python tf_running.py -c ./output/tf_deeptfactor_20200527_06/checkpoint.pt -enz ../../../SeqData/y-ome/processedNonTF_DNA_Seq_extended_20200523.fasta -o ./output/tf_deeptfactor_20200527_06/nontf_dna_binding_seq_results -g cuda:1

python tf_training.py -enz ../../../SeqData/y-ome/DeepTFactor_Dataset_preparation/processedTFSeq_extended_20200527_1.fasta -nonenz ../../../SeqData/y-ome/DeepTFactor_Dataset_preparation/processedNonTFSeq_extended_20200527_1.fasta -o ./output/tf_deeptfactor_20200527_07 -g cuda:1  -e 50 -b 128 -r 3e-4
python tf_running.py -c ./output/tf_deeptfactor_20200527_07/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200527_07/prediction_result -g cuda:1
python tf_running.py -c ./output/tf_deeptfactor_20200527_07/checkpoint.pt -enz ../../../SeqData/y-ome/processedNonTF_DNA_Seq_extended_20200523.fasta -o ./output/tf_deeptfactor_20200527_07/nontf_dna_binding_seq_results -g cuda:1

python tf_training.py -enz ../../../SeqData/y-ome/DeepTFactor_Dataset_preparation/processedTFSeq_extended_20200527_1.fasta -nonenz ../../../SeqData/y-ome/DeepTFactor_Dataset_preparation/processedNonTFSeq_extended_20200527_1.fasta -o ./output/tf_deeptfactor_20200527_08 -g cuda:1  -e 50 -b 128 -r 1e-3
python tf_running.py -c ./output/tf_deeptfactor_20200527_08/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200527_08/prediction_result -g cuda:1
python tf_running.py -c ./output/tf_deeptfactor_20200527_08/checkpoint.pt -enz ../../../SeqData/y-ome/processedNonTF_DNA_Seq_extended_20200523.fasta -o ./output/tf_deeptfactor_20200527_08/nontf_dna_binding_seq_results -g cuda:1

python tf_training.py -enz ../../../SeqData/y-ome/DeepTFactor_Dataset_preparation/processedTFSeq_extended_20200527_1.fasta -nonenz ../../../SeqData/y-ome/DeepTFactor_Dataset_preparation/processedNonTFSeq_extended_20200527_1.fasta -o ./output/tf_deeptfactor_20200527_09 -g cuda:1  -e 50 -b 128 -r 3e-3
python tf_running.py -c ./output/tf_deeptfactor_20200527_09/checkpoint.pt -enz ./Dataset/ecoli_k12_mg1655.fasta -o ./output/tf_deeptfactor_20200527_09/prediction_result -g cuda:1
python tf_running.py -c ./output/tf_deeptfactor_20200527_09/checkpoint.pt -enz ../../../SeqData/y-ome/processedNonTF_DNA_Seq_extended_20200523.fasta -o ./output/tf_deeptfactor_20200527_09/nontf_dna_binding_seq_results -g cuda:1