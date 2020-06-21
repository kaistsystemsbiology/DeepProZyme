python cnn_training.py -t ./Dataset/DeepEC_v2_input_sequences.fa -o ./output/deepec_v2_16 -g cuda:1 -e 50 -b 512 -r 0.003 -p 5 -third False
python cnn_training.py -t ./Dataset/DeepEC_v2_input_sequences.fa -o ./output/deepec_v2_15 -g cuda:1 -e 50 -b 512 -r 0.001 -p 5 -third False
python cnn_training.py -t ./Dataset/DeepEC_v2_input_sequences.fa -o ./output/deepec_v2_14 -g cuda:1 -e 50 -b 512 -r 0.0003 -p 5 -third False
python cnn_training.py -t ./Dataset/DeepEC_v2_input_sequences.fa -o ./output/deepec_v2_13 -g cuda:1 -e 50 -b 512 -r 0.0001 -p 5 -third False
python cnn_training.py -t ./Dataset/DeepEC_v2_input_sequences.fa -o ./output/deepec_v2_12 -g cuda:1 -e 50 -b 256 -r 0.003 -p 5 -third False
python cnn_training.py -t ./Dataset/DeepEC_v2_input_sequences.fa -o ./output/deepec_v2_11 -g cuda:1 -e 50 -b 256 -r 0.001 -p 5 -third False