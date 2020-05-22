#DeepEC reconstruction
##Procedure

**Note**: 
This source code was developed in Linux, and has been tested in Ubuntu 16.06 with Python 3.7.

1. Clone the repository

        git clone https://anlito@bitbucket.org/anlito/deepec_2.git

2. Create and activate virtual environment

        conda env create -f environment.yml
        conda activate torch_env


##Example


- Train CNN1

        python cnn1_training.py -o ./output/cnn1 -g cuda:0 -e 30 -b 64 -r 1e-4 -p 3 -c checkpoint.pt 

- Train CNN2

        python cnn_training.py -t ./Dataset/processedUniRefSeq.fasta -o ./output/ec7_cnn3_01 -g cuda:0 -e 30 -b 128 -r 1e-3 -p 5 -third False

- Train CNN3

        python cnn_training.py -t ./Dataset/processedUniRefSeq.fasta -o ./output/ec7_cnn3_01 -g cuda:0 -e 30 -b 128 -r 1e-3 -p 5 -third False

- Train CAM

        python cnn3_CAM_training.py -t ./Dataset/processedUniRefSeq.fasta -o ./output/cnn_CAM -g cuda:1 -e 30 -b 128 -r 1e-3 -third False

- Evaluate DeepEC

        python deepec_evaluate.py -o ./output/deepec_evaluation -g cuda:0 -b 1024 -c1 ./output/cnn1_01/checkpoint.pt -c2 ./output/ec7_cnn2_01/checkpoint.pt -c3 ./output/ec7_cnn3_03/checkpoint.pt -t ./Dataset/processedUniRefSeq.fasta

- Train DeepTFactor

        python tf_training.py -enz ./Dataset/processedTFSeq_extended.fasta -nonenz ./Dataset/processedNonTFSeq_extended.fasta -o ./output/tf_depptfactor -g cuda:1  -e 30 -b 256 -r 1e-3

- Run DeepTFactor

        python tf_running.py -c ./trained_model/DeepTFactor_checkpoint.pt -enz ./Dataset/example_tf.fasta -o ./output/tf0/tf_depptfactor -g cuda:0

- Run DeepEC predicion

        python cnn_running.py -enz ./Dataset/example_tf.fasta -c ./output/ec7_cnn3_03/checkpoint.pt -o ./output/ec7_cnn3_03/prediction_result -g cuda:0