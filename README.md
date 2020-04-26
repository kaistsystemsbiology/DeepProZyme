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

        python cnn_CAM_training.py -o ./output/cnn_CAM -g cuda:1 -e 30 -b 128 -r 1e-3 -third False

- Evaluate DeepEC

        python deepec_evaluate.py -o ./output/deepec_evaluation -g cuda:0 -b 1024 _c_cnn1 checkpoint_CNN1.pt -c_cnn1 checkpoint_CNN1.pt -c_cnn2 checkpoint_CNN2.pt -c_cnn3 checkpoint_CNN3.pt
