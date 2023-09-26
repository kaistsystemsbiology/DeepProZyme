#DeepECtransformer
##Procedure

**Note**: 
This source code was developed in Linux, and has been tested in Ubuntu 16.04 with Python 3.6.

1. Clone the repository

        git clone https://github.com/kaistsystemsbiology/DeepProZyme.git

2. Create and activate virtual environment (takes 5~10 min)

        conda env create -f environment.yml
        conda activate deepectransformer

3. To use gpus properly, install the pytorch and cuda for your gpus. This code was tested on pytorch=1.7.0 using cuda version 10.2



##Example


- Run DeepECtransformer (takes < 1 min)

        python run_deepectransformer.py -i ./example/mdh_ecoli.fa -o ./example/results -g cpu -b 128 -cpu 2
        python run_deepectransformer.py -i ./example/mdh_ecoli.fa -o ./example/results -g cuda:3 -b 128 -cpu 2
