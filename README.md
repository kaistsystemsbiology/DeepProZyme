#Enzymatic Data Integration into GEM of Escherichia coli
#Simultaneous prediction of metabolic genes/reactions for metabolic engineering

##Procedure

**Note**: 
This source code was developed in Linux, and has been tested in Ubuntu 16.06 with Python 2.7.
It should be noted that Python 3.7 is currently not supported.

1. Clone the repository

        git clone https://anlito@bitbucket.org/anlito/engem.git

2. Create and activate virtual environment

        virtualenv venv
        source venv/bin/activate

    or you can use pre-built anaconda environemnt

        conda env create -f environment.yml
        conda activate engem_env

3. Install gurobipy

    In our case, we installed gurobipy in the root of a server, and created its symbolic link in venv:

        ln -s /usr/local/lib/python2.7/dist-packages/gurobipy/ $HOME/Engem/venv/lib/python2.7/site-packages/

        ln -s /usr/local/lib/python3.6/dist-packages/gurobipy ../../anaconda3/envs/engem/lib/python2.7/site-packages/

4. Change the directory

        cd engem

5. Install packages

    If you created the environment by conda, you can skip this step

        pip install pip --upgrade
        pip install -r requirements.txt

##Example

- Run modeling code with retrieving relavant data

        python anlito.py -o ./output -i ./input_data/iML1515.xml -v versionX -r

- Run modeling code without retrieving relavant data

        python anlito.py -o ./output -i ./input_data/iML1515.xml -v versionX

- Run analytic codes for constructed model

        chmod 773 ./AnalysisEcoMBEL.sh
        ./AnalysisEcoMBEL.sh

- Run targeting algorithm

        python runTargetingSimulation.py -o ./output/targeting_result/ -i ./output/REDU_models/REDU_iML1515_20190906_MILP_relieve05_1.xml -t EX_ac_e -c 8 -n 5 -k 1 -d 1 -a 1