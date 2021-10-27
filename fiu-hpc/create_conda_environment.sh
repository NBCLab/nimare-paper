#!/bin/bash
module add miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
conda create -p /home/data/nbc/misc-projects/Salo_NiMARE/conda_env/ pip python=3.7
source activate /home/data/nbc/misc-projects/Salo_NiMARE/conda_env/
conda install pip
/home/data/nbc/misc-projects/Salo_NiMARE/conda_env/bin/pip install pip -U
/home/data/nbc/misc-projects/Salo_NiMARE/conda_env/bin/pip install -r /home/data/nbc/misc-projects/Salo_NiMARE/binder/requirements.txt
