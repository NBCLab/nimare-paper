#!/bin/bash
module add miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
conda create -p conda_env/ pip python=3.7
source activate conda_env/
conda install pip
/home/data/nbc/misc-projects/Salo_NiMARE/nimare-paper/conda_env/bin/pip install pip -U
/home/data/nbc/misc-projects/Salo_NiMARE/nimare-paper/conda_env/bin/pip install -r requirements.txt
