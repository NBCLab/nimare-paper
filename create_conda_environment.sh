#!/bin/bash
module add miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
conda create -n nimare_env pip python=3.7
source activate -n nimare_env
conda install -n nimare_env pip numpy scipy libgfortran sympy ply funcsigs cython matplotlib seaborn pandas numexpr scikit-learn tornado accelerate Biopython dateutil
pip install pip -U
pip install ipython jupyter-notebook ipywidgets nibabel nilearn
pip install -e /home/data/nbc/misc-projects/Salo_NiMARE/NiMARE/
conda list > python_requirements.txt
