#!/bin/bash
module add miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
conda create -p conda_env/ pip python=3.7
source activate conda_env/
conda install pip numpy scipy libgfortran sympy ply funcsigs cython matplotlib seaborn pandas numexpr scikit-learn tornado accelerate Biopython dateutil
/home/data/nbc/misc-projects/Salo_NiMARE/nimare-paper/conda_env/bin/pip install pip -U
/home/data/nbc/misc-projects/Salo_NiMARE/nimare-paper/conda_env/bin/pip install ipython jupyterlab ipywidgets nibabel nilearn
/home/data/nbc/misc-projects/Salo_NiMARE/nimare-paper/conda_env/bin/pip install -e /home/data/nbc/misc-projects/Salo_NiMARE/NiMARE/
conda list > python_requirements.txt
