#!/bin/bash
#---Number of cores
#SBATCH -c 8

#---Job's name in SLURM system
#SBATCH -J nimare-paper

#---Error file
#SBATCH -e /home/data/nbc/misc-projects/Salo_NiMARE/nimare-paper/err

#---Output file
#SBATCH -o /home/data/nbc/misc-projects/Salo_NiMARE/nimare-paper/out

#---Queue name
#SBATCH --account iacc_nbc

#---Partition
#SBATCH -p 16C_48G
########################################################
export NPROCS=`echo $LSB_HOSTS | wc -w`
export OMP_NUM_THREADS=4
. $MODULESHOME/../global/profile.modules
source /home/data/nbc/misc-projects/Salo_NiMARE/nimare-paper/activate_environment

python listings_figures_and_tables.py
