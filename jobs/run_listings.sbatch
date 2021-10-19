#!/bin/bash
#---Number of cores
#SBATCH -c 8

#---Job's name in SLURM system
#SBATCH -J nimare-paper

#---Error file
#SBATCH -e /home/data/nbc/misc-projects/Salo_NiMARE/jobs/slurm_stderr

#---Output file
#SBATCH -o /home/data/nbc/misc-projects/Salo_NiMARE/jobs/slurm_stdout

#---Queue name
#SBATCH --account iacc_nbc

#---Partition
#SBATCH -p IB_16C_96G
########################################################
export NPROCS=`echo $LSB_HOSTS | wc -w`
export OMP_NUM_THREADS=4
export NIMARE_DATA="/home/data/nbc/misc-projects/Salo_NiMARE/data/"
. $MODULESHOME/../global/profile.modules
source /home/data/nbc/misc-projects/Salo_NiMARE/activate_environment

jupyter-book clean /home/data/nbc/misc-projects/Salo_NiMARE/content/
jupyter-book build /home/data/nbc/misc-projects/Salo_NiMARE/content/
