#!/bin/bash
#---Number of CPUs
#SBATCH -c 1

#---Limit to 3GB
#SBATCH --mem-per-cpu=3000

#---Job's name in SLURM system
#SBATCH -J nimare-paper

#---Error file
#SBATCH -e /home/data/nbc/misc-projects/Salo_NiMARE/fiu-hpc/slurm_stderr

#---Output file
#SBATCH -o /home/data/nbc/misc-projects/Salo_NiMARE/fiu-hpc/slurm_stdout

#---Queue name
#SBATCH --account iacc_nbc

#---Partition
#SBATCH -p investor
########################################################
export NPROCS=`echo $LSB_HOSTS | wc -w`
# export OMP_NUM_THREADS=2
export NIMARE_DATA="/home/data/nbc/misc-projects/Salo_NiMARE/data/nimare-paper/data"
. $MODULESHOME/../global/profile.modules
source /home/data/nbc/misc-projects/Salo_NiMARE/fiu-hpc/activate_environment

jupyter-book build /home/data/nbc/misc-projects/Salo_NiMARE/content/
