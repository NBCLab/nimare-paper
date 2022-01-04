#!/bin/bash
#---Number of cores
#SBATCH -c 1

#---Job's name in SLURM system
#SBATCH -J nimare-paper-profile

#---Error file
#SBATCH -e /home/data/nbc/misc-projects/Salo_NiMARE/fiu-hpc/profile.err

#---Output file
#SBATCH -o /home/data/nbc/misc-projects/Salo_NiMARE/fiu-hpc/profile.out

#---Queue name
#SBATCH --account iacc_nbc

#---Partition
#SBATCH -p IB_16C_96G
########################################################
export NPROCS=`echo $LSB_HOSTS | wc -w`
export OMP_NUM_THREADS=2
export NIMARE_DATA="/home/data/nbc/misc-projects/Salo_NiMARE/data/nimare-paper/data"
. $MODULESHOME/../global/profile.modules
source /home/data/nbc/misc-projects/Salo_NiMARE/fiu-hpc/activate_environment

python /home/data/nbc/misc-projects/Salo_NiMARE/fiu-hpc/profile_book.py
