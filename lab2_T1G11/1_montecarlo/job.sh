#!/bin/bash

#SBATCH --job-name=ex1
#SBATCH -p std
#SBATCH --output=out_montecarlo.out
#SBATCH --error=out_montecarlo.err
#SBATCH --tasks-per-node=1
#SBATCH --ntasks=12
#SBATCH --nodes=12
#SBATCH --time=00:05:00

module purge
module load gcc/13.3.0 openmpi/5.0.3

make >> make.out || exit 1

mpirun -np 12 montecarlo 4 100000000 10
