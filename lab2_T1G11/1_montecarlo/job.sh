#!/bin/bash

#SBATCH --job-name=ex1
#SBATCH -p std
#SBATCH --output=out_montecarlo.out
#SBATCH --error=out_montecarlo.err
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --time=00:05:00

module purge
module load gcc/13.3.0 openmpi/5.0.3

make >> make.out || exit 1

mpirun -np 4 montecarlo
