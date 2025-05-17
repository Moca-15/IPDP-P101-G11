#!/bin/bash

#SBATCH --job-name=ex2
#SBATCH -p std
#SBATCH --output=out_controller.out
#SBATCH --error=out_controller.err
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --time=00:05:00

module purge
module load gcc/13.3.0 openmpi/5.0.3

make >> make.out || exit 1

mpirun -np 2 fc_mpi input_planes_test.txt 2 0 0
