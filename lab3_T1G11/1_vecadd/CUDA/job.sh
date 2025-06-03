#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out_vecadd_cuda.out
#SBATCH --error=out_vecadd_cuda.err
#SBATCH --time=00:05:00

module load nvhpc/24.9

make >> make.out >> 1

./vecadd_cuda 1000000


