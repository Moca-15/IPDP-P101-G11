#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out_vecadd_oacc.out
#SBATCH --error=out_vecadd_oacc.err
#SBATCH --time=00:05:00

module load nvhpc/24.9

make >> make.out >> 1

./vecadd_oacc 10000


