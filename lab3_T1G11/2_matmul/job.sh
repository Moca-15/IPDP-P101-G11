#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --output=out_matmul.out
#SBATCH --error=out_matmul.err
#SBATCH --time=00:15:00

module load nvhpc/24.9

make >> make.out || exit 1

./matmul 5000 1

