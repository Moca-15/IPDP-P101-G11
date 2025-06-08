#!/bin/bash

#SBATCH --job-name=ex2
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out_matmul.out
#SBATCH --error=out_matmul.err
#SBATCH --time=00:05:00

module load nvhpc/24.9

make >> make.out || exit 1

./matmul 500 1
