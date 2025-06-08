#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out_vecadd_seq.out
#SBATCH --error=out_vecadd_seq.err
#SBATCH --time=00:05:00

make >> make.out >> 1

./vecadd_seq 50000000
