#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out_partis_seq.out
#SBATCH --error=out_partis_seq.err
#SBATCH --time=01:05:00

module load nvhpc/24.9

make >> make.out >> 1

d=1000
./partis_seq $d 1
