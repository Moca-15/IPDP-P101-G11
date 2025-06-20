#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out_partis_oacc_async.out
#SBATCH --error=out_partis_oacc_async.err
#SBATCH --time=00:05:00

module load nvhpc/24.9

make >> make.out >> 1

nsys profile ./partis_oacc_async 10000 0

