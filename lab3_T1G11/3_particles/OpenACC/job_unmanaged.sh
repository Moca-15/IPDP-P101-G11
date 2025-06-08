#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out_partis_oacc_um.out
#SBATCH --error=out_partis_oacc_um.err
#SBATCH --time=00:05:00

module load nvhpc/24.9

make unmanaged >> make.out >> 1

nsys profile ./partis_oacc_unmanaged 5000000 0

