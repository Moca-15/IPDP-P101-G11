#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out_partis_oacc_uni_mem.out
#SBATCH --error=out_partis_oacc_uni_mem.err
#SBATCH --time=00:05:00

module load nvhpc/24.9

make managed >> make.out >> 1

d=500
./partis_oacc_uni_mem $d 0

d=5000
./partis_oacc_uni_mem $d 0

d=50000
./partis_oacc_uni_mem $d 0

d=500000
./partis_oacc_uni_mem $d 0

d=5000000
./partis_oacc_uni_mem $d 0

d=50000000
./partis_oacc_uni_mem $d 0

d=500000000
./partis_oacc_uni_mem $d 0
