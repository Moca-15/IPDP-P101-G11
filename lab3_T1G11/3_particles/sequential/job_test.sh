#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out_partis_oacc_m.out
#SBATCH --error=out_partis_oacc_m.err
#SBATCH --time=01:05:00

module load nvhpc/24.9

make >> make.out >> 1

d=500
./partis_seq $d 0

d=5000
./partis_seq $d 0

d=50000
./partis_seq $d 0

d=500000
./partis_seq $d 0

d=5000000
./partis_seq $d 0

d=50000000
./partis_seq $d 0

d=500000000
./partis_seq $d 0
