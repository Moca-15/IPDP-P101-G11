#!/bin/bash

#SBATCH --job-name=ex2_stats
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=statistics.out
#SBATCH --error=statistics.err
#SBATCH --time=00:30:00

module load nvhpc/24.9

# 8 cpus per la sequential
make >> make.out || exit 1

echo "\tRunning for N = 5"
./matmul 5 1
echo "\tRunning for N = 25"
./matmul 25 1
echo "\tRunning for N = 100"
./matmul 100 1
echo "\tRunning for N = 500"
./matmul 500 1
echo "\tRunning for N = 1000"
./matmul 1000 1
echo "\tRunning for N = 5000"
./matmul 5000 1
echo "\tRunning for N = 10000"
./matmul 10000 1

