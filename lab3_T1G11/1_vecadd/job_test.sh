#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out_vecadd_e1.out
#SBATCH --error=out_vecadd_e1.err
#SBATCH --time=00:05:00

module load nvhpc/24.9

make -C ./seq  >> make.out >> 1
make -C ./CUDA >> make.out >> 1

d=500
./seq/vecadd_seq $d
./CUDA/vecadd_cuda $d
echo ">>>>>>>>>>>>>>>>>>>>>>>>"
d=5000
./seq/vecadd_seq $d
./CUDA/vecadd_cuda $d
echo ">>>>>>>>>>>>>>>>>>>>>>>>"
d=50000
./seq/vecadd_seq $d
./CUDA/vecadd_cuda $d
echo ">>>>>>>>>>>>>>>>>>>>>>>>"
d=500000
./seq/vecadd_seq $d
./CUDA/vecadd_cuda $d
echo ">>>>>>>>>>>>>>>>>>>>>>>>"
d=5000000
./seq/vecadd_seq $d
./CUDA/vecadd_cuda $d
echo ">>>>>>>>>>>>>>>>>>>>>>>>"
d=50000000
./seq/vecadd_seq $d
./CUDA/vecadd_cuda $d
echo ">>>>>>>>>>>>>>>>>>>>>>>>"
d=500000000
./seq/vecadd_seq $d
./CUDA/vecadd_cuda $d

