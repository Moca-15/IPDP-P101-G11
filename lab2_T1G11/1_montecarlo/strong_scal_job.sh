#!/bin/bash

#SBATCH --job-name=ex1
#SBATCH -p std
#SBATCH --output=out_montecarlo_strong.out
#SBATCH --error=out_montecarlo_strong.err
#-BATCH --cpus-per-task=1
#SBATCH --tasks-per-node=24
#SBATCH --nodes=8
#SBATCH --time=01:00:00

module purge
module load gcc/13.3.0 openmpi/5.0.3

make >> make.out || exit 1

echo "cores,e_ratio,r_ratio,err,time"

mpirun -np 1 montecarlo 10 100000000 10

for i in {12..192..12}
do
mpirun -np $i montecarlo 10 100000000 10
done
