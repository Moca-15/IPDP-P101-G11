#!/bin/bash

#SBATCH --job-name=ex1
#SBATCH -p std
#SBATCH --output=out_montecarlo_weak.out
#SBATCH --error=out_montecarlo_weak.err
#-BATCH --cpus-per-task=1
#SBATCH --tasks-per-node=24
#SBATCH --nodes=8
#SBATCH --time=01:00:00

module purge
module load gcc/13.3.0 openmpi/5.0.3

make >> make.out || exit 1
echo "cores,e_ratio,r_ratio,err,time"
for i in {0..192..12}
do
if (($i == 0))
then
	mpirun -np 1 montecarlo 10 100000000 10
else
	samples=$((100000000*$i/12*2 ))
	mpirun -np $i montecarlo 10 $samples 10
fi
done
