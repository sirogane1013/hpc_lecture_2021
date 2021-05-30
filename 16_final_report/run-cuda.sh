#! /bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=0:01:00
module load cuda openmpi
mpirun -np 16 ./cuda