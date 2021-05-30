#! /bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=0:01:00
module purge
module load gcc intel-mpi
mpirun -np 4 ./omp