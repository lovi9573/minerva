#!/bin/sh


for T in *cpu-mpi.py; do 
	echo $T
	mpirun --mca mpi_cuda_support 0 -n 4 python $T
done
