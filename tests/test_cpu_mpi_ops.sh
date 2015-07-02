#!/bin/sh

LIBMPI=/usr/lib64/openmpi/lib/libmpi.so

for T in *cpu-mpi.py; do 
	echo $T
	LD_PRELOAD=$LIBMPI mpirun --mca mpi_cuda_support 0 -n 4 python $T
done
