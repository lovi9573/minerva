#!/bin/sh

LIBMPI=/usr/lib64/openmpi/lib/libmpi.so
TESTROOT=$HOME/git/myminerva/tests/

for T in $TESTROOT*cpu-mpi.py; do 
	echo $T
	#For OpenMpi
	#LD_PRELOAD=$LIBMPI mpiexec --mca mpi_cuda_support 0 -n 4 python $T
	mpiexec -n 16 python $T
done
