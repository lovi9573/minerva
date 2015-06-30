#!/bin/sh


for T in *cpu-mpi.py; do 
	echo $T
	mpirun -n 4 python $T
done
