#!/bin/sh


for T in *.mpi.py; do 
	echo $T
	mpirun -n 4 python $T
done
