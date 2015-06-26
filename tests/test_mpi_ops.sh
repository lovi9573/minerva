#!/bin/sh


for T in *.mpi.py; do 
	echo $T
	mpirun -n 3 python $T
done
