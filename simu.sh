#!/bin/sh
OMP_NUM_THREADS=8 mpiexec -n 16 ./ring_mono.out -o result/temp/snap -T 1000 -d 1024 -t 0.2 -i result/temp/snap00000.dat --log_file result/temp/log
