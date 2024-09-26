#!/bin/sh
OMP_NUM_THREADS=8 mpiexec -n 16 ./ring_mono.out -o result/d_1024/snap -T 1000 -d 1024 -i result/d_1024/snap00155.dat --log_file result/d_1024/log2
