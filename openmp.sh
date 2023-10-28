#!/bin/bash

gcc OpenMP-simple_instances.c -o OpenMP-simple_instances -fopenmp

g++ OpenMP-matrix_multiplication.cpp -o OpenMP-matrix_multiplication -fopenmp
    
gcc OpenMP-Matrix_Vector_Multiplication.c -o OpenMP-Matrix_Vector_Multiplication -fopenmp