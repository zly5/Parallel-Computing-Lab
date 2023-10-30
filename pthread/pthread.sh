#!/bin/bash

gcc PThread-matrix_multiplication.c -o PThread-matrix_multiplication -pthread

gcc PThread-simple_instances.c -o PThread-simple_instances -pthread
    
gcc PThread-synchronization.c -o PThread-synchronization -pthread