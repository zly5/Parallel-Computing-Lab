#!/bin/bash

nvcc OpenCL_vectorAdd.c -o OpenCL_vectorAdd -lOpenCL

nvcc OpenCL_Mixer.c -o OpenCL_Mixer -lOpenCL
