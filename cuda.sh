#!/bin/bash

nvcc cu_vectorAdd.cu -o cu_vectorAdd

nvcc cu_managed_Matrixmultiplication.cu -o cu_managed_Matrixmultiplication

nvcc Transformer_Encoder.cu -o Transformer_Encoder -lcublas