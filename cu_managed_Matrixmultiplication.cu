#include <iostream>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
	
#define BLOCK_SIZE 16

const int MAXSUN = 1000;
__managed__ int a[MAXSUN * MAXSUN];
__managed__ int b[MAXSUN * MAXSUN];
__managed__ int c_gpu[MAXSUN * MAXSUN];
__managed__ int c_cpu[MAXSUN * MAXSUN];
	
__global__ void gpu_matrix_mult(int* a, int* b, int* c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (col < k && row < m)
    {
      for (int i = 0; i < n; i++)
       {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}
	
void cpu_matrix_mult(int* a, int* b, int* h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i)
    {
       for (int j = 0; j < k; ++j)
       {
           int tmp = 0.0;
           for (int h = 0; h < n; ++h)
           {
               tmp += a[i * n + h] * b[h * k + j];
           }
            h_result[i * k + j] = tmp;
        }
    }
}
	
int main(int argc, char const* argv[])
{
    int m = 200;
    int n = 200;
    int k = 200;
	
    cudaEvent_t start, stop_cpu, stop_gpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_cpu);
    cudaEventCreate(&stop_gpu);

    //初始化矩阵A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i * n + j] = 0 * rand() % 1024 + 1;
        }
    }
    //初始化矩阵B
    for (int i = 0; i < n; ++i) {
       for (int j = 0; j < k; ++j) {
            b[i * k + j] = 0 * rand() % 1024 + 1;
        }
    }

    cudaEventRecord(start);
    cudaEventQuery(start);
	
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	    
    gpu_matrix_mult << <dimGrid, dimBlock >> > (a, b, c_gpu, m, n, k);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
	
    cpu_matrix_mult(a, b, c_cpu, m, n, k);
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    float elapsed_time_cpu, elapsed_time_gpu;
    cudaEventElapsedTime(&elapsed_time_gpu, start, stop_gpu);
    cudaEventElapsedTime(&elapsed_time_cpu, stop_gpu, stop_cpu);
    printf("GPU Time = %g ms.\n", elapsed_time_gpu);
    printf("CPU Time = %g ms.\n", elapsed_time_cpu);

    cudaEventDestroy(start);
    cudaEventDestroy(stop_cpu);
    cudaEventDestroy(stop_gpu);

	
	
    int ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
        //检验GPU运算结果和CPU运算结果是否相等
           if (fabs(c_gpu[i * k + j] - c_cpu[i * k + j]) > (1.0e-10))
         {

              ok = 0;
         }
         //printf("\n");
       }
    }

    if (ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }

    return 0;
}
