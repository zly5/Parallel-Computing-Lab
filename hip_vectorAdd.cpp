#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

__global__ void vectorAdd(float *d_A,float  *d_B,float  *d_C,int numElements)
 {
     int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
     if(i<numElements)
     {
     d_C[i] = d_A[i] + d_B[i];
     }
 }

 int main(int argc,char **argv)
{

    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

//1.申请Host内存并初始化
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

//1.申请Device内存
    float *d_A = NULL;
    hipMalloc((void **)&d_A, size);
    float *d_B = NULL;
    hipMalloc((void **)&d_B, size);
    float *d_C = NULL;
    hipMalloc((void **)&d_C, size);

 //2.将两个向量从Host端提交到Device端
     hipMemcpy(d_A,h_A,size,hipMemcpyHostToDevice);
     hipMemcpy(d_B,h_B,size,hipMemcpyHostToDevice);
 
//3.调用hip核函数    
     int threadsPerBlock = 256;
     int blocksPerGrid =(numElements+ threadsPerBlock - 1) / threadsPerBlock;
     hipLaunchKernelGGL(vectorAdd,blocksPerGrid, threadsPerBlock,0,0,d_A,d_B,d_C,numElements);
     printf("HIP kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  
//4.将两个向量相乘的结果从Device端传回Host端
    hipMemcpy(h_C,d_C,size,hipMemcpyDeviceToHost);
    //对比CPU和GPU计算结果误差
   for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-8)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
 
//5.释放内存
      hipFree(d_A);
      hipFree(d_B);
      hipFree(d_C);
      free(h_A);
      free(h_B);
      free(h_C);
 
     return 0;
 }
