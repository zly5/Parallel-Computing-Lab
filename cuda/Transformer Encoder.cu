
#include<bits/stdc++.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
// cuda API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
        }                                                                                          \
    } while (0)


#define batch_count 8
#define M 197
#define N 768

using data_type = float;
//定义输入和用于residual结构的张量
std::vector<std::vector<data_type>> tensor(batch_count, std::vector<data_type>(M* N));
std::vector<std::vector<data_type>> tensor_copy(batch_count, std::vector<data_type>(M* N));
//定义三个liner层的参数
std::vector<std::vector<data_type>> w1(batch_count, std::vector<data_type>(768 * 2304));
std::vector<std::vector<data_type>> b1(batch_count, std::vector<data_type>(197 * 2304));
std::vector<std::vector<data_type>> w2(batch_count, std::vector<data_type>(768 * 3072));
std::vector<std::vector<data_type>> b2(batch_count, std::vector<data_type>(197 * 3072));
std::vector<std::vector<data_type>> w3(batch_count, std::vector<data_type>(3072 * 768));
std::vector<std::vector<data_type>> b3(batch_count, std::vector<data_type>(197 * 768));
//定义QKV及其中间结果
std::vector<std::vector<data_type>> Q(batch_count * 8, std::vector<data_type>(197 * 96));
std::vector<std::vector<data_type>> K(batch_count * 8, std::vector<data_type>(197 * 96));
std::vector<std::vector<data_type>> V(batch_count * 8, std::vector<data_type>(197 * 96));
std::vector<std::vector<data_type>> QK(batch_count * 8, std::vector<data_type>(197 * 197));

//**********************************initial*******************************************
//初始化tensor数据
void tensor_initial(int input_dim, int output_dim)
{
    const int m = input_dim;
    const int n = output_dim;
    for (int b = 0; b < batch_count; b++)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                tensor[b][j * m + i] = (float)(rand() % 101) / 101;
            }
        }
}
//初始化liner层w和b数据
void liner_initial() {
    //第一个liner层（B*197*768————B*197*2304）
    const int m1 = 197;
    const int k1 = 768;
    const int n1 = 2304;
    for (int b = 0; b < batch_count; b++)
        for (int i = 0; i < k1; i++) {
            for (int j = 0; j < n1; j++) {
                w1[b][j * k1 + i] = (float)(rand() % 101) / 101;
            }
        }
    for (int b = 0; b < batch_count; b++)
        for (int i = 0; i < m1; i++) {
            for (int j = 0; j < n1; j++) {
                b1[b][j * m1 + i] = (float)(rand() % 101) / 101;
            }
        }
    //第二个liner层（B*197*768————B*197*3072）
    const int m2 = 197;
    const int k2 = 768;
    const int n2 = 3072;
    for (int b = 0; b < batch_count; b++)
        for (int i = 0; i < k2; i++) {
            for (int j = 0; j < n2; j++) {
                w2[b][j * k2 + i] = (float)(rand() % 101) / 101;
            }
        }
    for (int b = 0; b < batch_count; b++)
        for (int i = 0; i < m2; i++) {
            for (int j = 0; j < n2; j++) {
                b2[b][j * m2 + i] = (float)(rand() % 101) / 101;
            }
        }
    //第三个liner层（B*197*3072————B*197*768）
    const int m3 = 197;
    const int k3 = 3072;
    const int n3 = 768;
    for (int b = 0; b < batch_count; b++)
        for (int i = 0; i < k3; i++) {
            for (int j = 0; j < n3; j++) {
                w3[b][j * k3 + i] = (float)(rand() % 101) / 101;
            }
        }
    for (int b = 0; b < batch_count; b++)
        for (int i = 0; i < m3; i++) {
            for (int j = 0; j < n3; j++) {
                b3[b][j * m3 + i] = (float)(rand() % 101) / 101;
            }
        }
}
//初始化liner的w和b
void qkv_initial(int input_dim, int output_dim) {
    const int m1 = input_dim;
    const int n1 = output_dim;
    const int m_batch_count = batch_count * 8;
    for (int b = 0; b < m_batch_count; b++)
        for (int i = 0; i < m1; i++) {
            for (int j = 0; j < n1; j++) {
                Q[b][j * m1 + i] = (float)(rand() % 101) / 101;
                K[b][j * m1 + i] = (float)(rand() % 101) / 101;
                V[b][j * m1 + i] = (float)(rand() % 101) / 101;
            }
        }
}
//******************************MultiHeadAttention************************************
//计算b1=tensor*w1+b1，得到的b1（B×197×2304）为结果
void liner_1(int input_dim, int output_dim) {
    const int m = 197;
    const int n = output_dim;
    const int k = input_dim;
    const int lda = m;
    const int ldb = k;
    const int ldc = m;

    const data_type alpha = 1.0;
    const data_type beta = 1.0;

    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    data_type** d_A_array = nullptr;
    data_type** d_B_array = nullptr;
    data_type** d_C_array = nullptr;

    std::vector<data_type*> d_A(batch_count, nullptr);
    std::vector<data_type*> d_B(batch_count, nullptr);
    std::vector<data_type*> d_C(batch_count, nullptr);

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&d_A[i]), sizeof(data_type) * tensor[i].size()));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&d_B[i]), sizeof(data_type) * w1[i].size()));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&d_C[i]), sizeof(data_type) * b1[i].size()));
    }

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&d_A_array), sizeof(data_type*) * batch_count));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&d_B_array), sizeof(data_type*) * batch_count));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&d_C_array), sizeof(data_type*) * batch_count));

    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaMemcpyAsync(d_A[i], tensor[i].data(), sizeof(data_type) * tensor[i].size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B[i], w1[i].data(), sizeof(data_type) * w1[i].size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_C[i], b1[i].data(), sizeof(data_type) * b1[i].size(),
            cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, d_A.data(), sizeof(data_type*) * batch_count,
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, d_B.data(), sizeof(data_type*) * batch_count,
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(data_type*) * batch_count,
        cudaMemcpyHostToDevice, stream));
   
    /* step 3: compute */
    cublasSgemmBatched(cublasH, transa, transb, m, n, k, &alpha, d_A_array, lda,
        d_B_array, ldb, &beta, d_C_array, ldc, batch_count);

    /* step 4: copy data to host */
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaMemcpyAsync(b1[i].data(), d_C[i], sizeof(data_type) * b1[i].size(),
            cudaMemcpyDeviceToHost, stream));
    }
    /* free resources */
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_B_array));
    CUDA_CHECK(cudaFree(d_C_array));
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    //CUDA_CHECK(cudaDeviceReset());

}
//将b1（B×197×2304）划分得到qkv（B×8×197×96）
void Permute_1() {
    
}
//qk=(q@transpose.k)×1/sqrt(Dk)
void multiplication_1() {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int m = 197;
    const int n = 197;
    const int k = 96;
    const int lda = m;
    const int ldb = n; //k转置了
    const int ldc = m;
    const int m_batch_count = batch_count*8;

    const data_type alpha = 1 / pow(24,-0.5); //放缩系数
    const data_type beta = 0;

    data_type** d_A_array = nullptr;
    data_type** d_B_array = nullptr;
    data_type** d_C_array = nullptr;

    std::vector<data_type*> d_A(m_batch_count, nullptr);
    std::vector<data_type*> d_B(m_batch_count, nullptr);
    std::vector<data_type*> d_C(m_batch_count, nullptr);

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_T; //k转置

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    for (int i = 0; i < m_batch_count; i++) {
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&d_A[i]), sizeof(data_type) * Q[i].size()));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&d_B[i]), sizeof(data_type) * K[i].size()));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&d_C[i]), sizeof(data_type) * QK[i].size()));
    }

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&d_A_array), sizeof(data_type*) * batch_count));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&d_B_array), sizeof(data_type*) * batch_count));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&d_C_array), sizeof(data_type*) * batch_count));

    for (int i = 0; i < m_batch_count; i++) {
        CUDA_CHECK(cudaMemcpyAsync(d_A[i], Q[i].data(), sizeof(data_type) * Q[i].size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B[i], K[i].data(), sizeof(data_type) * K[i].size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_C[i], QK[i].data(), sizeof(data_type) * QK[i].size(),
            cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, d_A.data(), sizeof(data_type*) * batch_count,
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, d_B.data(), sizeof(data_type*) * batch_count,
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(data_type*) * batch_count,
        cudaMemcpyHostToDevice, stream));
    /* step 3: compute */
    cublasSgemmBatched(cublasH, transa, transb, m, n, k, &alpha, d_A_array, lda,
        d_B_array, ldb, &beta, d_C_array, ldc, batch_count);

    /* step 4: copy data to host */
    for (int i = 0; i < m_batch_count; i++) {
        CUDA_CHECK(cudaMemcpyAsync(QK[i].data(), d_C[i], sizeof(data_type) * QK[i].size(),
            cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    /* free resources */
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_B_array));
    CUDA_CHECK(cudaFree(d_C_array));
    for (int i = 0; i < m_batch_count; i++) {
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    //CUDA_CHECK(cudaDeviceReset());
}
//对qk进行softmax操作
void softmax() {
    const int m = 197;
    const int n = 197;
    const int count = batch_count * 8 * 197;
    float s[count] = {};
    //归约
    int t = 0;
    for (int b = 0; b < batch_count*8; b++)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                s[t]= s[t]+ exp(QK[b][j * m + i]);
            }
            t++;
        }
    //求softmax
    t = 0;
    for (int b = 0; b < batch_count * 8; b++)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                QK[b][j * m + i] = exp(QK[b][j * m + i]) / s[t];
            }
            t++;
        }
}
//计算qk@v,将结果存到q中
void multiplication_2() {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int m = 197;
    const int n = 96;
    const int k = 197;
    const int lda = m;
    const int ldb = k; 
    const int ldc = m;
    const int m_batch_count = batch_count * 8;

    const data_type alpha = 1.0;
    const data_type beta = 0;

    data_type** d_A_array = nullptr;
    data_type** d_B_array = nullptr;
    data_type** d_C_array = nullptr;

    std::vector<data_type*> d_A(m_batch_count, nullptr);
    std::vector<data_type*> d_B(m_batch_count, nullptr);
    std::vector<data_type*> d_C(m_batch_count, nullptr);

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N; 

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    for (int i = 0; i < m_batch_count; i++) {
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&d_A[i]), sizeof(data_type) * QK[i].size()));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&d_B[i]), sizeof(data_type) * V[i].size()));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&d_C[i]), sizeof(data_type) * Q[i].size()));
    }

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&d_A_array), sizeof(data_type*) * batch_count));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&d_B_array), sizeof(data_type*) * batch_count));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&d_C_array), sizeof(data_type*) * batch_count));

    for (int i = 0; i < m_batch_count; i++) {
        CUDA_CHECK(cudaMemcpyAsync(d_A[i], QK[i].data(), sizeof(data_type) * QK[i].size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B[i], V[i].data(), sizeof(data_type) * V[i].size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_C[i], Q[i].data(), sizeof(data_type) * Q[i].size(),
            cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, d_A.data(), sizeof(data_type*) * batch_count,
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, d_B.data(), sizeof(data_type*) * batch_count,
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(data_type*) * batch_count,
        cudaMemcpyHostToDevice, stream));
    /* step 3: compute */
    cublasSgemmBatched(cublasH, transa, transb, m, n, k, &alpha, d_A_array, lda,
        d_B_array, ldb, &beta, d_C_array, ldc, batch_count);

    /* step 4: copy data to host */
    for (int i = 0; i < m_batch_count; i++) {
        CUDA_CHECK(cudaMemcpyAsync(Q[i].data(), d_C[i], sizeof(data_type) * Q[i].size(),
            cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    /* free resources */
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_B_array));
    CUDA_CHECK(cudaFree(d_C_array));
    for (int i = 0; i < m_batch_count; i++) {
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    //CUDA_CHECK(cudaDeviceReset());
}
//将q（B×8×197×96）转化为B×197×768赋值给tensor
void Permute_2() {

}
//*************************************MLP******************************************
//计算b2=tensor*w2+b2，得到的b2（B×197×3072）为结果
void liner_2(int input_dim, int output_dim) {
    const int m = 197;
    const int n = output_dim;
    const int k = input_dim;
    const int lda = m;
    const int ldb = k;
    const int ldc = m;

    const data_type alpha = 1.0;
    const data_type beta = 1.0;

    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    data_type** d_A_array = nullptr;
    data_type** d_B_array = nullptr;
    data_type** d_C_array = nullptr;

    std::vector<data_type*> d_A(batch_count, nullptr);
    std::vector<data_type*> d_B(batch_count, nullptr);
    std::vector<data_type*> d_C(batch_count, nullptr);

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&d_A[i]), sizeof(data_type) * tensor[i].size()));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&d_B[i]), sizeof(data_type) * w2[i].size()));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&d_C[i]), sizeof(data_type) * b2[i].size()));
    }

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&d_A_array), sizeof(data_type*) * batch_count));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&d_B_array), sizeof(data_type*) * batch_count));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&d_C_array), sizeof(data_type*) * batch_count));

    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaMemcpyAsync(d_A[i], tensor[i].data(), sizeof(data_type) * tensor[i].size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B[i], w2[i].data(), sizeof(data_type) * w2[i].size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_C[i], b2[i].data(), sizeof(data_type) * b2[i].size(),
            cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, d_A.data(), sizeof(data_type*) * batch_count,
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, d_B.data(), sizeof(data_type*) * batch_count,
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(data_type*) * batch_count,
        cudaMemcpyHostToDevice, stream));

    /* step 3: compute */
    cublasSgemmBatched(cublasH, transa, transb, m, n, k, &alpha, d_A_array, lda,
        d_B_array, ldb, &beta, d_C_array, ldc, batch_count);

    /* step 4: copy data to host */
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaMemcpyAsync(b2[i].data(), d_C[i], sizeof(data_type) * b2[i].size(),
            cudaMemcpyDeviceToHost, stream));
    }
    /* free resources */
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_B_array));
    CUDA_CHECK(cudaFree(d_C_array));
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    //CUDA_CHECK(cudaDeviceReset());
}
//对b2进行GELU操作
__global__ void gelu(float* x, int n)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    if (ix < n)
        x[ix] = 0.5 * x[ix] * (1 + tanh(sqrt(2 / 3.1415926) + 0.004715 * pow(x[ix], 3)));
        
}
void GELU() {

    std::vector<data_type*> d_A(batch_count, nullptr);

    cudaStream_t stream = NULL;

    /* copy data to device */
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&d_A[i]), sizeof(data_type) * b2[i].size()));
        CUDA_CHECK(
            cudaMemcpyAsync(d_A[i], b2[i].data(), sizeof(data_type) * QK[i].size(), cudaMemcpyHostToDevice, stream));
        
        gelu << <608, 1024 >> > (d_A[i], QK[i].size());

        CUDA_CHECK(cudaMemcpyAsync(b2[i].data(), d_A[i], sizeof(data_type) * b1[i].size(),
            cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* free resources */
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaFree(d_A[i]));
    }
    //CUDA_CHECK(cudaStreamDestroy(stream));

    //CUDA_CHECK(cudaDeviceReset());
}
//计算b3=b2*w3+b3，得到的b3（B×197×768）为结果
void liner_3(int input_dim, int output_dim) {
    const int m = 197;
    const int n = output_dim;
    const int k = input_dim;
    const int lda = m;
    const int ldb = k;
    const int ldc = m;

    const data_type alpha = 1.0;
    const data_type beta = 1.0;

    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    data_type** d_A_array = nullptr;
    data_type** d_B_array = nullptr;
    data_type** d_C_array = nullptr;

    std::vector<data_type*> d_A(batch_count, nullptr);
    std::vector<data_type*> d_B(batch_count, nullptr);
    std::vector<data_type*> d_C(batch_count, nullptr);

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&d_A[i]), sizeof(data_type) * b2[i].size()));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&d_B[i]), sizeof(data_type) * w3[i].size()));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&d_C[i]), sizeof(data_type) * b3[i].size()));
    }

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&d_A_array), sizeof(data_type*) * batch_count));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&d_B_array), sizeof(data_type*) * batch_count));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&d_C_array), sizeof(data_type*) * batch_count));

    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaMemcpyAsync(d_A[i], b2[i].data(), sizeof(data_type) * b2[i].size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B[i], w3[i].data(), sizeof(data_type) * w3[i].size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_C[i], b3[i].data(), sizeof(data_type) * b3[i].size(),
            cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, d_A.data(), sizeof(data_type*) * batch_count,
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, d_B.data(), sizeof(data_type*) * batch_count,
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(data_type*) * batch_count,
        cudaMemcpyHostToDevice, stream));

    /* step 3: compute */
    cublasSgemmBatched(cublasH, transa, transb, m, n, k, &alpha, d_A_array, lda,
        d_B_array, ldb, &beta, d_C_array, ldc, batch_count);

    /* step 4: copy data to host */
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaMemcpyAsync(b3[i].data(), d_C[i], sizeof(data_type) * b3[i].size(),
            cudaMemcpyDeviceToHost, stream));
    }
    /* free resources */
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_B_array));
    CUDA_CHECK(cudaFree(d_C_array));
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

   // CUDA_CHECK(cudaDeviceReset());

}
//*******************************Transformer Encoder**************************************
//对输入的tensor进行LN处理
__global__ void ss(float* x, int n, float avg) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    if (ix < n)
        x[ix] = pow(x[ix] - avg, 2);
}
__global__ void ln(float* x, int n, float avg, float S) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    if (ix < n)
        x[ix] = (x[ix]-avg)/sqrt(S+ 1e-5);
}
void LayerNorm() {
    std::vector<data_type*> d_A(batch_count, nullptr);
    std::vector<data_type*> d_A_copy(batch_count, nullptr);
    float sum = 0.0, S=0.0;
    cudaStream_t stream = NULL;

    cublasHandle_t handle;
    cublasCreate(&handle);

    /* copy data to device */

    for (int i = 0; i < batch_count; i++) {
        sum = 0.0;
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&d_A[i]), sizeof(data_type) * tensor[i].size()));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&d_A_copy[i]), sizeof(data_type) * tensor[i].size()));
        CUDA_CHECK(
            cudaMemcpyAsync(d_A[i], tensor[i].data(), sizeof(data_type) * tensor[i].size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(
            cudaMemcpyAsync(d_A_copy[i], tensor[i].data(), sizeof(data_type) * tensor[i].size(), cudaMemcpyHostToDevice, stream));
        //先求和
        cublasSasum(handle, 197 * 768, d_A[i], 1, &sum);
        //求方差
        ss << <160, 1024 >> > (d_A_copy[i], 197 * 768, sum / (197 * 768));
        cublasSasum(handle, 197 * 768, d_A_copy[i], 1, &S);
        //求LN
        ln << <160, 1024 >> > (d_A[i], 197 * 768, sum / (197 * 768), S );
        //将结果赋值给tensor
        CUDA_CHECK(cudaMemcpyAsync(tensor[i].data(), d_A[i], sizeof(data_type) * tensor[i].size(),
            cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* free resources */
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_A_copy[i]));
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    //CUDA_CHECK(cudaStreamDestroy(stream));

    //CUDA_CHECK(cudaDeviceReset());
}
//MultiHeadAttention block 
void MultiHeadAttention() {
    //计算b1=tensor*w1+b1，得到的b1（B×197×2304）为结果
    liner_1(768, 2304);
    //将b1（B×197*2304）划分得到qkv（3×B×8×197×96）
    Permute_1();
    //qk=(q@transpose.k)*1/sqrt(Dk)
    multiplication_1();
    //对qk进行softmax操作
    softmax();
    //计算qk@v,将结果存到q中
    multiplication_2();
    //将q（B×8×197×96）转化为B×197×768赋值给tensor
    Permute_2();
}
//MLP block 
void MLP() {
    //计算b2=tensor*w2+b2，得到的b2（B×197×3072）为结果
    liner_2(768,3072);
    //对b2进行GELU操作
    GELU();
    //计算b3=b2*w3+b3，得到的b3（B×197×768）为结果
    liner_3(3072,768);
    //将结果赋值给tensor
    tensor = b3;
}
//residual结构(tensor=tensor+tensor_copy)
void tensor_add() {
    //先化成一维vector
    std::vector<data_type> A(batch_count * 197 * 768);
    std::vector<data_type> B(batch_count * 197 * 768);
    const int m = 197;
    const int n = 768;
 
    int t1 = 0;
    for (int b = 0; b < batch_count; b++)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[t1] = tensor[b][j * m + i];
                B[t1] = tensor_copy[b][j * m + i];
                t1++;
            }
        }

    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const data_type alpha = 1.0;
    const int incx = 1;
    const int incy = 1;

    data_type* d_A = nullptr;
    data_type* d_B = nullptr;

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(data_type) * B.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice,
        stream));

    /* step 3: compute */
    CUBLAS_CHECK(cublasSaxpy(cublasH, A.size(), &alpha, d_A, incx, d_B, incy));

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(B.data(), d_B, sizeof(data_type) * B.size(), cudaMemcpyDeviceToHost,
        stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    //CUDA_CHECK(cudaDeviceReset());

    //将B赋值给tensor
    int t2 = 0;
    for (int b = 0; b < batch_count; b++)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                tensor[b][j * m + i]=B[t2];
                t2++;
            }
        }
}

//**********************************main()************************************************
int main(int argc, char** argv)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //tensor初始化
    tensor_initial(M,N);
    //tensor_copy初始化(用于residual结构)
    tensor_copy = tensor;
    //for(int b = 0; b < batch_count; b++)for (int i = 0; i < M; i++) {for (int j = 0; j < N; j++)std::cout << tensor[b][i * N + j] << " ";std::cout << std::endl;}
    //初始化liner的w和b
    liner_initial();
    //初始化q,k,v
    qkv_initial(197, 96);

    /* GPU warm up */
    for(int i=0;i<100;i++)
        LayerNorm();

    /* compute */
    cudaEventRecord(start, 0);

    LayerNorm();
    //MultiHeadAttention block 
    MultiHeadAttention();
    //tensor=tensor+tensor_copy
    tensor_add();
    tensor_copy = tensor;

    LayerNorm();
    //MLP block 
    MLP();
    //tensor=tensor+tensor_copy
    tensor_add();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Transformer Encoder time(Batch_Size=8): %.2f ms\n", elapsedTime);
    return 0;
}





