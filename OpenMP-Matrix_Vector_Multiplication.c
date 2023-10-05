#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>

const int NUM_THREADS = 4; //设置线程数量
int N = 10000;
int M = 10000;
int mat[10000][10000]; //矩阵mat
int vec[10000], ans[10000]; //向量vec

void makeRandomMatrix()  //生成矩阵
{
    srand(time(NULL));
    int i, j;
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            mat[i][j] = rand() % 10 + 1;
        }
    }
}

void makeRandomVector() //生成向量
{
    srand(time(NULL));
    int i;
    for (i = 0; i < N; i++)
    {
        vec[i] = rand() % 10 + 1;
    }
}

void funy(int a[], int cur)  //计算矩阵和矢量乘的部分结果
{
    int i;
    for (i = 0; i < N; i++)
    {
        ans[cur] += a[i] * vec[i];
    }
}

void f()  //串行计算
{
    int i;
    for (i = 0; i < M; i++)
    {
        funy(mat[i], i);
    }
}

void fp() //并行计算
{
    int i;
    #pragma omp parallel for num_threads(NUM_THREADS)
        for (i = 0; i < M; i ++)
        {
            funy(mat[i], i);
        }
}

int main()
{
    printf("Makeing matrix(%d*%d) & vector(%d*1)...\n",N,M,N);
    makeRandomMatrix(); 
    makeRandomVector();
    double start_time = omp_get_wtime();
    f();
    double end_time = omp_get_wtime();
    printf("串行 --- Running time=%f s\n", end_time - start_time);
    start_time = omp_get_wtime();
    fp();
    end_time = omp_get_wtime();
    printf("%d threads --- Running time=%f s\n", NUM_THREADS,end_time - start_time);
    return 0;
}
