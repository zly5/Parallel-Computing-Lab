#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include<math.h>
int tickets = 20;
pthread_mutex_t mutex;

void *mythread1(void)
{
    while (1)
    {
        pthread_mutex_lock(&mutex); //给互斥量上锁
        if (tickets > 0)
        {
            usleep(1000);
            printf("ticketse1 sells ticket:%d\n", tickets--);
            pthread_mutex_unlock(&mutex); //给互斥量解锁
        }
        else
        {
            pthread_mutex_unlock(&mutex); //给互斥量解锁
            break;
        }
        sleep(1);
    }
    return (void *)0;
}
void *mythread2(void)
{
    while (1)
    {
        pthread_mutex_lock(&mutex); //给互斥量上锁
        if (tickets > 0)
        {
            usleep(1000);
            printf("ticketse2 sells ticket:%d\n", tickets--);
            pthread_mutex_unlock(&mutex); //给互斥量解锁
        }
        else
        {
            pthread_mutex_unlock(&mutex); //给互斥量解锁
            break;
        }
        sleep(1);
    }
    return (void *)0;
}

int main(int argc, const char *argv[])
{
    //int i = 0;
    int ret = 0;
    pthread_t id1, id2;

    ret = pthread_create(&id1, NULL, (void *)mythread1, NULL); //创建线程1
    if (ret)
    {
        printf("Create pthread error!\n");
        return 1;
    }

    ret = pthread_create(&id2, NULL, (void *)mythread2, NULL); //创建线程2
    if (ret)
    {
        printf("Create pthread error!\n");
        return 1;
    }

    pthread_join(id1, NULL); //等待线程结束
    pthread_join(id2, NULL);

    return 0;
}
*/


/*   //矩阵乘法
#include<stdio.h>   
#include<time.h>
#include<pthread.h>
#include<stdlib.h>
#include<unistd.h>
#include<memory.h>
 

#define M 600
#define N 600
int matrixA[M][N];
int matrixB[N][M];
int result[M][N];

void *func(void *arg);

const int NUM_THREADS =8 ;   //线程数
pthread_t tids[NUM_THREADS];  //线程
int L;                       //每个线程计算的块大小

void makeRandomMatrix_A()  //生成矩阵
{
    srand(time(NULL));
    int i, j;
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            matrixA[i][j] = rand() % 10 + 1;
        }
    }
}

void makeRandomMatrix_B()  //生成矩阵
{
    srand(time(NULL));
    int i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            matrixB[i][j] = rand() % 10 + 1;
        }
    }
}

//子线程函数 
void *func(void *arg)                                  
{
    int s=*(int *)arg;      //接收传入的参数（此线程从哪一行开始计算）
    int t=s+L;              //线程算到哪一行为止
    for(int i=s;i<t;i++)                                 
        for(int j=0;j<M;j++)
            for(int k=0;k<N;k++)
                result[i][j]+=matrixA[i][k]*matrixB[k][j];                               
}

void fp(){                              //串多线程函数
    int i;
    int j = 0;    
    int t[NUM_THREADS];    //传参索引
    L = M / NUM_THREADS;  //按设置的线程数分配工作块（单个线程所要计算的行数L）

    for(i=0;i<M;i+=L)
        {
            t[j] = i;
            if (pthread_create(&tids[j], NULL, func, (void *)&(t[j]))) //产生线程，去完成矩阵相乘的部分工作量
            {
                perror("pthread_create");
                exit(1);
            }
            j++;
       } 

    for(i=0;i<NUM_THREADS;i++)
        pthread_join(tids[i],NULL);                         //等所有的子线程计算结束
}

void f(){                                                //串行程序函数
    int res[M][M]={0};                                  //保存矩阵相乘的结果。非全局变量一定要显示初始化为0,否则为随机的一个数
    for(int i=0;i<M;i++)                                 
        for(int j=0;j<M;j++)
            for(int k=0;k<N;k++)
                res[i][j]+=matrixA[i][k]*matrixB[k][j];               
}

int main()
{
    makeRandomMatrix_A();                                      //用随机数产生两个待相乘的矩阵，并分别存入两个文件中
    makeRandomMatrix_B();                                      //从两个文件中读出数据赋给matrixA和matrixB
    printf("Makeing matrix(%d*%d) & matrix(%d*%d)...\n",N,M,M,N);

    //串行计算
    clock_t start2=clock();                              //开始计时
    f();                                               //串行程序
    clock_t finish2=clock();                             //结束计算
    printf("串行 --- Running time=%f s\n", (double)(finish2 - start2) / CLOCKS_PER_SEC);

    //多线程计算
    clock_t start1=clock();                              //开始计时
    fp();                                               //多线程
    clock_t finish1=clock();                             //结束计算
    printf("%d threads --- Running time=%f s\n", NUM_THREADS,(double)(finish1 - start1) / CLOCKS_PER_SEC);    

    return 0;
}
