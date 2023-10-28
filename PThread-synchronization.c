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
