#include <stdio.h>   
#include <pthread.h>
#include <unistd.h>
#include <malloc.h>

void* thread(void *id){
        pthread_t newthid;

        newthid = pthread_self();
        printf("this is a new thread, thread ID is %u\n", newthid);
        return NULL;
}

int main(){
        int num_thread = 5;
        pthread_t *pt = (pthread_t *)malloc(sizeof(pthread_t) * num_thread);

        printf("main thread, ID is %u\n", pthread_self());
        for (int i = 0; i < num_thread; i++){
                if (pthread_create(&pt[i], NULL, thread, NULL) != 0){
                        printf("thread create failed!\n");
                        return 1;
                }
        }
        sleep(2);
        free(pt);
        return 0;
}
