#include<stdio.h>  //hello
#include<omp.h>

int main()
{
    int nthreads,thread_id;
    printf("I am the main thread.\n");
    #omp_set_num_threads(32); 
    #pragma omp parallel private(nthreads,thread_id) 
    {
        nthreads=omp_get_num_threads();     
        thread_id=omp_get_thread_num();    
        printf("Helllo I am thread %d out of a team of %d\n",thread_id,nthreads);
    }
    printf("Here I am,back to the main thread.\n");
    return 0;
}
